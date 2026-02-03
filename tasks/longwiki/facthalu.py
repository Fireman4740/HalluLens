# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import json
import hashlib

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Dict

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from tasks.longwiki import prompt_templates
from segtok.segmenter import split_single

from transformers import AutoTokenizer
from tasks.longwiki.longwiki_retrieval import LongWikiRetrieval, LongWikiDB
import tasks.longwiki.longwiki_utils as utils


@dataclass
class Claim:
    claim: str
    sentence: object
    refernce: Optional[str] = None
    topic: Optional[str] = None
    search_results: Optional[List[Dict[str, str]]] = None
    prompt: Optional[str] = None
    is_supported: Optional[bool] = None
    question: Optional[str] = None  # same as generation.prompt


@dataclass
class Sentence:
    sentence: str
    generation: object
    prompt: Optional[str]
    claims: Optional[List[Claim]] = None


@dataclass
class Generation:
    generation: str
    prompt: str
    sentences: Optional[List[Sentence]] = None
    abstain: Optional[bool] = None
    reference: Optional[str] = None
    topic: Optional[str] = None

    def __hash__(self) -> int:
        return hash(self.generation + self.prompt)

    def __eq__(self, other) -> bool:
        return self.generation == other.generation and self.prompt == other.prompt


# Use GPT-2 tokenizer instead of gated Llama model for compatibility with OpenRouter
# The tokenizer is only used for token counting, so any reasonably-sized tokenizer works
encoding = AutoTokenizer.from_pretrained("gpt2")


class FactHalu:
    def __init__(
        self,
        generations_file_path,  #: str | Path
        output_csv: str,
        abstain_evaluator: str = "meta-llama/Llama-3.1-70B-Instruct",
        refusal_evaluator: str | None = None,
        claim_extractor: str = "meta-llama/Llama-3.1-405B-Instruct-FP8",
        verifier: str = "meta-llama/Llama-3.1-405B-Instruct-FP8",
        k: int = 32,
        eval_cache_path="/data/facthalu_longform/.cache",
        db_path="/data/wiki_data/.cache/enwiki-20230401.db",
        args=None,
    ):
        self.args = args
        self.generations_file_path = generations_file_path
        self.output_csv = output_csv
        self.prepare_path()

        self.abstain_evaluator = abstain_evaluator
        self.refusal_evaluator = (
            refusal_evaluator if refusal_evaluator is not None else abstain_evaluator
        )
        self.claim_extractor = claim_extractor
        self.ref_src = "retrieval_relevant"
        self.verifier = verifier

        self.k = k

        self.generations = None
        self.db = LongWikiDB(db_path=db_path)

        eval_cache_path = (eval_cache_path or "").strip()
        if not eval_cache_path or eval_cache_path.startswith("#"):
            eval_cache_path = "data/longwiki/.cache"
        self.CACHE_BASE_PATH = eval_cache_path  # used for retrieval cache
        self.embedded_cache_path = (
            f"{self.CACHE_BASE_PATH}/embedding/embed_cache_all.pkl"
        )
        if not os.path.exists(f"{self.CACHE_BASE_PATH}/embedding/"):
            os.makedirs(f"{self.CACHE_BASE_PATH}/embedding/")

        print("Cache path:", self.embedded_cache_path)

    def prepare_path(self):
        self.refusal_path = str(self.output_csv).replace(".csv", "_abstain.jsonl")
        self.extracted_claims_path = str(self.output_csv).replace(
            ".csv", "_all_claims.jsonl"
        )
        self.extracted_claims_by_prompt_path = str(self.output_csv).replace(
            ".csv", "_all_claims_by_prompt.jsonl"
        )
        self.extracted_claims_prompts_path = str(self.output_csv).replace(
            ".csv", "_all_claims_prompts.jsonl"
        )
        self.parsed_claims_path = str(self.output_csv).replace(
            ".csv", "_all_parsed_claims.jsonl"
        )
        self.verification_path = str(self.output_csv).replace(
            ".csv", "_verification_results.jsonl"
        )

    def _hash_prompt(self, prompt: str) -> str:
        return hashlib.sha1(prompt.encode("utf-8")).hexdigest()

    def _read_prompt_cache(self, path: str):
        cache = {}
        if not os.path.exists(path):
            return cache
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt_hash = rec.get("prompt_hash")
                eval_res = rec.get("eval_res")
                prompt = rec.get("prompt")
                if not prompt_hash:
                    continue
                cache[prompt_hash] = {
                    "prompt_hash": prompt_hash,
                    "prompt": prompt,
                    "eval_res": eval_res,
                }
        return cache

    def _write_prompt_cache(self, path: str, cache: dict):
        with open(path, "w") as f:
            for prompt_hash in sorted(cache.keys()):
                f.write(json.dumps(cache[prompt_hash], ensure_ascii=False) + "\n")

    def _read_prompt_hashes(self, path: str):
        if not os.path.exists(path):
            return None
        hashes = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    return None
                prompt_hash = rec.get("prompt_hash")
                if not prompt_hash:
                    return None
                hashes.append(prompt_hash)
        return hashes

    def _write_prompt_hashes(self, path: str, hashes: List[str]):
        with open(path, "w") as f:
            for h in hashes:
                f.write(json.dumps({"prompt_hash": h}, ensure_ascii=False) + "\n")

    def run(self):
        """
        Evaluate longwiki from model error.
        Saves results to output_csv as jsonl with one line per prompt.
        """
        # check if result output_csv exists
        if os.path.exists(self.output_csv):
            print(f"Output file {self.output_csv} already exists. Exiting.\n")
            final_results_df = pd.read_csv(self.output_csv)
            final_results_df = utils.print_all_metrics(final_results_df, k=self.k)
            return

        now = time.time()

        ### [[STEP #0]] Load generations
        self.load_generations()

        ### [[STEP #1]] False Refusal Test
        print("\n[[Step 1]] False Refusal Test starts")
        no_abstains = self.eval_abstention()

        ### [[STEP #2]] Extract claims
        print("\n[[Step 2]] Extracting Claims starts")
        all_claims = self.extract_claims(no_abstains)

        if self.args.do_extract_only:
            print("Extract only mode. Exiting.")
            return

        # if there is no claim for a generation, mark it as abstain
        no_abstains = [
            generation for generation in self.generations if not generation.abstain
        ]

        ### [[STEP #3]] Verify claims
        print(f"\n[[Step 3]] Verifying Claims starts. Len: {len(all_claims)}")
        all_verification_responses = self.verify_claims(all_claims)

        for claim, verification_response in zip(all_claims, all_verification_responses):
            claim.is_supported = verification_response["is_supported"]

        ### [[[ STEP #4]]] Calculate metrics: precision, recall@k, f1, response ratio
        print(f"[[Step 4]] Calculating metrics")
        final_results = []
        for generation in no_abstains:
            for sentence in generation.sentences:
                if not sentence.claims:
                    final_results.append(
                        {
                            "prompt": generation.prompt,
                            "is_supported": None,
                            "claim": "no claims",
                            "sentence": sentence.sentence,
                            "title": generation.topic,
                        }
                    )
                else:
                    for claim in sentence.claims:
                        final_results.append(
                            {
                                "prompt": generation.prompt,
                                "is_supported": claim.is_supported,
                                "claim": claim.claim,
                                "sentence": sentence.sentence,
                                "title": generation.topic,
                            }
                        )
        print("---------------------------------")
        print("Abstain ratio:", "%.3f" % (1 - len(no_abstains) / len(self.generations)))
        final_results_df = pd.DataFrame(final_results)
        if final_results_df.empty or "prompt" not in final_results_df.columns:
            print("No claims found; skipping metric calculation.")
            final_results_df.to_csv(self.output_csv, index=False)
            return

        final_results_df = utils.calculate_all_metrics(final_results_df, k=self.k)
        final_results_df.to_csv(self.output_csv, index=False)
        print("---------------------------------")
        print(f"Saved detailed results in {self.output_csv}")
        print(f"Took {time.time() - now} seconds")
        print("---------------------------------")

    ##########################################################################################
    ##########################################################################################
    def load_generations(self):
        self.generations = []
        with open(self.generations_file_path, "r") as f:
            for line in f:
                l = json.loads(line)
                generation = Generation(
                    generation=l["generation"],
                    prompt=l["prompt"],
                )
                # Adding reference article here to replace search
                if self.ref_src == "default":
                    if l.get("reference", None) != None:
                        generation.reference = l["reference"]
                generation.topic = l["title"]
                self.generations.append(generation)

    def eval_abstention(self):
        refusal_path = self.refusal_path

        abstain_prompts = [
            prompt_templates.ABSTAIN_PROMPT.format(
                prompt=generation.prompt.strip(), generation=generation.generation
            ).strip()
            for generation in self.generations
        ]

        abstains_eval_raw = utils.read_eval_raw(refusal_path)
        if len(abstains_eval_raw) == len(abstain_prompts):
            print("Read from cache {}".format(refusal_path))
        else:
            abstains_eval_raw = utils.model_eval_step(
                self.refusal_evaluator,
                abstain_prompts,
                max_token=128,
                batch_size=64,
                max_workers=getattr(self.args, "eval_max_workers", 16),
            )
            utils.save_eval_raw(abstains_eval_raw, output_file=refusal_path)

        abstains_eval = utils.jsonify_ans(
            raw_responses=abstains_eval_raw,
            eval_prompts=abstain_prompts,
            evaluator=self.refusal_evaluator,
            key="is_knowledgeable",
        )

        for generation, abstain in zip(self.generations, abstains_eval):
            generation.abstain = not abstain["is_knowledgeable"]

        no_abstains = [
            generation for generation in self.generations if not generation.abstain
        ]
        return no_abstains

    def extract_claims(self, no_abstains: List[Generation]):
        extracted_claims_path = self.extracted_claims_path
        all_claim_extractions = []

        all_sentences = [
            sentence
            for g in no_abstains
            for sentence in make_claim_extraction_prompts(
                g, claim_extractor=self.claim_extractor
            )
        ]

        all_prompts = [a.prompt for a in all_sentences]
        prompt_hashes = [self._hash_prompt(p) for p in all_prompts]
        prompt_cache = self._read_prompt_cache(self.extracted_claims_by_prompt_path)
        cached_prompt_hashes = self._read_prompt_hashes(
            self.extracted_claims_prompts_path
        )
        legacy_claim_extractions = utils.read_eval_raw(extracted_claims_path)
        force_cache = bool(getattr(self.args, "force_cache", False))

        all_claim_extractions = [None] * len(all_sentences)

        if prompt_cache:
            for i, (prompt_hash, prompt) in enumerate(zip(prompt_hashes, all_prompts)):
                cached = prompt_cache.get(prompt_hash)
                if cached and cached.get("prompt") == prompt:
                    all_claim_extractions[i] = cached.get("eval_res")

        legacy_cache_valid = (
            cached_prompt_hashes is not None
            and cached_prompt_hashes == prompt_hashes
            and len(legacy_claim_extractions) == len(prompt_hashes)
        )
        legacy_prefix_valid = (
            cached_prompt_hashes is not None
            and len(legacy_claim_extractions) > 0
            and cached_prompt_hashes[: len(legacy_claim_extractions)]
            == prompt_hashes[: len(legacy_claim_extractions)]
        )

        if legacy_cache_valid:
            print("***** [2-1] Reading extracted claims from cache (validated)")
            for i, res in enumerate(legacy_claim_extractions):
                if all_claim_extractions[i] is None:
                    all_claim_extractions[i] = res
                if res is not None:
                    prompt_cache[prompt_hashes[i]] = {
                        "prompt_hash": prompt_hashes[i],
                        "prompt": all_prompts[i],
                        "eval_res": res,
                    }
        elif legacy_prefix_valid:
            print(
                f"***** [2-1] Resuming extraction from cache (validated prefix), starting from {len(legacy_claim_extractions)}\n"
            )
            for i, res in enumerate(legacy_claim_extractions):
                if all_claim_extractions[i] is None:
                    all_claim_extractions[i] = res
                if res is not None:
                    prompt_cache[prompt_hashes[i]] = {
                        "prompt_hash": prompt_hashes[i],
                        "prompt": all_prompts[i],
                        "eval_res": res,
                    }
        elif force_cache and legacy_claim_extractions:
            print(
                "***** [2-1] Force-cache enabled; using legacy cache by index (may be misaligned)"
            )
            for i in range(
                min(len(legacy_claim_extractions), len(all_claim_extractions))
            ):
                res = legacy_claim_extractions[i]
                all_claim_extractions[i] = res
                if res is not None:
                    prompt_cache[prompt_hashes[i]] = {
                        "prompt_hash": prompt_hashes[i],
                        "prompt": all_prompts[i],
                        "eval_res": res,
                    }

        mini_bsz = 100
        missing = [
            (i, prompt_hashes[i], all_prompts[i])
            for i, res in enumerate(all_claim_extractions)
            if res is None
        ]
        if not missing and len(all_claim_extractions) > 0 and not legacy_cache_valid:
            print("***** [2-1] Reading extracted claims from cache (prompt-hash)")
        if missing:
            print(
                f"***** [2-1] Extracting {len(missing)} missing prompts (cache hit: {len(all_claim_extractions) - len(missing)})"
            )
        for i in range(0, len(missing), mini_bsz):
            batch = missing[i : i + mini_bsz]
            batch_prompts = [p for _, _, p in batch]
            batch_results = utils.model_eval_step(
                self.claim_extractor,
                batch_prompts,
                max_token=512,
                batch_size=8,
                max_workers=getattr(self.args, "eval_max_workers", 16),
            )
            for (idx, prompt_hash, prompt), res in zip(batch, batch_results):
                all_claim_extractions[idx] = res
                if res is not None:
                    prompt_cache[prompt_hash] = {
                        "prompt_hash": prompt_hash,
                        "prompt": prompt,
                        "eval_res": res,
                    }
            utils.save_eval_raw(all_claim_extractions, output_file=extracted_claims_path)
            self._write_prompt_cache(
                self.extracted_claims_by_prompt_path, prompt_cache
            )
            self._write_prompt_hashes(
                self.extracted_claims_prompts_path, prompt_hashes
            )
            if i % 500 == 0:
                print(
                    f"Processed {min(i + 100, len(missing))} missing prompts. out of {len(missing)}"
                )

        utils.save_eval_raw(all_claim_extractions, output_file=extracted_claims_path)
        self._write_prompt_cache(self.extracted_claims_by_prompt_path, prompt_cache)
        self._write_prompt_hashes(self.extracted_claims_prompts_path, prompt_hashes)

        print("***** [2-2] Parsing extracted claims")
        all_claims = []
        deduplicate = defaultdict(set)
        if len(all_claim_extractions) != len(all_sentences):
            print(
                "***** [2-2] Warning: cache size mismatch; padding/truncating to align"
            )
            if len(all_claim_extractions) > len(all_sentences):
                all_claim_extractions = all_claim_extractions[: len(all_sentences)]
            else:
                all_claim_extractions.extend(
                    [""] * (len(all_sentences) - len(all_claim_extractions))
                )
            utils.save_eval_raw(
                all_claim_extractions, output_file=extracted_claims_path
            )

        for claim_extraction, sentence in zip(all_claim_extractions, all_sentences):
            if (
                (not claim_extraction)
                or claim_extraction.strip() == "No verifiable claim."
                or claim_extraction.strip() == "No available facts"
                or claim_extraction.strip() == "No available facts."
            ):
                sentence.claims = []
                continue

            parsed_claim_extraction = utils.parse_claim_extraction(
                claim_extraction, self.claim_extractor
            )

            sentence_claims = []
            for claim_text in parsed_claim_extraction:
                if (
                    claim_text.strip() != ""
                    and claim_text not in deduplicate[sentence.generation]
                ):
                    deduplicate[sentence.generation].add(claim_text)
                    claim = Claim(
                        claim=claim_text,
                        sentence=sentence,
                        refernce=sentence.generation.reference,
                        topic=sentence.generation.topic,
                        question=sentence.generation.prompt,
                    )
                    sentence_claims.append(claim)
                    all_claims.append(claim)

            sentence.claims = sentence_claims

        for generation in self.generations:
            if not deduplicate[
                generation
            ]:  # no claims for a generation -> also abstains
                generation.abstain = True

        all_claims_text = [str(c.claim) for c in all_claims]
        utils.save_eval_raw(all_claims_text, output_file=self.parsed_claims_path)

        return all_claims

    def verify_claims(self, all_claims: List[Claim]):
        verification_path = self.verification_path

        claim_verification_res = utils.read_eval_raw(verification_path)

        print("***** [3] Ref Src: ", self.ref_src)
        # 1. Prepare the prompt for verification
        retrieval_batch_size = int(os.getenv("RETRIEVAL_BATCH_SIZE", "64"))
        retrieval = LongWikiRetrieval(
            self.db,
            cache_base_path=self.CACHE_BASE_PATH,
            embed_cache_path=self.embedded_cache_path,
            retrieval_type="gtr-t5-large",
            batch_size=retrieval_batch_size,
        )
        questions = list(set([claim.question for claim in all_claims]))
        if retrieval.use_ner:
            retrieval.make_ner_cache(questions)
        precompute_env = os.getenv("VERIFY_PROMPT_PRECOMPUTE_QUERIES", "true")
        precompute_queries = str(precompute_env).lower() not in ("0", "false", "no")
        query_vectors = None
        if precompute_queries and all_claims:
            retrieval.load_encoder()
            query_batch_size = int(
                os.getenv("RETRIEVAL_QUERY_BATCH_SIZE", str(retrieval.batch_size or 32))
            )
            retrieval_queries = [
                f"{claim.topic} {claim.claim.strip()}" for claim in all_claims
            ]
            query_vectors = retrieval.encoder.encode(
                retrieval_queries,
                batch_size=query_batch_size,
                device=retrieval.encoder.device,
            )
        profile_env = os.getenv("VERIFY_PROMPT_PROFILE", "false")
        profile = str(profile_env).lower() in ("1", "true", "yes")
        if profile:
            import threading

            timing_lock = threading.Lock()
            timing = {"retrieval_s": 0.0, "format_s": 0.0, "n": 0}
        prompt_workers_env = os.getenv("VERIFY_PROMPT_MAX_WORKERS")
        if prompt_workers_env is not None:
            prompt_workers = max(1, int(prompt_workers_env))
        else:
            prompt_workers = max(1, min(8, getattr(self.args, "eval_max_workers", 16)))

        def _build_prompt(indexed_claim) -> str:
            idx, claim = indexed_claim
            query_vector = query_vectors[idx] if query_vectors is not None else None
            t0 = time.perf_counter() if profile else None
            passages = retrieval.get_topk_related_passages(
                topic=claim.topic,
                claim=claim.claim,
                question=claim.question,
                k=5,
                query_vector=query_vector,
            )
            t1 = time.perf_counter() if profile else None
            context_parts = []
            for _, psg in enumerate(reversed(passages)):
                context_parts.append(
                    "Title: {}\nText: {}\n\n".format(
                        psg["title"], psg["text"].replace("<s>", "").replace("</s>", "")
                    )
                )
            context = "".join(context_parts)
            prompt = prompt_templates.VERIFICATION_TEMPLATE_W_REFERENCE_RETRIEVAL.format(
                claim=claim.claim, reference=context
            )
            if profile:
                t2 = time.perf_counter()
                with timing_lock:
                    timing["retrieval_s"] += (t1 - t0) if (t0 is not None and t1 is not None) else 0.0
                    timing["format_s"] += (t2 - t1) if (t1 is not None) else 0.0
                    timing["n"] += 1
            return prompt

        indexed_claims = list(enumerate(all_claims))
        if prompt_workers == 1:
            for indexed_claim in tqdm(
                indexed_claims, desc="Preparing verification prompts"
            ):
                _, claim = indexed_claim
                claim.prompt = _build_prompt(indexed_claim)
        else:
            prompts = thread_map(
                _build_prompt,
                indexed_claims,
                max_workers=prompt_workers,
                desc="Preparing verification prompts",
            )
            for claim, prompt in zip(all_claims, prompts):
                claim.prompt = prompt
        print("***** Prepared all verification prompts")
        if profile and timing["n"] > 0:
            n = timing["n"]
            print(
                "***** Step 3 prompt prep timings (avg/claim): "
                f"retrieval={timing['retrieval_s'] / n:.3f}s, "
                f"format={timing['format_s'] / n:.3f}s, "
                f"total={(timing['retrieval_s'] + timing['format_s']) / n:.3f}s"
            )
        retrieval.flush_embed_cache()

        # 2. Verify the claims
        verification_prompts = [c.prompt for c in all_claims]
        if len(claim_verification_res) == len(all_claims):
            print(
                "***** [3] Reading verification results from cache {}\n".format(
                    verification_path
                )
            )
        else:
            for i in range(0, len(verification_prompts), 100):
                batch_prompts = verification_prompts[i : i + 100]
                batch_results = utils.model_eval_step(
                    self.verifier,
                    batch_prompts,
                    max_token=512,
                    batch_size=8,
                    max_workers=getattr(self.args, "eval_max_workers", 16),
                )
                claim_verification_res.extend(batch_results)
                utils.save_eval_raw(
                    claim_verification_res, output_file=verification_path
                )
            utils.save_eval_raw(claim_verification_res, output_file=verification_path)

        assert len(claim_verification_res) == len(all_claims)
        # 3. post process the verification result
        calim_verification_results = utils.jsonify_ans(
            raw_responses=claim_verification_res,
            eval_prompts=verification_prompts,
            evaluator=self.verifier,
            key="is_supported",
        )

        return calim_verification_results


def make_claim_extraction_prompts(
    generation: Generation, claim_extractor="meta-llama/Llama-3.1-405B-Instruct-FP8"
):
    """
    Given a model output
    - split into sentences
    - go para by para, always add the first sent of the para into context1
    - snippet = (context1 = 0-3 sentence) <SOS>Sent<EOS> (context2 = 0-1 sentence)
    Return list of {"prompt": prompt_text, "sentence": target_sentence}
    """
    sentences = []
    # split the text into sentences
    sentences_text = [x.strip() for x in split_single(generation.generation)]
    question = generation.prompt.replace("Answer in one paragraph.", "").strip()
    response = generation.generation.strip()

    if claim_extractor == "finetuned":
        for i, sentence in list(enumerate(sentences_text)):
            input = f"Questions:\n{question.strip()}\nResponse:\n{response.strip()}"
            snippet = input.replace(sentence, f"<SOS>{sentence}<EOS>")
            prompt_text = prompt_templates.EXTRACT_CLAIMS_TEMPLATE_FINETUNED.format(
                snippet, ""
            )
            sentences.append(
                Sentence(prompt=prompt_text, sentence=sentence, generation=generation)
            )
    else:
        for i, sentence in list(enumerate(sentences_text)):
            if len(sentence) < 5:
                continue
            context1 = " ".join(sentences_text[max(0, i - 3) : i])
            target_sentence = sentences_text[i]
            sentence = f"<SOS>{target_sentence.strip()}<EOS>"
            context2 = " ".join(sentences_text[i + 1 : i + 2])
            snippet = (
                f"{context1.strip()} {sentence.strip()} {context2.strip()}".strip()
            )
            prompt_text = prompt_templates.EXTRACT_CLAIMS_TEMPLATE.format(
                snippet=snippet, sentence=sentence
            )
            # check token
            prompt_len = len(encoding.encode(prompt_text))
            if prompt_len > 3500:
                context1 = " ".join(sentences_text[max(0, i - 2) : i])
                snippet = (
                    f"{context1.strip()} {sentence.strip()} {context2.strip()}".strip()
                )

                prompt_text = prompt_templates.EXTRACT_CLAIMS_SHORT_TEMPLATE.format(
                    snippet=snippet, sentence=sentence
                )

                if len(encoding.encode(prompt_text)) > 3500:
                    prompt_text = (
                        prompt_templates.EXTRACT_CLAIMS_EXTREME_SHORT_TEMPLATE.format(
                            snippet=snippet, sentence=sentence
                        )
                    )

                    if len(encoding.encode(prompt_text)) > 3500:
                        prompt_text = prompt_templates.EXTRACT_CLAIMS_EXTREME_EXTREME_SHORT_TEMPLATE.format(
                            snippet=snippet, sentence=sentence
                        )

                assert len(encoding.encode(prompt_text)) <= 3500

            sentences.append(
                Sentence(
                    prompt=prompt_text, sentence=target_sentence, generation=generation
                )
            )
    generation.sentences = sentences
    return sentences
