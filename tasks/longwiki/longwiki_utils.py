# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import time
import numpy as np
from utils import lm
from typing import List
from tqdm.contrib.concurrent import thread_map


def print_all_metrics(final_results_df, k=32):
    if "prompt" not in final_results_df.columns:
        if final_results_df.index.name == "prompt":
            final_results_df = final_results_df.reset_index()
        else:
            print("Missing 'prompt' column; skipping metric printing.")
            return

    overall_recall = final_results_df.groupby("prompt").recall.first().mean()
    overall_precision = final_results_df.groupby("prompt").precision.first().mean()
    overall_f1 = final_results_df.groupby("prompt").f1.first().mean()
    final_results_df["overall_recall"] = overall_recall
    final_results_df["overall_precision"] = overall_precision
    final_results_df["overall_f1"] = overall_f1

    med_n_claims = final_results_df.groupby("prompt").n_claims.first().median()
    print("Precision:", "%.3f" % overall_precision)
    print(f"Recall@{k}:", "%.3f" % overall_recall)
    print(f"F1@{k}", "%.3f" % overall_f1)
    print(f"med_n_claims", "%.3f" % med_n_claims)

def calculate_all_metrics(final_results_df, k=32):
    if "prompt" not in final_results_df.columns:
        if final_results_df.index.name == "prompt":
            final_results_df = final_results_df.reset_index()
        else:
            print("Missing 'prompt' column; skipping metric calculation.")
            return final_results_df

    try:
        grouped = final_results_df.groupby("prompt")
        precision = grouped.is_supported.mean()
        recall = grouped.is_supported.apply(lambda g: min(g.sum() / k, 1))
        denom = precision + recall
        f1 = (2 * precision * recall).div(denom.replace(0, np.nan)).fillna(0)

        final_results_df["precision"] = final_results_df["prompt"].map(precision)
        final_results_df["recall"] = final_results_df["prompt"].map(recall)
        final_results_df["f1"] = final_results_df["prompt"].map(f1)
        final_results_df["k"] = k
        final_results_df["n_claims"] = grouped.is_supported.transform("count")
    except KeyError:
        print("Missing 'prompt' column during metric calculation; skipping.")
        return final_results_df

    overall_recall = final_results_df.groupby("prompt").recall.first().mean()
    overall_precision = final_results_df.groupby("prompt").precision.first().mean()
    overall_f1 = final_results_df.groupby("prompt").f1.first().mean()

    final_results_df["overall_recall"] = overall_recall
    final_results_df["overall_precision"] = overall_precision
    final_results_df["overall_f1"] = overall_f1

    med_n_claims = final_results_df.groupby("prompt").n_claims.first().median()

    print("Precision:", "%.3f" % overall_precision)
    print(f"Recall@{k}:", "%.3f" % overall_recall)
    print(f"F1@{k}", "%.3f" % overall_f1)
    print(f"med_n_claims", "%.3f" % med_n_claims)
    
    return final_results_df

def f1_score(g):
    prec = g.precision.iloc[0]
    rec = g.recall.iloc[0]
    if (prec + rec) == 0:
        f1 = 0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    g["f1"] = f1
    return g

def remove_prefix(text:str , prefix:str):
    # for < python 3.9 
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def parse_claim_extraction(claim_extraction: List[str], claim_extractor="meta-llama/Llama-3.1-405B-Instruct-FP8"):
    res = []
    if claim_extractor == "finetuned":
        res = [x.strip() for x in claim_extraction.split("\n")]
    else:
        for claim in claim_extraction.split('\n'):
            if not claim.startswith('- ') or \
                remove_prefix(claim, '- ').strip() == '':
                continue
            claim_text = remove_prefix(claim, '- ').strip()
    
            if claim_text== "No available facts" or \
                claim_text == "No available facts.":
                continue

            res.append(claim_text)
    return res

def save_eval_raw(
        raw_eval_list: List[str],
        output_file):
    
    with open(output_file, "w") as f:
        for r in raw_eval_list:
            f.write(json.dumps({"eval_res": r}) + "\n")

def read_eval_raw(eval_raw_file):
    eval_raw_res = []
    if os.path.exists(eval_raw_file):
        with open(eval_raw_file, "r") as f:
            eval_raw_res = [json.loads(line)["eval_res"] for line in f]
    return eval_raw_res
    
def model_eval_step(evaluator, prompts, max_token=512, batch_size=16, max_workers=16, api_i=0):
    max_retries = int(os.getenv("EVAL_EMPTY_RETRIES", "3"))
    base_sleep = float(os.getenv("EVAL_RETRY_BASE_SECONDS", "1.0"))

    def _safe_generate(p):
        for attempt in range(max_retries + 1):
            try:
                return lm.generate(p, evaluator, i=api_i)
            except Exception as e:
                if attempt >= max_retries:
                    print(f"Eval request failed after retries: {e}")
                    return ""
                sleep_s = min(30.0, base_sleep * (2**attempt))
                time.sleep(sleep_s)

    eval_raw_res = thread_map(
        _safe_generate,
        prompts,
        max_workers=max_workers,
        desc=f"using OpenRouter {evaluator}",
    )
    return eval_raw_res

def jsonify_ans(raw_responses, eval_prompts, evaluator, key):

    def check_validity(gen):
        if not gen:
            return -1
        if '{{"{}":false}}'.format(key) in gen.lower():
            return '{{"{}":false}}'.format(key)
        elif '{{"{}":true}}'.format(key) in gen.lower():
            return '{{"{}":true}}'.format(key)
        else:
            return -1

    def extract_json_candidate(text):
        if not text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start : end + 1]
        
    jsonifyed_res  = []
    for r, p in zip(raw_responses, eval_prompts):
        
        if check_validity(r) != -1:
            jsonifyed_res.append(json.loads(check_validity(r)))
            continue
        else:
            r = (r or "").split("\n")[0]
            try:
                json_candidate = extract_json_candidate(r)
                jsonifyed_res.append(json.loads(json_candidate or r))
            except Exception:
                print(f"Error in eval_answer: {r}")
                error_count = 0
                while True:
                    try:
                        re_eval = lm.generate(
                            p,
                            evaluator,
                            temperature=0.0,
                            top_p=1.0,
                            max_tokens=128,
                        )
                    except Exception as e:
                        print(f"\n** RETRY ERROR: {e}")
                        error_count += 1
                        if error_count > 3:
                            print("Error count exceeded 3. Skipping this prompt.")
                            jsonifyed_res.append(
                                {"error": "Error count exceeded 3. Skipping this prompt."}
                            )
                            break
                        continue
                    try:
                        print("\n** RETRY:", re_eval)
                        if check_validity(re_eval) != -1:
                            json_res = json.loads(check_validity(re_eval))
                        else:
                            re_eval = (re_eval or "").split("\n")[0]
                            json_candidate = extract_json_candidate(re_eval)
                            json_res = json.loads(json_candidate or re_eval)
                        jsonifyed_res.append(json_res)
                        print("<<< PASS >>>")
                        break
                    except Exception:
                        print("*** trying again** \n")
                        error_count += 1
                        if error_count > 3:
                            print("Error count exceeded 3. Skipping this prompt.")
                            jsonifyed_res.append(
                                {"error": "Error count exceeded 3. Skipping this prompt."}
                            )
                            break

    return jsonifyed_res
