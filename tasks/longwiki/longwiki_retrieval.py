# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import threading
from collections import OrderedDict

import sqlite3
import numpy as np
import pickle as pkl
import torch
from typing import List

from utils.cache import Cache
from tasks.longwiki.retrieval import DocDB
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

class LongWikiDB(DocDB):
    def __init__(self, db_path:str, data_path:str = None):
        # import DocDB from FactScore
        super(LongWikiDB, self).__init__(db_path, data_path)
        self.title_db_path = db_path.replace(".db", "-title.db")
        self.title_connection = sqlite3.connect(self.title_db_path, check_same_thread=False)
        self.SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"
        self._local = threading.local()
        self._ensure_titles_table()

    def _get_doc_connection(self):
        conn = getattr(self._local, "doc_conn", None)
        if conn is None:
            # Read-only connections allow concurrent reads without locking the writer connection.
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro",
                uri=True,
                check_same_thread=False,
                timeout=30.0,
            )
            try:
                conn.execute("PRAGMA query_only=ON")
            except Exception:
                pass
            self._local.doc_conn = conn
        return conn

    def _get_title_connection(self):
        conn = getattr(self._local, "title_conn", None)
        if conn is None:
            conn = sqlite3.connect(
                f"file:{self.title_db_path}?mode=ro",
                uri=True,
                check_same_thread=False,
                timeout=30.0,
            )
            try:
                conn.execute("PRAGMA query_only=ON")
            except Exception:
                pass
            self._local.title_conn = conn
        return conn

    def _ensure_titles_table(self):
        cursor = self.title_connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='titles';")
        has_titles = cursor.fetchone() is not None
        if not has_titles:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS titles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title_name TEXT NOT NULL
                );
                """
            )
            # Populate titles from the main documents DB
            doc_cursor = self.connection.cursor()
            doc_cursor.execute("SELECT title FROM documents")
            titles = [r[0] for r in doc_cursor.fetchall()]
            doc_cursor.close()
            cursor.executemany(
                "INSERT INTO titles (title_name) VALUES (?)",
                [(title,) for title in titles],
            )
            self.title_connection.commit()
        cursor.close()

    def get_relevant_titles(self, entity: str, limit: int = 0, mode: str = "contains"):
        conn = self._get_title_connection()
        cursor = conn.cursor()
        entity = entity.replace("'", "''")
        if mode == "exact":
            query = "SELECT title_name FROM titles WHERE title_name = ?"
            params = (entity,)
        elif mode == "prefix":
            query = "SELECT title_name FROM titles WHERE title_name LIKE ?"
            params = (entity + "%",)
        else:
            query = "SELECT title_name FROM titles WHERE title_name LIKE ?"
            params = ("%" + entity + "%",)
        if limit and limit > 0:
            query += " LIMIT ?"
            params = params + (limit,)
        cursor.execute(query, params)
        results = cursor.fetchall()
        cursor.close()
        results = [r[0] for r in results]
        return results
    
    def get_whole_passages(self):
        conn = self._get_doc_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT title, text FROM documents")
        results = cursor.fetchall()
        results = [r for r in results]
        results = [{"title": r[0], "text": para} for r in results for para in r[1].split(self.SPECIAL_SEPARATOR)]
        return results

    def get_text_from_title(self, title):
        conn = self._get_doc_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT text FROM documents WHERE title = ?", (title,))
        results = cursor.fetchall()
        cursor.close()
        if not results or len(results) != 1:
            raise KeyError(title)
        results = [{"title": title, "text": para} for para in results[0][0].split(self.SPECIAL_SEPARATOR)]
        if not results:
            raise KeyError(title)
        return results

    def get_texts_from_titles(self, titles: List[str]):
        if not titles:
            return {}
        conn = self._get_doc_connection()
        cursor = conn.cursor()
        placeholders = ",".join(["?"] * len(titles))
        cursor.execute(
            f"SELECT title, text FROM documents WHERE title IN ({placeholders})",
            titles,
        )
        rows = cursor.fetchall()
        cursor.close()
        out = {}
        for title, text in rows:
            out[title] = [{"title": title, "text": para} for para in text.split(self.SPECIAL_SEPARATOR)]
        return out
    

class LongWikiRetrieval(object):
    def __init__(self, db, cache_base_path, embed_cache_path,
                 retrieval_type="gtr-t5-large", batch_size=None, debugging=False):
        
        self.db = db
        self.CACHE_BASE_PATH = cache_base_path
        self.embed_cache_path = embed_cache_path
        self.load_cache()

        self.retrieval_type = retrieval_type
        self.batch_size = batch_size
        self.page_cache_size = int(os.getenv("RETRIEVAL_PAGE_CACHE_SIZE", "1024"))
        self.db_fetch_batch_titles = int(os.getenv("RETRIEVAL_DB_BATCH_TITLES", "128"))
        self.page_cache = OrderedDict() if self.page_cache_size > 0 else None
        self.page_cache_lock = threading.Lock()
        self.embed_cache_lock = threading.Lock()
        self.encoder_lock = threading.Lock()
        self.not_existing_pages_lock = threading.Lock()
        self.embed_cache_save_every = int(os.getenv("EMBED_CACHE_SAVE_EVERY", "50"))
        self.embed_cache_dirty = 0
        embed_write_env = os.getenv("EMBED_CACHE_WRITE", "true")
        self.embed_cache_write = str(embed_write_env).lower() not in ("0", "false", "no")
        cache_write_env = os.getenv("RETRIEVAL_CACHE_WRITE", "true")
        self.cache_write = str(cache_write_env).lower() not in ("0", "false", "no")
        use_ner_env = os.getenv("RETRIEVAL_USE_NER", "true")
        self.use_ner = str(use_ner_env).lower() not in ("0", "false", "no")
        use_relevant_env = os.getenv("RETRIEVAL_USE_RELEVANT_TITLES", "true")
        self.use_relevant_titles = str(use_relevant_env).lower() not in ("0", "false", "no")
        self.max_ners = int(os.getenv("RETRIEVAL_MAX_NER", "0"))
        self.min_ner_len = int(os.getenv("RETRIEVAL_MIN_NER_LEN", "0"))
        self.max_titles_per_ner = int(os.getenv("RETRIEVAL_MAX_TITLES_PER_NER", "0"))
        self.title_match_mode = os.getenv("RETRIEVAL_TITLE_MATCH_MODE", "contains").lower()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline_device = 0 if device == "cuda" else -1

        ner_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
        ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")
        ner_batch_size = int(os.getenv("NER_BATCH_SIZE", "32"))
        self.ner = pipeline(
            "ner",
            model=ner_model,
            tokenizer=ner_tokenizer,
            aggregation_strategy="simple",
            batch_size=ner_batch_size,
            device=pipeline_device,
        )
        self.ner_batch_size = ner_batch_size
        self.device = device
        
        self.encoder = None
        self.not_existing_pages = set()
        self.debugging = debugging
    
    def load_encoder(self):
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("sentence-transformers/" + self.retrieval_type, device=self.device)
        self.encoder = encoder
        assert self.batch_size is not None
    
    def load_cache(self):
        # cache
        self.relevant_pages_cache_path = f"{self.CACHE_BASE_PATH}/relevant_pages_cache.json" # key: entity, value: list of relevant page titles
        self.Q_NER_cache_path = f"{self.CACHE_BASE_PATH}/question_to_ner_cache.json"
        self.cache_path =f"{self.CACHE_BASE_PATH}/cache.json"
        self.embed_cache_path = self.embed_cache_path

        self.add_n = 0
        self.add_n_embed = 0

        # embedding cache
        if os.path.exists(self.embed_cache_path):
            with open(self.embed_cache_path, "rb") as f:
                self.embed_cache = pkl.load(f)
        else:
            self.embed_cache = {}

        # question to NER cache
        self.Q_NER_cache = Cache(self.Q_NER_cache_path)

        # single entity to relevant pages cache
        self.relevant_pages_cache = Cache(self.relevant_pages_cache_path)

        # prompt query to top-5 passage cache 
        self.cache = Cache(self.cache_path)

    
    def _save_embed_cache(self):
        if not self.embed_cache_write:
            return
        with self.embed_cache_lock:
            with open(self.embed_cache_path, "wb") as f:
                pkl.dump(self.embed_cache, f)

    def flush_embed_cache(self):
        if self.embed_cache_dirty <= 0:
            return
        self._save_embed_cache()
        self.embed_cache_dirty = 0

    def _get_cached_passages(self, title: str):
        if self.page_cache is None:
            return self.db.get_text_from_title(title)
        with self.page_cache_lock:
            cached = self.page_cache.get(title)
            if cached is not None:
                self.page_cache.move_to_end(title)
                return cached
        pages = self.db.get_text_from_title(title)
        with self.page_cache_lock:
            self.page_cache[title] = pages
            if len(self.page_cache) > self.page_cache_size:
                self.page_cache.popitem(last=False)
        return pages
        
    def get_topk_passages(self, topic, retrieval_query, key_passages, k=5, query_vector=None):
        if self.encoder is None:
            with self.encoder_lock:
                if self.encoder is None:
                    self.load_encoder()

        ordered_titles = []
        passages_all = []
        missing_titles = []
        missing_inputs = []
        missing_slices = []

        for title, passages in key_passages.items():
            ordered_titles.append(title)
            passages_all.extend(passages)

            with self.embed_cache_lock:
                passage_vectors = self.embed_cache.get(title)
            if passage_vectors is None:
                inputs = [
                    psg["title"]
                    + " "
                    + psg["text"].replace("<s>", "").replace("</s>", "")
                    for psg in passages
                ]
                if not inputs:
                    continue
                start = len(missing_inputs)
                missing_inputs.extend(inputs)
                end = len(missing_inputs)
                missing_titles.append(title)
                missing_slices.append((start, end))

        if missing_titles:
            with self.encoder_lock:
                encoded = self.encoder.encode(
                    missing_inputs,
                    batch_size=self.batch_size,
                    device=self.encoder.device,
                )
            with self.embed_cache_lock:
                for title, (start, end) in zip(missing_titles, missing_slices):
                    self.embed_cache[title] = encoded[start:end]
            self.embed_cache_dirty += len(missing_titles)
            if self.embed_cache_save_every > 0 and self.embed_cache_dirty >= self.embed_cache_save_every:
                self._save_embed_cache()
                self.embed_cache_dirty = 0

        passage_vectors_list = []
        with self.embed_cache_lock:
            for title in ordered_titles:
                vec = self.embed_cache.get(title)
                if vec is None:
                    # Should not happen; keep alignment between passages and vectors.
                    return []
                passage_vectors_list.append(vec)

        if not passage_vectors_list:
            return []
        passage_vectors_all = (
            np.concatenate(passage_vectors_list, axis=0)
            if len(passage_vectors_list) > 1
            else passage_vectors_list[0]
        )

        if query_vector is None:
            with self.encoder_lock:
                query_vectors = self.encoder.encode(
                    [retrieval_query],
                    batch_size=self.batch_size,
                    device=self.encoder.device,
                )[0]
        else:
            query_vectors = query_vector
        
        scores = np.inner(query_vectors, passage_vectors_all)
        indices = np.argsort(-scores)[:k]

        return [passages_all[i] for i in indices]

    def make_ner_cache(self, questions: List[str]):
        missing = [q for q in questions if self.Q_NER_cache.get_item(q) is None]
        if not missing:
            return

        batch_results = self.ner(missing)
        if (
            batch_results
            and isinstance(batch_results, list)
            and batch_results
            and isinstance(batch_results[0], dict)
        ):
            batch_results = [batch_results]
        for question, ner_results in zip(missing, batch_results):
            ners = [r["word"] for r in ner_results if "#" not in r["word"]]
            self.Q_NER_cache.set_item(question, ners)
                
    def get_topk_related_passages(self, topic, claim, question, k=5, use_cache=True, query_vector=None):
        """
            NER based top-k passage retrieval
            Extract named entities from question, get relevant pages for each entity.
            -> Passage pool: topic (where question is generated), NERs, NER-relevant pages
            Out of the passage pool, get top-k similar passages to the query using the encoder (self.get_topk_passages)
            return top-k passages
        """
        #### Function called from facthalu.py
        retrieval_query = topic + " " + claim.strip()
        cache_key = topic + "#" + claim.strip()

        # check cache
        cache_res = self.cache.get_item(cache_key)
        if use_cache and cache_res is not None:
            return cache_res
        
        # Using NER to get named entities from question
        ners, ner_relevant_titles = [], []
        if self.use_ner:
            ners = self.Q_NER_cache.get_item(question) or []
            # Dedup while preserving order
            seen = set()
            filtered = []
            for ner in ners:
                if self.min_ner_len and len(ner) < self.min_ner_len:
                    continue
                if ner in seen:
                    continue
                seen.add(ner)
                filtered.append(ner)
                if self.max_ners and len(filtered) >= self.max_ners:
                    break
            ners = filtered
            if self.use_relevant_titles and ners:
                for ner in ners:
                    pgs_selected = self.relevant_pages_cache.get_item(ner)
                    if pgs_selected:
                        pgs = pgs_selected
                    else:
                        pgs = self.db.get_relevant_titles(
                            ner,
                            limit=self.max_titles_per_ner,
                            mode=self.title_match_mode,
                        )
                        if not pgs:
                            continue
                        self.relevant_pages_cache.set_item(ner, pgs)
                    ner_relevant_titles += [
                        pg
                        for pg in pgs
                        if ((pg.lower() in claim.lower()) or (pg.lower() in question.lower()))
                    ]
        
        # Get all relevant pages
        combined = [topic] + ner_relevant_titles + ners
        all_related_pages = list(set(combined))

        # get all passages
        key_passages = {}
        to_fetch = []
        for title in all_related_pages:
            title = title.replace("_", " ")
            with self.not_existing_pages_lock:
                if title in self.not_existing_pages:
                    continue
            if self.page_cache is not None:
                with self.page_cache_lock:
                    cached = self.page_cache.get(title)
                    if cached is not None:
                        self.page_cache.move_to_end(title)
                        key_passages[title] = cached
                        continue
            to_fetch.append(title)

        if to_fetch:
            # Batch DB reads to reduce per-title cursor overhead.
            batch_n = min(900, max(1, self.db_fetch_batch_titles))
            fetched = {}
            for i in range(0, len(to_fetch), batch_n):
                chunk = to_fetch[i : i + batch_n]
                try:
                    fetched.update(self.db.get_texts_from_titles(chunk))
                except Exception:
                    # Fall back to per-title fetch for unexpected DB errors.
                    for t in chunk:
                        try:
                            fetched[t] = self.db.get_text_from_title(t)
                        except Exception:
                            pass

            if self.page_cache is not None and fetched:
                with self.page_cache_lock:
                    for t, pages in fetched.items():
                        self.page_cache[t] = pages
                        self.page_cache.move_to_end(t)
                        if len(self.page_cache) > self.page_cache_size:
                            self.page_cache.popitem(last=False)

            for t, pages in fetched.items():
                key_passages[t] = pages

            missing = set(to_fetch) - set(fetched.keys())
            if missing:
                with self.not_existing_pages_lock:
                    self.not_existing_pages.update(missing)
        
        top_k_related_passages = self.get_topk_passages(
            topic, retrieval_query, key_passages, k, query_vector=query_vector
        )
        if self.cache_write:
            self.cache.set_item(cache_key, top_k_related_passages)

        self.add_n += 1
        return top_k_related_passages
