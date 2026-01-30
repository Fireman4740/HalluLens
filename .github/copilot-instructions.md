# Copilot instructions for HalluLens

## Architecture & data flow

- Three benchmark tasks live under [tasks/](tasks):
  - PreciseWikiQA: [tasks/shortform/precise_wikiqa.py](tasks/shortform/precise_wikiqa.py)
  - LongWiki: [tasks/longwiki/longwiki_main.py](tasks/longwiki/longwiki_main.py)
  - NonExistentRefusal: [tasks/refusal_test/round_robin_nonsense_name.py](tasks/refusal_test/round_robin_nonsense_name.py)
- All tasks follow the same pipeline flags: `--do_generate_prompt` → `--do_inference` → `--do_eval` (see scripts in [scripts/](scripts)).
- Inference is centralized in `utils/exp.run_exp` in [utils/exp.py](utils/exp.py). It writes JSONL generations to output/{task}/{model}/generation.jsonl; evaluators read from that path (e.g., `PreciseQAEval` in [tasks/shortform/precise_wikiqa.py](tasks/shortform/precise_wikiqa.py)).
- LongWiki evaluation uses FactHalu + retrieval; document storage is an SQLite DB built/loaded in [tasks/longwiki/retrieval.py](tasks/longwiki/retrieval.py) and typically points at data/wiki_data/.cache/enwiki-20230401.db.
- NonExistentRefusal generation double-checks with web search via Brave API in [tasks/refusal_test/search.py](tasks/refusal_test/search.py).

## Critical workflows

- Install deps: see [requirements.txt](requirements.txt). Data is downloaded/prepared via [scripts/download_data.sh](scripts/download_data.sh).
- Task runners (recommended):
  - PreciseWikiQA: [scripts/task1_precisewikiqa.sh](scripts/task1_precisewikiqa.sh)
  - LongWiki: [scripts/task2_longwiki.sh](scripts/task2_longwiki.sh)
  - GeneratedEntities: [scripts/task3-2_generatedentities.sh](scripts/task3-2_generatedentities.sh)

## LLM integration & environment

- LLM calls are routed through [utils/lm.py](utils/lm.py):
  - `call_vllm_api()` for vLLM servers (default in scripts).
  - `openai_generate()` uses OPENAI_KEY.
  - `custom_api()` uses OpenRouter (OPENROUTER_API_KEY) and is the default in `generate()`.
- Match `--inference_method` to the backend in `utils/exp.run_exp` (`vllm`, `openai`, `custom`).
- NonExistentRefusal requires BRAVE_API_KEY for search (see [tasks/refusal_test/search.py](tasks/refusal_test/search.py)).

## Project conventions

- Model names are parsed via `model.split("/")[-1]` to build output paths; keep this in mind when adding new models or naming outputs.
- Many steps reuse cached artifacts if files already exist (e.g., JSONL under data/.../save or output/...); check existing files before regenerating.
- Python entrypoints are typically invoked as modules (python -m tasks.<...>) in scripts; keep new task CLIs consistent.
