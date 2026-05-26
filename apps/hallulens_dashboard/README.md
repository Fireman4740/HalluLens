# HalluLens Streamlit Dashboard

This app centralizes interactive analysis of hallucination metrics and creativity metrics across experiment runs.

## Features

- Load runs from `output/*` (requires `run_config.json`, `generation.jsonl`, `output.csv`)
- Prompt-level aggregation of metrics:
  - `n_claims`
  - `support_rate`
  - `hallucination_rate`
  - `response_length_words`
  - `response_length_tokens` (tiktoken when available, fallback approximation otherwise)
- Dashboard pages:
  - `Impact Hallucinations`: parameter impact exploration (`line`, `points`, `box`, `violin`, Spearman, advanced analysis, export)
  - `Creativity Prism`: strict scored-mode analysis based on `creativity.jsonl`
  - `Responses & Claims`: deep exploration of LLM responses, extracted claims, and verification outcomes
  - `LLM Export`: consolidated, LLM-ready report text combining insights from both pages with project context
- Responses & Claims page capabilities:
  - claim-level filtering by dataset/model/task/creativity/temperature/length/status
  - prompt drill-down (prompt text, generated response, per-claim verification table)
  - global risk views (status distributions, support-rate heatmap, risky prompts)
  - verification diagnostics (precision/recall/f1 scatter + grouped summary table)
  - CSV export of filtered claim rows
- Creativity page capabilities:
  - run-level coverage table (`missing`, `partial`, `complete`)
  - TTCT and TTCW (TTWT alias supported) parsing and merge with hallucination metrics
  - scatter/box/heatmap visualizations for creativity vs hallucinations
  - creativity-metrics vs creativity-level distribution chart (metrics on X, classes on Y)
  - CSV export for filtered scored data
- LLM export page capabilities:
  - generate a structured markdown report for chatbot analysis
  - include project context + Impact results + Creativity results + analysis instructions
  - one-click copy button + markdown download

## Architecture

The app is split into modules under `apps/hallulens_dashboard/hallulens_dashboard/`:

- `app_main.py`: Streamlit orchestration (page layout + section wiring)
- `data_loading.py`: run discovery + prompt-level dataset construction
- `creativity_loading.py`: creativity JSONL parsing + strict scored dataset merge + coverage table
- `claims_loading.py`: claim-level dataset build (output.csv + generation merge) + prompt summaries
- `plotting.py`: Plotly figure builders (line/box/violin)
- `creativity_plotting.py`: Plotly builders for creativity scatter/heatmap/boxplot/GLM forest
- `analytics.py`: impact summary + detailed Spearman analysis
- `creativity_analytics.py`: robust correlations, partial correlations, paired contrasts, GLM binomial
- `utils.py`: generic helpers (normalization, filtering, defaults)
- `config.py`: constants and default UI selections

Entry point:

- `apps/hallulens_dashboard/app.py` (thin launcher only)

## Run

```bash

pip install -r apps/hallulens_dashboard/requirements.txt

conda activate crea 
streamlit run apps/hallulens_dashboard/app.py
```

## Notes

- The app computes prompt-level rates from claim rows in `output.csv`:
  - `support_rate = supported_claims / valid_claims`
  - `hallucination_rate = 1 - support_rate`
- Rows where claim is `"no claims"` are excluded from the denominator.
- On `Creativity Prism`, strict mode is enabled: only rows with both hallucination metrics and valid creativity scores are retained.
