# Copilot Instructions for HalluLens

## 1) Objectif du repo

HalluLens est un benchmark d'hallucinations LLM avec deux couches:

- pipeline d'évaluation (scripts + tasks) qui génère/exécute/évalue les runs;
- dashboard Streamlit pour l'analyse expérimentale avancée.

Ce repo contient à la fois le code de benchmark d'origine et une couche d'analyse étendue pour la recherche sur l'impact des paramètres de génération et de la créativité.

## 2) Architecture globale

Entrées principales:

- `tasks/`: définition des benchmarks et des évaluateurs;
- `scripts/`: runners des expériences;
- `utils/`: orchestration d'inférence, appels LLM, utilitaires;
- `output/*`: artefacts de runs (source unique des analyses);
- `apps/hallulens_dashboard/`: app Streamlit d'analyse.

Tasks benchmark:

- `tasks/shortform/precise_wikiqa.py`
- `tasks/longwiki/longwiki_main.py`
- `tasks/refusal_test/round_robin_nonsense_name.py`

Pipeline commun côté scripts/tasks:

- `--do_generate_prompt` -> `--do_inference` -> `--do_eval`

Orchestration inférence:

- `utils/exp.py` (`run_exp`)
- backends dans `utils/lm.py` (`vllm`, `openai`, `custom`)

## 3) Contrat de données des runs (important)

La plupart des analyses supposent ces fichiers par run:

- `run_config.json`
- `generation.jsonl`
- `output.csv`

Fichiers optionnels selon analyses:

- `creativity.jsonl` (Creativity Prism)
- `output_abstain.jsonl`, `output_verification_results.jsonl`, etc. (pipeline étendu)

Règles de robustesse déjà implémentées:

- parsing défensif (fallback `DataFrame` vide si fichier invalide);
- normalisation de prompt/claim (`normalize_text`) pour aligner les sources;
- booléens de verdict via `parse_bool`;
- token length avec `tiktoken`, fallback regex robuste.

## 4) Dashboard Streamlit: état actuel (source de vérité)

Code principal:

- entrée: `apps/hallulens_dashboard/app.py`
- orchestration: `apps/hallulens_dashboard/hallulens_dashboard/app_main.py`
- config pages: `apps/hallulens_dashboard/hallulens_dashboard/config.py`

Pages actives:

- `Impact Hallucinations`
- `Creativity Prism`
- `Responses & Claims`
- `Evaluator Agreement`
- `LLM Export`

### 4.1 Impact Hallucinations

Implémenté:

- filtres complets (dataset/model/task/creativity/temp/length/response lengths);
- graphiques `line`, `points`, `box`, `violin`;
- résumé `build_impact_summary`;
- Spearman détaillé (numérique + catégoriel one-hot, incluant `model_name`);
- section statistique fusionnée (guide + résultats).

Advanced analysis (persistant disque via cache):

- corrélations robustes (Pearson/Spearman, bootstrap CI, permutation p-value, FDR BH);
- corrélations partielles (contrôle `response_length_words`, `n_claim_rows`);
- contrastes appariés par intention créative;
- GLM binomial (OR + CI + FDR);
- probabilités prédites GLM;
- médiation `creativity -> n_claims -> hallucination_rate`;
- MixedLM (`(1|prompt_id)`);
- stratification par modèle (`focus_model` vs `others_pooled`);
- test d'homogénéité de l'effet `creativity_level`.

### 4.2 Creativity Prism

Implémenté:

- mode strict scoré (`metrics + scores créativité`) avec exclusion des runs incomplets;
- parsing `creativity.jsonl` TTCT + TTCW (alias TTWT supporté);
- merge avec métriques hallucinations;
- bloc qualité des données (`missing/partial/complete`);
- visuels: scatter créativité vs hallucination, heatmap corr, distributions par `creativity_level`;
- export CSV filtré.

### 4.3 Responses & Claims

Implémenté:

- loader claim-level dédié: `claims_loading.py`;
- couverture run avec statut `loaded/missing_files`;
- filtres via formulaire (évite recompute continu);
- KPI claim-level (`supported/hallucinated/no_claim/unverified`);
- vue globale (distribution statuts, heatmap support task x creativity, prompts à risque);
- drill-down prompt (prompt, génération, claims, comparaison A/B);
- diagnostics precision/recall/f1;
- exports CSV.

Performance/UX:

- cache session_state pour filtres/drilldown/lookups;
- rendu par section active pour limiter les recalculs.

### 4.4 Evaluator Agreement

Implémenté:

- page dédiée aux runs dont le dossier contient `evaluation_test`;
- loader dédié: `evaluator_agreement.py`;
- extraction des modèles pipeline:
  - `claim_extractor_model`
  - `abstain_evaluator_model`
  - `verifier_model`
  - `evaluator_label` (signature finale)
- consensus des votes claim-level et pairwise agreement (`agreement_rate`, `kappa`);
- analyse par composant (extractor/abstain/verifier/final), global + par tâche;
- heatmaps pairwise;
- support-rate overall + par tâche;
- diagnostic d'interchangeabilité;
- mode `Strict stage control` optionnel;
- alerte si composantes couplées (même modèle sur a/c/v, non-identifiabilité des effets séparés).

### 4.5 LLM Export

Implémenté:

- génération markdown prêt à coller dans un chatbot;
- consolidation multi-pages:
  - Impact
  - Creativity
  - Responses & Claims
  - Evaluator Agreement
- reprise optionnelle des analyses avancées depuis cache Impact (sans recalcul lourd);
- inclusion avancée complète:
  - `corr_df`, `partial_df`, `intent_df`, `glm_df`, `mediation_df`, `mixedlm_df`
  - `glm_strat_df`, `mixedlm_strat_df`, `homogeneity_df`
- bouton copy + téléchargement `.md`.

## 5) Modules dashboard à connaître

Dans `apps/hallulens_dashboard/hallulens_dashboard/`:

- `data_loading.py`: dataset prompt-level à partir des runs
- `analytics.py`: résumés et Spearman détaillé
- `creativity_loading.py`: parsing créativité + couverture
- `creativity_analytics.py`: analyses robustes/GLM/MixedLM/médiation/stratification
- `creativity_plotting.py`: figures Plotly créativité/GLM/corr task-model
- `claims_loading.py`: claim rows + agrégations prompt-level
- `evaluator_agreement.py`: accord inter-évaluateurs
- `app_main.py`: wiring UI/caches/rendu pages
- `config.py`: constantes de pages/options par défaut
- `utils.py`: helpers génériques

## 6) Tests et validation

Tests dashboard existants:

- `tests/test_dashboard_creativity_loading.py`
- `tests/test_dashboard_creativity_analytics.py`
- `tests/test_dashboard_creativity_plotting.py`
- `tests/test_dashboard_claims_loading.py`
- `tests/test_dashboard_evaluator_agreement.py`

Commande minimale avant merge (dashboard):

```bash
pytest -q tests/test_dashboard_creativity_loading.py \
         tests/test_dashboard_creativity_analytics.py \
         tests/test_dashboard_claims_loading.py \
         tests/test_dashboard_evaluator_agreement.py
```

## 7) Conventions de maintenabilité

Principes à respecter:

- garder les loaders purs et tolérants aux erreurs (retour vide > crash);
- centraliser les calculs dans modules analytics/loading, pas dans le rendu UI;
- ne pas dupliquer la logique: préférer réutiliser les fonctions `build_*`;
- maintenir des colonnes stables (`task`, `model_name`, `creativity_level`, `run_id`, `prompt_clean`);
- utiliser `@st.cache_data` pour toute étape lourde, avec `persist=True` pour résultats stables;
- conserver les clés `st.session_state` explicites et versionnables.

Règles sur les nouvelles métriques:

- définir une sémantique claire (numérateur/dénominateur);
- documenter exclusions (`no_claim`, NaN, min_n);
- exposer la métrique dans la page concernée + export si utile;
- ajouter test unitaire sur un dataset synthétique minimal.

Règles UI:

- préférer filtres en `form` quand le dataset est grand;
- ajouter `download_button` pour tables importantes;
- afficher un message explicite si données insuffisantes.

## 8) Guide rapide pour ajouter une nouvelle fonctionnalité

1. Définir le contrat d'entrée (quels fichiers/colonnes de run sont requis).
2. Ajouter/étendre un loader dans `*_loading.py`.
3. Ajouter calculs dans `*_analytics.py` (fonctions pures).
4. Ajouter visualisation dans `*_plotting.py` si nécessaire.
5. Brancher dans `app_main.py` avec cache + filtres + états vides.
6. Ajouter export CSV et, si pertinent, section dans `LLM Export`.
7. Ajouter tests dans `tests/test_dashboard_*.py`.
8. Valider sur un sous-ensemble de runs réels + tests unitaires.

## 9) Pièges connus

- Les noms de modèles sont souvent normalisés par suffixe (`model.split('/')[-1]`).
- Les runs peuvent être partiels/incomplets: toujours coder des fallbacks.
- Les comparaisons d'accord peuvent devenir non-identifiables si toutes les étapes du pipeline changent ensemble.
- Les analyses dépendantes de `statsmodels/scipy` doivent dégrader proprement si indisponibles.

## 10) Variables d'environnement importantes

- `OPENAI_KEY` (backend OpenAI legacy dans ce repo)
- `OPENROUTER_API_KEY` (backend custom/OpenRouter)
- `BRAVE_API_KEY` (workflow refusal/search)

## 11) Rappel contribution

- Ne pas casser les formats `output/*` existants.
- Préserver la compatibilité ascendante des exports CSV.
- Toute nouvelle page/section doit gérer explicitement:
  - état vide,
  - données insuffisantes,
  - performance (cache),
  - export.
