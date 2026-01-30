# Configuration de la Température et Max Tokens pour l'Inférence

## Objectif
Permettre la configuration dynamique de la **température** et du **max_tokens** pour le modèle d'inférence (génération de réponses) dans le benchmark LongWiki.

## Fichiers Modifiés

### 1. `tasks/longwiki/longwiki_main.py`
**Changements :**
- Ajout de l'argument CLI `--temperature` (défaut: 0.0)
- Passage des paramètres `temperature` et `max_workers` à `exp.run_exp()`

**Exemple d'utilisation :**
```bash
python -m tasks.longwiki.longwiki_main \
  --exp_mode longwiki \
  --do_inference \
  --model "deepseek/deepseek-v3.2" \
  --temperature 0.7 \
  --max_tokens 2048 \
  --max_workers 32 \
  --N 10
```

### 2. `utils/exp.py`
**Changements :**
- Ajout du paramètre `temperature` avec valeur par défaut `0.0`
- Utilisation du paramètre au lieu de la valeur codée en dur
- **Rétrocompatibilité** : Les anciens appels sans `temperature` continuent de fonctionner

### 3. `scripts/task2_longwiki_openrouter.sh`
**Changements :**
- Ajout de variables configurables via environnement :
  - `TEMPERATURE` (défaut: 0.0)
  - `MAX_TOKENS` (défaut: 1024)
  - `MAX_WORKERS` (défaut: 64)

**Exemple d'utilisation :**
```bash
# Utiliser les valeurs par défaut
bash scripts/task2_longwiki_openrouter.sh

# Override avec des valeurs personnalisées
TEMPERATURE=0.7 MAX_TOKENS=2048 bash scripts/task2_longwiki_openrouter.sh
```

### 4. `scripts/task2_longwiki.sh`
**Changements identiques** au script OpenRouter.

## Impact sur les Températures

### Avant
- **Inférence (génération de réponses)** : `0.0` (codé en dur)
- **Génération de questions** : `0.7` (dans generate_question.py)
- **Answerability check** : `0.3` (dans generate_question.py)

### Après
- **Inférence (génération de réponses)** : **Configurable** via CLI (défaut: `0.0`)
- **Génération de questions** : `0.7` (inchangé)
- **Answerability check** : `0.3` (inchangé)

## Tests de Validation

### 1. Vérifier que les arguments CLI sont disponibles
```bash
python -m tasks.longwiki.longwiki_main --help | grep temperature
```

### 2. Tester avec différentes températures
```bash
# Température déterministe (défaut)
python -m tasks.longwiki.longwiki_main \
  --do_inference \
  --temperature 0.0 \
  --model "deepseek/deepseek-v3.2" \
  --N 5

# Température créative
python -m tasks.longwiki.longwiki_main \
  --do_inference \
  --temperature 0.9 \
  --model "deepseek/deepseek-v3.2" \
  --N 5
```

### 3. Vérifier la rétrocompatibilité
```bash
# Les autres tâches qui utilisent exp.run_exp() doivent continuer à fonctionner
python -m tasks.shortform.precise_wikiqa --do_inference --model "test-model"
```

## Rétrocompatibilité

✅ **Tous les appels existants à `exp.run_exp()` continuent de fonctionner** sans modification, car :
- Le paramètre `temperature` a une valeur par défaut (`0.0`)
- Les fichiers suivants ne nécessitent **aucune modification** :
  - `tasks/refusal_test/nonsense_name.py`
  - `tasks/refusal_test/round_robin_nonsense_name.py`
  - `tasks/shortform/precise_wikiqa.py`

## Recommandations d'Utilisation

### Pour une génération déterministe (reproductible)
```bash
--temperature 0.0
```
**Cas d'usage :** Benchmarks, tests, comparaisons de modèles

### Pour une génération équilibrée
```bash
--temperature 0.3
```
**Cas d'usage :** Production, chatbots, réponses variées mais contrôlées

### Pour une génération créative
```bash
--temperature 0.7
```
**Cas d'usage :** Brainstorming, génération de contenu créatif

### Pour une génération très diverse
```bash
--temperature 0.9
```
**Cas d'usage :** Expérimentation, échantillonnage large

## Notes Techniques

- La température contrôle l'aléatoire dans l'échantillonnage des tokens
- `0.0` = déterministe (toujours le token le plus probable)
- `1.0` = distribution de probabilité non modifiée
- `>1.0` = augmente l'aléatoire (peut produire du texte incohérent)

## Migration Notes

**Aucune action requise** pour les utilisateurs existants. Le comportement par défaut reste identique (température = 0.0).

Pour bénéficier de la nouvelle fonctionnalité, ajoutez simplement `--temperature <value>` à vos commandes.
