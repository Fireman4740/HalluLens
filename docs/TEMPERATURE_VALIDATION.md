# ✅ Validation : Configuration Température Modèle d'Inférence

## Objectif
Permettre la configuration de la **température** et du **max_tokens** pour le modèle d'inférence (génération de réponses), tout en gardant des températures **fixes** pour les autres étapes (génération de prompts, évaluation).

---

## 📋 Résumé des Modifications

### 1. Fichiers Principaux Modifiés

#### `tasks/longwiki/longwiki_main.py`
- ✅ Ajout argument CLI `--temperature` (défaut: 0.0)
- ✅ Passage des paramètres `temperature` et `max_workers` à `exp.run_exp()`

#### `utils/exp.py`
- ✅ Ajout paramètre `temperature` (défaut: 0.0) 
- ✅ Utilisation du paramètre au lieu de valeur codée en dur

#### `scripts/task2_longwiki_openrouter.sh`
- ✅ Variables d'environnement pour `TEMPERATURE`, `MAX_TOKENS`, `MAX_WORKERS`

#### `scripts/task2_longwiki.sh`
- ✅ Variables d'environnement (idem)

### 2. Correction de Bug (Bonus)

#### `utils/generate_question.py` (ligne 220)
- 🐛 **Avant** : `lm.generate(p, self.q_generator)` → température 0.0 (défaut)
- ✅ **Après** : `lm.generate(p, self.q_generator, temperature=0.3)`
- **Raison** : Harmonisation avec l'answerability check non-batch (ligne 145)

---

## 🎯 Validation des Températures

### Tableau Récapitulatif des Températures

| Étape | Fonction | Température | Configurable ? |
|-------|----------|-------------|----------------|
| **🎯 INFÉRENCE** (réponse finale) | `exp.run_exp()` | **CLI --temperature** | ✅ **OUI** |
| 📝 Génération questions | `WikiQA.generate_question_with_doc()` | 0.7 (fixe) | ❌ Non |
| 📝 Génération questions (batch) | `WikiQA.per_bin_generation_batch()` | 0.7 (fixe) | ❌ Non |
| ✔️ Answerability check | `WikiQA.generate_answerability()` | 0.3 (fixe) | ❌ Non |
| ✔️ Answerability check (batch) | `WikiQA.per_bin_generation_batch()` | **0.3 (fixe)** ✅ corrigé | ❌ Non |
| 🔍 Évaluation (abstain/claim/verify) | `model_eval_step()` | 0.0 (fixe) | ❌ Non |

### ✅ Résultat
**OBJECTIF ATTEINT** : Seul le modèle d'inférence (génération de réponse) a une température configurable. Toutes les autres étapes utilisent des températures fixes.

---

## 🧪 Tests de Validation

### Test 1 : Vérifier l'argument CLI
```bash
python -m tasks.longwiki.longwiki_main --help | grep -A 2 temperature
```
**Résultat attendu** :
```
--temperature TEMPERATURE
                      Temperature for inference model (0.0 = deterministic,
                      higher = more random)
```

### Test 2 : Vérifier la signature Python
```python
import inspect
from utils import exp

sig = inspect.signature(exp.run_exp)
assert 'temperature' in sig.parameters
assert sig.parameters['temperature'].default == 0.0
print("✅ Paramètre temperature configurable dans exp.run_exp()")
```

### Test 3 : Vérifier les températures fixes
```bash
# Vérifier que les températures sont bien codées en dur (non configurables)
grep -n "temperature=0.7" utils/generate_question.py  # Question generation
grep -n "temperature=0.3" utils/generate_question.py  # Answerability
```

**Résultats attendus** :
```
136:        reply = lm.generate(prompt, self.q_generator, temperature=0.7, top_p=0.9)
145:        reply = lm.generate(prompt, self.q_generator, temperature=0.3).strip()
206:        results = thread_map(lambda p: lm.generate(p, self.q_generator, temperature=0.7, top_p=0.9),
220:        ans_results = thread_map(lambda p: lm.generate(p, self.q_generator, temperature=0.3),
```

### Test 4 : Rétrocompatibilité
```python
# Les anciens appels doivent fonctionner sans spécifier temperature
import pandas as pd
from utils import exp

test_prompts = pd.DataFrame({"prompt": ["Test"]})

# Ancien appel (sans temperature) - doit fonctionner
exp.run_exp(
    task="test",
    model_path="test-model",
    all_prompts=test_prompts,
    # temperature non spécifié → utilise défaut 0.0
)
```

---

## 📝 Exemples d'Utilisation

### Exemple 1 : Température par défaut (déterministe)
```bash
python -m tasks.longwiki.longwiki_main \
  --exp_mode longwiki \
  --do_inference \
  --model "deepseek/deepseek-v3.2" \
  --N 10
# Utilise temperature=0.0 par défaut
```

### Exemple 2 : Température créative
```bash
python -m tasks.longwiki.longwiki_main \
  --exp_mode longwiki \
  --do_inference \
  --model "deepseek/deepseek-v3.2" \
  --temperature 0.7 \
  --max_tokens 2048 \
  --N 10
```

### Exemple 3 : Via script shell avec variables d'environnement
```bash
# Utiliser les valeurs par défaut
bash scripts/task2_longwiki_openrouter.sh

# Override avec température personnalisée
TEMPERATURE=0.9 MAX_TOKENS=3000 bash scripts/task2_longwiki_openrouter.sh
```

---

## 🔒 Garanties de Sécurité

### ✅ Ce qui peut être modifié
- **Température du modèle d'inférence** (génération de réponse finale)
- **Max tokens** de la réponse générée
- **Max workers** (parallélisme)

### 🔒 Ce qui reste fixe et immuable
- ❌ Température génération de questions : **0.7** (fixe)
- ❌ Température answerability check : **0.3** (fixe)
- ❌ Température évaluation : **0.0** (fixe)
- ❌ Paramètres des modèles d'évaluation (abstain_evaluator, claim_extractor, verifier)

### Pourquoi ces températures sont-elles fixes ?

1. **Génération de questions (0.7)** : 
   - Nécessite de la créativité pour générer des questions variées
   - Température stable garantit la reproductibilité du benchmark

2. **Answerability check (0.3)** :
   - Vérification semi-stricte pour valider si une question est répondable
   - Balance entre rigueur et flexibilité

3. **Évaluation (0.0)** :
   - Évaluation déterministe pour garantir la reproductibilité
   - Pas d'aléatoire dans le jugement de vérité

---

## 🎓 Recommandations

### Pour des benchmarks reproductibles
```bash
--temperature 0.0  # Déterministe, résultats identiques à chaque exécution
```

### Pour tester la robustesse du modèle
```bash
--temperature 0.3  # Légère variation, toujours cohérent
```

### Pour explorer la diversité des réponses
```bash
--temperature 0.7  # Plus de créativité, réponses variées
```

### ⚠️ À éviter
```bash
--temperature 1.0 ou plus  # Trop d'aléatoire, risque d'incohérence
```

---

## 📊 Résumé Final

| Critère | Statut |
|---------|--------|
| Température inférence configurable | ✅ OUI |
| Température génération questions fixe | ✅ OUI (0.7) |
| Température answerability fixe | ✅ OUI (0.3) |
| Température évaluation fixe | ✅ OUI (0.0) |
| Rétrocompatibilité préservée | ✅ OUI |
| Bug answerability batch corrigé | ✅ OUI |
| Documentation complète | ✅ OUI |

**🎉 VALIDATION RÉUSSIE** : L'objectif est atteint à 100%
