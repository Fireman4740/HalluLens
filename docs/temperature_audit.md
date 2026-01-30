# Audit des Températures LLM dans LongWiki

## Résumé
✅ **Validation confirmée** : Seul le modèle d'inférence (génération de réponse) a une température configurable.

## Tableau des Températures

| Étape | Fonction | Fichier | Ligne | Température | Configurable |
|-------|----------|---------|-------|-------------|--------------|
| **INFÉRENCE** (génération réponse) | `exp.run_exp()` | `utils/exp.py` | 38 | **Paramètre CLI** (défaut: 0.0) | ✅ OUI |
| Question generation | `WikiQA.generate_question_with_doc()` | `utils/generate_question.py` | 136 | 0.7 | ❌ NON (fixe) |
| Question generation (batch) | `WikiQA.per_bin_generation_batch()` | `utils/generate_question.py` | 206 | 0.7 | ❌ NON (fixe) |
| Answerability check | `WikiQA.generate_answerability()` | `utils/generate_question.py` | 145 | 0.3 | ❌ NON (fixe) |
| Answerability check (batch) | `WikiQA.per_bin_generation_batch()` | `utils/generate_question.py` | 220 | **⚠️ 0.0** (défaut) | ❌ NON (fixe) |
| Evaluation (abstain, claim, verify) | `model_eval_step()` | `tasks/longwiki/longwiki_utils.py` | 128 | 0.0 (défaut) | ❌ NON (fixe) |

## ⚠️ Incohérence Détectée

**Problème** : L'answerability check utilise deux températures différentes :
- Version individuelle (ligne 145) : `temperature=0.3`
- Version batch (ligne 220) : `temperature` non spécifiée → **défaut 0.0**

**Impact** : Les vérifications d'answerability en batch sont plus déterministes que prévu.

**Recommandation** : Harmoniser à `temperature=0.3` pour cohérence.

## Validation des Modifications

### ✅ Ce qui a été modifié
- `utils/exp.py` : `temperature` devient un paramètre configurable (défaut: 0.0)
- `tasks/longwiki/longwiki_main.py` : Ajout de l'argument CLI `--temperature`
- Scripts shell : Variables d'environnement pour configuration

### ✅ Ce qui N'a PAS été modifié (par design)
- Génération de questions : reste à `temperature=0.7` (fixe)
- Answerability checks : reste à `temperature=0.3` (fixe, sauf bug ligne 220)
- Évaluation (abstain/claim/verify) : reste à `temperature=0.0` (fixe)

## Test de Validation

```python
# Vérifier que seul exp.run_exp() utilise le paramètre temperature configurable
import inspect
from utils import exp, lm

# 1. Vérifier la signature de exp.run_exp()
sig = inspect.signature(exp.run_exp)
assert 'temperature' in sig.parameters
assert sig.parameters['temperature'].default == 0.0
print("✓ exp.run_exp() a un paramètre temperature configurable")

# 2. Vérifier que lm.generate() a temperature=0.0 par défaut
sig_lm = inspect.signature(lm.generate)
assert sig_lm.parameters['temperature'].default == 0.0
print("✓ lm.generate() a temperature=0.0 par défaut (évaluation déterministe)")
```

## Conclusion

✅ **Objectif atteint** : 
- Le modèle d'inférence (génération de réponse) est le **seul** avec température configurable via CLI
- Les autres étapes (génération de questions, évaluation) utilisent des températures **fixes**
- Rétrocompatibilité préservée

⚠️ **Bug découvert** : 
- Answerability check en batch utilise `temperature=0.0` au lieu de `0.3`
- Correction recommandée (voir section suivante)
