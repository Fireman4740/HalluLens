#!/usr/bin/env python
"""
Script de test pour valider la configuration de température et max_tokens.
Usage: python test_temperature_config.py
"""

import sys
from pathlib import Path

# Ensure project root is in path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from utils import exp
import pandas as pd


def test_temperature_parameter():
    """Test que le paramètre temperature est bien passé à exp.run_exp()"""
    print("=" * 60)
    print("Test: Validation des paramètres temperature et max_tokens")
    print("=" * 60)

    # Créer des données de test minimales
    test_prompts = pd.DataFrame(
        {
            "prompt": [
                "What is the capital of France?",
                "Explain quantum computing in simple terms.",
            ]
        }
    )

    print("\n✓ Test data created")

    # Test 1: Vérifier que temperature=0.0 fonctionne (comportement par défaut)
    print("\n[Test 1] Default temperature (0.0)...")
    try:
        # Note: Ce test ne fait que valider la signature de fonction
        # Il ne fait pas d'appel API réel
        import inspect

        sig = inspect.signature(exp.run_exp)
        params = sig.parameters

        assert "temperature" in params, "Parameter 'temperature' missing"
        assert params["temperature"].default == 0.0, "Default temperature should be 0.0"
        print("✓ Parameter 'temperature' exists with default 0.0")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

    # Test 2: Vérifier que max_tokens est bien un paramètre
    print("\n[Test 2] max_tokens parameter...")
    try:
        assert "max_tokens" in params, "Parameter 'max_tokens' missing"
        assert params["max_tokens"].default == 512, "Default max_tokens should be 512"
        print("✓ Parameter 'max_tokens' exists with default 512")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

    # Test 3: Vérifier que max_workers est bien un paramètre
    print("\n[Test 3] max_workers parameter...")
    try:
        assert "max_workers" in params, "Parameter 'max_workers' missing"
        assert params["max_workers"].default == 64, "Default max_workers should be 64"
        print("✓ Parameter 'max_workers' exists with default 64")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

    # Test 4: Vérifier la rétrocompatibilité
    print("\n[Test 4] Backward compatibility...")
    try:
        # Simuler un ancien appel sans temperature
        sig_params = list(params.keys())
        required_params = [
            p for p in sig_params if params[p].default == inspect.Parameter.empty
        ]
        print(f"  Required parameters: {required_params}")

        # Les anciens appels ne devraient avoir que ces paramètres requis
        assert "task" in required_params
        assert "model_path" in required_params
        assert "all_prompts" in required_params
        assert "temperature" not in required_params, (
            "temperature should have a default value"
        )
        print("✓ Backward compatibility maintained (temperature is optional)")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_temperature_parameter()
    sys.exit(0 if success else 1)
