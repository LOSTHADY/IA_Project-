"""Sortie des tests quand CTranslate2 a servi.

Une fois tous les tests passés, Python plante parfois en se terminant
(« Segmentation fault » pendant la libération des bibliothèques natives),
toujours après avoir utilisé CTranslate2 (test_deploiement). Constaté en CI
le 17 octobre 2026 : « 16 passed », puis code de sortie 139, donc un échec
alors que rien n'avait échoué. Quand CTranslate2 est chargé, on sort donc
directement avec le vrai résultat des tests, sans cette libération.
"""

from __future__ import annotations

import os
import sys

_STATUT: dict = {}


def pytest_sessionfinish(session, exitstatus):
    _STATUT["code"] = int(exitstatus)


def pytest_unconfigure(config):
    # Appelé après le résumé (« N passed ») : rien n'est perdu.
    if "ctranslate2" in sys.modules and "code" in _STATUT:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(_STATUT["code"])
