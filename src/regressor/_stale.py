
import numpy as np
from regressor.config import GENS, MODEL_ENUM, MODELS, PVIVA_CONST, PVVA_CONST, VALIDATOR_COUNTS_CONST

VALIDATOR_COUNTS = VALIDATOR_COUNTS_CONST

# =========================
#   Estimate Vpos and Vneg if not provided
# =========================
def get_pViv_full(gens, VALIDATOR_COUNTS):
    pViva = {}

    stats = np.zeros_like(VALIDATOR_COUNTS[0])
    for i in gens:
        stats += VALIDATOR_COUNTS[GENS.index(i)]

    pViva = {k: (stats[i][0])/(stats[i][0] + stats[i][1]) for i, k in enumerate(MODELS)}
    pVva = {k: (stats[i][3])/(stats[i][2] + stats[i][3]) for i, k in enumerate(MODELS)}

    return pViva, pVva

_piv, _pv = get_pViv_full(GENS, VALIDATOR_COUNTS)
PVVA = PVVA_CONST if PVVA_CONST is not None else np.array([_pv[m] for m in MODELS])
PVIVA = PVIVA_CONST if PVIVA_CONST is not None else np.array([_piv[m] for m in MODELS])