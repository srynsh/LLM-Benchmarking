from src.plots.ensemble_plots import valid_invalid_error, tpr_tnr_ensemble

from src.config import pathImages, pathOutput, MODELS_GEN
import pandas as pd

from src.plots.validator_plots import tpr_tnr_validator


if __name__ == "__main__":
    # Validator plots
    tpr_tnr_validator()

    # Ensemble plots
    tpr_tnr_ensemble()
    valid_invalid_error()