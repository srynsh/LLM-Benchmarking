from src.config import VALIDATOR_REPAIR_NAME
from src.plots.ensemble_plots import valid_invalid_error, tpr_tnr_ensemble
from src.plots.validator_plots import plot_failure_counts, tpr_tnr_validator, plot_validator_valid, plot_validator_invalid, plot_validator_valid_invalid_cumsum
from src.plots.repair_plots import repair_vs_error

# =========================
#   CUSTOM IMPORTS
# =========================
if VALIDATOR_REPAIR_NAME == 'fcnhp':
    from src.validation.generated_scripts.summary_fcnhp import (ensemble_max_errors, ensemble_tpr, ensemble_tnr, error_message_counts_validator)
    from src.validation.generated_scripts.llm_as_judge_fcnhp import validator_tpr, validator_tnr
elif VALIDATOR_REPAIR_NAME == '':
    from src.validation.generated_scripts.summary_noRepair import (ensemble_max_errors, ensemble_tpr, ensemble_tnr, error_message_counts_validator)
    from src.validation.generated_scripts.llm_as_judge_noRepair import validator_tpr, validator_tnr
else:
    raise ValueError(f"Unknown VALIDATOR_REPAIR_STR: {VALIDATOR_REPAIR_NAME}. Please check the configuration.")

# =========================
#   MAIN
# =========================
if __name__ == "__main__":
    # Validator plots
    plot_failure_counts(error_message_counts_validator)
    tpr_tnr_validator(validator_tpr, validator_tnr, ensemble_tpr, ensemble_tnr)
    plot_validator_valid()
    plot_validator_invalid()
    plot_validator_valid_invalid_cumsum()

    # Ensemble plots
    tpr_tnr_ensemble(ensemble_tpr, ensemble_tnr, validator_tpr, validator_tnr)
    valid_invalid_error(ensemble_max_errors)

    # Repair plots
    repair_vs_error()