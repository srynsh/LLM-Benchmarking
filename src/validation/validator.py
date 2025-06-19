"""
Refactored validator using Pydantic models and modular design.
"""

import sys
import json
import numpy as np
import pprint
import datetime
from typing import List, Optional
from tqdm import tqdm
import pandas as pd

sys.path.append("..")

from src.config import NUM_SIDS, Model, pathLogs
from src.validation.service import ValidationRunner, ValidationService
from src.validation.data import DataProvider
from src.validation.models import ValidationBatch, ValidationResult
from src.utils import print_warning, print_error
from src.validation.utils import merge_df_y_yhat, merge_df_merged_yhat, calculate_confusion_matrix, tpr_tnr_list
from src.validation.ensemble import ensemble_prediction
from src.regressor.config import pGa_CONST

####################
# Config
####################

### Missing annotations
# CLAUDE_3_OPUS: 2 SIDs
# GPT_4_TURBO: 17 SIDs
# GPT_4O: 2 SIDs


####################
# Printing
####################
def print_failed_matrix(dataProvider):
    """
    Print the failed SIDs and FIDs in a formatted matrix.
    """
    failed_sids = dataProvider.get_failed_sids()
    failed_fids = dataProvider.get_failed_fids()

    print("Failed SIDs:")
    for sid in failed_sids:
        print(f" - {sid}")

    print("Failed FIDs:")
    for fid in failed_fids:
        print(f" - {fid}")

####################
# LLM-as-a-Judge
####################

def validate_model(MODEL_GENS, MODEL_VALS):
    """
    Validate a specific model and return the results as a batch.
    
    Args:
        modelGen: Model used for generation
        modelVal: Model used for validation
        use_ground_truth: Whether to use ground truth data

    Returns:
        ValidationBatch: The validation results
    """
    count_invalids = 0
    count_invalid_max = 0
    label_max = None
    count_valids = 0
    confusion_matrices_gen = []
    confusion_matrices_val = []
    failed_sids = []
    failed_fids = []
    GV = []
    dfs = {}
    

    for modelGen in MODEL_GENS:
        row_gen = []
        row_gv = []
        failed_sids_row = []
        failed_fids_row = []
        df_merged = pd.DataFrame()

        for j, modelVal in enumerate(MODEL_VALS):
            dataProvider = DataProvider(modelGen, modelVal)
            if len(dataProvider.get_failed_sids()) > count_invalid_max:
                count_invalid_max = len(dataProvider.get_failed_sids())
                label_max = f'{modelGen} -> {modelVal}'

            failed_sids_row += [len(dataProvider.get_failed_sids())]
            failed_fids_row += [len(dataProvider.get_failed_fids())]

            count_invalids += len(dataProvider.get_failed_sids())
            count_valids += len(dataProvider.get_successful_results())

            # Create DataFrame with specified columns, filtering for successful validations only
            df_yhat = dataProvider.validation_batch.create_dataframe()
            df_y = dataProvider.generation_batch.create_dataframe()

            if df_merged.empty:
                df_merged = df_y.copy()
            df_merged = merge_df_merged_yhat(df_merged, df_yhat, modelVal)

            confusion_matrix = calculate_confusion_matrix(df_y, df_yhat)
            # print(f"Confusion Matrix for {modelGen} vs {modelVal}: {confusion_matrix}")

            # Add confusion matrix calculation
            row_gen.append(confusion_matrix)

            if len(confusion_matrices_val) <= j:
                confusion_matrices_val.append([])
                
            prev_row = confusion_matrices_val[j]
            confusion_matrices_val[j] = prev_row + [confusion_matrix]

            tn, fp, fn, tp = confusion_matrix
            percentage_valid = (tp + fp) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0

            print(f"Confusion Matrix for {modelGen} vs {modelVal}: {confusion_matrix}, percentage_valid={percentage_valid:.2f}")
            row_gv.append(percentage_valid)

        df_merged.to_csv(f'{pathLogs}/ensemble/df_{modelGen}.csv', index=False)
        failed_sids.append(failed_sids_row)
        failed_fids.append(failed_fids_row)

        confusion_matrices_gen.append(row_gen)
        GV.append(row_gv)
        dfs[modelGen] = df_merged
    
    print(f"\n{'='*60}")
    print('Failed SIDs Matrix')
    print(f"\n{'='*60}")
    pprint.pprint(failed_sids)
    print(f"\n{'='*60}")
    print('Failed FIDs Matrix')
    print(f"\n{'='*60}")
    pprint.pprint(failed_fids)

    tprs = []
    tnrs = []
    for j, modelVal in enumerate(MODEL_VALS):
        tpr, tnr = tpr_tnr_list(confusion_matrices_val[j])
        tprs.append(tpr)
        tnrs.append(tnr)

    return count_valids, count_invalids, confusion_matrices_gen, tprs, tnrs, GV, dfs, label_max


def llm_judge_errors(MODEL_GENS, MODEL_VALS, GV_gen: List[List[float]]):
    max_error_max = 0
    max_error_min = float('inf')
    mean_error_max = 0
    mean_error_min = float('inf')

    for j, modelVal in enumerate(MODEL_VALS):
        max_error_val = 0
        mean_error_val = 0

        for i, modelGen in enumerate(MODEL_GENS):
            valueActual = pGa_CONST[modelGen]
            valuePredicted = GV_gen[i][j]
            error = abs(valueActual - valuePredicted)


            max_error_val = max(max_error_val, error)
            mean_error_val += error

        mean_error_val /= len(MODEL_GENS)

        max_error_max = max(max_error_max, max_error_val)
        max_error_min = min(max_error_min, max_error_val)
        mean_error_max = max(mean_error_max, mean_error_val)
        mean_error_min = min(mean_error_min, mean_error_val)

    print(f"Max Error range: ({max_error_min*100}, {max_error_max*100})")
    print(f"Mean Error range: ({mean_error_min*100}, {mean_error_max*100})")


####################
# Main
####################

if __name__ == "__main__":
    MODEL_GENS = [Model.GPT_4O.value, Model.GPT_4_TURBO.value, Model.CLAUDE_3_OPUS.value, Model.GEMINI_1_5_PRO.value, Model.QWEN_CODER_PLUS.value, Model.DEEPSEEK_CHAT.value]
    # MODEL_GENS = [Model.GEMINI_1_5_PRO.value]

    MODEL_VALS = [
        Model.GPT_4_TURBO.value, Model.GPT_4O_MINI.value, Model.GPT_4O.value,
        Model.CLAUDE_3_OPUS.value, Model.CLAUDE_3_5_SONNET.value,
        Model.GEMINI_1_5_FLASH.value, Model.GEMINI_1_5_PRO.value,
        Model.QWEN_CODER_PLUS.value,
        Model.DEEPSEEK_CHAT.value,
        Model.CLAUDE_3_5_HAIKU.value, Model.GEMINI_2_5_FLASH.value, Model.GEMINI_2_5_PRO.value, Model.GPT_4_1.value, Model.GPT_4_1_MINI.value
    ]

    # MODEL_VALS = [
    #     Model.GPT_4O.value,
    #     Model.GEMINI_1_5_FLASH.value
    # #     Model.GEMINI_2_5_PRO.value,
    # #     Model.CLAUDE_3_5_HAIKU.value,
        
    # ]

    # Route 1
    count_valids_gen, count_invalids_gen, confusion_matrices_gen, tprs_gen, tnrs_gen, GV_gen, dfs_gen, label_max = validate_model(MODEL_GENS, MODEL_VALS)

    # Round 2 for all gens
    count_valids_all, count_invalids_all, confusion_matrices_all, tprs_all, tnrs_all, GV_all, dfs_all, label_max = validate_model(MODEL_VALS, MODEL_VALS)

    print("\nConfusion Matrices:")
    pprint.pprint(confusion_matrices_gen)

    print("\nValidation TPR:")
    pprint.pprint(tprs_gen)

    print("\nValidation TNR:")
    pprint.pprint(tnrs_gen)
    
    print("\nGV Array:")
    pprint.pprint(GV_all)

    llm_judge_errors(MODEL_GENS, MODEL_VALS, GV_gen)

    ensemble_results = ensemble_prediction(dfs_gen, MODEL_GENS, MODEL_VALS)

    print(f'Validation counts: invalid={count_invalids_all}, percentage={count_invalids_all / (count_valids_all + count_invalids_all) * 100:.2f}%')
    print(f'Worst pair: {label_max}')