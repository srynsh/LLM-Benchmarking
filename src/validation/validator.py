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
from src.regressor.config import pGa_CONST

####################
# Config
####################

### Missing annotations
# CLAUDE_3_OPUS: 2 SIDs
# GPT_4_TURBO: 17 SIDs
# GPT_4O: 2 SIDs


def main():
    """Main entry point for validation operations."""
    
    # TODO: remove hardcoding
    # Configuration
    generator_model = 'claude_3.5_haiku'
    validator_model = 'gemini-2.5-pro-preview-03-25'
    use_ground_truth = False
    
    # Option 1: Resume from existing file (for quota exceeded errors)
    resume_mode = True
    existing_file = './new_logs/new_labeller_gen_claude_3.5_haiku_val_gemini-2.5-pro-preview-03-25_2025-05-05_13-06-52.json'
    
    if resume_mode and existing_file:
        # Extract failed SIDs from existing file
        failed_sids = filter_quota_exceeded_sids(existing_file)
        
        if not failed_sids:
            print("No SIDs with quota exceeded errors found.")
            return
        
        print(f"Found {len(failed_sids)} SIDs with quota exceeded errors: {failed_sids}")
        
        # Create validation runner
        runner = ValidationRunner(generator_model, validator_model, use_ground_truth)
        
        # Run quota recovery
        runner.run_quota_recovery(existing_file)
        
    else:
        # Option 2: Run fresh validation
        sids = list(range(1, NUM_SIDS + 1))  # Replace NUM_SIDS with actual number of SIDs
        # sids = list(range(1, 367))  # All SIDs
        sids = [1, 2, 3, 4, 6, 8, 9, 12, 13, 15]  # Sample SIDs for testing
        
        # Create validation runner
        runner = ValidationRunner(generator_model, validator_model, use_ground_truth)
        
        # Run validation
        runner.run_validation(sids)
    
    print("Validation completed successfully!")

def validate_specific_sids(sids: List[int], generator_model: str, validator_model: str, 
                          use_ground_truth: bool = False) -> ValidationBatch:
    """
    Validate specific SIDs and return results as a batch.
    
    Args:
        sids: List of SIDs to validate
        generator_model: Model used for generation
        validator_model: Model used for validation
        use_ground_truth: Whether to use ground truth data
        
    Returns:
        ValidationBatch: Batch of validation results
    """
    service = ValidationService(generator_model, validator_model, use_ground_truth)
    
    results = []
    for sid in tqdm(sids, desc="Validating SIDs"):
        result = service.validate_single_sid(sid)
        results.append(result)
    
    # Create batch
    attempt_name = f"batch_gen_{generator_model}_val_{validator_model}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    batch = ValidationBatch(
        attempt_name=attempt_name,
        generator_model=generator_model,
        validator_model=validator_model,
        results=results,
        use_ground_truth=use_ground_truth
    )
    
    return batch

def compare_validation_runs(file_paths: List[str]) -> None:
    """
    Compare multiple validation runs and print comparative statistics.
    
    Args:
        file_paths: List of paths to validation result files
    """
    from src.validation.utils import load_validation_batch_from_file
    
    batches = []
    for file_path in file_paths:
        batch = load_validation_batch_from_file(file_path)
        if batch:
            batches.append(batch)
        else:
            print_warning(f"Could not load batch from {file_path}")
    
    if len(batches) < 2:
        print_error("Need at least 2 valid batches to compare")
        return
    
    print(f"\n{'='*80}")
    print(f"VALIDATION COMPARISON ({len(batches)} runs)")
    print(f"{'='*80}")
    
    for i, batch in enumerate(batches):
        stats = batch.get_summary_stats()
        print(f"\nRun {i+1}: {batch.attempt_name}")
        print(f"  Generator: {batch.generator_model}")
        print(f"  Validator: {batch.validator_model}")
        print(f"  Success Rate: {stats['success_rate']:.2%}")
        print(f"  Precision: {stats['precision']:.4f}")
        print(f"  Total Valid: {stats['total_valid']}")
        print(f"  Total Invalid: {stats['total_invalid']}")

def merge_df_y_yhat(df_y: pd.DataFrame, df_yhat: pd.DataFrame) -> pd.DataFrame:
    # Convert key columns to string type for consistent merging
    df_yhat = df_yhat.copy()
    df_yhat['sid'] = df_yhat['sid'].astype(str)
    df_yhat['line_number'] = df_yhat['line_number'].astype(str) 
    df_yhat['feedback'] = df_yhat['feedback'].astype(str)

    df_y['sid'] = df_y['sid'].astype(str)
    df_y['line_number'] = df_y['line_number'].astype(str)
    df_y['feedback'] = df_y['feedback'].astype(str)

    merged_df = df_y.merge(
        df_yhat, 
        on=['sid', 'line_number', 'feedback'], 
        how='left', 
        suffixes=('_true', '_pred')
    )
    return merged_df

def merge_df_merged_yhat(df_merged: pd.DataFrame, df_yhat: pd.DataFrame, model) -> pd.DataFrame:
    # Convert key columns to string type for consistent merging
    df_merged['sid'] = df_merged['sid'].astype(str)
    df_merged['line_number'] = df_merged['line_number'].astype(str)
    df_merged['feedback'] = df_merged['feedback'].astype(str)

    df_yhat = df_yhat.copy()
    df_yhat['sid'] = df_yhat['sid'].astype(str)
    df_yhat['line_number'] = df_yhat['line_number'].astype(str) 
    df_yhat['feedback'] = df_yhat['feedback'].astype(str)

    
    merged_df = pd.merge(
        df_merged, 
        df_yhat, 
        on=['sid', 'line_number', 'feedback'], 
        how='left', 
        suffixes=('', f'_{model}')
    )
    return merged_df

def calculate_confusion_matrix(df_y: pd.DataFrame, df_yhat: pd.DataFrame) -> List[int]:
    """
    Calculate confusion matrix [TN, FN, FP, TP] from ground truth and predictions.
    
    Args:
        df_y: Ground truth DataFrame with columns [sid, line_number, feedback, classification]
        df_yhat: Predictions DataFrame with columns [sid, line_number, feedback, classification]
    
    Returns:
        List[int]: [TN, FN, FP, TP] counts
    """
    # Perform left outer join on sid and line_number
    merged_df = merge_df_y_yhat(df_y, df_yhat)

    y_true = merged_df['classification_true']
    y_pred = merged_df['classification_pred']

    # Calculate confusion matrix components
    tn = ((y_true == 0) & (y_pred == 0)).sum()  # True Negative
    fp = ((y_true == 0) & (y_pred == 1)).sum()  # False Positive
    fn = ((y_true == 1) & (y_pred == 0)).sum()  # False Negative
    tp = ((y_true == 1) & (y_pred == 1)).sum()  # True Positive
    
    return [tn, fp, fn, tp]

def tpr_tnr(tn, fp, fn, tp):
    """
    Calculate True Positive Rate (TPR) and True Negative Rate (TNR).
    
    Args:
        tn: True Negatives
        fp: False Positives
        fn: False Negatives
        tp: True Positives

    Returns:
        Tuple[float, float]: (TPR, TNR)
    """
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    return tpr, tnr

def tpr_tnr_list(confusion_matrices):
    """
    Calculate TPR and TNR for a list of confusion matrices.
    
    Args:
        confusion_matrices: List of confusion matrices [TN, FP, FN, TP]
    Returns:

        List[Tuple[float, float]]: List of (TPR, TNR) tuples
    """
    tn_cumulative = 0
    fp_cumulative = 0
    fn_cumulative = 0
    tp_cumulative = 0
    

    for cm in confusion_matrices:
        tn, fp, fn, tp = cm
        tn_cumulative += tn
        fp_cumulative += fp
        fn_cumulative += fn
        tp_cumulative += tp

    tpr, tnr = tpr_tnr(tn_cumulative, fp_cumulative, fn_cumulative, tp_cumulative)
    return (tpr, tnr)

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
    GV = []
    dfs = {}
    

    for modelGen in MODEL_GENS:
        row_gen = []
        row_gv = []
        df_merged = pd.DataFrame()

        for j, modelVal in enumerate(MODEL_VALS):
            dataProvider = DataProvider(modelGen, modelVal)
            if len(dataProvider.get_failed_sids()) > count_invalid_max:
                count_invalid_max = len(dataProvider.get_failed_sids())
                label_max = f'{modelGen} -> {modelVal}'
            
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
        

        confusion_matrices_gen.append(row_gen)
        GV.append(row_gv)
        dfs[modelGen] = df_merged
    
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

def invalid_voting(df, validation_columns, invalid_count):
    """
    Perform invalid voting based on the validation columns.
    
    Args:
        df: DataFrame containing validation results
        validation_columns: List of columns to consider for voting
    
    Returns:
        Series: Invalid votes count for each row
    """
    # Count how many models predict invalid (0) for each row
    invalid_votes = (df[validation_columns] == 0).sum(axis=1)
    
    # Ensemble rule: predict invalid (0) if at least 4 models predict invalid (0)
    df[f'ensemble_i{invalid_count}'] = (invalid_votes >= invalid_count).astype(int)
    
    # Flip the logic: when enough models say invalid (0), we predict invalid (0)
    df[f'ensemble_i{invalid_count}'] = 1 - df[f'ensemble_i{invalid_count}']

    return df

def valid_voting(df, validation_columns, valid_count):
    """
    Perform valid voting based on the validation columns.
    
    Args:
        df: DataFrame containing validation results
        validation_columns: List of columns to consider for voting
    
    Returns:
        Series: Valid votes count for each row
    """
    # Count how many models predict valid (1) for each row
    valid_votes = (df[validation_columns] == 1).sum(axis=1)
    
    # Ensemble rule: predict valid (1) if at least 4 models predict valid (1)
    df[f'ensemble_v{valid_count}'] = (valid_votes >= valid_count).astype(int)

    return df

def calculate_ensemble_accuracy(df, model_gen, ensemble_label):
    """
    Calculate ensemble accuracy for a specific model generation.

    Args:
        df: DataFrame containing validation results
        ensemble_label: Ensemble label for the model generation

    Returns:
        float: Ensemble accuracy for the model generation
    """
    # Calculate confusion matrix for ensemble vs ground truth
    y_true = df['classification']
    y_pred = df[ensemble_label]
    
    # Remove NaN values
    # mask = ~(y_true.isna() | y_pred.isna())
    # y_true = y_true[mask]
    # y_pred = y_pred[mask]
    
    # Calculate confusion matrix components
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Predicted error
    predScore = (tp + fp) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    error = abs(predScore - pGa_CONST[model_gen])

    return error


def ensemble_prediction(dfs_gen, MODEL_GENS, MODEL_VALS, valid_count=None, invalid_count=None):
        """
        Create ensemble predictions and calculate errors for each dataset.
        
        Args:
            dfs_gen: Dictionary of dataframes for different generator models
        
        Returns:
            Dictionary containing ensemble results and errors for each model
        """
        # Get model validation columns (exclude 'classification' which is ground truth)
        validation_columns = [f'classification_{model}' for model in MODEL_VALS]
        num_validators = len(validation_columns)

        ensemble_max_errors = {f'v{i}': 0 for i in range(1, num_validators + 1)}
        ensemble_mean_errors = {f'v{i}': 0 for i in range(1, num_validators + 1)}
        ensemble_max_errors_modelGen = {f'v{i}': None for i in range(1, num_validators + 1)}

        ensemble_max_errors.update({f'i{i}': 0 for i in range(1, num_validators + 1)})
        ensemble_mean_errors.update({f'i{i}': 0 for i in range(1, num_validators + 1)})
        ensemble_max_errors_modelGen.update({f'i{i}': None for i in range(1, num_validators + 1)})
        

        for model_gen in MODEL_GENS:
            df = dfs_gen[model_gen].copy()

            # Perform valid and invalid voting for each validator
            for i in range(1, num_validators + 1):
                df = valid_voting(df, validation_columns, i)
                error = calculate_ensemble_accuracy(df, model_gen, f'ensemble_v{i}')
                max_error = max(ensemble_max_errors[f'v{i}'], error)
                ensemble_max_errors[f'v{i}'] = max_error
                ensemble_max_errors_modelGen[f'v{i}'] = model_gen if max_error == error else ensemble_max_errors_modelGen[f'v{i}']
                ensemble_mean_errors[f'v{i}'] += error

                df = invalid_voting(df, validation_columns, i)
                error = calculate_ensemble_accuracy(df, model_gen, f'ensemble_i{i}')
                max_error = max(ensemble_max_errors[f'i{i}'], error)
                ensemble_max_errors[f'i{i}'] = max_error
                ensemble_max_errors_modelGen[f'i{i}'] = model_gen if max_error == error else ensemble_max_errors_modelGen[f'i{i}']
                ensemble_mean_errors[f'i{i}'] += error

            # Replace column names that start with 'classification' to start with 'y'
            df.columns = [col.replace('classification', 'y') if col.startswith('classification') else col for col in df.columns]
            df.to_csv(f'{pathLogs}/ensemble/df_{model_gen}.csv', index=False)

        # Calculate mean errors
        for key in ensemble_mean_errors:
            ensemble_mean_errors[key] /= len(MODEL_GENS)

        # Print the max and mean errors for each ensemble
        print("\nEnsemble Max Errors:")
        for key, value in ensemble_max_errors.items():
            print(f"\t{key}: {value * 100:.2f}% ({ensemble_max_errors_modelGen[key]})")
        print("\nEnsemble Mean Errors:")
        for key, value in ensemble_mean_errors.items():
            print(f"\t{key}: {value * 100:.2f}%")


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