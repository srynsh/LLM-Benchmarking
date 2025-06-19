from src.regressor.config import pGa_CONST
from src.config import NUM_SIDS, Model, pathLogs

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