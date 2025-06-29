from src.config import Model, pathImages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import MODELS_SHORT, MODELS_VAL, VALIDATOR_REPAIR_SUFFIX, ValidatorRepairConfig
from src.plots.validator_plots import get_df_validator_tpr_tnr

pathEnsemble = f'{pathImages}/ensemble'

####################
# True Positive Rate vs True Negative Rate Plot
####################
def get_df_ensemble_tpr_tnr(ensemble_tpr, ensemble_tnr, match):
    rows = []

    for key in ensemble_tpr.keys():
        if key.startswith(match):
            tpr = ensemble_tpr[key] * 100  # Convert to percentage
            tnr = ensemble_tnr[key] * 100  # Convert to percentage
            rows.append({'ensemble': int(key[1:]), 'tpr': tpr, 'tnr': tnr})

    rows.sort(key=lambda x: x['ensemble'])  # Sort by ensemble index

    df = pd.DataFrame(rows)
    return df

def tpr_tnr_ensemble(ensemble_tpr, ensemble_tnr, validator_tpr, validator_tnr):
    '''Plot the TPR and TNR for the ensemble models.'''
    # Parse the data
    df_v = get_df_ensemble_tpr_tnr(ensemble_tpr, ensemble_tnr, 'v')
    df_i = get_df_ensemble_tpr_tnr(ensemble_tpr, ensemble_tnr, 'i')
    df_validator = get_df_validator_tpr_tnr(validator_tpr, validator_tnr)

    # Create the plot
    plt.figure(figsize=(8, 4))
    plt.plot(df_v['tpr'], df_v['tnr'], marker='^', linewidth=2, label='valid voting', color='blue')
    plt.plot(df_i['tpr'], df_i['tnr'], marker='v', linewidth=2, label='invalid voting', color='red')
    plt.scatter(df_validator['tpr'], df_validator['tnr'], marker='o', color='green', s=20, label='validator models')

    plt.legend(['valid voting', 'invalid voting', 'LLM as judge'], fontsize=12)

    # Add small labels for valid and invalid voting next to markers.
    for i, row in df_v.iterrows():
        plt.text(row['tpr'] - 1, row['tnr'] - 8, f'v{int(row["ensemble"])}', fontsize=10, color='black', ha='left')
    for i, row in df_i.iterrows():
        plt.text(row['tpr'] + 1, row['tnr'] + 5, f'i{int(row["ensemble"])}', fontsize=10, color='black', ha='right')

    plt.ylim(0,100)
    plt.xlim(40,100)

    # Labels and fontsize
    plt.xlabel('True Positive Rate %', fontsize=14)
    plt.ylabel('True Negative Rate %', fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{pathEnsemble}/tpr_tnr_ensemble{VALIDATOR_REPAIR_SUFFIX}.pdf', bbox_inches='tight')


####################
# Valid and Invalid Voting Errors
####################
def valid_invalid_error(ensemble_max_errors):
    '''Plot the valid and invalid voting errors from the ensemble summary data.'''

    # Parse the data
    v_data = []
    i_data = []

    for key, value in ensemble_max_errors.items():
        value = value * 100  # Convert to percentage
        if key.startswith('v'):
            v_data.append({'index': int(key[1:]), 'value': value})
        elif key.startswith('i'):
            i_data.append({'index': int(key[1:]), 'value': value})

    # Sort the data by index
    v_data.sort(key=lambda x: x['index'])
    i_data.sort(key=lambda x: x['index'])

    # Create dataframes
    df_v = pd.DataFrame(v_data)
    df_i = pd.DataFrame(i_data)

    # Create the plot
    plt.figure(figsize=(8, 4))
    plt.plot(df_i['index'], df_i['value'], marker='s', linewidth=2, label='invalid voting', color='red')
    plt.plot(df_v['index'], df_v['value'], marker='o', linewidth=2, label='valid voting', color='blue')

    plt.legend(['invalid voting', 'valid voting'], fontsize=12)

    # Adjust the labels and sizing
    plt.xlabel('Votes', fontsize=14)
    plt.ylabel('Error (%)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 15), fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 50)  # Set y-axis limit to 0-50% for better visibility
    plt.tight_layout()
    plt.savefig(f'{pathEnsemble}/valid_invalid_error{VALIDATOR_REPAIR_SUFFIX}.pdf')


####################
# Repair vs ensemble error
####################
def get_repair_suffixes():
    # List of flags in ValidatorRepairConfig
    VALIDATOR_REPAIR_FLAGS = ValidatorRepairConfig.flags()
    validator_repair = ValidatorRepairConfig()
    
    # Set them false initially and append to suffixes
    for flag in VALIDATOR_REPAIR_FLAGS:
        setattr(validator_repair, flag, False)
    suffixNoRepair = validator_repair.getSuffix()

    # Set them all to true initially and append to suffixes
    for flag in VALIDATOR_REPAIR_FLAGS:
        setattr(validator_repair, flag, True)
    suffixAllTrue = validator_repair.getSuffix()

    # Cumulatively, grow the suffixes with each flag set to True
    suffixes = [suffixNoRepair]
    suffixCummulative = suffixAllTrue[0]
    for key in suffixAllTrue[1:]:
        suffixCummulative += key
        suffixes.append(suffixCummulative)

    return suffixes

def get_repair_ensemble_data(suffixes):
    '''Load the ensemble max errors for a given repair suffix.'''
    rows = []

    for suffix in suffixes:
        repair = ValidatorRepairConfig.getName(suffix)

        try:
            # Load the module dynamically based on the suffix
            module = __import__(f'src.validation.generated_scripts.summary{suffix}', fromlist=['ensemble_max_errors'])
            ensemble_max_errors = module.ensemble_max_errors
            count_invalids_all = module.count_invalids_all
            percentage_invalids = module.percentage_invalids

            # If success
            if ensemble_max_errors:
                ensemble_max_error_majority = ensemble_max_errors['v8'] * 100  # Majority voting error
                ensemble_max_error_best = min(ensemble_max_errors.values()) * 100  # Best voting error
                

                # Attach to DF rows
                row = [f'{repair}', count_invalids_all, percentage_invalids, ensemble_max_error_majority, ensemble_max_error_best]
                rows.append(row)
        except ImportError as e:
            pass
            # print(f"Error importing module for suffix {suffix}: {e}")

    df = pd.DataFrame(rows, columns=['repair', 'count_invalids_all', 'percentage_invalids', 'error_majority', 'error_best'])
    return df

def plot_repair_vs_ensemble(df):
    '''Plot the repair vs ensemble errors. On x-axis, we have the repair configurations, and on y1-axis, we have the count of invalids, and on y2-axis, we have the error percentages.'''
    # Set the figure size
    plt.figure(figsize=(10, 6))

    # Create a twin axis for the count of invalids
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # Plot the count of invalids on the first y-axis
    sns.barplot(x='repair', y='percentage_invalids', data=df, ax=ax1, color='lightgray', alpha=0.7)
    ax1.set_ylabel('Missing Values (%)', fontsize=14)
    ax1.set_xlabel('Repair Undertaken', fontsize=14)
    ax1.tick_params(axis='y', labelsize=12)

    # Plot the error percentages on the second y-axis
    sns.lineplot(x='repair', y='error_majority', data=df, ax=ax2, marker='s', color='blue', label='Ensemble Majority', dashes=(2, 2))
    sns.lineplot(x='repair', y='error_best', data=df, ax=ax2, marker='o', color='red', label='Ensemble Best', dashes=(4, 4), linestyle=':')

    ax2.set_ylabel('Error Percentage (%)', fontsize=14)
    ax2.tick_params(axis='y', labelsize=12)

    # Set the x-ticks to be the repair configurations
    ax1.set_xticks(range(len(df['repair'])))
    ax1.set_xticklabels(df['repair'], fontsize=12) #  rotation=45, ha='right', 
    ax1.set_ylim(0, 15)
    ax2.set_ylim(0, 15)  # Set y-axis limit to 0-100% for better visibility

    # Add a legend
    # ax2.legend(['Majority Error', 'Best Error'], fontsize=12, loc='upper left')

    plt.tight_layout()
    plt.savefig(f'{pathEnsemble}/repair_vs_ensemble_error{VALIDATOR_REPAIR_SUFFIX}.pdf', bbox_inches='tight')

def repair_vs_ensemble_error():
    '''Plot the repair vs ensemble errors.'''

    # Get the suffixes for the repair configurations
    suffixes = get_repair_suffixes()

    # Load the DF for each suffix
    df = get_repair_ensemble_data(suffixes)

    # Plot the data
    plot_repair_vs_ensemble(df)
