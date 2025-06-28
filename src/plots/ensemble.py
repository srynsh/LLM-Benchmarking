from src.validation.generated_scripts.afterRepair.summary import (ensemble_max_errors)
from src.config import pathImages
import pandas as pd
import matplotlib.pyplot as plt

pathEnsemble = f'{pathImages}/ensemble'

def valid_invalid_error():
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
    plt.title('Comparison of Valid voting and Invalid voting Errors', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 15), fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 50)  # Set y-axis limit to 0-50% for better visibility
    plt.tight_layout()
    plt.savefig(f'{pathEnsemble}/valid_invalid_error.pdf')