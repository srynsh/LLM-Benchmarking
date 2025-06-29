from src.config import Model, pathImages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import MODELS_SHORT, MODELS_VAL, VALIDATOR_REPAIR_SUFFIX

pathValidator = f'{pathImages}/validator'

####################
# True Positive Rate vs True Negative Rate Plot
####################
def get_df_validator_tpr_tnr(validator_tpr, validator_tnr):
    # Create the DF
    df_validator = pd.DataFrame({
        'validator': [model for model in MODELS_VAL],
        'tpr': [tpr * 100 for tpr in validator_tpr],
        'tnr': [tnr * 100 for tnr in validator_tnr]
    })
            
    return df_validator

def add_offsets(df_validator):
    # Improved annotation positioning to avoid overlaps in the 10-30 TNR range and 1200-1300 Elo range
    model_offsets = {
        Model.GPT_4_TURBO.value: (-30, 10),       # Move up and right
        Model.GEMINI_1_5_FLASH.value: (15, -5), # Move down and right

        Model.GPT_4O_MINI.value: (10, -5),      # Move down and right
        Model.GPT_4O.value: (30, 10),          # Move down and left
        Model.CLAUDE_3_OPUS.value: (-20, -65),    # Move up and left
        Model.CLAUDE_3_5_SONNET.value: (40, -65),   # Move right

        Model.GEMINI_1_5_PRO.value: (-10, -30),  # Move down and left
        Model.QWEN_CODER_PLUS.value: (-90, -25),  # Move up and left
        Model.DEEPSEEK_CHAT.value: (-45, -60),   # Move down and left
        Model.CLAUDE_3_5_HAIKU.value: (-45, -30), # Move down and right
        Model.GPT_4_1.value: (-256, -15),  # Move up and left
        Model.GPT_4_1_MINI.value: (-250, -10), # Move up and right

        Model.GEMINI_2_5_FLASH.value: (-25, -25), # Keep default
        Model.GEMINI_2_5_PRO.value: (-25, -25),   # Keep default
    }

    # Add annotations for each point
    for idx, row in df_validator.iterrows():
        shorthand = MODELS_SHORT.get(row['validator'], row['validator'])
        x_offset, y_offset = model_offsets.get(row['validator'], (0, 0))
        
        if shorthand in [MODELS_SHORT[Model.GEMINI_1_5_FLASH.value], 
                        MODELS_SHORT[Model.GEMINI_2_5_PRO.value], 
                        MODELS_SHORT[Model.GEMINI_2_5_FLASH.value], 
                        MODELS_SHORT[Model.QWEN_CODER_PLUS.value]]: 
            
            # Add arrow pointing to the marker
            plt.annotate(shorthand,
                            xy=(row['tpr'], row['tnr']), 
                            xytext=(x_offset, y_offset),
                            textcoords='offset points',
                            fontsize=12,
                            alpha=0.8,
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', alpha=0.6))

def tpr_tnr_validator(validator_tpr, validator_tnr):
    plt.figure(figsize=(8, 4))

    df_validator = get_df_validator_tpr_tnr(validator_tpr, validator_tnr)
    sns.scatterplot(data=df_validator, x='tpr', y='tnr', hue='validator', style='validator', palette='deep')

    # Add offsets to the annotations
    add_offsets(df_validator)

    # Use shorthand names for the legend instead of full model names
    handles, labels = plt.gca().get_legend_handles_labels()
    shorthand_labels = [MODELS_SHORT.get(label, label) for label in labels]
    plt.legend(handles, shorthand_labels,  bbox_to_anchor=(0.5, 1.05), loc='lower center', ncol=6)

    plt.ylim(0,60)
    plt.xlim(82,100)

    # Labels and fontsize
    plt.xlabel('True Positive Rate %', fontsize=14)
    plt.ylabel('True Negative Rate %', fontsize=14)

    plt.axhline(y=25, color='gray', linestyle='--', alpha=0.2)
    plt.axvline(x=96, color='gray', linestyle='--', alpha=0.2)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{pathValidator}/tpr_tnr_validator{VALIDATOR_REPAIR_SUFFIX}.pdf', bbox_inches='tight')