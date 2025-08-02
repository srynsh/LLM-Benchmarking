import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import ELO_RATING, MODELS_VAL, MODELS_SHORT, pathImages
from src.validation.generated_scripts.llm_as_judge_fcnhp import ensemble_predicted_precisions

def elo_vs_precision():
    """
    Plot Elo vs Precision for the ensemble.
    """

    plt.figure(figsize=(8, 4))

    rows = [(modelName, ensemble_predicted_precisions[i]*100, ELO_RATING[modelName]) for i, modelName in enumerate(MODELS_VAL)]

    df_plot_rest = pd.DataFrame(rows, columns=['generator', 'precision', 'elo'])
    print(df_plot_rest)
    # df_plot_human = pd.DataFrame(rows, columns=['generator', 'precision'])

    sns.scatterplot(data=df_plot_rest, x='elo', y='precision', hue='generator', palette='deep', s=150)
    # sns.scatterplot(data=df_plot_human, x='elo', y='precision', hue='generator', palette='deep')
    # plt.title('Elo Rating vs Precision')
        
    # Add annotations for each point
    rows = list(df_plot_rest.iterrows()) + list(df_plot_rest.iterrows())
    for idx, row in rows:
        shorthand = MODELS_SHORT.get(row['generator'], row['generator'])
        # Use a star symbol for models in MODELS_GEN to indicate "ground truth"
        # if row['generator'] in df_pg['generator'].values:
        #     plt.scatter(row['elo'], row['precision'], color='gold', marker='*', s=150, zorder=4, label='Ground Truth' if idx == 0 else "")
        
        elo = row['elo']
        prec = row['precision']


        offset = (0, -20)
        if shorthand == 'Gemini 2.5-P':
            offset = (-90, -5)
        elif shorthand == 'DeepSeek 2.5':
            offset = (-20, -30)
        elif shorthand == 'Qwen Coder-P':
            offset = (-10, -30)
        elif shorthand == 'Sonnet 3.5':
            offset = (-70, 30)
        elif shorthand == 'Opus 3':
            offset = (-60, 20)
        elif shorthand == 'Gemini 1.5-F':
            offset = (0, -25)
        elif shorthand == 'Gemini 1.5-P':
            offset = (20, -15)
        plt.annotate(shorthand,  # Use shorthand for model name for clarity
                xy=(row['elo'], row['precision']),
                xytext=offset,  # Offset text slightly to the right
                textcoords='offset points',
                fontsize=12,
                alpha=0.8,
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', alpha=0.6))

    plt.ylim(84, 100)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.title('Elo Rating vs Precision', fontsize=16)
    plt.xlabel('Elo Rating', fontsize=14)
    plt.ylabel('Precision %', fontsize=14)
    plt.legend().remove()  # Remove the legend since we have annotations
    plt.savefig(f'{pathImages}/others/elo_precision.pdf', bbox_inches='tight')

    asdf

