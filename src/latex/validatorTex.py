
from dataframe.validator import tpr_tnr_prec

# =========================
#   TPR & TNR of a model
# =========================
def getValAcc(chosenGen, MODELS, dfs, valAcc):
    for generator in chosenGen:
        for validator in MODELS:
            y = dfs[generator]['y'] if 'y' in dfs[generator] else None
            y_hat = dfs[generator][validator]
            TPR, TNR, ACC, precision, precision_hat = tpr_tnr_prec(y, y_hat)
            
            if generator not in valAcc:
                valAcc[generator] = {}
            valAcc[generator][validator] = {'tpr': TPR, 'tnr': TNR, 'acc': ACC}
            
def generate_tpr_tnr_model(dfs, generators, MODELS, mapping, fname):
    symbolHash = {'tpr': '\\tpr{j}', 'tnr': '\\tnr{j}', 'acc': '$\\accuracyV{j}$'}
    valAcc = {}

    getValAcc(generators, MODELS, dfs, valAcc)

    for generator in generators:
        # Model name
        name = mapping[generator]
        print(f'\\multirow{{3}}{{*}}{{\\textbf{{{name}}}}}', end='')

        # TPR TNR and ACC
        for metric in ['tpr', 'tnr', 'acc']:
            symbol = symbolHash[metric]
            print(f' & {symbol} ', end='')
            
            for validator in MODELS:
                value = valAcc[generator][validator][metric]
                value = round(value, 1)
                print(f'& {value}\\%', end=' ')

            print('\\\\')

# =========================
#   TPR & TNR across models
# =========================
