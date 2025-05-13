import json
import os
import numpy as np
import pandas as pd
import math
import re

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc


# =========================
#   TPR & TNR & PRECISION
# =========================
def tpr_tnr_prec(y_true, y_hat, rounding=3):
    # TP counts
    TP, TN, FP, FN = None, None, None, None
    if y_true is not None:
        TP = np.sum((y_true == 1) & (y_hat == 1))
        TN = np.sum((y_true == 0) & (y_hat == 0))
        FP = np.sum((y_true == 0) & (y_hat == 1))
        FN = np.sum((y_true == 1) & (y_hat == 0))

    # Valid counts
    valid = np.sum(y_true == 1) if y_true is not None else None
    invalid = np.sum(y_true == 0) if y_true is not None else None
    valid_hat = np.sum(y_hat == 1)
    invalid_hat = np.sum(y_hat == 0)

    # Prec
    precision = valid / (valid + invalid) if y_true is not None else None
    precision_hat = valid_hat / (valid_hat + invalid_hat)

    # TPR
    TPR = TP / (TP + FN) if y_true is not None and (TP + FN) > 0 else None
    TNR = TN / (TN + FP) if y_true is not None and (TN + FP) > 0 else None
    ACC = (TP + TN) / (TP + TN + FP + FN) if y_true is not None else None

    TPR = round(TPR*100, rounding) if TPR else None
    TNR = round(TNR*100, rounding) if TNR else None
    ACC = round(ACC*100, rounding) if ACC else None
    precision = round(precision*100, rounding) if precision else None
    precision_hat = round(precision_hat*100, rounding)

    return TPR, TNR, ACC, precision, precision_hat
    

# =========================
#   PREDICTED MATRIX
# =========================
def generate_predicted_matrix_latex(dfs, MODELS, mapping, fnamePredMatrix):
    genMeanHash = {}
    genPrecHash = {}
    latexStr = ''

    for generator in MODELS:
        # Name
        name = mapping[generator]
        latexStr += f'\\textbf{{{name}}} & '

        # Pij
        precs = []
        for validator in MODELS:
            y = dfs[generator]['y'] if 'y' in dfs[generator] else None
            y_hat = dfs[generator][validator]

            TPR, TNR, ACC, precision, precision_hat = tpr_tnr_prec(y, y_hat)
            precs.append(precision_hat)
            p = round(precision_hat,1)
            latexStr += f'{p}\\% & '
        
        # Pij mean
        p = round(np.mean(precs),1)
        genMeanHash[generator] = p
        genPrecHash[generator] = precision
        latexStr += f'{p}\\% & '

        # Ground truth precision
        if y is not None:
            p = round(precision,1)
            latexStr += f'{p}\\%'

        latexStr += ' \\\\ \n'

    open(fnamePredMatrix, 'w').write(latexStr)

    return genMeanHash, genPrecHash

# =========================
#   PREDICTED MATRIX ERROR
# =========================
def generate_predicted_matrix_error(dfs, genMeanHash, genPrecHash, MODELS, MODELS_GEN, mapping, fnamePredMatrix):
    errorsHash = {}
    latexStr = ''

    def printErr(fun, errorsHash):
        tempStr = ''
        for validator in MODELS:
            errors = errorsHash[validator]
            error = round(fun(errors),1)
            tempStr += f'{error}\% & '

        return tempStr

    def meanPrinting(fun, errorsHash):
        tempStr = ''
        errors = [abs(genMeanHash[generator] - genPrecHash[generator])
            for generator in MODELS_GEN]
        error = round(fun(errors),1)
        tempStr += f'{error}\% & '

        return tempStr

    for validator in MODELS:
        precs = []
        errors = []
        
        for generator in MODELS_GEN:
            y = dfs[generator]['y']
            y_hat = dfs[generator][validator]

            TPR, TNR, ACC, precision, precision_hat = tpr_tnr_prec(y, y_hat)
            precs.append(precision_hat)
            error = abs(precision - precision_hat) if precision is not None else 0
            if precision is not None:
                errors.append(error) 
        
        errorsHash[validator] = errors

    latexStr += '\\textbf{Min Error} & '
    latexStr += printErr(np.min, errorsHash)
    latexStr += meanPrinting(np.min, errorsHash)
    latexStr += '-- \\\\ \n'

    latexStr += '\\textbf{Mean Error} & '
    latexStr += printErr(np.mean, errorsHash)
    latexStr += meanPrinting(np.mean, errorsHash)
    latexStr += '-- \\\\ \n'

    latexStr += '\\textbf{Max Error} & '
    latexStr += printErr(np.max, errorsHash)
    latexStr += meanPrinting(np.max, errorsHash)
    latexStr += '-- \\\\ \n'

    open(fnamePredMatrix, 'a').write(latexStr)