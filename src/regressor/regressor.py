import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import   get_GV, get_VALIDATOR_COUNTS, get_precision
from config import PV_START, PVIV_START, PG_START, NUM_RUNS, ERR_EPSILON_PG, ERR_EPSILON_PIV, ERR_EPSILON_PV, GENS, MODELS, MODEL_NAMES, MODEL_ENUM, MAX_WORKERS
from config import VALIDATOR_COUNTS_CONST, pGa_CONST, GV_CONST, PVVA_CONST, PVIVA_CONST
import sys
np.random.seed(42)
scipy_seed = 42

import warnings
warnings.filterwarnings("always", category=UserWarning)

"""CONSTANTS"""

pGa = pGa_CONST if pGa_CONST is not None else get_precision()
GV = GV_CONST if GV_CONST is not None else get_GV()

"""-----------------CONSTANTS END HERE----------------"""

def loss_function(pVv, pViv, pG, GV):
    """
    Calculate the cross-entropy loss between the predicted and actual gene-variant matrices.
    
    This function computes a binary cross-entropy loss between the generator precision 
    prediction by validator (GV) and its estimate by regression method (GV_hat). 
    The GV_hat matrix is constructed using outer products of generator and validator performance vectors.
    
    Parameters:
    -----------
    pVv : numpy.ndarray
        Probability vector of a validator classifying output as valid.
    pViv : numpy.ndarray
        Probability vector of a validator classifying output as invalid.
    pG : numpy.ndarray
        Probability vector for generator output being valid.
    GV : numpy.ndarray
        Generator precision prediction by validator matrix.
    
    Returns:
    --------
    float
        Negative mean of the cross-entropy values, representing the loss.
        
    Notes:
    ------
    A small epsilon (1e-9) is added to avoid log(0) errors through clipping.
    """
    G_hat = np.outer(pG, pVv) + np.outer((1 - pG), (1 - pViv))
    epsilon = 1e-9
    return -np.mean([
        GV[i, j]*np.log(np.clip(G_hat[i, j], epsilon, 1))
        + (1 - GV[i, j])*np.log(np.clip(1 - G_hat[i, j], epsilon, 1))
        for i in range(GV.shape[0])
        for j in range(GV.shape[1])
    ])

def reg(p_hat, p, id):
    """
    Calculate the root mean squared error (RMSE) between predicted and actual values.

    This function computes the RMSE for a specific subset of indices, comparing
    predictions from a model to the actual values.

    Parameters:
    ----------
    p_hat : dict
        Dictionary mapping model names to their predictions.
    p : list or array-like
        The actual/ground truth values.
    id : list or array-like
        Indices for which to calculate the error. If empty, returns 0.

    Returns:
    -------
    float
        The RMSE between predictions and actual values for the specified indices.
        Returns 0 if `id` is empty.
    """
    return np.sqrt(np.mean([(p[i] - p_hat[MODEL_ENUM[i]])**2 for i in id]) if len(id) > 0 else 0)

GLOBAL_LOSS_COUNT = 0
def total_loss(x, GV, pVva, pViva, pGa, idV, idG, w):
    NUM_VALIDATORS = GV.shape[1]
    NUM_GENERATORS = GV.shape[0]
    l1 = loss_function(x[:NUM_VALIDATORS], x[NUM_VALIDATORS:2*NUM_VALIDATORS], x[2*NUM_VALIDATORS:], GV)
    
    l2 = reg(x[2*NUM_VALIDATORS:], pGa, idG)
    l3_1 = reg(x[:NUM_VALIDATORS], pVva, idV)
    l3_2 = reg(x[NUM_VALIDATORS:2*NUM_VALIDATORS], pViva, idV)

    global GLOBAL_LOSS_COUNT
    GLOBAL_LOSS_COUNT += 1
    # print(f"{GLOBAL_LOSS_COUNT}. Loss GV: {l1}, pG: {l2}, pV+: {l3_1}, pV-: {l3_2}")

    return l1 + w[0]*l2 + w[1]*l3_1 + w[2]*l3_2

def estimate_probs(GV, pVva, pViva, pGa, idV, idG, w = [1, 0, 0]):
    res = None
    val = np.inf

    NUM_VALIDATORS = GV.shape[1]
    NUM_GENERATORS = GV.shape[0]

    if PG_START == 'mean':
        pG_ = np.mean(GV, axis=1)
    elif PG_START == 'uniform':
        pG_ = np.random.uniform(0, 1, NUM_GENERATORS)
    else:
        pG_ = np.ones(NUM_GENERATORS) * PG_START

    if PV_START == 'uniform':
        pVv_hat_ = np.random.uniform(0, 1, NUM_VALIDATORS)
    else:
        pVv_hat_ = np.ones(NUM_VALIDATORS) * PV_START

    if PVIV_START == 'uniform':
        pViv_hat_ = np.random.uniform(0, 1, NUM_VALIDATORS)
    else:
        pViv_hat_ = np.ones(NUM_VALIDATORS) * PVIV_START

    print(f"pG: {pG_}, pVv: {pVv_hat_}, pViv: {pViv_hat_}")
    for run_count in range (NUM_RUNS):
        if run_count != 0:
            pG = pG_ + np.random.uniform(-ERR_EPSILON_PG/2, ERR_EPSILON_PG/2, NUM_GENERATORS)
            pVv_hat = pVv_hat_ + np.random.uniform(-ERR_EPSILON_PV/2, ERR_EPSILON_PV/2, NUM_VALIDATORS)
            pViv_hat = pViv_hat_ + np.random.uniform(-ERR_EPSILON_PIV/2, ERR_EPSILON_PIV/2, NUM_VALIDATORS)
        else:
            pG, pVv_hat, pViv_hat = pG_, pVv_hat_, pViv_hat_

        pG = np.clip(pG, 0, 1)
        pVv_hat = np.clip(pVv_hat, 0, 1)
        pViv_hat = np.clip(pViv_hat, 0, 1)  

        x = np.concatenate([pVv_hat, pViv_hat, pG])

        res1 = minimize(total_loss, x, args=(GV, pVva, pViva, pGa, idV, idG, w), bounds=[(0, 1)] * len(x), method='L-BFGS-B')
        if(res1.success == False):
            print(f"\033[91mRun {run_count} failed to converge\033[0m")

        res1 = res1.x
        if total_loss(res1, GV, pVva, pViva, pGa, idV, idG, w) < val:
            val = total_loss(res1, GV, pVva, pViva, pGa, idV, idG, w)
            res = res1

    pVv = res[:NUM_VALIDATORS]
    pViv = res[NUM_VALIDATORS:2*NUM_VALIDATORS]
    pG = res[2*NUM_VALIDATORS:]

    return pVv, pViv, pG

def print_GV_hat(pVv, pViv, pG_min, pG_max, pG_mean, PVVA, PVIVA):
    G_hat = np.outer(pG_mean, pVv) + np.outer((1 - pG_mean), (1 - pViv))
    
    s = r'''
  \begin{adjustbox}{max width=\textwidth}
     \begin{tabular}{@{}ccccccccccc||cccc}
        \toprule
        &  & \multicolumn{10}{c}{\textbf{Validators}} \\
        \cmidrule(l){2-15}
         \textbf{Generators} & \textbf{3.5-turbo} & \textbf{4-turbo} & \textbf{4o-mini} & \textbf{4o} & \textbf{3 opus} & \textbf{3.5 sonnet} & \textbf{1.5 flash} & \textbf{1.5 pro} & \textbf{qwen} & \textbf{deepseek} & \textbf{$\hat{\pgen}$} & \textbf{$\max(\hat{\pgen})$} & \textbf{$\min(\hat{\pgen})$} & GT \\
    \midrule
    '''

    print(s)

    for i, row in enumerate(G_hat):
        print(f'\\textbf{{{MODEL_NAMES[i]}}} & ', end='')

        for val in row:
            print(f"{(100*val):.1f}\\%", end=' & ')

        print(f"{(100*pG_mean[i]):.1f}\\%", end=' & ')
        print(f"{(100*pG_max[i]):.1f}\\%", end=' & ')
        print(f"{(100*pG_min[i]):.1f}\\%", end=' & ')
        if pGa.get(MODELS[i], None) is not None:
            print(f"{(100*pGa[MODELS[i]]):.1f}\\% \\\\")
        else:
            print(f"\\\\")

    print(r'\midrule')

    s1 = r'''\bottomrule
    \end{tabular}%
  \end{adjustbox}
\end{table*}'''

    print(r'$\hat{\pvalid}$ ', end='')
    for x in pVv:
        print(f'& {100*x:.1f} ', end='')

    print(r'\\')

    print(r'$\pvalid$ ', end='')
    for x in PVVA:
        print(f'& {100*x:.1f} ', end='')

    print(r'\\')

    print(r'$\hat{\pinvalid}$ ', end='')
    for x in pViv:
        print(f'& {100*x:.1f} ', end='')

    print(r'\\')

    print(r'$\pinvalid$ ', end='')
    for x in PVIVA:
        print(f'& {100*x:.1f} ', end='')

    print(r'\\')

    print(s1)

def get_pViv_full(gens, VALIDATOR_COUNTS):
    pViva = {}

    stats = np.zeros_like(VALIDATOR_COUNTS[0])
    for i in gens:
        stats += VALIDATOR_COUNTS[GENS.index(i)]

    pViva = {k: (stats[i][0])/(stats[i][0] + stats[i][1]) for i, k in enumerate(MODELS)}
    pVva = {k: (stats[i][3])/(stats[i][2] + stats[i][3]) for i, k in enumerate(MODELS)}

    return pViva, pVva

def regress(GV, pGa, gens, k1, VALIDATOR_COUNTS, w=[1, 0, 0]):
    idG = list(combinations(pGa.keys(), k1))
    pGs = []

    errors = []
    avg_errors = []

    logs = []

    numComb = 0
    for j in idG:
        stats = np.zeros_like(VALIDATOR_COUNTS[0])

        for i in j:
            stats += VALIDATOR_COUNTS[gens.index(i)]

        if k1 == 0:
            pVv, pViv, pG = estimate_probs(GV, {}, {}, pGa, (), j, w=w)
        else:
            pVva  = {k: (stats[i][3])/(stats[i][2] + stats[i][3]) for i, k in enumerate(MODELS)}
            pViva = {k: (stats[i][0])/(stats[i][0] + stats[i][1]) for i, k in enumerate(MODELS)}

            pVv, pViv, pG = estimate_probs(GV, pVva, pViva, pGa, MODELS, j, w=w)

        logs.append((pVv, pViv, pG, j))
        pGs.append(pG)

        excluded = [candidate for candidate in gens if candidate not in j]
        p_temp = np.array([pGa[m] for m in excluded])
        p_hat_temp = np.array([pG[MODEL_ENUM[m]] for m in excluded])

        avg_errors.append(100*np.mean(abs(p_hat_temp - p_temp)))
        errors.append(100*max(abs(p_hat_temp - p_temp)) if len(p_temp) > 0 else 0)

        numComb += 1
        GLOBAL_LOSS_COUNT = 0
        print(f"\nC{numComb} k={k1} with {j}")
        print(f"\tpG: {pG}")
        print(f"\tpV+: {pVv}")
        print(f"\tpV-: {pViv}")
        print(f"\tMax Error: {errors[-1]}")
        print(f"\tAvg Error: {avg_errors[-1]}")


    errors = []
    for (pv1, pv2, pg, combo) in logs:
        idxs = [MODEL_ENUM[m] for m in pGa]
        ref = np.array([pGa[m] for m in pGa])
        errors.append(np.sum(np.abs(pg[idxs] - ref)))
    m_idx = np.argsort(errors)[len(errors)//2]
    mean_pG = logs[m_idx][2]
    min_pG = logs[np.argmin(errors)][2]
    max_pG = logs[np.argmax(errors)][2]

    mean_pVv = np.mean([combi[0] for combi in logs], axis=0)
    mean_pViv = np.mean([combi[1] for combi in logs], axis=0)

    _piv, _pv = get_pViv_full(GENS, VALIDATOR_COUNTS)

    PVVA = PVVA_CONST if PVVA_CONST is not None else np.array([_pv[m] for m in MODELS])
    PVIVA = PVIVA_CONST if PVIVA_CONST is not None else np.array([_piv[m] for m in MODELS])

    print('Mean GV_hat', k1)
    print_GV_hat(mean_pVv, mean_pViv, min_pG, max_pG, mean_pG, PVVA, PVIVA)

    return pGs, errors, avg_errors, logs

def plot_data_no_exclude(all_logs, pGa, pG_mean):
    means = []
    maxs = []
    base_mean = []
    all_errors = []
    base_max = []

    line_styles = [
        '-', '--', '-.', ':',
        (0, (5,2)), (0, (5,1)), (0, (3,1,1,1)), (0, (1,2)), (0, (3,5,1,5)), (0, (1,2)),
        (0, (2,3,2,3)), (0, (3,1,1,1,1,1)), (0, (4,2,1,2)), (0, (2,2,1,2,1,2)), (0, (3,2,1,2,1,2)), 
        (0, (4,2,1,2,1,2)), (0, (3,2,1,2,1,2,1,2)), (0, (4,2,1,2,1,2,1,2)), (0, (3,2,1,2,1,2,1,2,1,2)),
        (0, (4,2,1,2,1,2,1,2,1,2)), (0, (3,2,1,2,1,2,1,2,1,2,1,2)), (0, (4,2,1,2,1,2,1,2,1,2,1,2)), (0, (3,2,1,2,1,2,1,2,1,2,1,2,1,2)),
        (0, (4,2,1,2,1,2,1,2,1,2,1,2,1,2)), (0, (3,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2)), (0, (4,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2)),
        (0, (1, 1)), (0, (3, 3, 1, 3)), (0, (5, 5, 1, 5)), (0, (7, 7, 1, 7)),
        (0, (1, 1, 1, 1)), (0, (3, 3, 3, 3)), (0, (5, 5, 5, 5)), (0, (7, 7, 7, 7)),
        (0, (1, 1, 3, 1)), (0, (3, 3, 5, 3)), (0, (5, 5, 7, 5)), (0, (7, 7, 9, 7)),
        (0, (1, 1, 1, 3)), (0, (3, 3, 3, 5)), (0, (5, 5, 5, 7)), (0, (7, 7, 7, 9)),
        (0, (1, 1, 3, 3)), (0, (3, 3, 5, 5)), (0, (5, 5, 7, 7)), (0, (7, 7, 9, 9)),
        (0, (1, 1, 5, 1)), (0, (3, 3, 7, 3)), (0, (5, 5, 9, 5)), (0, (7, 7, 11, 7)),
        (0, (1, 1, 7, 1)), (0, (3, 3, 9, 3)), (0, (5, 5, 11, 5)), (0, (7, 7, 13, 7)),
        (0, (1, 1, 9, 1)), (0, (3, 3, 11, 3)), (0, (5, 5, 13, 5)), (0, (7, 7, 15, 7))
    ]
    cmap = plt.get_cmap('tab10', len(all_logs))

    fig, ax = plt.subplots()
    for k, log in enumerate(all_logs):
        # if 0 <= k < len(all_logs)-1:
        if k in [0, 1, 2]:
            tmp_errors = []
            for (v1, v2, pG_est, _) in log:
                idxs = [MODEL_ENUM[m] for m in pGa]
                target = [pGa[m] for m in pGa]
                tmp_errors.append(np.sum(np.abs(pG_est[idxs] - target)))
            m_idx = np.argsort(tmp_errors)[len(tmp_errors)//2]
            median_log = log[m_idx][2]
            ax.plot(median_log*100, label=f'Median k = {k}', linestyle=line_styles[k], color=cmap(k))

    ax.plot(pG_mean*100, color='black', label='Mean Prediction')
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODEL_NAMES, rotation=50, fontsize=12)
    ax.tick_params(axis='x', length=10)

    # for i, key in enumerate(pGa.keys()):
    #     ax.scatter(MODEL_ENUM[key], pGa[key]*100, color='black')

    for i, key in enumerate(pGa.keys()):
        if i == 0:
            ax.scatter(MODEL_ENUM[key], pGa[key]*100, color='black', label='Ground Truth')
        else:
            ax.scatter(MODEL_ENUM[key], pGa[key]*100, color='black')

    ax.legend()
    plt.tight_layout()
    plt.savefig(f'images/pngs/all_regressor.png')
    plt.savefig(f'images/pdfs/all_regressor.pdf', format='pdf')


    fig, ax = plt.subplots()
    for k, log in enumerate(all_logs):
        # if 0 <= k < len(all_logs)-1:
        if k in [0, 2]:
            tmp_errors = []
            for (v1, v2, pG_est, _) in log:
                idxs = [MODEL_ENUM[m] for m in pGa]
                target = [pGa[m] for m in pGa]
                tmp_errors.append(np.sum(np.abs(pG_est[idxs] - target)))
            min_log = log[np.argmin(tmp_errors)][2]
            ax.plot(min_log*100, label=f'Peek: {k}', linestyle=line_styles[k], color=cmap(k))

    ax.plot(pG_mean*100, color='black', label='Mean')
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODEL_NAMES, rotation=50, fontsize=12)
    ax.tick_params(axis='x', length=10)

    for i, key in enumerate(pGa.keys()):
        ax.scatter(MODEL_ENUM[key], pGa[key]*100, color='black')

    ax.legend()
    plt.tight_layout()
    plt.savefig(f'images/pngs/all_regressor_min.png')
    plt.savefig(f'images/pdfs/all_regressor_min.pdf', format='pdf')


    fig, ax = plt.subplots()
    for k, log in enumerate(all_logs):
        # if 0 <= k < len(all_logs)-1:
        if k in [0, 2]:
            tmp_errors = []
            for (v1, v2, pG_est, _) in log:
                idxs = [MODEL_ENUM[m] for m in pGa]
                target = [pGa[m] for m in pGa]
                tmp_errors.append(np.sum(np.abs(pG_est[idxs] - target)))
            min_log = log[np.argmax(tmp_errors)][2]
            ax.plot(min_log*100, label=f'Peek: {k}', linestyle=line_styles[k], color=cmap(k))

    ax.plot(pG_mean*100, color='black', label='Mean')
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODEL_NAMES, rotation=50, fontsize=12)
    ax.tick_params(axis='x', length=10)

    for i, key in enumerate(pGa.keys()):
        ax.scatter(MODEL_ENUM[key], pGa[key]*100, color='black')

    ax.legend()
    plt.tight_layout()
    plt.savefig(f'images/pngs/all_regressor_max.png')
    plt.savefig(f'images/pdfs/all_regressor_max.pdf', format='pdf')


    for k, log in enumerate(all_logs):
        means.append([])
        maxs.append([])
        all_errors.append([])
        base_mean.append([])
        base_max.append([])

        fig, ax = plt.subplots()

        if 0 < k < len(all_logs)-1:
            ax.plot(np.mean([combi[2]*100 for combi in log], axis=0), color='red', label='Mean peek')

        for index, combi in enumerate(log):
            pVv, pViv, pG, j = combi

            if k == 0:
                ax.plot(pG * 100, linestyle=line_styles[index], color=cmap(index), label='k=0')
            else:
                ax.plot(pG * 100, linestyle=line_styles[index], color=cmap(index), label=str(j))

            excluded = [candidate for candidate in pGa.keys() if candidate not in j]

            avg_error, pG_mean_error = 0, 0
            for m in excluded:
                all_errors[k].append(abs(pG[MODEL_ENUM[m]] - pGa[m]))
                avg_error += abs(pG[MODEL_ENUM[m]] - pGa[m])
                pG_mean_error += abs(pG_mean[MODEL_ENUM[m]] - pGa[m])

            avg_error = avg_error/len(excluded) if len(excluded) > 0 else 0
            pG_mean_error = pG_mean_error/len(excluded) if len(excluded) > 0 else 0
            max_error = max(abs(pG[MODEL_ENUM[m]] - pGa[m]) for m in excluded) if len(excluded) > 0 else 0
            base_max_error = max(abs(pG_mean[MODEL_ENUM[m]] - pGa[m]) for m in excluded) if len(excluded) > 0 else 0

            if k == 0:
                for m in excluded:
                    print(abs(100* (pG_mean[MODEL_ENUM[m]] - pGa[m])))
                print(pG_mean_error/len(excluded) if len(excluded) > 0 else 0)

            
            means[k].append(avg_error)
            maxs[k].append(max_error)
            base_mean[k].append(pG_mean_error)
            base_max[k].append(base_max_error)
            
        ax.plot(pG_mean*100, color='black', label='Mean')

        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(MODEL_NAMES, rotation=50, fontsize=12)

        for i, key in enumerate(pGa.keys()):
            ax.scatter(MODEL_ENUM[key], pGa[key]*100, color='black')

        plt.tight_layout()
        plt.savefig(f'images/pngs/regressor_{k}.png')
        plt.savefig(f'images/pdfs/regressor_{k}.pdf', format='pdf')


    fig, ax = plt.subplots()

    
    data = [[val * 100 for val in sublist] for sublist in maxs[:-1]]
    means_vals = [np.mean(d) for d in data]
    # stds = [np.std(d) for d in data]
    yerrs = np.transpose([(np.mean(d) - np.min(d), np.max(d) - np.mean(d)) for d in data])
    x_pos = range(1, len(all_logs))

    ax.errorbar(x_pos, means_vals, yerr=yerrs, fmt='-o', label='Regression', color='black', capsize=5)
    ax.set_xticks(x_pos)
    ax.fill_between(range(1, len(all_logs)), 2.7, 15.9, color='blue', alpha=0.1, label='Individual LLM')
    ax.axhline(y=4.7, color='red', label='Ensemble', linestyle='-.')
    ax.axhline(y=base_max[0][0]*100, color='gray', label='Mean Prediction', linestyle='--')

    ax.set_xticklabels(range(0, len(all_logs) - 1), fontsize=14)
    ax.set_xlabel('k', fontsize=16)
    ax.set_ylabel('Maximum Error', fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    plt.tight_layout()

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [3,0,1,2]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=16, loc='upper right')

    plt.savefig('images/pngs/regressor_max_comparison.png')
    plt.savefig('images/pdfs/regressor_max_comparison.pdf', format='pdf')


    fig, ax = plt.subplots()
    ax.axhline(y=base_max[0][0]*100, color='gray', label='Baseline Mean', linestyle='--')
    ax.boxplot(data, positions=x_pos, showmeans=True)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(range(0, len(all_logs) - 1))
    ax.set_xlabel('k')
    ax.set_ylabel('Maximum Error')
    plt.tight_layout()
    plt.legend()
    plt.savefig('images/pngs/regressor_max_comparison_boxplot.png')
    plt.savefig('images/pdfs/regressor_max_comparison_boxplot.pdf', format='pdf')
    

    print(base_max[0][0]*100, end=' ')
    # for m in means_vals:
    #     print(f"{m:.3f}", end='\\% & ')
    # print()

    print(base_mean[0][0]*100)

    data_means = [[val * 100 for val in sublist] for sublist in means[:-1]]
    means_vals_means = [np.mean(d) for d in data_means]
    yerrs_means = np.transpose([(np.mean(d) - np.min(d), np.max(d) - np.mean(d)) for d in data_means])

    for m, o1 in zip(means_vals_means, yerrs_means[0]):
        print(f"{m - o1:.1f}", end='\\% & ')
    print()

    for m in means_vals_means:
        print(f"{m:.1f}", end='\\% & ')
    print()

    for m, o1 in zip(means_vals_means, yerrs_means[1]):
        print(f"{m + o1:.1f}", end='\\% & ')
    print()
    print()

    for ma, o2 in zip(means_vals, yerrs[0]):
        print(f"{ma - o2:.1f}", end='\\% & ')
    print()

    for ma in means_vals:
        print(f"{ma:.1f}", end='\\% & ')
    print()

    for ma, o2 in zip(means_vals, yerrs[1]):
        print(f"{ma + o2:.1f}", end='\\% & ')
    print()
    print()

    fig, ax = plt.subplots()
    ax.errorbar(x_pos, means_vals_means, yerr=yerrs_means, fmt='-o', label='Regression', color='black', capsize=5)
    ax.axhline(y=2.8, color='red', label='Ensemble', linestyle='-.')
    ax.fill_between(range(1, len(all_logs)), 1.8, 5.5, color='blue', alpha=0.1, label='Individual LLM')
    ax.axhline(y=base_mean[0][0]*100, color='gray', label='Mean Prediction', linestyle='--')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(range(0, len(all_logs) - 1), fontsize=14)
    ax.set_xlabel('k', fontsize=16)
    ax.set_ylabel('Mean Error', fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    plt.tight_layout()
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [3,1,0,2]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=16, loc='upper right')
    plt.savefig('images/pngs/regressor_mean_comparison.png')
    plt.savefig('images/pdfs/regressor_mean_comparison.pdf', format='pdf')

if __name__ == '__main__':
    VALIDATOR_COUNTS = VALIDATOR_COUNTS_CONST if VALIDATOR_COUNTS_CONST is not None else get_VALIDATOR_COUNTS()

    redo = True
    if redo:
        all_errors = []
        all_avg_errors = []
        all_logs = []

        futures = {}
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # for k in range(0, len(GENS)+1):
            for k in [2]:
                futures[executor.submit(regress, GV, pGa, GENS, k, VALIDATOR_COUNTS, w=[10, 1, 10])] = k

        results = []
        for future in as_completed(futures):
            k = futures[future]
            results.append((k, future.result()))

        results.sort(key=lambda x: x[0])

        for k, (pGs, errors, avg_error, logs) in results:
            all_errors.append(errors)
            all_logs.append(logs)
            all_avg_errors.append(avg_error)
            print("Done with k =", k)


        with open('all_logs.pkl', 'wb') as f:
            pickle.dump(all_logs, f)
    else:
        with open('all_logs.pkl', 'rb') as f:
            all_logs = pickle.load(f)
    
    print('============================================')
    plot_data_no_exclude(all_logs, pGa, np.mean(GV, axis=1))
    print('============================================')

    print('Baseline: ', np.round(100*np.mean(GV, axis=1), 1))