import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import   get_GV, get_VALIDATOR_COUNTS, get_precision
from config import PV_START, PVIV_START, PG_START, NUM_RUNS, ERR_EPSILON, GENS, MODELS, MODEL_NAMES, MODEL_ENUM, MAX_WORKERS
from config import VALIDATOR_COUNTS_CONST, pGa_CONST, GV_CONST, PVVA_CONST, PVIVA_CONST
np.random.seed(42)

"""CONSTANTS"""

# pGa = get_precision()
pGa = pGa_CONST if pGa_CONST is not None else get_precision()

# GV = get_GV()
GV = GV_CONST if GV_CONST is not None else get_GV()

"""-----------------CONSTANTS END HERE----------------"""

# def loss_function(pVv, pViv, pG, GV):
#     G_hat = np.outer(pG, pVv) + np.outer((1 - pG), (1 - pViv))
#     return np.mean((GV - G_hat) ** 2)

def loss_function(pVv, pViv, pG, GV):
    G_hat = np.outer(pG, pVv) + np.outer((1 - pG), (1 - pViv))
    epsilon = 1e-9
    return -np.mean([
        GV[i, j]*np.log(np.clip(G_hat[i, j], epsilon, 1))
        + (1 - GV[i, j])*np.log(np.clip(1 - G_hat[i, j], epsilon, 1))
        for i in range(GV.shape[0])
        for j in range(GV.shape[1])
    ])

# def reg(p_hat, p, id):
#     epsilon = 1e-9
#     return -np.sum([
#         p[i]*np.log(np.clip(p_hat[MODEL_ENUM[i]], epsilon, 1 - epsilon))
#         + (1 - p[i])*np.log(np.clip(1 - p_hat[MODEL_ENUM[i]], epsilon, 1 - epsilon))
#         for i in id
#     ]) / len(id) if len(id) > 0 else 0

# def reg(p_hat, p, id):
#     return np.sqrt(np.mean([(p[i] - p_hat[MODEL_ENUM[i]])**2 for i in id])) if len(id) > 0 else 0

def reg(p_hat, p, id):
    return np.sqrt(np.mean([(p[i] - p_hat[MODEL_ENUM[i]])**2 for i in id]) if len(id) > 0 else 0)

# def reg(p_hat, p, id):
#     return np.mean([(p[i] - p_hat[MODEL_ENUM[i]])**2 for i in id]) if len(id) > 0 else 0

def reg_KL(p_hat, p, id):
    epsilon = 1e-9
    return np.sum([
        p[i]*np.log(np.clip(p[i]/np.clip(p_hat[MODEL_ENUM[i]], epsilon, 1 - epsilon), epsilon, 1 - epsilon))
        + (1 - p[i])*np.log(np.clip((1 - p[i])/(1 - np.clip(p_hat[MODEL_ENUM[i]], epsilon, 1 - epsilon)), epsilon, 1 - epsilon))
        for i in id
    ]) / len(id) if len(id) > 0 else 0

def total_loss(x, GV, pVva, pViva, pGa, idV, idG, w):
    NUM_VALIDATORS = GV.shape[1]
    NUM_GENERATORS = GV.shape[0]
    l1 = loss_function(x[:NUM_VALIDATORS], x[NUM_VALIDATORS:2*NUM_VALIDATORS], x[2*NUM_VALIDATORS:], GV)
    
    l2 = reg(x[2*NUM_VALIDATORS:], pGa, idG)
    l3_1 = reg(x[:NUM_VALIDATORS], pVva, idV)
    l3_2 = reg(x[NUM_VALIDATORS:2*NUM_VALIDATORS], pViva, idV)

    # l2 = reg_cross_entropy(x[2*NUM_VALIDATORS:], pGa, idG)
    # l3_1 = reg_cross_entropy(x[:NUM_VALIDATORS], pVva, idV)
    # l3_2 = reg_cross_entropy(x[NUM_VALIDATORS:2*NUM_VALIDATORS], pViva, idV)

    # l2 = reg_KL(x[2*NUM_VALIDATORS:], pGa, idG)
    # l3_1 = reg_KL(x[:NUM_VALIDATORS], pVva, idV)
    # l3_2 = reg_KL(x[NUM_VALIDATORS:2*NUM_VALIDATORS], pViva, idV)

    # l2 = reg(x[2*NUM_VALIDATORS:], pGa, idG) + reg_cross_entropy(x[2*NUM_VALIDATORS:], pGa, idG) + reg_KL(x[2*NUM_VALIDATORS:], pGa, idG)
    # l3_1 = reg(x[:NUM_VALIDATORS], pVva, idV) + reg_cross_entropy(x[:NUM_VALIDATORS], pVva, idV) + reg_KL(x[:NUM_VALIDATORS], pVva, idV)
    # l3_2 = reg(x[NUM_VALIDATORS:2*NUM_VALIDATORS], pViva, idV) + reg_cross_entropy(x[NUM_VALIDATORS:2*NUM_VALIDATORS], pViva, idV) + reg_KL(x[NUM_VALIDATORS:2*NUM_VALIDATORS], pViva, idV)

    return l1 + w[0]*l2 + w[1]*l3_1 + w[2]*l3_2

def numerical_gradient(loss_fn, x, *args, eps=1e-6):
    grad = np.zeros_like(x)
    fx = loss_fn(x, *args)
    for i in range(len(x)):
        old = x[i]
        x[i] = old + eps
        fxh = loss_fn(x, *args)
        grad[i] = (fxh - fx) / eps
        x[i] = old
    return grad

def gradient_descent(loss_fn, x0, args, lr=1e-4, max_iter=1000, tol=1e-7):
    x = np.clip(x0.copy(), 0, 1)
    for _ in range(max_iter):
        g = numerical_gradient(loss_fn, x, *args)
        x_new = np.clip(x - lr*g, 0, 1)
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x

def estimate_grad_desc(GV, pVva, pViva, pGa, idV, idG, w = [1, 0, 0]):
    nV = GV.shape[1]
    nG = GV.shape[0]

    pVv_hat = np.mean(GV, axis=0) + np.random.uniform(-0.05, 0.05, GV.shape[1])
    pVv_hat = np.clip(pVv_hat, 0, 1)
    pViv_hat = np.mean(GV, axis=0) + np.random.uniform(-0.05, 0.05, GV.shape[1])
    pViv_hat = np.clip(pViv_hat, 0, 1)
    pG = np.mean(GV, axis=1) + np.random.uniform(-0.05, 0.05, GV.shape[0])
    pG = np.clip(pG, 0, 1)

    # x0 = np.random.rand(2 * nV + nG)

    x0 = np.concatenate([pVv_hat, pViv_hat, pG])

    result = gradient_descent(
        total_loss,
        x0,
        (GV, pVva, pViva, pGa, idV, idG, w)
    )

    pVv = result[:nV]
    pViv = result[nV:2*nV]
    pG = result[2*nV:]
    return pVv, pViv, pG

def estimate_probs(GV, pVva, pViva, pGa, idV, idG, w = [1, 0, 0]):
    # return estimate_probs_ga(GV, pVva, pViva, pGa, idV, idG, w)
    # pVv, pViv, pG = estimate_probs_ga(GV, pVva, pViva, pGa, idV, idG, w)
    res = None
    # val = total_loss(np.concatenate([pVv, pViv, pG]), GV, pVva, pViva, pGa, idV, idG, w)
    val = np.inf

    NUM_VALIDATORS = GV.shape[1]
    NUM_GENERATORS = GV.shape[0]

    if PG_START == 'mean':
            pG = np.mean(GV, axis=1)
    elif PG_START == 'uniform':
        pG = np.random.uniform(0, 1, NUM_GENERATORS)
    else:
        pG = np.ones(NUM_GENERATORS) * PG_START

    if PV_START == 'uniform':
        pVv_hat = np.random.uniform(0, 1, NUM_VALIDATORS)
    else:
        pVv_hat = np.ones(NUM_VALIDATORS) * PV_START

    if PVIV_START == 'uniform':
        pViv_hat = np.random.uniform(0, 1, NUM_VALIDATORS)
    else:
        pViv_hat = np.ones(NUM_VALIDATORS) * PVIV_START

    for run_count in range (NUM_RUNS):
        if run_count != 0:
            pG = pG + np.random.uniform(-ERR_EPSILON/2, ERR_EPSILON/2, NUM_GENERATORS)
            pVv_hat = pVv_hat + np.random.uniform(-ERR_EPSILON/2, ERR_EPSILON/2, NUM_VALIDATORS)
            pViv_hat = pViv_hat + np.random.uniform(-ERR_EPSILON/2, ERR_EPSILON/2, NUM_VALIDATORS)

        pG = np.clip(pG, 0, 1)
        pVv_hat = np.clip(pVv_hat, 0, 1)
        pViv_hat = np.clip(pViv_hat, 0, 1)  

        x = np.concatenate([pVv_hat, pViv_hat, pG])

        res1 = minimize(total_loss, x, args=(GV, pVva, pViva, pGa, idV, idG, w), bounds=[(0, 1)] * len(x)).x
        # res1 = gradient_descent(total_loss, x, args=(GV, pVva, pViva, pGa, idV, idG, w))
        # res1 = minimize(total_loss, x, args=(GV, pVva, pViva, pGa, idV, idG, w), bounds=[(0.5, 1)]*NUM_VALIDATORS + [(0, 0.6)]*NUM_VALIDATORS + [(0,1)]*NUM_GENERATORS).x

        if total_loss(res1, GV, pVva, pViva, pGa, idV, idG, w) < val:
            val = total_loss(res1, GV, pVva, pViva, pGa, idV, idG, w)
            res = res1

    pVv = res[:NUM_VALIDATORS]
    pViv = res[NUM_VALIDATORS:2*NUM_VALIDATORS]
    pG = res[2*NUM_VALIDATORS:]

    return pVv, pViv, pG

def estimate_probs_ga(GV, pVva, pViva, pGa, idV, idG, w=[1, 0, 0], population_size=50, generations=100, mutation_rate=0.1, crossover_rate=0.5):
    nV = GV.shape[1]
    nG = GV.shape[0]
    dim = 2 * nV + nG

    # Initialize a population with random solutions in [0, 1]
    population = np.random.rand(population_size, dim)
    # Set the last nG entries (pG part) to the mean of GV along axis 1 for each candidate
    nG = GV.shape[0]
    population[:, -nG:] = np.tile(np.mean(GV, axis=1), (population_size, 1))

    best_candidate = None
    best_score = np.inf

    def tournament_selection(population, scores, tournament_size=3):
        # Select a random subset and return the best individual
        indices = np.random.choice(len(population), tournament_size, replace=False)
        best_idx = indices[np.argmin(scores[indices])]
        return population[best_idx]

    # Begin evolution over generations
    for gen in range(generations):
        # Evaluate fitness for each individual
        scores = np.array([total_loss(ind, GV, pVva, pViva, pGa, idV, idG, w) for ind in population])
        # Update best solution found
        current_best_idx = np.argmin(scores)
        if scores[current_best_idx] < best_score:
            best_score = scores[current_best_idx]
            best_candidate = population[current_best_idx].copy()

        new_population = []
        # Create new population via selection, crossover and mutation
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, scores)
            parent2 = tournament_selection(population, scores)
            # Uniform crossover
            mask = np.random.rand(dim) < crossover_rate
            child = np.where(mask, parent1, parent2)
            # Mutation step: add small Gaussian noise
            if np.random.rand() < mutation_rate:
                child += np.random.normal(0, 0.05, size=dim)
            child = np.clip(child, 0, 1)
            new_population.append(child)
        population = np.array(new_population)

    res = best_candidate
    pVv = res[:nV]
    pViv = res[nV:2*nV]
    pG = res[2*nV:]
    return pVv, pViv, pG

def print_GV_hat(pVv, pViv, pG_min, pG_max, pG_mean, PVVA, PVIVA):
    G_hat = np.outer(pG_mean, pVv) + np.outer((1 - pG_mean), (1 - pViv))
    # print(np.round(G_hat, 1))
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

    for j in idG:
        stats = np.zeros_like(VALIDATOR_COUNTS[0])

        for i in j:
            stats += VALIDATOR_COUNTS[gens.index(i)]

        if k1 == 0:
            pVv, pViv, pG = estimate_probs(GV, {}, {}, pGa, (), j, w=w)
        else:
            pVva  = {k: (stats[i][3])/(stats[i][2] + stats[i][3]) for i, k in enumerate(MODELS)}
            pViva = {k: (stats[i][0])/(stats[i][0] + stats[i][1]) for i, k in enumerate(MODELS)}

            # pViva, _ = get_pViv_full(j)

            pVv, pViv, pG = estimate_probs(GV, pVva, pViva, pGa, MODELS, j, w=w)

        logs.append((pVv, pViv, pG, j))
        pGs.append(pG)

        excluded = [candidate for candidate in gens if candidate not in j]
        p_temp = np.array([pGa[m] for m in excluded])
        p_hat_temp = np.array([pG[MODEL_ENUM[m]] for m in excluded])

        avg_errors.append(100*np.mean(abs(p_hat_temp - p_temp)))
        errors.append(100*max(abs(p_hat_temp - p_temp)) if len(p_temp) > 0 else 0)

    # mean_pG = np.mean(pGs, axis=0)
    # min_pG = np.min(pGs, axis=0)
    # max_pG = np.max(pGs, axis=0)

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
    # PVVA = np.array([_pv[m] for m in MODELS])
    # PVIVA = np.array([_piv[m] for m in MODELS])

    PVVA = PVVA_CONST if PVVA_CONST is not None else np.array([_pv[m] for m in MODELS])
    PVIVA = PVIVA_CONST if PVIVA_CONST is not None else np.array([_piv[m] for m in MODELS])

    print('Mean GV_hat', k1)
    print_GV_hat(mean_pVv, mean_pViv, min_pG, max_pG, mean_pG, PVVA, PVIVA)

    return pGs, errors, avg_errors, logs

def print_pred_accuracy(VALIDATOR_COUNTS):
    for i in range(len(GENS)):
        print(f"\\textbf{{{GENS[i]}}} & ", end='')
        x = []
        for j in range(10):
            x.append((VALIDATOR_COUNTS[i][j][0], VALIDATOR_COUNTS[i][j][1], VALIDATOR_COUNTS[i][j][2], VALIDATOR_COUNTS[i][j][3]))
            accuracy = 100*(VALIDATOR_COUNTS[i][j][0] + VALIDATOR_COUNTS[i][j][3])/(VALIDATOR_COUNTS[i][j][0] + VALIDATOR_COUNTS[i][j][1] + VALIDATOR_COUNTS[i][j][2] + VALIDATOR_COUNTS[i][j][3])
            print(f"{accuracy:.1f}\\% & ", end='')

        print(f"{100*np.mean([a[0] + a[3] for a in x])/(np.mean([a[0] + a[1] + a[2] + a[3] for a in x])):.1f}\\% \\\\")

def print_pred_prec(VALIDATOR_COUNTS):
    for i in range(len(GENS)):
        print(f"\\textbf{{{GENS[i]}}} & ", end='')
        x = []
        for j in range(10):
            x.append((VALIDATOR_COUNTS[i][j][2], VALIDATOR_COUNTS[i][j][3]))
            accuracy = 100*(VALIDATOR_COUNTS[i][j][3])/(VALIDATOR_COUNTS[i][j][2] + VALIDATOR_COUNTS[i][j][3])
            print(f"{accuracy:.1f}\\% & ", end='')
    
        print(f"{100*np.mean([a[1] for a in x])/(np.mean([a[0] for a in x]) + np.mean([a[1] for a in x])):.1f}\\% \\\\", end='')
        print()

    print(r'\textbf{Mean} & ', end='')
    for j in range(10):
        mean_accuracy = 100 * np.mean([VALIDATOR_COUNTS[i][j][3] / (VALIDATOR_COUNTS[i][j][2] + VALIDATOR_COUNTS[i][j][3]) for i in range(len(GENS))])
        print(f"{mean_accuracy:.1f}\\% & ", end='')

    print(r'\\')


    for j in range(10):
        mean_accuracy = np.mean([VALIDATOR_COUNTS[i][j][3] / (VALIDATOR_COUNTS[i][j][2] + VALIDATOR_COUNTS[i][j][3]) for i in range(len(GENS))])
        print(f"{mean_accuracy}, ", end='')


    # print(f"{100 * np.mean([VALIDATOR_COUNTS[i][j][3] for i in range(len(GENS)) for j in range(10)]) / np.mean([VALIDATOR_COUNTS[i][j][2] + VALIDATOR_COUNTS[i][j][3] for i in range(len(GENS)) for j in range(10)]):.1f}\\% \\\\")
    # np.mean([VALIDATOR_COUNTS[i][j][3] / (VALIDATOR_COUNTS[i][j][2] + VALIDATOR_COUNTS[i][j][3]) for i in range(len(GENS)) for j in range(10)])

    print()

def print_pred_prec_invalid(VALIDATOR_COUNTS):
    for i in range(len(GENS)):
        print(f"\\textbf{{{GENS[i]}}} & ", end='')
        x = []
        for j in range(10):
            x.append((VALIDATOR_COUNTS[i][j][0], VALIDATOR_COUNTS[i][j][1]))
            accuracy = 100*(VALIDATOR_COUNTS[i][j][0])/(VALIDATOR_COUNTS[i][j][1] + VALIDATOR_COUNTS[i][j][0])
            print(f"{accuracy:.1f}\\% & ", end='')
    
        print(f"{100*np.mean([a[0] for a in x])/(np.mean([a[0] for a in x]) + np.mean([a[1] for a in x])):.1f}\\% \\\\ ", end='')
        print()

    print(r'\textbf{Mean} & ', end='')
    for j in range(10):
        mean_accuracy = 100 * np.mean([VALIDATOR_COUNTS[i][j][0] / (VALIDATOR_COUNTS[i][j][0] + VALIDATOR_COUNTS[i][j][1]) for i in range(len(GENS))])
        print(f"{mean_accuracy:.1f}\\% & ", end='')

    print(r'\\')
    print()

    for j in range(10):
        mean_accuracy = np.mean([VALIDATOR_COUNTS[i][j][0] / (VALIDATOR_COUNTS[i][j][0] + VALIDATOR_COUNTS[i][j][1]) for i in range(len(GENS))])
        print(f"{mean_accuracy}, ", end='')

    print()

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

    # print('-----------------------------info-----------------------------')
    # print(x_pos, means, stds, base_max[0][0])

    # # print(means)
    # # print(stds)

    ax.errorbar(x_pos, means_vals, yerr=yerrs, fmt='-o', label='Regression', color='black', capsize=5)
    ax.set_xticks(x_pos)
    ax.fill_between(range(1, len(all_logs)), 2.7, 15.9, color='blue', alpha=0.1, label='Individual LLM')
    ax.axhline(y=4.7, color='red', label='Ensemble', linestyle='-.')
    ax.axhline(y=base_max[0][0]*100, color='gray', label='Mean Prediction', linestyle='--')

    # ax.set_xticklabels(range(0, len(all_logs) - 1))
    ax.set_xticklabels(range(0, len(all_logs) - 1), fontsize=14)
    ax.set_xlabel('k', fontsize=16)
    ax.set_ylabel('Maximum Error', fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    plt.tight_layout()

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [3,0,1,2]
    # order = [0,1,2,3]
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

def find_lambda_error(GV, pGa, gens, k1, w, VALIDATOR_COUNTS):
    idG = list(combinations(pGa.keys(), k1))
    mean_maxs = []

    # for _ in tqdm(range(10), desc="Iterations"):
    for _ in range(10):
        errors = []
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

            err = total_loss(np.concatenate([pVv, pViv, pG]), GV, pVva, pViva, pGa, MODELS, j, w)

            excluded = [candidate for candidate in gens if candidate not in j]
            p_temp = np.array([pGa[m] for m in excluded])
            p_hat_temp = np.array([pG[MODEL_ENUM[m]] for m in excluded])

            errors.append(100*max(abs(p_hat_temp - p_temp)) if len(p_temp) > 0 else 0)
        mean_maxs.append(np.mean(errors))

    mean_max_error = min(errors)

    return mean_max_error

def plot_lambda_errors(GV, pGa, gens, pos):
    cmap = plt.get_cmap('tab10', 5)
    linestlyes = ['-', '--', '-.', ':', (0, (5,2))]
    w = [1, 0, 0]

    fig, ax = plt.subplots(figsize=(6, 3))

    ax.set_ylim(0, 10)
    ax.set_xlabel(fr'$\lambda_{pos + 1}$')
    ax.set_ylabel('Max Error')

    for k in tqdm(range(0, 5)):
        errors = []

        l1s = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
        l1s = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 10000000, 100000000]
        labels = [f'$10^{{{round(np.log10(l), 1)}}}$' if l != 0 else '0' for l in l1s]
        for l1 in tqdm(l1s, desc=f"Lambda {pos + 1}, k = {k}"):
            w[pos] = l1
            max_error = find_lambda_error(GV, pGa, gens, k, w=w)
            errors.append(max_error)
        
        ax.plot(labels, errors, label=f'k = {k}', color=cmap(k), linestyle=linestlyes[k])
        plt.xticks(rotation=45)
    
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'images/pngs/lambda_error_{pos+1}.png')
        plt.savefig(f'images/pdfs/lambda_error_{pos+1}.pdf', format='pdf')

def find_lambda_loss(GV, pGa, gens, k1, w, VALIDATOR_COUNTS):
    idG = list(combinations(pGa.keys(), k1))

    errors = []
    for j in idG:
        stats = np.zeros_like(VALIDATOR_COUNTS[0])

        for i in j:
            stats += VALIDATOR_COUNTS[gens.index(i)]

        if k1 == 0:
            pVv, pViv, pG = estimate_probs(GV, {}, {}, pGa, (), j, w=w)
            err = total_loss(np.concatenate([pVv, pViv, pG]), GV, {}, {}, pGa, (), j, w)
        else:
            pVva  = {k: (stats[i][3])/(stats[i][2] + stats[i][3]) for i, k in enumerate(MODELS)}
            pViva = {k: (stats[i][0])/(stats[i][0] + stats[i][1]) for i, k in enumerate(MODELS)}
            pVv, pViv, pG = estimate_probs(GV, pVva, pViva, pGa, MODELS, j, w=w)

            err = total_loss(np.concatenate([pVv, pViv, pG]), GV, pVva, pViva, pGa, MODELS, j, w)
        errors.append(err)

    mean_error = np.mean(errors)

    return mean_error

def plot_lambda_loss(GV, pGa, gens, pos):
    cmap = plt.get_cmap('tab10', 5)
    linestlyes = ['-', '--', '-.', ':', (0, (5,2))]
    w = [1, 0, 1]

    fig, ax = plt.subplots(figsize=(6, 3))

    # ax.set_ylim(0, 10)
    ax.set_xlabel(fr'$\lambda_{pos + 1}$')
    ax.set_ylabel('Final Loss')

    for k in tqdm(range(0, 5)):
        errors = []

        l1s = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
        l1s = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 10000000, 100000000]
        labels = [f'$10^{{{round(np.log10(l), 1)}}}$' if l != 0 else '0' for l in l1s]
        for l1 in tqdm(l1s, desc=f"Lambda {pos + 1}, k = {k}"):
            w[pos] = l1
            max_error = find_lambda_loss(GV, pGa, gens, k, w=w)
            errors.append(max_error)
        
        ax.plot(labels, errors, label=f'k = {k}', color=cmap(k), linestyle=linestlyes[k])
        plt.xticks(rotation=45)
    
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'images/pngs/lambda_loss_{pos+1}.png')
        plt.savefig(f'images/pdfs/lambda_loss_{pos+1}.pdf', format='pdf')

if __name__ == '__main__':
    # print('Accuracy')
    # print_pred_accuracy()
    # print('Precision')
    # print_pred_prec()
    # print('Invalid Precision')
    # print_pred_prec_invalid()
    # exit()
    # VALIDATOR_COUNTS = get_VALIDATOR_COUNTS()
    VALIDATOR_COUNTS = VALIDATOR_COUNTS_CONST if VALIDATOR_COUNTS_CONST is not None else get_VALIDATOR_COUNTS()

    redo = True
    if redo:
        all_errors = []
        all_avg_errors = []
        all_logs = []

        futures = {}
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for k in range(0, len(GENS)+1):
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
    # print_latex_table(all_logs, pGa, np.mean(GV, axis=1))

    print('Baseline: ', np.round(100*np.mean(GV, axis=1), 1))
    
    # plot_lambda_errors(GV, pGa, GENS, 0)
    # plot_lambda_loss(GV, pGa, GENS, 0)