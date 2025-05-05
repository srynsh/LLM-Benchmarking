import json
import os
import numpy as np
import sys
from config import GENS, MODELS, MODEL_NAMES, MODEL_ENUM
# from data import *
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if parent_dir not in sys.path:
#     sys.path.insert(0, parent_dir)

parent_dir = ".."
sys.path.append(parent_dir)

from data import get_data, get_row_4o, get_row_4_turbo, get_row_3_opus, get_row_35, get_row_gemini_15_pro, get_row_deepseek_chat, get_row_qwen_coder_plus
import logging

def get_counts(feedback):
    TP, FP, ec = 0, 0, 0

    for line in feedback:
        try:
            if line['classification'] == 'valid':
                TP += 1
            elif line['classification'] == 'invalid':
                FP += 1
        except:
            ec += 1

    return TP, FP, ec

def get_counts_total(all_feedbacks):
    TP, FP, EC = 0, 0, 0

    for sub in all_feedbacks:
        try:
            labelled_feedback = sub['output']['feedback_lines']
        except:
            labelled_feedback = []

        tp, fp, ec = get_counts(labelled_feedback)
        TP += tp
        FP += fp
        EC += ec



    return TP, FP, EC

def get_counts_total_from_file(file):
    with open(file) as f:
        all_feedbacks = json.load(f)

    return get_counts_total(all_feedbacks)

def get_counts_total_from_dir(directory):
    
    counts = {}

    vals = MODELS

    for file in os.listdir(directory):
        if not file.endswith(".json"):
            continue
        TP, FP, EC = 0, 0, 0

        # print(file)
        gen = file.split("_gen_")[1].split("_val_")[0]

        validator = file.split("_val_")[1].split("2024")[0][:-1]

        if validator not in vals:
            validator = file.split("_val_")[1].split("2025")[0][:-1]

        tp, fp, ec = get_counts_total_from_file(f"{directory}/{file}")
        TP += tp
        FP += fp
        EC += ec

        counts[(gen, validator)] = (TP, FP, EC)


    # sort by gen, val 
    counts = sorted(counts.items(), key=lambda x: (vals.index(x[0][0]), vals.index(x[0][1])))

    return counts

def get_GV(directory="../../data/validator"): # This gives the GV array
    counts = get_counts_total_from_dir(directory)
    
    c = 0
    x = []
    i = 0
    for (gen, val), (TP, FP, EC) in counts:
        if c % 10 == 0:
            i += 1
            x.append([])
        # Avoid division by zero by checking if (TP + FP) > 0
        rate = TP / (TP + FP) if (TP + FP) > 0 else 0
        x[i - 1].append(rate)
        c += 1

    return np.array(x)

def get_VALIDATOR_COUNTS(directory="../../data/validator"):
    gens = GENS
    vals = MODELS

    tables = np.zeros((len(gens), len(vals), 4))

    for file in os.listdir(directory):
        if not file.endswith(".json"):
            continue

        gen = file.split("_gen_")[1].split("_val_")[0]
        val = file.split("_val_")[1].split("2024")[0].strip("_")
        if val not in vals:
            val = file.split("_val_")[1].split("2025")[0].strip("_")

        if gen not in gens:
            continue

        # print(f"Gen: {gen}, Val: {val}")

        iv_iv, iv_v, v_iv, v_v, ec = 0, 0, 0, 0, 0

        with open(f"{directory}/{file}", "r") as f:
            gen_labels = json.load(f)


        for i, row in enumerate(gen_labels):
            sid = row['sid']
            if gen == "gpt-4o":
                labelled_feedback, unlabelled_feedback, student_code, correct_code, question, all_testcases, prefix, suffix, _ = get_data(get_row_4o(sid))
            elif gen == "gpt-4-turbo":
                labelled_feedback, unlabelled_feedback, student_code, correct_code, question, all_testcases, prefix, suffix, _ = get_data(get_row_4_turbo(sid))
            elif gen == "gpt-3.5-turbo":
                labelled_feedback, unlabelled_feedback, student_code, correct_code, question, all_testcases, prefix, suffix, _ = get_data(get_row_35(sid))
            elif gen == "claude_3_opus":
                labelled_feedback, unlabelled_feedback, student_code, correct_code, question, all_testcases, prefix, suffix, _ = get_data(get_row_3_opus(sid))
            elif gen == "gemini-1.5-pro":
                labelled_feedback, unlabelled_feedback, student_code, correct_code, question, all_testcases, prefix, suffix, _ = get_data(get_row_gemini_15_pro(sid))
            elif gen == "qwen-coder-plus":
                labelled_feedback, unlabelled_feedback, student_code, correct_code, question, all_testcases, prefix, suffix, _ = get_data(get_row_qwen_coder_plus(sid))
            elif gen == "deepseek-chat":
                labelled_feedback, unlabelled_feedback, student_code, correct_code, question, all_testcases, prefix, suffix, _ = get_data(get_row_deepseek_chat(sid))
            else:
                raise Exception("Invalid gen")

            try:
                gen_lab_fb = row['output']['feedback_lines']
            except:
                gen_lab_fb = []
                
            for i in range(len(gen_lab_fb)):
                try:
                    if gen_lab_fb[i]['classification'] == 'invalid':
                        if labelled_feedback[i]['category'] in ['FP-I', 'FP-H']:
                            iv_iv += 1
                        elif labelled_feedback[i]['category'] in ['TP', 'FP-E', 'FP-R']:
                            v_iv += 1
                    elif gen_lab_fb[i]['classification'] == 'valid':
                        if labelled_feedback[i]['category'] in ['FP-I', 'FP-H']:
                            iv_v += 1
                        elif labelled_feedback[i]['category'] in ['TP', 'FP-E', 'FP-R']:
                            v_v += 1
                except:
                    ec += 1

        try:
            tables[gens.index(gen)][vals.index(val)] = [iv_iv, iv_v, v_iv, v_v]
        except:
            # print(f"Gen: {gen}, Val: {val} not found in tables")
            pass
        try:
            accuracy = (iv_iv + v_v) / (iv_iv + iv_v + v_iv + v_v)
            recall = v_v/(v_v + iv_v)
            precision = v_v/(v_v + v_iv)
            f_score = 2 * (precision * recall) / (precision + recall)
        except:
            accuracy, recall, precision, f_score = 0, 0, 0, 0

        # print(f"IV-IV: {iv_iv}, IV-V: {iv_v}, V-IV: {v_iv}, V-V: {v_v}, EC: {ec}, Total: {iv_iv + iv_v + v_iv + v_v }")
        # print(f"Precision: {precision}")

    tables = tables.astype(int)
    arr = tables.tolist()
    return arr

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =========================
#   Validator Vpos and Vneg
# =========================
def get_pViv_full(gens, VALIDATOR_COUNTS):
    pViva = {}

    stats = np.zeros_like(VALIDATOR_COUNTS[0])
    for i in gens:
        stats += VALIDATOR_COUNTS[GENS.index(i)]

    pViva = {k: (stats[i][0])/(stats[i][0] + stats[i][1]) for i, k in enumerate(MODELS)}
    pVva = {k: (stats[i][3])/(stats[i][2] + stats[i][3]) for i, k in enumerate(MODELS)}

    return pViva, pVva

# =========================
#   GEN PRECISION ACTUAL
# =========================

def _calculate_feedback_counts(filepath):
    """
    Reads a JSON file and counts valid and invalid feedback lines.

    Args:
        filepath (str): The path to the JSON file.

    Returns:
        tuple: A tuple containing the count of valid feedback lines (v)
               and invalid feedback lines (iv). Returns (0, 0) if the file
               cannot be read or processed correctly.
    """
    valid_count, invalid_count = 0, 0
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        for item in data:
            try:
                feedback_lines = item.get('feedback', [])
                for line in feedback_lines:
                    category = line.get('category')
                    if category in ['TP', 'FP-R', 'FP-E']: # Redundant and Extra categories treated as valid
                        valid_count += 1
                    elif category in ['FP-H', 'FP-I']: # Hallucinated and Incorrect categories treated as invalid
                        invalid_count += 1
            except AttributeError:
                # Handle cases where item is not a dictionary or line is not structured as expected
                logging.warning(f"Skipping malformed item or line in {filepath}")
                continue
            except KeyError:
                # This specific catch might be less necessary now with .get()
                logging.warning(f"Skipping item due to missing 'feedback' key or malformed line in {filepath}")
                continue

    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return 0, 0
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from file: {filepath}")
        return 0, 0
    except Exception as e:
        logging.error(f"An unexpected error occurred while processing {filepath}: {e}")
        return 0, 0

    return valid_count, invalid_count

def get_precision(directory="../../data/generator"):
    """
    Calculates the precision for each generator based on feedback files in a directory.

    Precision is defined as: valid_count / (valid_count + invalid_count).

    Args:
        directory (str): The path to the directory containing generator feedback JSON files.
                         Files are expected to be named like '<generator_name>_feedback.json'.

    Returns:
        dict: A dictionary mapping generator names (str) to their calculated precision (float).
    """
    generator_precision = {}

    try:
        all_files = [f for f in os.listdir(directory) if f.endswith('.json') and '_feedback' in f]
    except FileNotFoundError:
        logging.error(f"Directory not found: {directory}")
        return {}
    except Exception as e:
        logging.error(f"Error listing files in directory {directory}: {e}")
        return {}


    for filename in all_files:
        filepath = os.path.join(directory, filename)

        # Extract generator name from filename
        try:
            # Assumes filename format like 'generator-name_feedback.json'
            generator_name = filename.split('_feedback')[0].strip()
            if not generator_name:
                 logging.warning(f"Could not extract generator name from filename: {filename}. Skipping.")
                 continue
        except Exception as e:
            logging.warning(f"Error parsing generator name from filename {filename}: {e}. Skipping.")
            continue

        # Calculate counts for the current file
        valid_count, invalid_count = _calculate_feedback_counts(filepath)

        # Calculate and store precision
        total_count = valid_count + invalid_count
        if total_count > 0:
            precision = valid_count / total_count
        else:
            precision = 0.0 # Avoid division by zero, assign 0 precision if no relevant feedback
            logging.info(f"No valid or invalid feedback found for generator '{generator_name}' in file {filename}. Precision set to 0.")

        generator_precision[generator_name] = precision
        logging.info(f"Processed {filename}: Generator='{generator_name}', Valid={valid_count}, Invalid={invalid_count}, Precision={precision:.4f}")


    return generator_precision