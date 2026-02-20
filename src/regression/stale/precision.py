import json
import os
import numpy as np
import sys
from src.regression.config import GENS, MODELS, MODEL_NAMES, MODEL_ENUM

from src.regression.stale.dataGenerator import get_data, get_row_4o, get_row_4_turbo, get_row_3_opus, get_row_35, get_row_gemini_15_pro, get_row_deepseek_chat, get_row_qwen_coder_plus
import logging

# from data import *
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if parent_dir not in sys.path:
#     sys.path.insert(0, parent_dir)

parent_dir = ".."
sys.path.append(parent_dir)


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