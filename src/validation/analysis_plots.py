"""
Validation Analysis Plots
Generates three visualizations for LLM-as-Judge experiments:
1. Generator success rate (1D heatmap)
2. Validator success rate per generator (2D heatmap)
3. Validator precision per generator (2D heatmap with valid/invalid counts)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from typing import Dict, List, Tuple
import sys

sys.path.append("..")
from src.config import NUM_SIDS, pathData, pathOutput, pathImages, MODELS_RELEASE_DATE
from datetime import datetime

# Paths
PATH_GENERATOR = f'{pathData}/generator/'
PATH_VALIDATOR = f'{pathData}/validator/'
PATH_OUTPUT_IMAGES = f'{pathImages}/validation_analysis/'

# Create output directory
os.makedirs(PATH_OUTPUT_IMAGES, exist_ok=True)

# Default release date for unknown models
DEFAULT_RELEASE_DATE = '2026-03-01'

def get_model_release_date(model_name: str) -> str:
    """Get release date for a model, return default if not found."""
    return MODELS_RELEASE_DATE.get(model_name, DEFAULT_RELEASE_DATE)

def sort_models_by_release(models: List[str]) -> List[str]:
    """Sort models by their release date."""
    return sorted(models, key=lambda m: datetime.strptime(get_model_release_date(m), '%Y-%m-%d'))

def get_generator_models() -> List[str]:
    """Extract all generator model names from generator files."""
    generator_files = [f for f in os.listdir(PATH_GENERATOR) if f.endswith('_feedback.json')]
    models = []
    for file in generator_files:
        # Remove '_feedback.json' suffix
        model_name = file.replace('_feedback.json', '')
        if model_name != 'generator' and model_name in MODELS_RELEASE_DATE:  # Skip generic file
            models.append(model_name)
    return sorted(models)

def get_validator_models(generator: str) -> List[str]:
    """Extract all validator model names for a given generator."""
    gen_dir = f'{PATH_VALIDATOR}/gen={generator}/'
    if not os.path.exists(gen_dir):
        return []
    
    validator_files = [f for f in os.listdir(gen_dir) if f.endswith('.json')]
    validators = []
    for file in validator_files:
        # Remove 'val=' prefix and '.json' suffix
        validator_name = file.replace('val=', '').replace('.json', '')
        if validator_name in MODELS_RELEASE_DATE:
            validators.append(validator_name)
    return sorted(validators)

def get_all_validators() -> List[str]:
    """Get unique set of all validators across all generators."""
    all_validators = set()
    generators = get_generator_models()
    for gen in generators:
        validators = get_validator_models(gen)
        all_validators.update(validators)
    return sorted(list(all_validators))

def analyze_generator_success() -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Analyze generator success rates.
    Success criteria: SID exists (1-366) AND feedback array is non-empty
    Returns: (model_names, success_counts, failure_counts)
    """
    generators = sort_models_by_release(get_generator_models())
    success_rates = []
    failure_rates = []
    
    for gen in generators:
        filepath = f'{PATH_GENERATOR}/{gen}_feedback.json'
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Create a dict of sid -> item for quick lookup
            sid_dict = {item.get('sid'): item for item in data if item.get('sid') is not None}
            
            successes = 0
            failures = 0
            
            # Check all expected SIDs (1 to NUM_SIDS)
            for sid in range(1, NUM_SIDS + 1):
                if sid in sid_dict:
                    item = sid_dict[sid]
                    # Success: SID exists AND feedback is non-empty
                    feedback = item.get('feedback', [])
                    if feedback and len(feedback) > 0:
                        successes += 1
                    else:
                        failures += 1
                else:
                    # SID doesn't exist - failure
                    failures += 1
            
            total = NUM_SIDS
            success_rates.append((successes / total) * 100 if total > 0 else 0)
            failure_rates.append((failures / total) * 100 if total > 0 else 0)
            
        except Exception as e:
            print(f"Error reading {gen}: {e}")
            success_rates.append(0)
            failure_rates.append(100)
    
    return generators, np.array(success_rates), np.array(failure_rates)

def analyze_validator_success() -> Tuple[List[str], List[str], np.ndarray]:
    """
    Analyze validator error rates for each generator.
    Error = success flag is False OR not all 366 SIDs present
    Returns: (generators, validators, error_matrix)
    """
    generators = sort_models_by_release(get_generator_models())
    validators = sort_models_by_release(get_all_validators())
    
    # Initialize matrix: -2 = no file, 0-100 = error percentage
    error_matrix = np.full((len(generators), len(validators)), -2.0)
    
    for i, gen in enumerate(generators):
        for j, val in enumerate(validators):
            filepath = f'{PATH_VALIDATOR}/gen={gen}/val={val}.json'
            
            if not os.path.exists(filepath):
                error_matrix[i, j] = -2  # File doesn't exist (black)
                continue
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Count errors: success=False OR missing SIDs
                total_sids = len(data)
                error_count = 0
                
                # Check if we have all 366 SIDs
                if total_sids != NUM_SIDS:
                    error_count += abs(NUM_SIDS - total_sids)
                
                # Count items with success=False
                for item in data:
                    output = item.get('output')
                    if output is None or not isinstance(output, dict):
                        error_count += 1
                    else:
                        feedback_lines = output.get('feedback_lines', [])
                        if not feedback_lines or len(feedback_lines) == 0:
                            error_count += 1
                
                # Calculate error percentage
                error_pct = (error_count / NUM_SIDS) * 100 if NUM_SIDS > 0 else 0
                error_matrix[i, j] = min(error_pct, 100)  # Cap at 100%
                
            except Exception as e:
                print(f"Error reading gen={gen}, val={val}: {e}")
                error_matrix[i, j] = -1  # Error reading file (yellow)
    
    return generators, validators, error_matrix

def analyze_validator_precision() -> Tuple[List[str], List[str], np.ndarray]:
    """
    Analyze validator precision (valid / (valid + invalid)) for each generator.
    Returns: (generators, validators, precision_matrix)
    """
    generators = sort_models_by_release(get_generator_models())
    validators = sort_models_by_release(get_all_validators())
    
    # Initialize matrix: -1 = error, -2 = no file
    precision_matrix = np.full((len(generators), len(validators)), -1.0)
    
    for i, gen in enumerate(generators):
        for j, val in enumerate(validators):
            filepath = f'{PATH_VALIDATOR}/gen={gen}/val={val}.json'
            
            if not os.path.exists(filepath):
                precision_matrix[i, j] = -2  # File doesn't exist (black)
                continue
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                valid_count = 0
                invalid_count = 0
                
                for item in data:
                    # Count valid/invalid classifications
                    output = item.get('output')
                    if output and isinstance(output, dict):
                        feedback_lines = output.get('feedback_lines', [])
                        for line in feedback_lines:
                            classification = line.get('classification', '')
                            # Handle list or string classification
                            if isinstance(classification, list):
                                classification = classification[0] if classification else ''
                            classification = str(classification).lower()
                            if classification == 'valid':
                                valid_count += 1
                            elif classification == 'invalid':
                                invalid_count += 1
                
                total = valid_count + invalid_count
                if total > 0:
                    precision = valid_count / total
                    precision_matrix[i, j] = precision
                else:
                    precision_matrix[i, j] = -1  # No valid data
                
            except Exception as e:
                print(f"Error reading gen={gen}, val={val}: {e}")
                precision_matrix[i, j] = -1
    
    return generators, validators, precision_matrix

def plot_generator_success(generators: List[str], success_rates: np.ndarray, 
                          failure_rates: np.ndarray):
    """Plot 1: Generator success/failure rates as horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(12, max(8, len(generators) * 0.4)))
    
    y_pos = np.arange(len(generators))
    
    # Create stacked horizontal bars
    ax.barh(y_pos, success_rates, color='#2ecc71', label='Success', alpha=0.8)
    ax.barh(y_pos, failure_rates, left=success_rates, color='#e74c3c', 
            label='Failure', alpha=0.8)
    
    # Add percentage labels
    for i, (success, failure) in enumerate(zip(success_rates, failure_rates)):
        if success > 5:  # Only show if there's enough space
            ax.text(success / 2, i, f'{success:.1f}%', 
                   ha='center', va='center', fontweight='bold', color='white')
        if failure > 5:
            ax.text(success + failure / 2, i, f'{failure:.1f}%', 
                   ha='center', va='center', fontweight='bold', color='white')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(generators)
    ax.set_xlabel('Percentage (%)', fontsize=12)
    ax.set_title(f'Generator Success/Failure Rates (Total Tasks: {NUM_SIDS})', 
                fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{PATH_OUTPUT_IMAGES}/1_generator_success_rates.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: 1_generator_success_rates.png")

def plot_validator_success(generators: List[str], validators: List[str], 
                          error_matrix: np.ndarray):
    """Plot 2: Validator error rates per generator (2D heatmap)."""
    fig, ax = plt.subplots(figsize=(max(12, len(validators) * 0.5), 
                                     max(8, len(generators) * 0.4)))
    
    # Create custom colormap: black (-2), yellow (-1), green gradient (low error)
    colors_list = ['#000000',  # -2: No file (black)
                   '#FFD700',  # -1: Error reading file (yellow)
                   '#006400',  # 0-2%: Very dark green (excellent)
                   '#228B22',  # 2-5%: Dark green (good)
                   '#32CD32',  # 5-10%: Medium green (ok)
                   '#90EE90',  # 10-20%: Light green (moderate errors)
                   '#FFA500',  # 20%+: Orange (high errors)
                   '#FF0000']  # Very high errors: Red
    
    bounds = [-2.5, -1.5, 0, 2, 5, 10, 20, 50, 100]
    cmap = ListedColormap(colors_list)
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Create heatmap
    im = ax.imshow(error_matrix, cmap=cmap, norm=norm, aspect='auto')
    
    # Add text annotations
    for i in range(len(generators)):
        for j in range(len(validators)):
            error = error_matrix[i, j]
            
            if error == -2:
                text = 'N/A'
                color = 'white'
            elif error == -1:
                text = 'ERR'
                color = 'black'
            else:
                text = f'{error:.1f}'
                color = 'white' if error > 10 or error < 2 else 'black'
            
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=8)
    
    # Labels
    ax.set_xticks(np.arange(len(validators)))
    ax.set_yticks(np.arange(len(generators)))
    ax.set_xticklabels(validators, rotation=45, ha='right')
    ax.set_yticklabels(generators)
    ax.set_xlabel('Validators', fontsize=12)
    ax.set_ylabel('Generators', fontsize=12)
    ax.set_title('Validator Error Rates per Generator (sorted by release date)\n(Black=No file, Green=Low errors, Orange/Red=High errors)', 
                fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Error Rate (%)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(f'{PATH_OUTPUT_IMAGES}/2_validator_success_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: 2_validator_success_matrix.png")

def plot_validator_precision(generators: List[str], validators: List[str], 
                            precision_matrix: np.ndarray):
    """Plot 3: Validator precision per generator (valid/(valid+invalid))."""
    fig, ax = plt.subplots(figsize=(max(14, len(validators) * 0.6), 
                                     max(10, len(generators) * 0.5)))
    
    # Custom colormap based on user specification
    colors_list = ['#000000',  # -2: No file (black)
                   '#FFD700',  # -1: Error (yellow)
                   '#f7fbff',  # 0-85%: Very light blue
                   '#c6dbef',  # 85-90%: Light blue
                   '#6baed6',  # 90-95%: Medium blue
                   '#2171b5',  # 95-98%: Dark blue
                   '#08306b']  # 98-100%: Very dark blue
    
    bounds = [-2.5, -1.5, 0, 0.85, 0.9, 0.95, 0.98, 1.0]
    cmap = ListedColormap(colors_list)
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Create heatmap
    im = ax.imshow(precision_matrix, cmap=cmap, norm=norm, aspect='auto')
    
    # Add text annotations
    for i in range(len(generators)):
        for j in range(len(validators)):
            precision = precision_matrix[i, j]
            
            if precision == -2:
                text = 'N/A'
                color = 'white'
            elif precision == -1:
                text = 'ERR'
                color = 'black'
            else:
                text = f'{precision:.2f}'
                color = 'white' if precision > 0.85 else 'black'
            
            ax.text(j, i, text, ha='center', va='center', color=color, 
                   fontsize=9, fontweight='bold')
    
    # Labels
    ax.set_xticks(np.arange(len(validators)))
    ax.set_yticks(np.arange(len(generators)))
    ax.set_xticklabels(validators, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(generators, fontsize=10)
    ax.set_xlabel('Validators', fontsize=13, fontweight='bold')
    ax.set_ylabel('Generators', fontsize=13, fontweight='bold')
    ax.set_title('Validator Precision per Generator (Valid/(Valid+Invalid)) - Sorted by Release Date\n' +
                'Black=No file, Yellow=Errors, Blue gradient=Precision', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Colorbar with better labels
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Precision', rotation=270, labelpad=25, fontsize=12, fontweight='bold')
    cbar.set_ticks([-2, -1, 0.425, 0.875, 0.925, 0.965, 0.99])
    cbar.set_ticklabels(['No file', 'Error', '<85%', '85-90%', '90-95%', '95-98%', '>98%'])
    
    plt.tight_layout()
    plt.savefig(f'{PATH_OUTPUT_IMAGES}/3_validator_precision_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: 3_validator_precision_matrix.png")

def print_summary_stats():
    """Print summary statistics."""
    generators = get_generator_models()
    validators = get_all_validators()
    
    print("\n" + "="*80)
    print("VALIDATION ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total Generators: {len(generators)}")
    print(f"Total Validators: {len(validators)}")
    print(f"Total SIDs per task: {NUM_SIDS}")
    print(f"\nGenerators found:")
    for gen in generators:
        print(f"  - {gen}")
    print(f"\nValidators found:")
    for val in validators:
        print(f"  - {val}")
    print("="*80)

def main():
    """Main execution function."""
    print("Starting validation analysis...")
    print_summary_stats()
    
    print("\n📊 Generating plots...")
    
    # Plot 1: Generator success rates
    print("\n[1/3] Analyzing generator success rates...")
    generators, success_rates, failure_rates = analyze_generator_success()
    plot_generator_success(generators, success_rates, failure_rates)
    
    # Plot 2: Validator error rates
    print("\n[2/3] Analyzing validator error rates per generator...")
    generators, validators, error_matrix = analyze_validator_success()
    plot_validator_success(generators, validators, error_matrix)
    
    # Plot 3: Validator precision
    print("\n[3/3] Analyzing validator precision per generator...")
    generators, validators, precision_matrix = analyze_validator_precision()
    plot_validator_precision(generators, validators, precision_matrix)
    
    print(f"\n✅ All plots saved to: {PATH_OUTPUT_IMAGES}")
    print("\nDone! 🎉")

if __name__ == "__main__":
    main()
