"""
Refactored validator using Pydantic models and modular design.
"""

import sys
import json
import datetime
from typing import List, Optional
from tqdm import tqdm
from src.config import NUM_SIDS, Model

sys.path.append("..")

from src.validation.service import ValidationRunner, ValidationService
from src.validation.data import filter_quota_exceeded_sids
from src.validation.utils import (
    print_validation_summary, 
    analyze_error_patterns
)
from src.validation.models import ValidationBatch, ValidationResult
from src.utils import print_warning, print_error


####################
# Config
####################

### Missing annotations
# CLAUDE_3_OPUS: 2 SIDs
# GPT_4_TURBO: 17 SIDs
# GPT_4O: 2 SIDs

# Current selected model
modelGen = Model.CLAUDE_3_OPUS.value
modelVal = Model.CLAUDE_3_OPUS.value

EXPORT_RESULTS_TO_CSV = False  # Set to True to export results to CSV

def main():
    """Main entry point for validation operations."""
    
    # TODO: remove hardcoding
    # Configuration
    generator_model = 'claude_3.5_haiku'
    validator_model = 'gemini-2.5-pro-preview-03-25'
    use_ground_truth = False
    
    # Option 1: Resume from existing file (for quota exceeded errors)
    resume_mode = True
    existing_file = './new_logs/new_labeller_gen_claude_3.5_haiku_val_gemini-2.5-pro-preview-03-25_2025-05-05_13-06-52.json'
    
    if resume_mode and existing_file:
        # Extract failed SIDs from existing file
        failed_sids = filter_quota_exceeded_sids(existing_file)
        
        if not failed_sids:
            print("No SIDs with quota exceeded errors found.")
            return
        
        print(f"Found {len(failed_sids)} SIDs with quota exceeded errors: {failed_sids}")
        
        # Create validation runner
        runner = ValidationRunner(generator_model, validator_model, use_ground_truth)
        
        # Run quota recovery
        runner.run_quota_recovery(existing_file)
        
    else:
        # Option 2: Run fresh validation
        sids = list(range(1, NUM_SIDS + 1))  # Replace NUM_SIDS with actual number of SIDs
        # sids = list(range(1, 367))  # All SIDs
        sids = [1, 2, 3, 4, 6, 8, 9, 12, 13, 15]  # Sample SIDs for testing
        
        # Create validation runner
        runner = ValidationRunner(generator_model, validator_model, use_ground_truth)
        
        # Run validation
        runner.run_validation(sids)
    
    print("Validation completed successfully!")


def validate_specific_sids(sids: List[int], generator_model: str, validator_model: str, 
                          use_ground_truth: bool = False) -> ValidationBatch:
    """
    Validate specific SIDs and return results as a batch.
    
    Args:
        sids: List of SIDs to validate
        generator_model: Model used for generation
        validator_model: Model used for validation
        use_ground_truth: Whether to use ground truth data
        
    Returns:
        ValidationBatch: Batch of validation results
    """
    service = ValidationService(generator_model, validator_model, use_ground_truth)
    
    results = []
    for sid in tqdm(sids, desc="Validating SIDs"):
        result = service.validate_single_sid(sid)
        results.append(result)
    
    # Create batch
    attempt_name = f"batch_gen_{generator_model}_val_{validator_model}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    batch = ValidationBatch(
        attempt_name=attempt_name,
        generator_model=generator_model,
        validator_model=validator_model,
        results=results,
        use_ground_truth=use_ground_truth
    )
    
    return batch


def analyze_validation_results(modelGen: str, modelVal: str) -> None:
    """
    Analyze validation results from a file and print comprehensive statistics.
    
    Args:
        file_path: Path to the validation results file
    """
    from src.validation.utils import load_validation_batch_from_file
    
    batch = load_validation_batch_from_file(modelGen, modelVal)
    
    # Print summary
    print_validation_summary(batch)
    
    # Analyze error patterns
    error_analysis = analyze_error_patterns(batch.results)
    if error_analysis['total_failed'] > 0:
        print(f"\nError Analysis:")
        print(f"Total Failed: {error_analysis['total_failed']}")
        for error_type, count in error_analysis['error_counts'].items():
            print(f"  {error_type}: {count} SIDs")
    
    # Export to CSV if required
    if EXPORT_RESULTS_TO_CSV:
        export_results_to_csv(batch, modelGen, modelVal)
    

def compare_validation_runs(file_paths: List[str]) -> None:
    """
    Compare multiple validation runs and print comparative statistics.
    
    Args:
        file_paths: List of paths to validation result files
    """
    from src.validation.utils import load_validation_batch_from_file
    
    batches = []
    for file_path in file_paths:
        batch = load_validation_batch_from_file(file_path)
        if batch:
            batches.append(batch)
        else:
            print_warning(f"Could not load batch from {file_path}")
    
    if len(batches) < 2:
        print_error("Need at least 2 valid batches to compare")
        return
    
    print(f"\n{'='*80}")
    print(f"VALIDATION COMPARISON ({len(batches)} runs)")
    print(f"{'='*80}")
    
    for i, batch in enumerate(batches):
        stats = batch.get_summary_stats()
        print(f"\nRun {i+1}: {batch.attempt_name}")
        print(f"  Generator: {batch.generator_model}")
        print(f"  Validator: {batch.validator_model}")
        print(f"  Success Rate: {stats['success_rate']:.2%}")
        print(f"  Precision: {stats['precision']:.4f}")
        print(f"  Total Valid: {stats['total_valid']}")
        print(f"  Total Invalid: {stats['total_invalid']}")

# TODO: (1) Run for a single validator file and print stats. (2) Run LLM on failures and write into a new file. (3) Run LLM on all generator and validator combinations and write into a new file.
if __name__ == "__main__":
    # main()
    analyze_validation_results(modelGen, modelVal)