
import os
import sys
import json
import numpy as np
import pprint
import datetime
from typing import List, Optional
from tqdm import tqdm
import pandas as pd

sys.path.append("..")

from src.config import NUM_SIDS, Model, pathLogs
from src.validation.service import ValidationRunner, ValidationService
from src.validation.data import DataProvider
from src.validation.models import ValidationBatch, ValidationResult
from src.utils import print_warning, print_error
from src.config import pGa_CONST, pathLogs

# Configuration: Set generator and validator models here
MODEL_GENERATOR = Model.GEMINI_1_5_PRO.value

# MODELS_VALIDATE = [
#     Model.CLAUDE_4_5_HAIKU.value,
#     Model.CLAUDE_4_5_SONNET.value,
#     Model.GEMINI_3_FLASH_PREVIEW.value,
#     Model.GPT_5_mini.value]
# MODEL_VALIDATOR = MODELS_VALIDATE[3]
MODEL_VALIDATOR = Model.GEMINI_3_PRO_PREVIEW.value
sids_restrict = None # [1,2] # Set to None to run all SIDs

# Constants (don't change)
use_ground_truth = False

def main():
    """Main entry point for validation operations."""
    # Determine SIDs to process
    if sids_restrict is None:
        sids = range(1, NUM_SIDS + 1)
    else:
        sids = sids_restrict

    # Create validation runner
    runner = ValidationRunner(MODEL_GENERATOR, MODEL_VALIDATOR, use_ground_truth)
    
    # Run quota recovery
    runner.run_validation(sids)
    
    print("Validation LLM invocation completed successfully!")

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


if __name__ == "__main__":
    main()