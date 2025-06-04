"""
Utility functions for validation operations.
"""

import os
import sys
import json
import datetime
from typing import List, Dict, Any, Optional

from src.config import pathValidator, pathLogs

from src.utils import print_warning, print_error
from src.validation.models import ValidationResult, ValidationBatch


def load_validation_batch_from_file(modelGen: str, modelVal: str) -> Optional[ValidationBatch]:
    """
    Load a validation batch from a JSON file.
    
    Args:
        modelGen: Model used for generation
        modelVal: Model used for validation

    Returns:
        ValidationBatch or None: Loaded batch if successful
    """
    try:
        file_path = f'{pathValidator}/gen={modelGen}/val={modelVal}.json'
        # file_path = f'{pathValidator}/new_labeller_gen_{modelGen}_val_{modelVal}_2025-05-05_13-06-52.json'
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # If data is a list (old format), convert to new format
        if isinstance(data, list):
            # Try to extract metadata from filename
            filename = os.path.basename(file_path)
            use_ground_truth = "cheat" in filename

            
            
            # Convert list to ValidationResult objects
            results = []
            for item in data:
                try:
                    # Ensure item has required fields and attach it
                    result = ValidationResult(**item)
                    results.append(result)

                except Exception as e:
                    # Handle parsing errors gracefully
                    error_lines = str(e).splitlines()
                    error_lines_alt = [error_lines[i] for i in range(1, len(error_lines), 3)]
                    error_str = " | ".join(error_lines_alt)
                    print_warning(f"Error parsing result for SID {item.get('sid', 'unknown')}: {error_str}")

                    # Try to atleast create a minimal result
                    minimal_result = ValidationResult(
                        sid=item.get('sid', -1),
                        success=False,
                        output=None,
                        raw_response=item.get('raw_response', ''),
                        error=error_str
                    )

                    results.append(minimal_result)

            return ValidationBatch(
                generator_model=modelGen,
                validator_model=modelVal,
                results=results,
                use_ground_truth=use_ground_truth
            )
        else:
            # New format
            return ValidationBatch(**data)
            
    except Exception as e:
        print_error(f"Error loading validation batch for gen={modelGen}, val={modelVal}: {e}")
        sys.exit(1)
        return None


def save_validation_batch_to_file(batch: ValidationBatch, file_path: str) -> bool:
    """
    Save a validation batch to a JSON file.
    
    Args:
        batch: ValidationBatch to save
        file_path: Path to save the batch
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(batch.dict(), f, indent=4)
        
        return True
        
    except Exception as e:
        print_error(f"Error saving validation batch to {file_path}: {e}")
        return False


def merge_validation_results(file_paths: List[str], output_path: str) -> bool:
    """
    Merge multiple validation result files into a single file.
    
    Args:
        file_paths: List of paths to validation result files
        output_path: Path to save the merged results
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        all_results = []
        
        for file_path in file_paths:
            batch = load_validation_batch_from_file(file_path)
            if batch:
                all_results.extend(batch.results)
            else:
                print_warning(f"Could not load results from {file_path}")
        
        if not all_results:
            print_error("No results to merge")
            return False
        
        # Create merged batch (use first batch's metadata)
        first_batch = load_validation_batch_from_file(file_paths[0])
        if not first_batch:
            print_error("Could not load first batch for metadata")
            return False
        
        merged_batch = ValidationBatch(
            generator_model=first_batch.generator_model,
            validator_model=first_batch.validator_model,
            results=all_results,
            use_ground_truth=first_batch.use_ground_truth
        )
        
        return save_validation_batch_to_file(merged_batch, output_path)
        
    except Exception as e:
        print_error(f"Error merging validation results: {e}")
        return False


def filter_successful_results(results: List[ValidationResult]) -> List[ValidationResult]:
    """
    Filter results to only include successful validations.
    
    Args:
        results: List of validation results
        
    Returns:
        List[ValidationResult]: Filtered successful results
    """
    return [r for r in results if r.success and r.output]


def get_failed_sids(results: List[ValidationResult]) -> List[int]:
    """
    Get list of SIDs that failed validation.
    
    Args:
        results: List of validation results
        
    Returns:
        List[int]: List of failed SIDs
    """
    return [r.sid for r in results if not r.success]


def print_validation_summary(batch: ValidationBatch) -> None:
    """
    Print a comprehensive summary of validation results.
    
    Args:
        batch: ValidationBatch to summarize
    """
    stats = batch.get_summary_stats()
    
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY: {batch.generator_model} -> {batch.validator_model}")
    print(f"{'='*60}")
    print(f"Generator Model: {batch.generator_model}")
    print(f"Validator Model: {batch.validator_model}")
    print(f"Used Ground Truth: {'Yes' if batch.use_ground_truth else 'No'}")
    print(f"")
    print(f"Total Results: {stats['total_results']}")
    print(f"Successful Results: {stats['successful_results']} ({stats['success_rate']:.2%})")
    print(f"")
    print(f"Classification Results:")
    print(f"  Valid Feedback: {stats['total_valid']}")
    print(f"  Invalid Feedback: {stats['total_invalid']}")
    print(f"  Precision: {stats['precision']:.4f}")
    print(f"{'='*60}")
    
    # Print failed SIDs if any
    failed_sids = get_failed_sids(batch.results)
    if failed_sids:
        print(f"\nFailed SIDs ({len(failed_sids)}): {failed_sids}")


def analyze_error_patterns(results: List[ValidationResult]) -> Dict[str, Any]:
    """
    Analyze error patterns in validation results.
    
    Args:
        results: List of validation results
        
    Returns:
        Dict containing error analysis
    """
    failed_results = [r for r in results if not r.success]
    
    error_patterns = {}
    for result in failed_results:
        # Extract error type from raw response
        error_key = "unknown_error"
        
        if "429" in result.raw_response:
            error_key = "quota_exceeded"
        elif "timeout" in result.raw_response.lower():
            error_key = "timeout"
        elif "connection" in result.raw_response.lower():
            error_key = "connection_error"
        elif "json" in result.raw_response.lower():
            error_key = "json_parse_error"
        elif "error" in result.raw_response.lower():
            error_key = "general_error"
        
        if error_key not in error_patterns:
            error_patterns[error_key] = []
        error_patterns[error_key].append(result.sid)
    
    return {
        "total_failed": len(failed_results),
        "error_patterns": error_patterns,
        "error_counts": {k: len(v) for k, v in error_patterns.items()}
    }