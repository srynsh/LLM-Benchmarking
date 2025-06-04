"""
Data access utilities for validation.
"""

from typing import Tuple, Dict, Any, List, Optional
import json
import os
import sys
from functools import lru_cache

from src.config import MODELS_GEN, NUM_SIDS, pathValidator

from src.utils import print_warning, print_error
from src.validation.models import (
    ValidationResult, ValidationBatch
)
from src.generation.models import GeneratorData
from src.generation.data import (load_existing_results, get_processed_results)


class DataProvider:
    """Centralized data provider for validation operations."""
    validation_batch: Optional[ValidationBatch] = None
    generation_batch: Optional[GeneratorData] = None

    @classmethod
    def load_validation_batch(cls, modelGen: str, modelVal: str) -> Optional[ValidationBatch]:
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

                cls.validation_batch = ValidationBatch(
                    generator_model=modelGen,
                    validator_model=modelVal,
                    results=results,
                    use_ground_truth=use_ground_truth
                )
            else:
                # New format
                cls.validation_batch = ValidationBatch(**data)

        except Exception as e:
            print_error(f"Error loading validation batch for gen={modelGen}, val={modelVal}: {e}")
            sys.exit(1)
            cls.validation_batch = None

    @classmethod
    def save_validation_batch_to_file(cls, file_path: str) -> bool:
        """
        Save a validation batch to a JSON file.
        
        Args:
            file_path: Path to save the batch

        Returns:
            bool: True if successful, False otherwise
        """
        if cls.validation_batch is None:
            print_warning("No validation batch to save.")
            return False

        try:
            with open(file_path, 'w') as f:
                json.dump(cls.validation_batch.dict(), f, indent=4)

            return True

        except Exception as e:
            print_error(f"Error saving validation batch to {file_path}: {e}")
            return False

    @classmethod
    def filter_successful_results(cls) -> List[ValidationResult]:
        """
        Filter results to only include successful validations.
        
        Args:
            results: List of validation results
            
        Returns:
            List[ValidationResult]: Filtered successful results
        """
        if cls.validation_batch is None:
            return []

        return [r for r in cls.validation_batch.results if r.success and r.output]
    
    @classmethod
    def get_failed_sids(cls) -> List[int]:
        """
        Get list of SIDs that failed validation.
        
        Args:
            results: List of validation results
            
        Returns:
            List[int]: List of failed SIDs
        """
        if cls.validation_batch is None:
            return []

        return [r.sid for r in cls.validation_batch.results if not r.success]

    @classmethod
    def print_validation_summary(cls) -> None:
        """
        Print a comprehensive summary of validation results.
    
        Args:
            batch: ValidationBatch to summarize
        """
        batch = cls.validation_batch
        if batch is None:
            print_warning("No validation batch loaded.")
            return
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
        failed_sids = cls.get_failed_sids()
        if failed_sids:
            print(f"\nFailed SIDs ({len(failed_sids)}): {failed_sids}")

    @classmethod
    def analyze_error_patterns(cls) -> Dict[str, Any]:
        """
        Analyze error patterns in validation results.
        
        Args:
            results: List of validation results
            
        Returns:
            Dict containing error analysis
        """
        failed_results = [r for r in cls.validation_batch.results if not r.success]
        
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
    
    @classmethod
    def load_generation_batch(cls, modelGen: str):
        print(f"\n{'='*60}")
        print(f"GENERATION SUMMARY: {modelGen}")
        print(f"{'='*60}")

        category_required = modelGen in MODELS_GEN
        results = load_existing_results(modelGen)
        processed_results = get_processed_results(results, category_required=category_required)
        cls.generation_batch = processed_results

    @classmethod
    def __init__(cls, modelGen: str, modelVal: str) -> None:
        cls.load_generation_batch(modelGen)
        cls.load_validation_batch(modelGen, modelVal)
        
        # Print summary
        cls.print_validation_summary()

        # Analyze error patterns
        error_analysis = cls.analyze_error_patterns()
        if error_analysis['total_failed'] > 0:
            print(f"\nError Analysis:")
            print(f"Total Failed: {error_analysis['total_failed']}")
            for error_type, count in error_analysis['error_counts'].items():
                print(f"  {error_type}: {count} SIDs")
        
