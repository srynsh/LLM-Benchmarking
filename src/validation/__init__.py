"""
Validation module for LLM feedback assessment.

This module provides a modular, Pydantic-based approach to validating
LLM-generated feedback using various validation models.
"""

from .models import (
    ValidationInput,
    ValidationOutput, 
    ValidationResult,
    ValidationBatch,
    FeedbackLine,
    ValidatedFeedbackLine,
    TestCase,
    GroundTruthFeedback
)

from .service import (
    ValidationService,
    ValidationRunner,
    create_attempt_name,
    calculate_precision_stats,
    print_validation_stats
)

from .data import (
    DataProvider,
    load_existing_validation_results,
    save_validation_results,
    filter_quota_exceeded_sids
)

from .utils import (
    print_validation_summary,
    analyze_error_patterns,
    load_validation_batch_from_file,
    save_validation_batch_to_file
)

from .prompt import (
    get_validation_prompt,
    get_validation_prompt_with_ground_truth
)

__all__ = [
    # Models
    'ValidationInput',
    'ValidationOutput',
    'ValidationResult', 
    'ValidationBatch',
    'FeedbackLine',
    'ValidatedFeedbackLine',
    'TestCase',
    'GroundTruthFeedback',
    
    # Services
    'ValidationService',
    'ValidationRunner',
    'create_attempt_name',
    'calculate_precision_stats',
    'print_validation_stats',
    
    # Data access
    'DataProvider',
    'load_existing_validation_results',
    'save_validation_results',
    'filter_quota_exceeded_sids',
    
    # Utilities
    'print_validation_summary', 
    'analyze_error_patterns',
    'load_validation_batch_from_file',
    'save_validation_batch_to_file',
    
    # Prompts
    'get_validation_prompt',
    'get_validation_prompt_with_ground_truth'
]