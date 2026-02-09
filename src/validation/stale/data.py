"""
Data access utilities for validation.
"""

from typing import Tuple, Dict, Any, List, Optional
import json
import os
from functools import lru_cache

from src.config import NUM_SIDS

from src.utils import print_warning, print_error
from src.validation.models import (
    ValidationInput, FeedbackLine, TestCase, GroundTruthFeedback,
    parse_feedback_from_json, parse_test_cases_from_json, convert_ground_truth_categories
)


# Import existing data functions from dataGenerator
try:
    from .dataGenerator import (
        get_data, get_row, get_row_4o_mini, get_row_4_1, get_row_3_opus, 
        get_annonated_data, get_row_35, get_row_35_sonnet, get_row_gemini_15_pro, 
        get_row_gemini_15_flash, get_row_qwen_coder_plus, get_row_deepseek_chat
    )
except ImportError as e:
    print_error(f"Error importing data functions: {e}")
    # Fallback functions can be defined here if needed


class DataProvider:
    """Centralized data provider for validation operations."""
    
    # TODO: Use the mapping from config.py instead of strings here
    # Model to data function mapping
    MODEL_DATA_MAP = {
        "gpt-4o": get_row,
        "gpt-4-turbo": get_row_4_1,
        "gpt-3.5-turbo": get_row_35,
        "gpt-4o-mini": get_row_4o_mini,
        "claude_3_opus": get_row_3_opus,
        "claude_3.5_sonnet": get_row_35_sonnet,
        "gemini-1.5-pro": get_row_gemini_15_pro,
        "gemini-1.5-flash": get_row_gemini_15_flash,
        "qwen-coder-plus": get_row_qwen_coder_plus,
        "deepseek-chat": get_row_deepseek_chat,
    }
    
    @classmethod
    def get_validation_data(cls, sid: int, generator_model: str) -> Tuple[str, List[FeedbackLine], str, str, str, List[TestCase]]:
        """
        Get validation data for a specific SID and generator model.
        
        Args:
            sid: Student ID
            generator_model: Name of the generator model
            
        Returns:
            Tuple containing: (question, feedback_lines, student_code, correct_code, all_testcases_str, test_cases)
        """
        # Get the appropriate data function
        data_func = cls.MODEL_DATA_MAP.get(generator_model)
        if data_func is None:
            # Default fallback
            data_func = lambda sid: get_row(sid, generator_model)
        
        try:
            # Get raw data
            labelled_feedback, unlabelled_feedback, student_code, correct_code, question, all_testcases, _, _, _ = get_data(data_func(sid))
            
            # Parse feedback
            feedback_lines = parse_feedback_from_json(unlabelled_feedback)
            
            # Parse test cases
            test_cases = parse_test_cases_from_json(all_testcases)
            
            return question, feedback_lines, student_code, correct_code, all_testcases, test_cases
            
        except Exception as e:
            print_error(f"Error getting validation data for SID {sid} with model {generator_model}: {e}")
            raise
    
    @classmethod
    def get_ground_truth_data(cls, sid: int) -> Optional[List[GroundTruthFeedback]]:
        """
        Get ground truth data for a specific SID.
        
        Args:
            sid: Student ID
            
        Returns:
            List of ground truth feedback or None if not available
        """
        try:
            ground_truth, _, _, _, _, _, _, _, _ = get_annonated_data(get_row_3_opus(sid))
            return convert_ground_truth_categories(ground_truth)
        except Exception as e:
            print_warning(f"Could not get ground truth data for SID {sid}: {e}")
            return None
    
    @classmethod
    def create_validation_input(cls, sid: int, generator_model: str, use_ground_truth: bool = False) -> ValidationInput:
        """
        Create a ValidationInput object for a specific SID.
        
        Args:
            sid: Student ID
            generator_model: Name of the generator model
            use_ground_truth: Whether to include ground truth data
            
        Returns:
            ValidationInput object
        """
        question, feedback_lines, student_code, correct_code, _, test_cases = cls.get_validation_data(sid, generator_model)
        
        ground_truth = None
        if use_ground_truth:
            ground_truth = cls.get_ground_truth_data(sid)
        
        return ValidationInput(
            question=question,
            student_code=student_code,
            correct_code=correct_code,
            feedback=feedback_lines,
            test_cases=test_cases,
            ground_truth=ground_truth
        )





def get_failed_sids_from_log(file_path: str, error_pattern: str = "429 You exceeded your current quota.") -> List[int]:
    """
    Extract SIDs that failed due to a specific error pattern from a log file.
    
    Args:
        file_path: Path to the log file
        error_pattern: Error pattern to search for
        
    Returns:
        List of SIDs that failed with the specified error
    """
    try:
        with open(file_path, 'r') as f:
            raw_responses = json.load(f)
        
        failed_sids = [
            resp['sid'] for resp in raw_responses 
            if error_pattern in str(resp.get('raw_response', ''))
        ]
        
        return failed_sids
        
    except Exception as e:
        print_error(f"Error reading log file {file_path}: {e}")
        return []


@lru_cache(maxsize=1)
def get_all_sids() -> List[int]:
    """
    Get all available SIDs from the dataset.
    
    Returns:
        List of all available SIDs
    """
    # This would need to be implemented based on your dataset structure
    # For now, return the range that seems to be used in the original code
    return list(range(1, NUM_SIDS))


def filter_quota_exceeded_sids(file_path: str) -> List[int]:
    """
    Filter out SIDs that failed due to quota exceeded errors.
    
    Args:
        file_path: Path to the log file
        
    Returns:
        List of SIDs that need to be retried
    """
    return get_failed_sids_from_log(file_path, "429 You exceeded your current quota.")


