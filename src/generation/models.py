"""
Pydantic models for Generator data validation.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, ValidationError, validator, Field, model_validator
from src.generation.data import load_student_code_mapping
from src.utils import print_warning, print_error
import pandas as pd
import os


class LLMFeedback(BaseModel):
    """Model for individual feedback from LLM output."""
    line_number: Union[str, int] = Field(..., description="Line number where the mistake occurs")
    feedback: str = Field(..., description="Feedback text for the student")
    
    @validator('line_number', pre=True)
    def validate_line_number(cls, v):
        """Convert line_number to string for consistency."""
        return str(v)


class LLMOutput(BaseModel):
    """Model for raw LLM output format."""
    correct_code: str = Field(..., description="The student code with minimal changes to pass all test cases")
    feedbacks: List[LLMFeedback] = Field(..., description="List of feedback items")


class ProcessedFeedback(BaseModel):
    """Model for processed feedback with category."""
    line_number: Union[str, int] = Field(..., description="Line number where the mistake occurs")
    feedback: str = Field(..., description="Feedback text for the student")
    category: Optional[str] = Field(None, description="Feedback category (TP, FP-H, FP-I, FP-E, FN)")


class RepairSuccess(BaseModel):
    """Model for repair success status."""
    success: bool = Field(..., description="Whether the repair was successful")
    status_code: int = Field(..., description="Status code of the repair operation")

class GeneratorData(BaseModel):
    """Model for complete Generator data structure."""
    sid: int = Field(..., description="Student ID")
    repaired_code: str = Field(..., description="Code after repair (equal to correct_code from LLM)")
    feedback: List[ProcessedFeedback] = Field(..., description="List of processed feedback with categories")
    student_code: str = Field(..., description="Original student code from dataset")
    pid: int = Field(..., description="Problem ID")
    category_required: bool = Field(default=True, description="Whether category field is required in feedback")
    
    @model_validator(mode='before')
    def validate_feedback_with_category_required(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate feedback list based on category_required setting."""
        feedback_list = values.get('feedback', [])
        category_required = values.get('category_required', True)
        
        if not isinstance(feedback_list, list):
            raise ValueError("feedback must be a list")
        
        for item in feedback_list:
            if not isinstance(item, dict):
                raise ValueError("Each feedback item must be a dictionary")
            
            required_fields = ['line_number', 'feedback']
            if category_required:
                required_fields.append('category')
            
            for field in required_fields:
                if field not in item:
                    raise ValueError(f"Missing required field '{field}' in feedback item")
            
            # Additional check for empty category when required
            if category_required and (not item.get('category') or item.get('category') == ''):
                raise ValueError("Category cannot be empty when category_required is True")
        
        return values


def validate_llm_output(llm_response: Dict[str, Any]) -> bool:
    """
    Validate that LLM output matches the expected format.
    
    Args:
        llm_response: Raw LLM response dictionary
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        LLMOutput(**llm_response)
        return True
    except ValidationError as e:
        print_error(f"LLM output validation failed: {e}")
        return False


def validate_generator_data(data: Dict[str, Any], category_required: bool = True) -> bool:
    """
    Validate that data matches the complete Generator data structure.
    
    Args:
        data: Data dictionary to validate
        category_required: Whether category field is required in feedback
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Add category_required to data if not present
        if 'category_required' not in data:
            data['category_required'] = category_required
        
        GeneratorData(**data)
        return True
    except ValidationError as e:
        print(f"- {e}")
        return False


def convert_llm_to_generator_data(llm_output: Dict[str, Any], sid: int, pid: int, 
                                student_code_mapping: Dict[int, str], category_required: bool = False) -> Optional[Dict[str, Any]]:
    """
    Convert LLM output to Generator data format with student code from dataset.
    
    Args:
        llm_output: Raw LLM output
        sid: Student ID
        pid: Problem ID
        student_code_mapping: Mapping of sid to student code
        category_required: Whether category field is required in feedback
        
    Returns:
        Dict or None: Converted data if successful, None otherwise
    """
    try:
        # Validate LLM output first
        if not validate_llm_output(llm_output):
            return None
        
        # Get student code from mapping
        student_code = student_code_mapping.get(sid)
        if student_code is None:
            print_warning(f"No student code found for sid {sid}")
            return None
        
        # Convert feedbacks to processed format (without categories initially)
        processed_feedback = []
        for fb in llm_output['feedbacks']:
            feedback_data = {
                'line_number': str(fb['line_number']),
                'feedback': fb['feedback']
            }
            
            # Only add category if it's required
            if category_required:
                feedback_data['category'] = ''  # Default category, should be updated by labeling process
            
            processed_feedback.append(feedback_data)
        
        # Create generator data structure
        generator_data = {
            'sid': sid,
            'repaired_code': llm_output['correct_code'],
            'feedback': processed_feedback,
            'student_code': student_code,
            'pid': pid,
            'category_required': category_required
        }
        
        # Validate the complete structure
        if validate_generator_data(generator_data, category_required):
            return generator_data
        else:
            return None
            
    except Exception as e:
        print_error(f"Error converting LLM output for sid {sid}: {e}")
        return None


def validate_json_file_data(data: List[Dict[str, Any]], category_required: bool = True) -> List[int]:
    """
    Validate data from a JSON file and return list of valid SIDs.
    
    Args:
        data: List of data entries from JSON file
        category_required: Whether category field is required in feedback
        
    Returns:
        List[int]: List of SIDs that have valid data structure
    """
    valid_sids = []
    student_code_mapping = load_student_code_mapping()
    
    for entry in data:
        try:
            # Add category_required to entry if not present
            if 'category_required' not in entry:
                entry['category_required'] = category_required
                
            # Validate the entry structure
            if validate_generator_data(entry, category_required):
                sid = entry['sid']
                
                # Check if student_code matches the dataset
                expected_student_code = student_code_mapping.get(sid)
                if expected_student_code is not None:
                    actual_student_code = entry.get('student_code', '')
                    
                    # Normalize whitespace for comparison
                    expected_normalized = ' '.join(expected_student_code.split())
                    actual_normalized = ' '.join(actual_student_code.split())
                    
                    if expected_normalized == actual_normalized:
                        valid_sids.append(sid)
                    else:
                        print_warning(f"Student code mismatch for sid {sid}")
                else:
                    print_warning(f"No expected student code found for sid {sid}")
            else:
                print_warning(f"SID {entry.get('sid', 'unknown')} validation failed")

        except Exception as e:
            print_error(f"Error validating entry: {e}")
            
    return valid_sids
