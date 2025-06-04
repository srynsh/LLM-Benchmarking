"""
Pydantic models for Validator data validation.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, ValidationError, validator, Field, model_validator
from src.utils import print_warning, print_error
import json

class GroundTruthFeedback(BaseModel):
    """Model for ground truth feedback with category."""
    line_number: Union[str, int] = Field(..., description="Line number where the feedback applies")
    feedback: str = Field(..., description="Feedback text for the student")
    category: str = Field(..., description="Ground truth category (valid/invalid)")
    
    @validator('line_number', pre=True)
    def validate_line_number(cls, v):
        """Convert line_number to string for consistency."""
        return str(v)


class ValidatedFeedbackLine(BaseModel):
    """Model for feedback line with validation results."""
    line_number: Union[str, int] = Field(..., description="Line number referenced by the feedback")
    feedback: str = Field(..., description="The feedback provided by the TA")
    analysis: str = Field(..., description="Analysis of the feedback's accuracy")
    classification: str = Field(..., description="Feedback validity classification (valid/invalid)")

    def __init__(self, **data):
        """Alias 'line_num' to 'line_number' for input data."""
        if 'line_num' in data and 'line_number' not in data:
            data['line_number'] = data.pop('line_num')
        super().__init__(**data)
    
    # TODO: Also ensure that the line number and feedback matches the generator's output
    @validator('line_number', pre=True)
    def validate_line_number(cls, v):
        """Convert line_number to string for consistency."""
        return str(v)
    
    @validator('classification')
    def validate_classification(cls, v):
        """Ensure classification is either 'valid' or 'invalid'."""
        if v.lower() not in ['valid', 'invalid']:
            raise ValueError("Classification must be either 'valid' or 'invalid'")
        return v.lower()


class ValidationOutput(BaseModel):
    """Model for LLM validation output format."""
    mistakes: Optional[List[str]] = Field(..., description="List of mistakes found in the student's code")
    fixes: Optional[List[str]] = Field(..., description="List of corrections proposed in the fixed code")
    feedback_lines: List[ValidatedFeedbackLine] = Field(..., description="List of validated feedback lines")


class ValidationResult(BaseModel):
    """Model for complete validation result."""
    sid: int = Field(..., description="Student ID")
    raw_response: str = Field(..., description="Raw LLM response")
    output: Optional[ValidationOutput] = Field(None, description="Parsed validation output")
    success: Optional[bool] = Field(default=True, description="Whether validation was successful")
    timestamp: Optional[str] = Field(None, description="Timestamp of validation")
    
    def get_classification_counts(self) -> Dict[str, int]:
        """Get counts of valid/invalid classifications."""
        if not self.output or not self.output.feedback_lines:
            return {"valid": 0, "invalid": 0}
        
        counts = {"valid": 0, "invalid": 0}
        for feedback_line in self.output.feedback_lines:
            classification = feedback_line.classification.lower()
            if classification in counts:
                counts[classification] += 1
        
        return counts


class ValidationBatch(BaseModel):
    """Model for batch validation results."""
    generator_model: str = Field(..., description="Model used for generation")
    validator_model: str = Field(..., description="Model used for validation")
    results: List[ValidationResult] = Field(..., description="List of validation results")
    use_ground_truth: bool = Field(default=False, description="Whether ground truth was used")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the batch."""
        total_results = len(self.results)
        successful_results = len([r for r in self.results if r.success])
        
        total_valid = 0
        total_invalid = 0
        
        for result in self.results:
            if result.success and result.output:
                counts = result.get_classification_counts()
                total_valid += counts["valid"]
                total_invalid += counts["invalid"]
        
        precision = total_valid / (total_valid + total_invalid) if (total_valid + total_invalid) > 0 else 0
        
        return {
            "total_results": total_results,
            "successful_results": successful_results,
            "success_rate": successful_results / total_results if total_results > 0 else 0,
            "total_valid": total_valid,
            "total_invalid": total_invalid,
            "precision": precision
        }


def validate_validation_output(llm_response: Dict[str, Any]) -> bool:
    """
    Validate that LLM output matches the expected validation format.
    
    Args:
        llm_response: Raw LLM response dictionary
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        ValidationOutput(**llm_response)
        return True
    except ValidationError as e:
        print_error(f"Validation output validation failed: {e}")
        return False



def convert_ground_truth_categories(ground_truth: List[Dict[str, Any]]) -> List[GroundTruthFeedback]:
    """
    Convert ground truth categories from detailed format to valid/invalid.
    
    Args:
        ground_truth: List of ground truth feedback with detailed categories
        
    Returns:
        List[GroundTruthFeedback]: Converted ground truth feedback
    """
    converted_feedback = []
    
    for item in ground_truth:
        # Convert detailed categories to valid/invalid
        category = item.get('category', '')
        if category in ['TP', 'FP-E', 'FP-R', 'TP-E', 'TP-R']:
            new_category = 'valid'
        elif category in ['FP-H', 'FP-I']:
            new_category = 'invalid'
        else:
            new_category = category  # Keep as is if already valid/invalid
        
        converted_item = {
            'line_number': item['line_number'],
            'feedback': item['feedback'],
            'category': new_category
        }
        
        try:
            converted_feedback.append(GroundTruthFeedback(**converted_item))
        except ValidationError as e:
            print_warning(f"Error converting ground truth item: {e}")
    
    return converted_feedback


def validate_json_file_data(file_path: str) -> List[ValidationResult]:
    """
    Validate and parse validation results from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing validation results
        
    Returns:
        List[ValidationResult]: List of valid validation results
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print_error(f"Expected list in JSON file {file_path}")
            return []
        
        valid_results = []
        for item in data:
            try:
                # Parse the output if it exists and is not empty
                output = None
                if item.get('output') and isinstance(item['output'], dict):
                    try:
                        output = ValidationOutput(**item['output'])
                    except ValidationError as e:
                        print_warning(f"Invalid output format for SID {item.get('sid', 'unknown')}: {e}")
                
                result = ValidationResult(
                    sid=item['sid'],
                    raw_response=item['raw_response'],
                    output=output,
                    success=output is not None,
                    generator_model=item.get('generator_model', 'unknown'),
                    validator_model=item.get('validator_model', 'unknown'),
                    timestamp=item.get('timestamp')
                )
                valid_results.append(result)
                
            except ValidationError as e:
                print_warning(f"Error validating result for SID {item.get('sid', 'unknown')}: {e}")
        
        return valid_results
        
    except Exception as e:
        print_error(f"Error reading validation file {file_path}: {e}")
        return []
