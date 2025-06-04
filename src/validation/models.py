"""
Pydantic models for Validator data validation.
"""

from typing import List, Dict, Any, Optional, Union
import pandas as pd
from pydantic import BaseModel, ValidationError, validator, Field, model_validator
from src.generation.models import GeneratorData
from src.utils import print_warning, print_error
import json
from fuzzywuzzy import fuzz

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
    generatorData: Optional[GeneratorData] = Field(None, description="Corresponding Generator data used for validation")
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
    
    @validator('output')
    def validate_output_against_generator(cls, v, values):
        """Ensure all generator feedback lines are accounted for in validation output."""
        if not v or not v.feedback_lines:
            return v
            
        generator_data = values.get('generatorData')
        if not generator_data or not generator_data.feedback:
            return v
        
        # Create lists of (line_number, feedback) tuples for comparison
        generator_feedback_items = []
        for fb in generator_data.feedback:
            generator_feedback_items.append((str(fb.line_number), fb.feedback))
        
        validation_feedback_items = []
        for fb_line in v.feedback_lines:
            validation_feedback_items.append((str(fb_line.line_number), fb_line.feedback))
        
        missing_items = set()
        # Track which validation items have been matched to avoid duplicates
        matched_validation_indices = set()
        
        for gen_line, gen_feedback in generator_feedback_items:
            # Find best match for this generator feedback
            best_match_score = 0
            best_match_index = -1
            
            for val_index, (val_line, val_feedback) in enumerate(validation_feedback_items):
                if val_index in matched_validation_indices:
                    continue  # Skip already matched items
                    
                if gen_line == val_line:  # Same line number
                    similarity = fuzz.ratio(gen_feedback, val_feedback)
                    if similarity > best_match_score:
                        best_match_score = similarity
                        best_match_index = val_index
            
            # If good match found, replace validation feedback with generator feedback
            if best_match_score >= 85:  # 85% similarity threshold
                matched_validation_indices.add(best_match_index)
                # Replace the feedback text in the validation feedback line
                v.feedback_lines[best_match_index].feedback = gen_feedback
            else:
                # If no good match found, mark as missing
                missing_items.add((gen_line, gen_feedback))
        if missing_items:
            missing_str = "; ".join([f"Line {ln}: {fb}" for ln, fb in missing_items])
            raise ValueError(f"Missing generator feedback lines in validation output: {missing_str}")
        
        return v


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
    
    def create_dataframe(self) -> pd.DataFrame:
        """
        Create a DataFrame from dataProvider.validation_batch containing specified columns.
        Only includes results where success is True (Pydantic validation succeeded).
        
        Args:
            dataProvider: DataProvider instance with loaded validation_batch
            
        Returns:
            pd.DataFrame: DataFrame with columns [sid, line_number, feedback, classification]
        """
        rows = []

        # Filter for successful results only
        successful_results = [r for r in self.results if r.success and r.output]

        for result in successful_results:
            if result.output and result.output.feedback_lines:
                for feedback_line in result.output.feedback_lines:
                    rows.append({
                        'sid': result.sid,
                        'line_number': feedback_line.line_number,
                        'feedback': feedback_line.feedback,
                        'classification': 1 if feedback_line.classification == 'valid' else 0
                    })
        
        df = pd.DataFrame(rows)
        df['sid'] = df['sid'].astype(str)
        df['line_number'] = df['line_number'].astype(str)
        df['feedback'] = df['feedback'].astype(str)
        df['classification'] = df['classification'].astype(int)
        print(f"Created DataFrame with {len(df)} rows from {len(successful_results)} successful validation results")
        
        return df


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

