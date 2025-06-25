"""
Pydantic models for Validator data validation.
"""

import sys
from typing import Annotated, List, Dict, Any, Optional, Union
import pandas as pd
from pydantic import BaseModel, BeforeValidator, ValidationError, validator, Field, model_validator
from src.config import VALIDATOR_REPAIR
from src.generation.models import GeneratorData
from src.utils import print_warning, print_error
import json
from fuzzywuzzy import fuzz


####################
# Pydantic Models
####################

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
        if VALIDATOR_REPAIR.partially_valid_label:
            # If partially valid is allowed, treat it as invalid for consistency
            if v.lower() == 'partially valid':
                return 'invalid'
            
        if v.lower() not in ['valid', 'invalid']:
            raise ValueError("Classification must be either 'valid' or 'invalid'")
        return v.lower()


class ValidationOutput(BaseModel):
    """Model for LLM validation output format."""
    mistakes: Optional[List[str]] = Field(..., description="List of mistakes found in the student's code")
    fixes: Optional[List[str]] = Field(..., description="List of corrections proposed in the fixed code")
    feedback_lines: List[ValidatedFeedbackLine] = Field(..., description="List of validated feedback lines")

# Contains one validation result for a given submission ID (sid)
class ValidationResult(BaseModel):
    """Model for complete validation result."""
    generatorData: Optional[GeneratorData] = Field(None, description="Corresponding Generator data used for validation")
    sid: int = Field(..., description="Student ID")
    raw_response: str = Field(..., description="Raw LLM response")
    output: Optional[ValidationOutput] = Field(None, description="Parsed validation output")
    success: Optional[bool] = Field(default=True, description="Whether validation was successful")
    fidFailureCount: int = Field(..., description="Count of failed feedback lines")
    timestamp: Optional[str] = Field(None, description="Timestamp of validation")

    @model_validator(mode='before')
    @classmethod
    def validate_output(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the output field."""
        # Ensure output is present and has feedback_lines
        if value is None or 'output' not in value or not value['output']:
            return value
        
        output = value['output']
        generator_data = value.get('generatorData')
        if not generator_data or not generator_data.feedback:
            return value
        
        # Check if output is a dict
        if 'feedback_lines' not in output or not output['feedback_lines']:
            raise ValueError("Validation output must contain 'output' with 'feedback_lines'")

        # Choose matching strategy
        if VALIDATOR_REPAIR.feedback_match_fuzzy:
            matched_items, missing_items = cls._fuzzy_match(generator_data, output)
        else:
            matched_items, missing_items = cls._exact_match(generator_data, output)

        value['fidFailureCount'] = len(missing_items)  # Count of missing feedback items
        cls._print_validate_output(value, generator_data, matched_items, missing_items)

        return value
    
    @classmethod
    def _print_validate_output(cls, value: dict[str, Any], generator_data, matched_items, missing_items) -> None:
        """Print the validation output for debugging."""
        # Ensure we have at least one matched item
        if not matched_items:
            raise ValueError("No matching feedback lines found in validation output")
        
        # If missing items, log a warning and add them as unsuccessful
        if missing_items:
            value['success'] = False  # Mark as unsuccessful if there are missing items
            missing_str = "; ".join([f"Line {ln}: {fb}" for ln, fb in missing_items])
            print_warning(f"Missing feedback items for SID {value.get('sid', 'unknown')}: {missing_str}")

            
    @classmethod
    def _get_feedback_items(cls, generator_data: GeneratorData, v: dict[str, Any]) -> List[tuple[str, str]]:
        """Extract feedback items from generator data."""
        if 'feedback_lines' not in v or not v['feedback_lines']:
            raise ValueError("Validation output must contain 'feedback_lines'")

        # Create sets of (line_number, feedback) tuples for comparison
        generator_feedback_items = set()
        for fb in generator_data.feedback:
            if not fb.line_number or not fb.feedback:
                raise ValueError("Generator feedback must contain 'line_number' and 'feedback'")

            generator_feedback_items.add((str(fb.line_number), fb.feedback))

        validation_feedback_items = set()
        for fb_line in v['feedback_lines']:
            if 'line_number' not in fb_line or 'feedback' not in fb_line:
                raise ValueError("Validation feedback lines must contain 'line_number' and 'feedback'")
            
            validation_feedback_items.add((str(fb_line['line_number']), fb_line['feedback']))

        return generator_feedback_items, validation_feedback_items

    @classmethod
    def _exact_match(cls, generator_data: dict[str, Any], output: dict[str, Any]) -> bool:
        generator_feedback_items, validation_feedback_items = cls._get_feedback_items(generator_data, output)

        # Check if all generator feedback items are present in validation output
        matched_items = generator_feedback_items & validation_feedback_items
        missing_items = generator_feedback_items - validation_feedback_items

        return matched_items, missing_items

    @classmethod
    def _fuzzy_match(cls, generator_data: dict[str, Any], output: dict[str, Any]) -> bool:
        # Create lists of (line_number, feedback) tuples for comparison
        generator_feedback_items, validation_feedback_items = cls._get_feedback_items(generator_data, output)
        
        # Sort feedback items to ensure deterministic processing
        generator_feedback_items = sorted(generator_feedback_items)
        validation_feedback_items = sorted(validation_feedback_items)

        matched_items = set()
        missing_items = set()
        # Track which validation items have been matched to avoid duplicates
        matched_validation_indices = set()
        
        for gen_line, gen_feedback in generator_feedback_items:
            # Find best match for this generator feedback
            best_match_score = 0
            best_match_index = -1
            # print_error(f"Generator feedback: {gen_line}. {gen_feedback}")

            for val_index, (val_line, val_feedback) in enumerate(validation_feedback_items):
                if val_index in matched_validation_indices:
                    continue  # Skip already matched items

                if val_line != gen_line:  # Different line number
                    continue  # Skip if line numbers don't match

                # Clip validation feedback if it's shorter than generator feedback. To handle cases where validator got "lazy"
                if VALIDATOR_REPAIR.clip_feedback_lazy:
                    if len(val_feedback) < len(gen_feedback):
                        gen_feedback = gen_feedback[:len(val_feedback)]

                if gen_line == val_line:  # Same line number
                    similarity = fuzz.ratio(gen_feedback, val_feedback)
                    if similarity > best_match_score:
                        best_match_score = similarity
                        best_match_index = val_index

                # if generator_data.sid == 162:
                # print_error(f"\tValidation feedback: {val_line}. {val_feedback}")
                # print_error(f"\tSimilarity score: {similarity}, Best match score: {best_match_score}, Best match index: {best_match_index}")
            
            # If good match found, replace validation feedback with generator feedback
            if best_match_score >= 85:  # 85% similarity threshold
                matched_validation_indices.add(best_match_index)
                # Replace the feedback text in the validation feedback line
                output['feedback_lines'][best_match_index]['feedback'] = gen_feedback
                matched_items.add((gen_line, gen_feedback))
            else:
                # If no good match found, mark as missing
                missing_items.add((gen_line, gen_feedback))

        return matched_items, missing_items

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

    def get_failure_count(self) -> int:
        """Get the count of failed to validate feedback lines."""
        return self.fidFailureCount if self.fidFailureCount is not None else 0
    
    def get_success_count(self) -> int:
        """Get the total count of feedback lines."""
        if not self.output or not self.output.feedback_lines:
            return 0
    
        return len(self.output.feedback_lines)

    def is_failed(self) -> bool:
        """Check if the validation result failed."""
        return not self.success or not self.output or self.get_failure_count() > 0

# Contains the list of validator results for a given validator and generator model
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

        fids = [fb for r in self.results
                          if r.output and r.output.feedback_lines
                          for fb in r.output.feedback_lines]

        total_fids  = len(fids)
        failure_fids = sum([r.fidFailureCount for r in self.results])
        successful_fids = total_fids - failure_fids

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
            "total_fids": total_fids,
            "successful_fids": successful_fids,
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
                    label = None # Default to None if classification is neither valid nor invalid
                    if feedback_line.classification == 'valid':
                        label = 1
                    elif feedback_line.classification == 'invalid':
                        label = 0
                    rows.append({
                        'sid': result.sid,
                        'line_number': feedback_line.line_number,
                        'feedback': feedback_line.feedback,
                        'classification': label
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
