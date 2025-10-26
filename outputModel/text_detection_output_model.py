"""
Pydantic models for text detection output
"""
from pydantic import BaseModel, Field
from typing import List, Dict


class IndividualModelScore(BaseModel):
    """Individual model prediction for text detection"""
    model_name: str = Field(description="Name of the model")
    prediction: str = Field(description="Prediction (bot or human)")
    confidence: float = Field(description="Confidence score (0.0 to 1.0)")
    bot_score: float = Field(description="Bot/AI-generated score (0.0 to 1.0)")
    human_score: float = Field(description="Human-written score (0.0 to 1.0)")


class TextDetectionResult(BaseModel):
    """Ensemble prediction result for text detection"""
    text: str = Field(description="Input text that was analyzed")
    prediction: str = Field(description="Final prediction (AI-generated or human-generated)")
    confidence: float = Field(description="Confidence score (0.0 to 1.0)")
    average_bot_score: float = Field(description="Average bot/AI-generated score across all models")
    average_human_score: float = Field(description="Average human-written score across all models")
    voting_method: str = Field(description="Voting method used (majority, weighted, or unanimous)")
    individual_scores: List[IndividualModelScore] = Field(
        default_factory=list,
        description="Individual results from each model"
    )
    voting_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Voting breakdown (bot and human vote counts)"
    )
    
    def to_json(self, indent: int = 2) -> str:
        """Convert model to JSON string"""
        return self.model_dump_json(indent=indent)
    
    def to_dict(self) -> dict:
        """Convert model to dictionary"""
        return self.model_dump()
