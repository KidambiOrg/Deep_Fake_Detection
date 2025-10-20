"""
Pydantic models for deepfake detection output
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class IndividualModelResult(BaseModel):
    """Individual model detection result"""
    model: str = Field(description="Name of the model")
    prediction: str = Field(description="Prediction (FAKE or REAL)")
    confidence: float = Field(description="Confidence score (0.0 to 1.0)")
    fakeScore: float = Field(default=0.0, description="Fake score (0.0 to 1.0)")
    realScore: float = Field(default=0.0, description="Real score (0.0 to 1.0)")


class VotingResult(BaseModel):
    """Voting breakdown"""
    fake: int = Field(default=0, description="Number of models voting FAKE")
    real: int = Field(default=0, description="Number of models voting REAL")
    total: int = Field(default=0, description="Total number of models")


class ImageDetectionResult(BaseModel):
    """Detection result for a single image"""
    imageFile: str = Field(description="Name of the image file")
    prediction: str = Field(description="Final prediction (FAKE or REAL)")
    confidence: float = Field(description="Confidence score (0.0 to 1.0)")
    averageFake: float = Field(description="Average fake score across all models")
    averageReal: float = Field(description="Average real score across all models")
    votingMethod: str = Field(description="Voting method used")
    voting: VotingResult = Field(description="Voting breakdown")
    individualModelResults: List[IndividualModelResult] = Field(
        default_factory=list,
        description="Individual results from each model"
    )


class DetectionOutput(BaseModel):
    """Complete detection output for all images"""
    markdown: str = Field(default="", description="Markdown formatted results")
    totalImages: int = Field(default=0, description="Total number of images processed")
    images: List[ImageDetectionResult] = Field(
        default_factory=list,
        description="Detection results for each image"
    )
    
    def to_json(self, indent: int = 2) -> str:
        """Convert model to JSON string"""
        return self.model_dump_json(indent=indent)
    
    def to_dict(self) -> dict:
        """Convert model to dictionary"""
        return self.model_dump()
