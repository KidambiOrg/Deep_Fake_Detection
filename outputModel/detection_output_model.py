"""
Pydantic models for deepfake detection output
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ExifData(BaseModel):
    """EXIF metadata from image"""
    make: Optional[str] = Field(default=None, description="Camera/device manufacturer")
    model: Optional[str] = Field(default=None, description="Camera/device model")
    software: Optional[str] = Field(default=None, description="Software used to process image")
    dateTime: Optional[str] = Field(default=None, description="Date and time image was taken")
    dateTimeOriginal: Optional[str] = Field(default=None, description="Original date and time")
    dateTimeDigitized: Optional[str] = Field(default=None, description="Date and time digitized")
    imageWidth: Optional[int] = Field(default=None, description="Image width in pixels")
    imageHeight: Optional[int] = Field(default=None, description="Image height in pixels")
    orientation: Optional[int] = Field(default=None, description="Image orientation")
    xResolution: Optional[float] = Field(default=None, description="X resolution (DPI)")
    yResolution: Optional[float] = Field(default=None, description="Y resolution (DPI)")
    resolutionUnit: Optional[int] = Field(default=None, description="Resolution unit")
    colorSpace: Optional[int] = Field(default=None, description="Color space")
    exifVersion: Optional[str] = Field(default=None, description="EXIF version")
    flashPixVersion: Optional[str] = Field(default=None, description="FlashPix version")
    gpsInfo: Optional[Dict[str, Any]] = Field(default=None, description="GPS information")
    additionalData: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional EXIF data")
    hasExif: bool = Field(default=False, description="Whether EXIF data is present")


class C2paManifestSummary(BaseModel):
    """Summary of C2PA manifest information"""
    claimGenerator: Optional[str] = Field(default=None, description="Claim generator identifier")
    title: Optional[str] = Field(default=None, description="Manifest title")
    format: Optional[str] = Field(default=None, description="Content format")
    instanceId: Optional[str] = Field(default=None, description="Unique instance identifier")
    claimGeneratorInfo: Optional[Dict[str, Any]] = Field(default=None, description="Claim generator details")
    assertions: List[Dict[str, Any]] = Field(default_factory=list, description="List of assertions")
    ingredients: List[Dict[str, Any]] = Field(default_factory=list, description="List of ingredients")
    signature: Optional[Dict[str, Any]] = Field(default=None, description="Signature information")


class C2paData(BaseModel):
    """C2PA (Content Credentials) metadata"""
    hasC2pa: bool = Field(default=False, description="Whether C2PA data is present")
    validationState: Optional[str] = Field(default=None, description="Validation state (e.g., Valid, Invalid)")
    activeManifest: Optional[C2paManifestSummary] = Field(default=None, description="Active manifest summary")
    manifests: Optional[List[str]] = Field(default=None, description="List of manifest identifiers")
    validationResults: Optional[Dict[str, Any]] = Field(default=None, description="Detailed validation results")
    isEmbedded: Optional[bool] = Field(default=None, description="Whether manifest is embedded in the asset")
    remoteUrl: Optional[str] = Field(default=None, description="Remote URL if manifest is stored remotely")
    error: Optional[str] = Field(default=None, description="Error message if extraction failed")


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
    exifData: Optional[ExifData] = Field(default=None, description="EXIF metadata from image")
    c2paData: Optional[C2paData] = Field(default=None, description="C2PA (Content Credentials) metadata")
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
