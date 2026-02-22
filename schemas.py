from pydantic import BaseModel, Field
from typing import Optional

class HousePredictionRequest(BaseModel):
    """Schema for house prediction request - validates all 13 features"""

    # Property listing features
    total_images: int = Field(..., ge=0, le=50, description="Number of property images (0-50)")

    # Basic property features
    beds: int = Field(..., ge=0, le=10, description="Number of bedrooms (0-10)")
    baths: float = Field(..., ge=0, le=10, description="Number of bathrooms (0-10, can be decimal like 2.5)")
    area: float = Field(..., gt=0, le=10000, description="Property area in square feet (must be positive, max 10000)")

    # Location features
    latitude: float = Field(..., ge=-90, le=90, description="Property latitude (-90 to 90)")
    longitude: float = Field(..., ge=-180, le=180, description="Property longitude (-180 to 180)")

    # Binary features (0 or 1)
    garden: int = Field(..., ge=0, le=1, description="Has garden: 0=No, 1=Yes")
    garage: int = Field(..., ge=0, le=1, description="Has garage: 0=No, 1=Yes")
    new_construction: int = Field(..., ge=0, le=1, description="Is new construction: 0=No, 1=Yes")
    pool: int = Field(..., ge=0, le=1, description="Has pool: 0=No, 1=Yes")
    terrace: int = Field(..., ge=0, le=1, description="Has terrace: 0=No, 1=Yes")
    air_conditioning: int = Field(..., ge=0, le=1, description="Has AC: 0=No, 1=Yes")
    parking: int = Field(..., ge=0, le=1, description="Has parking: 0=No, 1=Yes")

    class Config:
        json_schema_extra = {
            "example": {
                "total_images": 10,
                "beds": 3,
                "baths": 2.5,
                "area": 1800.0,
                "latitude": 40.7128,
                "longitude": -74.0060,
                "garden": 1,
                "garage": 1,
                "new_construction": 0,
                "pool": 0,
                "terrace": 1,
                "air_conditioning": 1,
                "parking": 1
            }
        }


class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    predicted_price: float = Field(..., description="The predicted house price")
    currency: str = Field("USD", description="Currency of the predicted price")
    model_version: str = Field(..., description="Version of the model used for prediction")
    


class ModelInfoResponse(BaseModel):
    """Schema for model information response"""
    model_type: str = Field(..., description="Type of model used (RandonForestRegressor)")
    version: str = Field(..., description="Model version")
    features: list[str] = Field(..., description="List of model input features in the correct order")
    training_date: str = Field(..., description="Date when the model was trained (YYYY-MM-DD)")
    rmse:float = Field(..., description="Root Mean Squared Error of the model")
    description: str = Field(..., description="Random forest regressor")
   


class HealthCheckResponse(BaseModel):
    """Schema for health check response"""
    status: str = Field(..., description='Health status: "healthy" or "unhealthy"')
    model_loaded: bool = Field(..., description="True if the model is loaded successfully")
    message: str = Field(..., description="Model health check message")
