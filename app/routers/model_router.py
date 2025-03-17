from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.model_service import ModelService

router = APIRouter(prefix="/api/model", tags=["model"])
model_service = ModelService()

class PredictionRequest(BaseModel):
    features: list

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float

@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        prediction = model_service.predict(request.features)
        return PredictionResponse(prediction=prediction, confidence=0.95)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

