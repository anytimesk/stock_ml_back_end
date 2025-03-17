from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.stock_prediction_service import StockPredictionService
from app.utils.data_loader import get_stock_dataframe
import json

router = APIRouter(prefix="/api/stock", tags=["stock"])
stock_service = StockPredictionService()

class PredictionResponse(BaseModel):
    prediction: float
    current_price: float
    change_percent: float

@router.get("/data")
def get_stock_data():
    """
    주식 데이터 전체 반환
    """
    df = get_stock_dataframe()
    return df.to_dict(orient="records")

@router.post("/train")
def train_model():
    """
    모델 학습
    """
    try:
        result = stock_service.train_model()
        return {"success": True, "metrics": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predict", response_model=PredictionResponse)
def predict_stock():
    """
    다음 날 주가 예측
    """
    try:
        prediction = stock_service.predict_next_day()
        
        # 현재 가격 (가장 최근 데이터)
        df = get_stock_dataframe()
        current_price = df.iloc[-1]['clpr']
        
        # 변화율 계산
        change_percent = ((prediction - current_price) / current_price) * 100
        
        return PredictionResponse(
            prediction=prediction,
            current_price=current_price,
            change_percent=change_percent
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))