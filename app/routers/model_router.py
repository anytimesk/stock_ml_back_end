from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import json
import pandas as pd
import os
from pathlib import Path
from app.ml.models.lstm_model import LSTMModel
from app.ml.preprocessing import prepare_stock_data
from app.ml.training import ModelTrainer
from pydantic import BaseModel

# 응답 모델 정의
class PredictionResponse(BaseModel):
    prediction: float
    current_price: float
    change_percent: float

class TrainingResponse(BaseModel):
    success: bool
    metrics: Dict[str, Any]

# 라우터 생성
router = APIRouter(prefix="/api/model", tags=["model"])

# 데이터 로드 함수
def load_stock_data() -> pd.DataFrame:
    """
    Samsung 주식 데이터를 JSON 파일에서 로드하여 DataFrame으로 변환
    """
    data_path = Path(__file__).parent.parent / "data" / "samsung_stock_data.json"
    
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # JSON 구조에서 실제 아이템 추출
    items = data["response"]["body"]["items"]["item"]
    
    # DataFrame으로 변환
    df = pd.DataFrame(items)
    
    # 필요한 데이터 전처리
    df["clpr"] = df["clpr"].astype(float)  # 종가 숫자형으로 변환
    df["basDt"] = pd.to_datetime(df["basDt"], format="%Y%m%d")  # 날짜 형식 변환
    
    # 날짜 기준으로 정렬
    df = df.sort_values("basDt")
    
    return df

# 모델 학습 엔드포인트
@router.post("/train", response_model=TrainingResponse)
async def train_model():
    """
    Samsung 주식 데이터를 사용하여 LSTM 모델 학습
    """
    try:
        # 데이터 로드
        df = load_stock_data()
        
        # LSTM 모델 초기화
        model = LSTMModel(params={
            "time_steps": 5,
            "units": 50,
            "dropout": 0.2
        })
        
        # 모델 학습기 초기화
        trainer = ModelTrainer(model)
        
        # 모델 학습
        metrics = trainer.train_model(
            df=df,
            target_col="clpr",
            epochs=50,
            batch_size=32
        )
        
        return TrainingResponse(success=True, metrics=metrics)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 주가 예측 엔드포인트
@router.get("/predict", response_model=PredictionResponse)
async def predict_stock():
    """
    학습된 모델을 사용하여 다음 날 Samsung 주가 예측
    """
    try:
        # 데이터 로드
        df = load_stock_data()
        
        # 현재 주가 (가장 최근 데이터)
        current_price = df.iloc[-1]["clpr"]
        
        # LSTM 모델 초기화
        model = LSTMModel()
        
        # 모델 학습기 초기화
        trainer = ModelTrainer(model)
        
        # 모델 로드 시도
        model_loaded = trainer.load_model("lstm_model")
        
        # 모델이 없으면 학습
        if not model_loaded:
            trainer.train_model(
                df=df,
                target_col="clpr"
            )
        
        # 다음 날 예측
        prediction_result = trainer.predict_next()
        prediction = prediction_result["prediction"]
        
        # 변화율 계산
        change_percent = ((prediction - current_price) / current_price) * 100
        
        return PredictionResponse(
            prediction=prediction,
            current_price=current_price,
            change_percent=change_percent
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 데이터 조회 엔드포인트
@router.get("/data")
async def get_stock_data():
    """
    Samsung 주식 데이터 전체 반환
    """
    try:
        df = load_stock_data()
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

