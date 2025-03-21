"""
모델 관련 스키마
"""
from typing import Optional, List
from pydantic import BaseModel
from enum import Enum

class ModelType(str, Enum):
    """모델 유형"""
    LSTM = "lstm"
    RNN = "rnn"
    # 추후 다른 모델 유형 추가 가능

class TrainModelMetrics(BaseModel):
    """모델 학습 메트릭"""
    train_loss: float
    val_loss: Optional[float] = None
    test_loss: Optional[float] = None
    test_mae: Optional[float] = None
    epochs_trained: int
    timestamp: str

class TrainModelResponse(BaseModel):
    """모델 학습 응답"""
    success: bool
    message: str
    metrics: Optional[TrainModelMetrics] = None
    model_path: Optional[str] = None

class PredictionResponse(BaseModel):
    """예측 응답"""
    success: bool
    message: str
    prediction: Optional[float] = None
    date: Optional[str] = None
    prediction_array: Optional[List[float]] = None 