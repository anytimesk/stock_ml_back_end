import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
from typing import Dict, Any, Tuple
from ..abs_model import AbsModel

class LSTMModel(AbsModel):
    """LSTM 기반 시계열 예측 모델"""
    
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.time_steps = params.get("time_steps", 10)
        self.learning_rate = params.get("learning_rate", 0.001)
        self.dropout = params.get("dropout", 0.1)
        self.model = None
        self.scaler = None
        
    def build(self, input_shape: Tuple[int, int]) -> None:
        """LSTM 모델 구축"""
        self.model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout),
            LSTM(units=30, return_sequences=False),
            Dropout(self.dropout),
            Dense(units=1)
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error'
        )
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             epochs: int = 100, batch_size: int = 32,
             validation_split: float = 0.2) -> Dict[str, float]:
        """모델 학습 수행"""
        if self.model is None:
            self.build(input_shape=(X_train.shape[1], X_train.shape[2]))
            
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        return {
            "loss": float(history.history["loss"][-1]),
            "val_loss": float(history.history["val_loss"][-1])
        }
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측 수행"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        return self.model.predict(X, verbose=0)
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """모델 평가"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
            
        loss = self.model.evaluate(X, y, verbose=0)
        return {
            "loss": float(loss),
            "mae": float(loss)  # MSE를 사용하므로 loss와 동일
        }
        
    def predict_next_value(self, sequence: np.ndarray) -> float:
        """다음 값 예측"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
            
        # 입력 데이터 reshape
        X = sequence.reshape(1, self.time_steps, 1)
        
        # 예측 수행
        prediction = self.model.predict(X, verbose=0)
        
        # 스케일링 복원
        if self.scaler is not None:
            prediction = self.scaler.inverse_transform(prediction)
            
        return float(prediction[0][0])
        
    def save(self, path: str) -> None:
        """모델 저장"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        self.model.save(path)
        
    def load(self, path: str) -> None:
        """모델 로드"""
        self.model = load_model(path)
        
    @property
    def model_type(self) -> str:
        """모델 타입 반환"""
        return "lstm" 