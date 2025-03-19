import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
from typing import Dict, Any, Optional
from .base_model import BaseModel

class LSTMModel(BaseModel):
    """LSTM 기반 시계열 예측 모델"""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.time_steps = params.get('time_steps', 3)
        self.units = params.get('units', 50)
        self.dropout = params.get('dropout', 0.2)
        self.scaler = None
        self.last_sequence = None
        
    def build(self):
        """LSTM 모델 구축"""
        model = Sequential()
        model.add(LSTM(self.units, return_sequences=True, 
                       input_shape=(self.time_steps, 1)))
        model.add(Dropout(self.dropout))
        model.add(LSTM(self.units, return_sequences=False))
        model.add(Dropout(self.dropout))
        model.add(Dense(1))
        
        model.compile(
            optimizer='adam', 
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train(self, X, y, **kwargs):
        """모델 학습"""
        if self.model is None:
            self.build()
            
        epochs = kwargs.get('epochs', 50)
        batch_size = kwargs.get('batch_size', 1)
        validation_split = kwargs.get('validation_split', 0.2)
        
        early_stopping = EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """예측 수행"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        return self.model.predict(X)
    
    def evaluate(self, X, y) -> Dict[str, float]:
        """모델 평가"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
            
        loss, mae = self.model.evaluate(X, y)
        return {
            "loss": loss,
            "mae": mae
        }
    
    def save(self, path: str):
        """모델 저장"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        
        # 필요한 경우 scaler와 last_sequence도 저장
        if self.scaler is not None:
            scaler_path = os.path.join(os.path.dirname(path), "scaler.npy")
            np.save(scaler_path, self.scaler.scale_)
            
        if self.last_sequence is not None:
            sequence_path = os.path.join(os.path.dirname(path), "last_sequence.npy")
            np.save(sequence_path, self.last_sequence)
        
    def load(self, path: str):
        """모델 로드"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {path}")
            
        self.model = load_model(path)
        
        # scaler와 last_sequence도 로드
        scaler_path = os.path.join(os.path.dirname(path), "scaler.npy")
        if os.path.exists(scaler_path):
            from sklearn.preprocessing import MinMaxScaler
            scale_ = np.load(scaler_path)
            self.scaler = MinMaxScaler()
            self.scaler.scale_ = scale_
            self.scaler.min_ = 0
            self.scaler.data_min_ = 0
            self.scaler.data_max_ = 1
            self.scaler.data_range_ = 1
            
        sequence_path = os.path.join(os.path.dirname(path), "last_sequence.npy")
        if os.path.exists(sequence_path):
            self.last_sequence = np.load(sequence_path)
        
    @property
    def model_type(self) -> str:
        return "lstm" 