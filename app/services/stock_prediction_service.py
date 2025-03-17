import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from app.utils.data_loader import get_stock_dataframe
import os
import pickle
from pathlib import Path

class StockPredictionService:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.time_steps = 3  # 기본 시퀀스 길이
        self.model_dir = Path(__file__).parent.parent / "models"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def prepare_data(self, df, target_col='clpr', train_ratio=0.8):
        """
        예측을 위한 데이터 준비
        """
        # 스케일링
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(df[target_col].values.reshape(-1, 1))
        
        # 시퀀스 데이터 생성
        X, y = [], []
        for i in range(len(scaled_data) - self.time_steps):
            features = []
            for step in range(self.time_steps):
                features.append(scaled_data[i + step, 0])
            X.append(features)
            y.append(scaled_data[i + self.time_steps, 0])
        
        X, y = np.array(X), np.array(y)
        
        # 학습/테스트 분할
        train_size = int(len(X) * train_ratio)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, X_test, y_train, y_test, scaled_data
    
    def train_model(self):
        """
        선형 회귀 모델 학습
        """
        # 데이터 가져오기
        df = get_stock_dataframe()
        
        # 데이터 준비
        X_train, X_test, y_train, y_test, scaled_data = self.prepare_data(df)
        
        # 모델 정의 및 학습
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        
        # 성능 평가
        train_predictions = self.model.predict(X_train)
        test_predictions = self.model.predict(X_test)
        
        # 스케일 역변환
        train_predictions = self.scaler.inverse_transform(train_predictions.reshape(-1, 1))
        test_predictions = self.scaler.inverse_transform(test_predictions.reshape(-1, 1))
        
        # 마지막 시퀀스 저장
        self.last_sequence = scaled_data[-self.time_steps:].reshape(1, -1)
        
        # 모델 저장
        self.save_model()
        
        return {
            "train_score": self.model.score(X_train, y_train),
            "test_score": self.model.score(X_test, y_test)
        }
    
    def predict_next_day(self):
        """
        다음 날 주가 예측
        """
        if self.model is None:
            self.load_model()
            if self.model is None:
                self.train_model()
        
        # 예측
        scaled_prediction = self.model.predict(self.last_sequence)
        prediction = self.scaler.inverse_transform(scaled_prediction.reshape(-1, 1))[0, 0]
        
        return prediction
    
    def save_model(self):
        """
        모델 저장
        """
        model_path = self.model_dir / "stock_prediction_model.pkl"
        scaler_path = self.model_dir / "stock_scaler.pkl"
        sequence_path = self.model_dir / "last_sequence.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(sequence_path, 'wb') as f:
            pickle.dump(self.last_sequence, f)
    
    def load_model(self):
        """
        저장된 모델 로드
        """
        model_path = self.model_dir / "stock_prediction_model.pkl"
        scaler_path = self.model_dir / "stock_scaler.pkl"
        sequence_path = self.model_dir / "last_sequence.pkl"
        
        if not model_path.exists():
            return False
            
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(sequence_path, 'rb') as f:
            self.last_sequence = pickle.load(f)
            
        return True