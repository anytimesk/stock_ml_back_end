from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

class AbsModel(ABC):
    """추상 모델 클래스"""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self.model = None
        self.scaler = None
        self.last_sequence = None
        self.metrics = {}
        self.model_dir = Path("storage/models")
        self.time_steps = self.params.get("time_steps", 10)
    
    @abstractmethod
    def build(self, input_shape: Tuple[int, int]) -> None:
        """모델 구축"""
        pass
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             epochs: int = 100, batch_size: int = 32,
             validation_split: float = 0.2) -> Dict[str, float]:
        """모델 학습"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측 수행"""
        pass
    
    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """모델 평가"""
        pass
        
    @abstractmethod
    def predict_next_value(self, sequence: np.ndarray) -> float:
        """다음 값 예측"""
        pass
    
    def train_from_dataframe(self, 
                           df: pd.DataFrame,
                           target_col: str,
                           isin_code: str,
                           **kwargs) -> Dict[str, Any]:
        """DataFrame으로부터 모델 학습"""
        print(f"\n=== 데이터 전처리 시작 ===")
        print(f"데이터 크기: {len(df)}")
        print(f"목표 변수: {target_col}")
        
        # 데이터 전처리
        train_data = self._prepare_data(df, target_col)
        
        print(f"\n=== 모델 학습 시작 ===")
        print(f"학습 데이터 크기: {train_data['X_train'].shape}")
        print(f"테스트 데이터 크기: {train_data['X_test'].shape}")
        
        # 모델 학습
        train_metrics = self.train(
            train_data['X_train'],
            train_data['y_train'],
            **kwargs
        )
        
        print(f"\n=== 모델 평가 시작 ===")
        # 모델 평가
        eval_metrics = self.evaluate(
            train_data['X_test'],
            train_data['y_test']
        )
        
        # 메트릭 저장
        self.metrics = {
            "train_loss": float(train_metrics.get("loss", 0.0)),
            "val_loss": float(train_metrics.get("val_loss", 0.0)),
            "test_loss": float(eval_metrics.get("loss", 0.0)),
            "test_mae": float(eval_metrics.get("mae", 0.0)),
            "epochs_trained": int(kwargs.get("epochs", 100)),
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"\n=== 모델 저장 시작 ===")
        # 모델 저장
        model_path = self._save_with_metadata(isin_code, target_col, **kwargs)
        
        print(f"\n=== 학습 완료 ===")
        print(f"모델 저장 경로: {model_path}")
        print(f"학습 메트릭스: {self.metrics}")
        
        return {
            "metrics": self.metrics,
            "model_path": model_path
        }
    
    def _prepare_data(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """데이터 전처리"""
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.model_selection import train_test_split
        
        # 결측치 처리
        df = df.dropna(subset=[target_col])
        
        # 스케일링
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(df[target_col].values.reshape(-1, 1))
        
        # 시퀀스 생성
        X, y = self._create_sequences(scaled_data)
        
        # 학습/테스트 분할
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # 마지막 시퀀스 저장 (다음 예측을 위해)
        self.last_sequence = scaled_data[-self.time_steps:]
        
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "scaled_data": scaled_data
        }
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """시계열 시퀀스 생성"""
        X, y = [], []
        for i in range(len(data) - self.time_steps):
            X.append(data[i:i + self.time_steps])
            y.append(data[i + self.time_steps])
        
        return np.array(X), np.array(y)
    
    def _save_with_metadata(self, isin_code: str, target_col: str, **kwargs) -> str:
        """모델과 메타데이터 저장"""
        # 모델 저장 경로 생성
        model_name = f"{isin_code}_{self.model_type}"
        model_dir = self.model_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 모델 저장
        model_path = model_dir / "model.keras"
        self.save(str(model_path))
        
        # 2. 파라미터 저장
        params = {
            "model_type": self.model_type,
            "target_col": target_col,
            "time_steps": self.time_steps,
            "learning_rate": self.params.get("learning_rate", 0.001),
            "dropout": self.params.get("dropout", 0.1),
            "training_params": {
                "epochs": kwargs.get("epochs", 100),
                "batch_size": kwargs.get("batch_size", 32),
                "validation_split": kwargs.get("validation_split", 0.2)
            }
        }
        
        params_path = model_dir / "params.json"
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=4, ensure_ascii=False)
            
        # 3. 메트릭 저장
        metrics_path = model_dir / "metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=4, ensure_ascii=False)
        
        # 4. 스케일러 저장
        if self.scaler is not None:
            scaler_path = model_dir / "scaler.pkl"
            import joblib
            joblib.dump(self.scaler, scaler_path)
        
        # 5. 마지막 시퀀스 저장
        if self.last_sequence is not None:
            sequence_path = model_dir / "last_sequence.npy"
            np.save(sequence_path, self.last_sequence)
        
        print(f"\n=== 모델 저장 완료 ===")
        print(f"저장 경로: {model_dir}")
        print(f"저장된 파일:")
        print(f"- model.keras: 모델 가중치")
        print(f"- params.json: 모델 파라미터")
        print(f"- metrics.json: 학습 메트릭")
        print(f"- scaler.pkl: 스케일러")
        print(f"- last_sequence.npy: 마지막 시퀀스")
        
        return str(model_dir)
    
    @abstractmethod
    def save(self, path: str) -> None:
        """모델 저장"""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """모델 로드"""
        pass
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """모델 타입 반환"""
        pass 