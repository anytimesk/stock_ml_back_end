from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path
from app.ml.models.base_model import BaseModel
from app.ml.preprocessing import prepare_stock_data

class ModelTrainer:
    """모델 학습 및 관리"""
    
    def __init__(self, model: BaseModel, model_dir: Optional[str] = None):
        self.model = model
        self.model_dir = Path(model_dir or Path(__file__).parent.parent / "models")
        os.makedirs(self.model_dir, exist_ok=True)
        self.metrics = {}
        
    def train_model(self, df: pd.DataFrame, target_col: str = 'clpr', 
                    time_steps: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        데이터프레임을 사용해 모델 학습
        
        Args:
            df: 학습 데이터 DataFrame
            target_col: 예측할 열 이름
            time_steps: 시퀀스 길이 (None인 경우 모델 기본값 사용)
            **kwargs: 추가 학습 매개변수
            
        Returns:
            Dict[str, Any]: 학습 결과 및 메트릭
        """
        # 시퀀스 길이 설정
        if time_steps is not None:
            self.model.time_steps = time_steps
            
        # 데이터 준비
        data = prepare_stock_data(
            df, 
            target_col=target_col, 
            time_steps=self.model.time_steps
        )
        
        # 모델 속성 설정
        self.model.scaler = data['scaler']
        self.model.last_sequence = data['last_sequence']
        
        # 모델 학습
        history = self.model.train(
            data['X_train'], 
            data['y_train'],
            **kwargs
        )
        
        # 모델 평가
        eval_metrics = self.model.evaluate(data['X_test'], data['y_test'])
        
        # 결과 저장
        self.metrics = {
            "train_loss": history.history['loss'][-1],
            "val_loss": history.history['val_loss'][-1] if 'val_loss' in history.history else None,
            "test_loss": eval_metrics.get('loss'),
            "test_mae": eval_metrics.get('mae'),
            "epochs_trained": len(history.history['loss']),
            "timestamp": datetime.now().isoformat()
        }
        
        # 모델 저장
        self.save_model()
        
        return self.metrics
    
    def save_model(self, model_name: Optional[str] = None) -> str:
        """
        학습된 모델 저장
        
        Args:
            model_name: 모델 이름 (None인 경우 자동 생성)
            
        Returns:
            str: 저장된 모델 경로
        """
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{self.model.model_type}_{timestamp}"
            
        model_path = self.model_dir / model_name
        self.model.save(str(model_path))
        
        # 메트릭 저장
        metrics_path = self.model_dir / f"{model_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f)
            
        return str(model_path)
    
    def load_model(self, model_name: str) -> bool:
        """
        저장된 모델 로드
        
        Args:
            model_name: 모델 이름
            
        Returns:
            bool: 로드 성공 여부
        """
        model_path = self.model_dir / model_name
        
        try:
            self.model.load(str(model_path))
            
            # 메트릭 로드
            metrics_path = self.model_dir / f"{model_name}_metrics.json"
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.metrics = json.load(f)
                    
            return True
        except Exception as e:
            print(f"모델 로드 오류: {e}")
            return False
            
    def predict_next(self, input_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        다음 값 예측
        
        Args:
            input_data: 입력 데이터 (None인 경우 저장된 last_sequence 사용)
            
        Returns:
            Dict[str, Any]: 예측 결과
        """
        if input_data is None:
            if self.model.last_sequence is None:
                raise ValueError("예측을 위한 입력 데이터가 제공되지 않았습니다.")
            input_data = self.model.last_sequence
            
        # 예측
        scaled_prediction = self.model.predict(input_data)
        
        # 역변환
        if self.model.scaler:
            prediction = self.model.scaler.inverse_transform(scaled_prediction)[0, 0]
        else:
            prediction = scaled_prediction[0, 0]
            
        return {
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        } 