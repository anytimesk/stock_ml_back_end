from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime
from pathlib import Path
from app.ml.models.abs_model import AbsModel
from app.ml.preprocessing import prepare_stock_data

class ModelTrainer:
    """모델 학습 및 관리"""
    
    def __init__(self, model: AbsModel, model_dir: Optional[str] = None):
        self.model = model
        self.model_dir = Path(model_dir or Path(__file__).parent.parent / "models")
        os.makedirs(self.model_dir, exist_ok=True)
        self.metrics = {}
        
    def train_model(self, df: pd.DataFrame, target_col: str = 'clpr', 
                    time_steps: Optional[int] = None, isin_code: str = None, **kwargs) -> Dict[str, Any]:
        """
        데이터프레임을 사용해 모델 학습
        
        Args:
            df: 학습 데이터 DataFrame
            target_col: 예측할 열 이름
            time_steps: 시퀀스 길이 (None인 경우 모델 기본값 사용)
            isin_code: 종목 코드
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
        model_path = self.save_model(
            target_col=target_col,
            time_steps=time_steps,
            scaler=data['scaler'],
            isin_code=isin_code,
            **kwargs
        )
        
        return self.metrics
    
    def save_model(self, target_col: str, time_steps: int, scaler: Any, 
                  isin_code: str = None, model_name: Optional[str] = None, **kwargs) -> str:
        """
        학습된 모델과 관련 정보를 저장
        
        Args:
            target_col: 예측 대상 컬럼명
            time_steps: 시퀀스 길이
            scaler: 데이터 스케일러
            isin_code: 종목 코드
            model_name: 모델 이름 (None인 경우 자동 생성)
            **kwargs: 추가 학습 파라미터
            
        Returns:
            str: 저장된 모델 디렉토리 경로
        """
        if model_name is None:
            # YYYYMMDD 형식의 날짜만 사용
            timestamp = datetime.now().strftime("%Y%m%d")
            if isin_code:
                model_name = f"{isin_code}_{self.model.model_type}_{timestamp}"
            else:
                model_name = f"{self.model.model_type}_{timestamp}"
            
        # 모델 디렉토리 생성
        model_dir = self.model_dir / model_name
        os.makedirs(model_dir, exist_ok=True)
        
        # 1. 모델 저장
        model_path = model_dir / "model.keras"
        self.model.save(str(model_path))
        
        # 2. 스케일러 저장
        scaler_path = model_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # 3. 파라미터 저장
        params = {
            "target_col": target_col,
            "time_steps": time_steps,
            "model_type": self.model.model_type,
            "model_params": {
                "units": getattr(self.model, 'units', None),
                "dropout": getattr(self.model, 'dropout', None)
            },
            "training_params": kwargs
        }
        
        params_path = model_dir / "params.json"
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=4)
        
        # 4. 메트릭스 저장
        metrics_path = model_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
            
        return str(model_dir)
    
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