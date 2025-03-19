from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseModel(ABC):
    """모든 머신러닝 모델의 기본 클래스"""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self.model = None
        
    @abstractmethod
    def build(self):
        """모델 구축"""
        pass
    
    @abstractmethod
    def train(self, X, y, **kwargs):
        """모델 학습"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """예측 수행"""
        pass
    
    @abstractmethod
    def evaluate(self, X, y) -> Dict[str, float]:
        """모델 평가 메트릭 계산"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """모델 저장"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """모델 로드"""
        pass
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """모델 유형 식별자"""
        pass 