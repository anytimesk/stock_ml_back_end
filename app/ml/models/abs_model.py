from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class AbsModel(ABC):
    """모든 머신러닝 모델의 추상 기본 클래스"""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self.model = None
    
    @classmethod
    def create(cls, model_type: str, params: Dict[str, Any]) -> 'AbsModel':
        """
        모델 타입에 따른 모델 인스턴스 생성
        
        Args:
            model_type: 모델 유형 ("lstm", "rnn" 등)
            params: 모델 파라미터
            
        Returns:
            AbsModel: 생성된 모델 인스턴스
            
        Raises:
            ValueError: 지원하지 않는 모델 유형일 경우
        """
        # 동적 임포트로 순환 참조 방지
        from .lstm_model import LSTMModel
        from .rnn_model import RNNModel
        
        if model_type == "lstm":
            return LSTMModel(params)
        elif model_type == "rnn":
            return RNNModel(params)
        else:
            raise ValueError(f"지원하지 않는 모델 유형입니다: {model_type}")
        
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