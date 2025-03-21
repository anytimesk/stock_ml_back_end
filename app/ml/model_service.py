"""
모델 생성 및 관리를 위한 서비스
"""
from typing import Dict, Any, Type
from .abs_model import AbsModel

class ModelService:
    """모델 생성 및 관리를 위한 서비스 클래스
    
    현재 기능:
    - 모델 클래스 등록
    - 모델 인스턴스 생성
    
    향후 확장 가능한 기능:
    - 모델 버전 관리
    - 모델 메타데이터 관리
    - 모델 성능 모니터링
    - 모델 라이프사이클 관리
    """
    
    _models = {}  # 등록된 모델 클래스를 저장할 딕셔너리
    
    @classmethod
    def register(cls, model_type: str, model_class: Type[AbsModel]):
        """새로운 모델 클래스 등록
        
        Args:
            model_type: 모델 타입 식별자 (예: "lstm", "rnn")
            model_class: 등록할 모델 클래스
        """
        cls._models[model_type] = model_class
    
    @classmethod
    def create(cls, model_type: str, params: Dict[str, Any]) -> AbsModel:
        """모델 인스턴스 생성
        
        Args:
            model_type: 생성할 모델의 타입
            params: 모델 생성에 필요한 파라미터
            
        Returns:
            생성된 모델 인스턴스
            
        Raises:
            ValueError: 지원하지 않는 모델 타입인 경우
        """
        if model_type not in cls._models:
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
        
        model_class = cls._models[model_type]
        return model_class(params) 