"""
머신러닝 모듈
"""
from app.ml.model_service import ModelService
from app.ml.models import LSTMModel, RNNModel

# 모델 등록
ModelService.register("lstm", LSTMModel)
ModelService.register("rnn", RNNModel)

__all__ = ['ModelService'] 