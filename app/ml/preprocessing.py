import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Any

def create_sequences(data: np.ndarray, time_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    시계열 데이터를 LSTM 모델용 시퀀스로 변환
    
    Args:
        data: 1차원 시계열 데이터
        time_steps: 시퀀스 길이
        
    Returns:
        X: 입력 시퀀스 배열 [samples, time_steps, features]
        y: 타겟 배열 [samples]
    """
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps, 0])
        y.append(data[i + time_steps, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y

def prepare_stock_data(df: pd.DataFrame, target_col: str = 'clpr', 
                     time_steps: int = 3, train_ratio: float = 0.8, 
                     scale: bool = True) -> Dict[str, Any]:
    """
    주식 데이터 전처리 및 시퀀스 생성
    
    Args:
        df: 주식 데이터 DataFrame
        target_col: 예측할 열 이름
        time_steps: 시퀀스 길이
        train_ratio: 학습/테스트 분할 비율
        scale: 데이터 스케일링 여부
        
    Returns:
        Dict[str, Any]: 전처리된 데이터 및 메타데이터
    """
    # 결측치 처리
    df = df.dropna(subset=[target_col])
    
    # 스케일링
    scaler = None
    scaled_data = None
    
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[target_col].values.reshape(-1, 1))
    else:
        scaled_data = df[target_col].values.reshape(-1, 1)
    
    # 시퀀스 생성
    X, y = create_sequences(scaled_data, time_steps)
    
    # 학습/테스트 분할
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 마지막 시퀀스 (다음 예측을 위한)
    last_sequence = scaled_data[-time_steps:].reshape(1, time_steps, 1)
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "last_sequence": last_sequence,
        "scaled_data": scaled_data,
        "original_data": df[target_col].values
    } 