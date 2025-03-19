import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from pathlib import Path

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    예측 성능 메트릭 계산
    
    Args:
        y_true: 실제 값
        y_pred: 예측 값
        
    Returns:
        Dict[str, float]: 계산된 메트릭
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape
    }
    
def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                    title: str = "실제 vs 예측", 
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    예측 결과 시각화
    
    Args:
        y_true: 실제 값
        y_pred: 예측 값
        title: 그래프 제목
        save_path: 저장 경로 (None인 경우 저장하지 않음)
        
    Returns:
        plt.Figure: 그래프 객체
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='실제 값')
    plt.plot(y_pred, label='예측 값', linestyle='--')
    plt.title(title)
    plt.xlabel('시간')
    plt.ylabel('값')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        plt.savefig(save_path)
    
    return plt.gcf()
    
def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                        title: str = "모델 평가", 
                        save_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    예측 결과 평가 및 시각화
    
    Args:
        y_true: 실제 값
        y_pred: 예측 값
        title: 평가 제목
        save_dir: 결과 저장 디렉토리 (None인 경우 저장하지 않음)
        
    Returns:
        Dict[str, Any]: 평가 결과 (메트릭 및 그래프 경로)
    """
    # 메트릭 계산
    metrics = calculate_metrics(y_true, y_pred)
    
    # 결과 저장
    result = {
        "title": title,
        "metrics": metrics,
        "plot_path": None
    }
    
    # 시각화 및 저장
    if save_dir:
        save_dir = Path(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        
        # 메트릭 저장
        metrics_path = save_dir / f"{title.lower().replace(' ', '_')}_metrics.csv"
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        
        # 그래프 저장
        plot_path = save_dir / f"{title.lower().replace(' ', '_')}_plot.png"
        plot_predictions(y_true, y_pred, title, str(plot_path))
        result["plot_path"] = str(plot_path)
    else:
        # 그래프만 생성 (저장 X)
        plot_predictions(y_true, y_pred, title)
    
    return result 