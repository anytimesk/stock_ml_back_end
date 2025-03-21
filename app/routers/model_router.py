from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, List
from app.data.clients.open_api_client import OpenApiClient
from app.utils import CSVHandler
from pydantic import BaseModel
from app.core.config import API_BASE_URL, API_KEY
import pandas as pd
from app.ml.models.abs_model import AbsModel
from app.ml.training import ModelTrainer
from enum import Enum
import os
from pathlib import Path
from datetime import datetime

# 응답 모델 정의
class CSVSaveResponse(BaseModel):
    success: bool
    message: str
    file_path: Optional[str] = None
    records_count: Optional[int] = None

# CSV 파일 정보 모델
class CSVFileInfo(BaseModel):
    filename: str
    path: str
    size_bytes: int
    created_at: str
    isin_code: str
    stock_name: str

# CSV 파일 목록 응답 모델
class CSVListResponse(BaseModel):
    success: bool
    message: str
    files: List[CSVFileInfo]
    count: int

# 모델 유형 Enum
class ModelType(str, Enum):
    lstm = "lstm"
    rnn = "rnn"
    # 추후 다른 모델 유형 추가 가능

# 모델 학습 응답 모델
class TrainModelResponse(BaseModel):
    success: bool
    message: str
    metrics: Optional[dict] = None
    model_path: Optional[str] = None

# 라우터 정의
router = APIRouter(prefix="/ml", tags=["ml"])

# OpenAPI 클라이언트 의존성
def get_open_api_client():
    return OpenApiClient(API_KEY, API_BASE_URL)

@router.get("/getStockData", response_model=CSVListResponse)
async def get_stock_data_files(
    itmsNm: Optional[str] = Query(None, description="종목명으로 필터링 (선택사항)")
):
    """
    storage/csv 디렉토리에 저장된 주식 데이터 CSV 파일 목록을 반환합니다.
    선택적으로 종목명으로 필터링할 수 있습니다.
    """
    try:
        # CSVHandler를 사용하여 파일 목록 조회
        files = CSVHandler.get_csv_files(stock_name=itmsNm)
        
        return CSVListResponse(
            success=True,
            message=f"총 {len(files)}개의 CSV 파일이 있습니다.",
            files=files,
            count=len(files)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV 파일 목록 조회 중 오류 발생: {str(e)}")

@router.get("/saveStockDataCSV", response_model=CSVSaveResponse)
async def save_stock_data_csv(
    itmsNm: str = Query(..., description="종목명"),
    client: OpenApiClient = Depends(get_open_api_client)
):
    """
    종목명으로 주식 가격 정보를 조회하고 CSV 파일로 저장합니다.
    파일명은 "isinCd값_itmsNm_생성일.csv" 형식으로 저장됩니다.
    """
    try:
        # 주식 데이터 조회
        result = await client.fetch_stock_data(
            itmsNm=itmsNm,
            pageNo=1,
            numOfRows=100
        )
        
        # 데이터 구조 확인 및 검증
        if "response" not in result or "body" not in result["response"]:
            return CSVSaveResponse(
                success=False,
                message="API 응답 형식이 올바르지 않습니다."
            )
        
        if "items" not in result["response"]["body"] or "item" not in result["response"]["body"]["items"]:
            return CSVSaveResponse(
                success=False,
                message="API 응답에 주식 데이터가 포함되어 있지 않습니다."
            )
        
        items = result["response"]["body"]["items"]["item"]
        
        if not items:
            return CSVSaveResponse(
                success=False,
                message="검색 결과가 없습니다."
            )
        
        # isinCd 값 추출 (첫 번째 항목의 isinCd 사용)
        isin_code = str(items[0].get("isinCd", "unknown"))
        
        # CSVHandler를 사용하여 CSV 파일 저장
        file_path = CSVHandler.save_to_csv(
            data=items,
            isin_code=isin_code,
            stock_name=itmsNm
        )
        
        return CSVSaveResponse(
            success=True,
            message=f"{itmsNm} 종목 데이터를 성공적으로 CSV 파일로 저장했습니다.",
            file_path=file_path,
            records_count=len(items)
        )
        
    except Exception as e:
        return CSVSaveResponse(
            success=False,
            message=f"CSV 저장 중 오류 발생: {str(e)}"
        )

@router.post("/trainModel", response_model=TrainModelResponse)
async def train_model(
    isin_code: str = Query(..., description="종목 코드"),
    model_type: ModelType = Query(..., description="모델 유형 (현재는 LSTM, RNN 지원)"),
    time_steps: int = Query(3, description="시계열 데이터의 시퀀스 길이"),
    epochs: int = Query(50, description="학습 에포크 수"),
    batch_size: int = Query(32, description="배치 크기"),
    validation_split: float = Query(0.2, description="검증 데이터 비율")
):
    """
    주식 데이터를 사용하여 머신러닝 모델을 학습합니다.
    현재는 LSTM, RNN 모델을 지원하며, 추후 다른 모델들도 추가될 수 있습니다.
    """

    try:
        # CSV 파일 목록 조회
        files = CSVHandler.get_csv_files()
        target_files = [f for f in files if f["isin_code"] == isin_code]
        
        if not target_files:
            return TrainModelResponse(
                success=False,
                message=f"종목 코드 {isin_code}에 해당하는 CSV 파일을 찾을 수 없습니다."
            )
            
        # 가장 최근 파일 사용
        latest_file = sorted(target_files, key=lambda x: x["created_at"], reverse=True)[0]
        
        # CSV 파일 읽기
        df = pd.read_csv(latest_file["path"])
        
        # 모델 파라미터 설정
        model_params = {
            "time_steps": time_steps,
            "units": 50,  # LSTM 유닛 수
            "dropout": 0.2  # 드롭아웃 비율
        }
        
        # 모델 초기화
        try:
            model = AbsModel.create(model_type.value, model_params)
        except ValueError as e:
            return TrainModelResponse(
                success=False,
                message=str(e)
            )
        
        # 모델 트레이너 초기화
        model_dir = Path("storage/models")
        trainer = ModelTrainer(model, str(model_dir))
        
        # 모델 학습
        metrics = trainer.train_model(
            df=df,
            target_col="CLPR",  # 종가 기준
            time_steps=time_steps,
            isin_code=isin_code,  # isin_code 전달
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )
        
        # 모델 저장 경로 생성
        timestamp = datetime.now().strftime("%Y%m%d")
        model_name = f"{isin_code}_{model_type}_{timestamp}"
        model_path = str(model_dir / model_name)
        
        return TrainModelResponse(
            success=True,
            message=f"{isin_code} 종목의 {model_type} 모델 학습이 완료되었습니다.",
            metrics=metrics,
            model_path=model_path
        )
    
    except Exception as e:
        print(f"\n=== Train Model Error ===")
        print(f"Error Type: {type(e)}")
        print(f"Error Message: {str(e)}")
        print("=======================\n")
        
        return TrainModelResponse(
            success=False,
            message=f"모델 학습 중 오류 발생: {str(e)}"
        )
