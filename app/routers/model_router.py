from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, List, Dict
from app.data.clients.open_api_client import OpenApiClient
from app.utils import CSVHandler
from app.core.config import API_BASE_URL, API_KEY
import pandas as pd
from app.ml import ModelService
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import json

# 스키마 import
from app.schemas.csv import CSVSaveResponse, CSVFileInfo, CSVListResponse
from app.schemas.model import (
    ModelType,
    TrainModelMetrics,
    TrainModelResponse,
    PredictionResponse
)

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
            numOfRows=1000
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
        
        # API 응답 데이터 구조 확인
        print("\n=== API 응답 데이터 구조 ===")
        print(f"데이터 개수: {len(items)}")
        if len(items) > 0:
            print("첫 번째 항목의 키:")
            for key in items[0].keys():
                print(f"- {key}")
            print("\n첫 번째 항목의 값:")
            for key, value in items[0].items():
                print(f"- {key}: {value}")
        
        # isinCd 값 추출 (첫 번째 항목의 isinCd 사용)
        isin_code = str(items[0].get("isinCd", "unknown"))
        print(f"\n=== 데이터 처리 시작 ===")
        print(f"원본 데이터 개수: {len(items)}")
        print(f"ISIN 코드: {isin_code}")
        
        try:
            # 데이터를 DataFrame으로 변환하고 날짜 기준으로 정렬
            print("\n=== DataFrame 변환 및 정렬 ===")
            df = pd.DataFrame(items)
            print(f"DataFrame 컬럼: {df.columns.tolist()}")
            
            # 날짜 컬럼 확인
            date_column = 'basDt'  # API 응답의 실제 컬럼명 사용
            if date_column not in df.columns:
                raise KeyError(f"날짜 컬럼 '{date_column}'을 찾을 수 없습니다. 사용 가능한 컬럼: {df.columns.tolist()}")
                
            print(f"날짜 데이터 샘플: {df[date_column].head().tolist()}")
            
            # 날짜 정렬
            df[date_column] = pd.to_datetime(df[date_column], format='%Y%m%d')
            print("날짜 변환 완료")
            
            df = df.sort_values(date_column)  # 날짜 기준 오름차순 정렬
            print("정렬 완료")
            
            df[date_column] = df[date_column].dt.strftime('%Y%m%d')  # 다시 원래 형식으로 변환
            print("날짜 형식 재변환 완료")
            
            items = df.to_dict('records')  # 정렬된 데이터를 리스트로 변환
            print(f"변환 후 데이터 개수: {len(items)}")
            
            # CSVHandler를 사용하여 CSV 파일 저장
            print("\n=== CSV 저장 시작 ===")
            file_path = CSVHandler.save_to_csv(
                data=items,
                isin_code=isin_code,
                stock_name=itmsNm
            )
            print(f"저장된 파일 경로: {file_path}")
            
            return CSVSaveResponse(
                success=True,
                message=f"{itmsNm} 종목 데이터를 성공적으로 CSV 파일로 저장했습니다.",
                file_path=file_path,
                records_count=len(items)
            )
            
        except Exception as e:
            print(f"\n=== 데이터 처리 중 오류 발생 ===")
            print(f"오류 유형: {type(e)}")
            print(f"오류 메시지: {str(e)}")
            raise  # 원래 except 블록으로 전달
        
    except Exception as e:
        print(f"\n=== CSV 저장 중 오류 발생 ===")
        print(f"오류 유형: {type(e)}")
        print(f"오류 메시지: {str(e)}")
        return CSVSaveResponse(
            success=False,
            message=f"CSV 저장 중 오류 발생: {str(e)}"
        )

@router.post("/trainModel", response_model=TrainModelResponse)
async def train_model(
    isin_code: str = Query(..., description="종목 코드"),
    model_type: ModelType = Query(..., description="모델 유형 (현재는 LSTM, RNN 지원)"),
    time_steps: int = Query(10, description="시계열 데이터의 시퀀스 길이"),
    epochs: int = Query(100, description="학습 에포크 수"),
    batch_size: int = Query(32, description="배치 크기"),
    validation_split: float = Query(0.2, description="검증 데이터 비율")
):
    """
    주식 데이터를 사용하여 머신러닝 모델을 학습합니다.
    """
    try:
        print("\n=== 모델 학습 시작 ===")
        print(f"ISIN 코드: {isin_code}")
        print(f"모델 타입: {model_type.value}")
        
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
        print(f"\n=== 데이터 로드 ===")
        print(f"파일 경로: {latest_file['path']}")
        
        # CSV 파일 읽기
        df = pd.read_csv(latest_file["path"])
        print(f"데이터 행 수: {len(df)}")
        
        # 모델 파라미터 설정
        model_params = {
            "time_steps": time_steps,
            "learning_rate": 0.001,
            "dropout": 0.1
        }
        
        # Factory를 통한 모델 생성
        model = ModelService.create(model_type.value, model_params)
        
        # 모델 학습
        result = model.train_from_dataframe(
            df=df,
            target_col="CLPR",
            isin_code=isin_code,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )
        
        print("\n=== 학습 완료 ===")
        print(f"모델 저장 경로: {result['model_path']}")
        print(f"최종 메트릭스: {result['metrics']}")
        
        return TrainModelResponse(
            success=True,
            message=f"{isin_code} 종목의 {model_type} 모델 학습이 완료되었습니다.",
            metrics=result['metrics'],
            model_path=result['model_path']
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

@router.post("/predict/{isin_code}")
async def predict_stock_price(
    isin_code: str,
    model_type: ModelType,
    model_date: str = Query(..., description="모델 날짜 (YYYYMMDD 형식)")
) -> PredictionResponse:
    """
    학습된 모델을 사용하여 다음날 주가 예측
    """
    try:
        # 모델 경로 생성
        model_dir = f"{isin_code}_{model_type.value}_{model_date}"
        model_path = Path("storage/models") / model_dir
        
        if not model_path.exists():
            return PredictionResponse(
                success=False,
                message=f"학습된 모델을 찾을 수 없습니다: {model_dir}"
            )
            
        # 모델 파라미터 로드
        params_path = model_path / "params.json"
        with open(params_path, 'r') as f:
            saved_params = json.load(f)
            
        # 모델 생성 및 로드
        model = ModelService.create(model_type.value, saved_params)
        model.load(str(model_path / "model.keras"))
        
        # 스케일러 로드
        scaler_path = model_path / "scaler.pkl"
        if scaler_path.exists():
            import joblib
            model.scaler = joblib.load(scaler_path)
            
        # 마지막 시퀀스 로드
        sequence_path = model_path / "last_sequence.npy"
        if not sequence_path.exists():
            return PredictionResponse(
                success=False,
                message="저장된 시퀀스 데이터를 찾을 수 없습니다."
            )
            
        last_sequence = np.load(sequence_path)
        
        # 예측 수행
        prediction = model.predict_next_value(last_sequence)
        print(f"예측 결과: {prediction:,.0f}원")
        
        # 날짜는 현재 날짜 기준 다음 영업일
        next_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        return PredictionResponse(
            success=True,
            message="예측이 성공적으로 완료되었습니다.",
            prediction=float(prediction),
            date=next_date
        )
        
    except Exception as e:
        return PredictionResponse(
            success=False,
            message=f"예측 중 오류가 발생했습니다: {str(e)}"
        )
