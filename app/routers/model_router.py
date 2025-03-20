from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, Any, Optional, List
from app.data.clients.open_api_client import OpenApiClient
import pandas as pd
import os
from pathlib import Path
import datetime
from pydantic import BaseModel
from app.core.config import API_BASE_URL, API_KEY, CSV_DIR, DATE_FORMAT, BASE_DIR

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
    stock_code: str
    stock_name: str

# CSV 파일 목록 응답 모델
class CSVListResponse(BaseModel):
    success: bool
    message: str
    files: List[CSVFileInfo]
    count: int

# 라우터 정의
router = APIRouter(prefix="/ml", tags=["ml"])

# OpenAPI 클라이언트 의존성
def get_open_api_client():
    return OpenApiClient(api_key=API_KEY, base_url=API_BASE_URL)

@router.get("/getStockData", response_model=CSVListResponse)
async def get_stock_data_files(
    itmsNm: Optional[str] = Query(None, description="종목명으로 필터링 (선택사항)")
):
    """
    storage/csv 디렉토리에 저장된 주식 데이터 CSV 파일 목록을 반환합니다.
    선택적으로 종목명으로 필터링할 수 있습니다.
    """
    try:
        # CSV 디렉토리 확인
        if not os.path.exists(CSV_DIR):
            return CSVListResponse(
                success=True,
                message="CSV 디렉토리가 아직 생성되지 않았습니다.",
                files=[],
                count=0
            )
        
        # 모든 CSV 파일 검색
        csv_files = list(CSV_DIR.glob("*.csv"))
        
        # 파일 정보 수집
        file_info_list = []
        for file_path in csv_files:
            # 파일명 분석 (isinCd_itmsNm_date.csv 형식)
            filename = file_path.name
            parts = filename.split('_')
            
            if len(parts) >= 3:
                stock_code = parts[0]
                
                # 날짜는 맨 뒤의 ".csv" 제거 후 마지막 요소
                date_part = parts[-1].replace(".csv", "")
                
                # 종목명은 중간 부분들 (코드와 날짜 사이의 모든 부분)
                stock_name = '_'.join(parts[1:-1])
                
                # 종목명 필터링이 있는 경우
                if itmsNm and itmsNm.lower() not in stock_name.lower():
                    continue
                
                # 파일 생성 시간 
                created_time = datetime.datetime.fromtimestamp(file_path.stat().st_ctime)
                created_at = created_time.strftime("%Y-%m-%d %H:%M:%S")
                
                # 상대 경로 계산 (프로젝트 루트 경로 제거)
                try:
                    relative_path = file_path.relative_to(BASE_DIR)
                    path_str = str(relative_path)
                except ValueError:
                    # 상대 경로 계산이 실패한 경우, 전체 경로에서 앞부분 제거
                    path_str = str(file_path).replace(str(BASE_DIR) + '/', '')
                
                file_info = CSVFileInfo(
                    filename=filename,
                    path=path_str,  # 상대 경로 사용
                    size_bytes=file_path.stat().st_size,
                    created_at=created_at,
                    stock_code=stock_code,
                    stock_name=stock_name
                )
                file_info_list.append(file_info)
        
        return CSVListResponse(
            success=True,
            message=f"총 {len(file_info_list)}개의 CSV 파일이 있습니다.",
            files=file_info_list,
            count=len(file_info_list)
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
        
        # 데이터프레임으로 변환
        df = pd.DataFrame(items)
        
        # 컬럼명을 대문자로 변환
        df.columns = [col.upper() for col in df.columns]
        
        # isinCd 값 추출 (첫 번째 항목의 isinCd 사용)
        isin_cd = str(items[0].get("isinCd", "unknown"))
        
        # 날짜 포맷 설정
        today = datetime.datetime.now().strftime(DATE_FORMAT)
        
        # 파일명 생성 (isinCd값_itmsNm_생성일.csv)
        safe_itms_nm = itmsNm.replace("/", "_").replace("\\", "_")  # 파일명에 사용할 수 없는 문자 처리
        csv_filename = f"{isin_cd}_{safe_itms_nm}_{today}.csv"
        csv_path = CSV_DIR / csv_filename
        
        # CSV 파일로 저장
        df.to_csv(csv_path, index=False)
        
        return CSVSaveResponse(
            success=True,
            message=f"{itmsNm} 종목 데이터를 성공적으로 CSV 파일로 저장했습니다.",
            file_path=str(csv_path),
            records_count=len(df)
        )
        
    except Exception as e:
        return CSVSaveResponse(
            success=False,
            message=f"CSV 저장 중 오류 발생: {str(e)}"
        )

