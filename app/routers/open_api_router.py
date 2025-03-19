from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, Any, Optional
from app.data.clients.open_api_client import OpenApiClient

# 환경변수나 설정 파일에서 가져오는 것이 좋음
BASE_URL = "https://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService"
OPEN_API_KEY = "wnxcssFNghF+kPdnOVHpDeCs/48Gz3wDcP8zTitIlcFQGm72/SM6ojdxcH7euW2bFNQYZ+npRLI2JJOm9RvtFg=="

# 라우터 정의
router = APIRouter(prefix="/stock", tags=["stock"])

# OpenAPI 클라이언트 의존성
def get_open_api_client():
    return OpenApiClient(api_key=OPEN_API_KEY, base_url=BASE_URL)

@router.get("/getStockPriceInfo")
async def get_stock_price_info(
    itmsNm: str = Query(..., description="종목명"),
    pageNo: int = Query(1, description="페이지 번호"),
    numOfRows: int = Query(10, description="페이지당 행 수"),
    client: OpenApiClient = Depends(get_open_api_client)
):
    """
    종목명으로 주식 가격 정보를 조회합니다.
    data.go.kr API를 통해 데이터를 가져옵니다.
    """
    try:
        result = await client.fetch_stock_data(
            itmsNm=itmsNm,
            pageNo=pageNo,
            numOfRows=numOfRows
        )
        print('result ', result)
            
        return result
    except Exception as e:
        # 예외 발생 시 오류 메시지 그대로 반환
        raise HTTPException(status_code=500, detail=str(e))
