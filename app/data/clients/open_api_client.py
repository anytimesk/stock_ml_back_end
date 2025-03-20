import aiohttp
import ssl
import certifi
import json
from typing import Dict, Any
from urllib.parse import quote_plus

class OpenApiClient:
    """
    data.go.kr Open API 클라이언트
    공공데이터 포털에서 주식 데이터를 가져오는 기능 제공
    """
    
    # 클래스 변수 선언
    api_key =""
    base_url = ""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url

    def encode_api_key(self,api_key):
        """
        API 키를 URL 안전한 형식으로 인코딩합니다.
        공공 데이터 포털 API는 특수문자(+, /, =)를 URL 안전한 형식(%2B, %2F, %3D)으로 인코딩해야 합니다.
        
        Args:
            api_key (str): 인코딩할 API 키
            
        Returns:
            str: 인코딩된 API 키
        """
        return quote_plus(api_key)

    async def fetch_aiohttps(self, endpoint, params=None):
        """
        aiohttp 라이브러리를 사용하여 주어진 엔드포인트와 파라미터로 API를 비동기로 호출합니다.
        
        Args:
            endpoint (str): API 엔드포인트 URL
            params (dict, optional): API 호출에 사용할 파라미터 딕셔너리
            
        Returns:
            str or None: 성공 시 API 응답 내용, 실패 시 None
        """
        
        url = endpoint
        if params:
            # 파라미터 문자열 수동 생성 (이미 인코딩된 값 보존)
            params_str = '&'.join([f"{k}={v}" for k, v in params.items()])
            url = f"{endpoint}?{params_str}"
            
        try:
            # 기본 SSL 컨텍스트 설정
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # 더 유연한 SSL 옵션 설정
            ssl_context.set_ciphers('DEFAULT')
            
            # 커스텀 헤더 설정
            headers = {
                'Accept': 'application/json'
            }
            
            # 클라이언트 타임아웃 설정
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                async with session.get(url, ssl=ssl_context) as response:
                    # 응답 확인
                    response.raise_for_status()
                    
                    # 응답 데이터 읽기
                    content = await response.text()
                    
                    return content
            
        except Exception as e:
            print(f"aiohttp 요청 실패: {e}")
            return None


    async def fetch_stock_data(self, 
                        itmsNm: str, 
                        pageNo: int = 1,
                        numOfRows: int = 10) -> Dict[str, Any]:
        """
        주식 데이터 조회 API 호출
        
        Args:
            itmsNm: 종목명
            pageNo: 페이지 번호
            numOfRows: 한 페이지 결과 수
            
        Returns:
            Dict[str, Any]: API 응답 데이터 (JSON 객체)
        """
        
        endpoint = f"{self.base_url}/getStockPriceInfo"
        
        # 파라미터를 딕셔너리로 정의
        params = {
            "itmsNm": itmsNm,
            "serviceKey": self.encode_api_key(self.api_key),  # 인코딩 함수 또는 저장된 값 사용
            "numOfRows": numOfRows,
            "pageNo": pageNo,
            "resultType": "json"  # 기본값으로 json 형식 사용
        }
        
        # aiohttp로 API 호출
        content = await self.fetch_aiohttps(endpoint, params)
        
        # API 응답 처리
        if content:
            try:
                # 문자열을 JSON 객체로 파싱
                json_data = json.loads(content)
                return json_data
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류: {e}")
                return {"error": "JSON 파싱 실패", "content": content[:100]}
        else:
            return {"error": "API 호출 실패"}
