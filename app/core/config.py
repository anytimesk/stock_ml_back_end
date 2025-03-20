import os
from pathlib import Path

# 기본 경로 설정
BASE_DIR = Path(__file__).parent.parent.parent  # 프로젝트 루트 디렉토리
STORAGE_DIR = os.environ.get('STORAGE_DIR', BASE_DIR / 'storage')
CSV_DIR = Path(STORAGE_DIR) / 'csv'

# API 설정
API_BASE_URL = "https://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService"
API_KEY = os.environ.get('OPEN_API_KEY', "wnxcssFNghF+kPdnOVHpDeCs/48Gz3wDcP8zTitIlcFQGm72/SM6ojdxcH7euW2bFNQYZ+npRLI2JJOm9RvtFg==")

# 파일 형식 설정
DATE_FORMAT = "%Y%m%d"

# 필요한 디렉토리 생성 함수
def ensure_directories():
    """필요한 디렉토리가 존재하는지 확인하고, 없으면 생성합니다."""
    os.makedirs(STORAGE_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    return True 