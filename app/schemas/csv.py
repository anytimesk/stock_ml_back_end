"""
CSV 파일 관련 스키마
"""
from typing import Optional, List
from pydantic import BaseModel

class CSVSaveResponse(BaseModel):
    """CSV 저장 응답 모델"""
    success: bool
    message: str
    file_path: Optional[str] = None
    records_count: Optional[int] = None

class CSVFileInfo(BaseModel):
    """CSV 파일 정보 모델"""
    filename: str
    path: str
    size_bytes: int
    created_at: str
    isin_code: str
    stock_name: str

class CSVListResponse(BaseModel):
    """CSV 파일 목록 응답 모델"""
    success: bool
    message: str
    files: List[CSVFileInfo]
    count: int 