from pathlib import Path
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
from app.core.config import CSV_DIR, DATE_FORMAT

class CSVHandler:
    @staticmethod
    def save_to_csv(data: List[Dict], isin_code: str, stock_name: str) -> str:
        """
        데이터를 CSV 파일로 저장합니다.
        
        Args:
            data: 저장할 데이터 리스트
            isin_code: 종목 코드
            stock_name: 종목 명
            
        Returns:
            저장된 CSV 파일의 경로
        """
        # CSV 디렉토리가 없으면 생성
        Path(CSV_DIR).mkdir(parents=True, exist_ok=True)
        
        # DataFrame 생성
        df = pd.DataFrame(data)
        
        # 컬럼명을 대문자로 변환
        df.columns = [col.upper() for col in df.columns]
        
        # 파일명 생성 (YYYYMMDD 형식의 날짜 포함)
        today = datetime.now().strftime(DATE_FORMAT)
        
        # 파일명에 사용할 수 없는 문자 처리
        safe_stock_name = stock_name.replace("/", "_").replace("\\", "_")
        filename = f"{isin_code}_{safe_stock_name}_{today}.csv"
        filepath = Path(CSV_DIR) / filename
        
        # CSV 파일로 저장
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        return str(filepath)
    
    @staticmethod
    def get_csv_files(stock_name: Optional[str] = None) -> List[Dict]:
        """
        저장된 CSV 파일 목록을 반환합니다.
        
        Args:
            stock_name: 검색할 종목명 (선택사항)
            
        Returns:
            CSV 파일 정보 리스트
        """
        # CSV 디렉토리가 없으면 빈 리스트 반환
        if not Path(CSV_DIR).exists():
            return []
            
        files = []
        for file_path in Path(CSV_DIR).glob("*.csv"):
            # 파일명에서 정보 추출
            parts = file_path.stem.split("_")
            if len(parts) >= 3:
                isin_code = parts[0]
                
                # 날짜는 맨 뒤의 ".csv" 제거 후 마지막 요소
                date_part = parts[-1]
                
                # 종목명은 중간 부분들 (코드와 날짜 사이의 모든 부분)
                name = '_'.join(parts[1:-1])
                
                # 종목명으로 필터링
                if stock_name and stock_name.lower() not in name.lower():
                    continue
                
                # 파일 생성 시간
                created_time = datetime.fromtimestamp(file_path.stat().st_ctime)
                created_at = created_time.strftime("%Y-%m-%d %H:%M:%S")
                
                files.append({
                    "filename": file_path.name,
                    "path": str(file_path.relative_to(Path.cwd())),
                    "size_bytes": file_path.stat().st_size,
                    "created_at": created_at,
                    "isin_code": isin_code,
                    "stock_name": name
                })
                
        return files

    @staticmethod
    def get_row_count(file_path: str) -> int:
        """
        CSV 파일의 행 수를 반환합니다.
        
        Args:
            file_path: CSV 파일 경로
            
        Returns:
            CSV 파일의 행 수 (헤더 제외)
        """
        try:
            df = pd.read_csv(file_path)
            return len(df)
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return 0 