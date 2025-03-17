import json
import os
import pandas as pd
from pathlib import Path

def load_stock_data():
    """
    JSON 파일에서 주식 데이터를 로드합니다.
    """
    file_path = Path(__file__).parent.parent / "data" / "samsung_stock_data.json"
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return data

def get_stock_dataframe():
    """
    주식 데이터를 pandas DataFrame으로 변환합니다.
    """
    data = load_stock_data()
    items = data['response']['body']['items']['item']
    df = pd.DataFrame(items)
    
    # 데이터 전처리
    df['basDt'] = pd.to_datetime(df['basDt'], format='%Y%m%d')
    df = df.sort_values('basDt')  # 날짜 순으로 정렬
    
    # 숫자형 컬럼 변환
    numeric_columns = ['clpr', 'mkp', 'hipr', 'lopr', 'trqu', 'trPrc', 'lstgStCnt', 'mrktTotAmt']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col])
    
    # fltRt 컬럼 처리 (특수 문자 제거 후 변환)
    df['fltRt'] = df['fltRt'].str.replace('-', '-', regex=False)  # 음수 부호 처리
    df['fltRt'] = pd.to_numeric(df['fltRt'])
    
    return df