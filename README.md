# 주식 예측 머신러닝 백엔드 서비스

삼성전자 주가 데이터를 기반으로 한 머신러닝 예측 서비스입니다.

## 설치 및 실행 방법

### 환경 설정

```bash
# conda 환경 생성
conda create -n ml-backend python=3.12 -y

# 환경 활성화
conda activate ml-backend

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 서버 실행

```bash
python -m uvicorn app.main:app --reload
```

## API 엔드포인트

- `GET /`: 서비스 정보 및 사용 가능한 엔드포인트 목록
- `GET /api/stock/data`: 삼성전자 주가 데이터 조회
- `POST /api/stock/train`: 예측 모델 학습
- `GET /api/stock/predict`: 다음 날 주가 예측

## API 문서

애플리케이션이 실행되면 다음 URL에서 API 문서를 확인할 수 있습니다:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
