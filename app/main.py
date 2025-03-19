from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import model_router
from app.routers import open_api_router

app = FastAPI(title="Stock Prediction Service", description="삼성전자 주식 예측 서비스")

# CORS 설정
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(model_router.router)
app.include_router(open_api_router.router)

@app.get("/")
def read_root():
    return {
        "message": "삼성전자 주가 예측 서비스가 실행 중입니다",
        "endpoints": [
            {"path": "/api/model/data", "method": "GET", "description": "삼성전자 주가 데이터 조회 (새 구조)"},
            {"path": "/api/model/train", "method": "POST", "description": "LSTM 모델 학습 (새 구조)"},
            {"path": "/api/model/predict", "method": "GET", "description": "다음 날 주가 예측 (새 구조)"},
            {"path": "/stock/getStockPriceInfo", "method": "GET", "description": "종목 검색 (OpenAPI)"},
            {"path": "/stock/{stock_code}", "method": "GET", "description": "종목 데이터 조회 (OpenAPI)"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)