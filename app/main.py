from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import stock_router

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
app.include_router(stock_router.router)

@app.get("/")
def read_root():
    return {
        "message": "삼성전자 주가 예측 서비스가 실행 중입니다",
        "endpoints": [
            {"path": "/api/stock/data", "method": "GET", "description": "삼성전자 주가 데이터 조회"},
            {"path": "/api/stock/train", "method": "POST", "description": "예측 모델 학습"},
            {"path": "/api/stock/predict", "method": "GET", "description": "다음 날 주가 예측"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)