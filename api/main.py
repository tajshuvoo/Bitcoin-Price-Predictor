# api/main.py

from fastapi import FastAPI
from prediction.one_day_model_prediction import predict_next_day
from prediction.seven_days_data import fetch_last_seven_days
from prediction.two_day_model_prediction import predict_next_day_with_two_day_model
from prediction.three_day_model_prediction import predict_next_day_with_three_day_model
from prediction.four_day_model_prediction import predict_next_day_with_four_day_model
from prediction.five_day_model_prediction import predict_next_day_with_five_day_model
from prediction.six_day_model_prediction import predict_next_day_with_six_day_model
from prediction.seven_day_model_prediction import predict_next_day_with_seven_day_model
from prediction.one_hour_model_prediction import predict_next_day_with_one_hour_model
from prediction.six_hour_model_prediction import predict_next_day_with_six_hour_model
from prediction.twelve_hour_model_prediction import predict_next_day_with_twelve_hour_model
from prediction.eighteen_hour_model_prediction import predict_next_day_with_eighteen_hour_model
from prediction.fifteen_min_model_prediction import predict_next_day_with_fifteen_min_model
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import json

app = FastAPI()

origins = [
    "http://localhost:5173", 
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] for all origins (less secure)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "✅ BTC Predictor API is running"}

predict_router = APIRouter(prefix="/predict")

@predict_router.get("/one_day")
def get_prediction():
    try:
        results = predict_next_day()
        return {"status": "success", "data": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@predict_router.get("/two_days")
def get_two_day_prediction():
    try:
        results = predict_next_day_with_two_day_model()
        return {"status": "success", "data": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@predict_router.get("/three_days")
def get_three_day_prediction():
    try:
        results = predict_next_day_with_three_day_model()
        return {"status": "success", "data": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@predict_router.get("/four_days")
def get_four_day_prediction():
    try:
        results = predict_next_day_with_four_day_model()
        return {"status": "success", "data": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@predict_router.get("/five_days")
def get_five_day_prediction():
    try:
        results = predict_next_day_with_five_day_model()
        return {"status": "success", "data": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@predict_router.get("/six_days")
def get_six_day_prediction():
    try:
        results = predict_next_day_with_six_day_model()
        return {"status": "success", "data": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@predict_router.get("/seven_days")
def get_seven_day_prediction():
    try:
        results = predict_next_day_with_seven_day_model()
        return {"status": "success", "data": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@predict_router.get("/one_hour")
def get_one_hour_prediction():
    try:
        results = predict_next_day_with_one_hour_model()
        return {"status": "success", "data": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@predict_router.get("/six_hour")
def get_six_hour_prediction():
    try:
        results = predict_next_day_with_six_hour_model()
        return {"status": "success", "data": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@predict_router.get("/twelve_hour")
def get_twelve_hour_prediction():
    try:
        results = predict_next_day_with_twelve_hour_model()
        return {"status": "success", "data": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@predict_router.get("/eighteen_hour")
def get_eighteen_hour_prediction():
    try:
        results = predict_next_day_with_eighteen_hour_model()
        return {"status": "success", "data": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@predict_router.get("/fifteen_min")
def get_fifteen_min_prediction():
    try:
        results = predict_next_day_with_fifteen_min_model()
        return {"status": "success", "data": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}


app.include_router(predict_router)


@app.get("/fetch_last_seven_days")
def get_last_seven_days_api():
    data = fetch_last_seven_days()
    return {"data": data}

MODEL_INFO_PATH = Path(__file__).parent / "model_info.json"
@app.get("/model_info")
def get_model_info():
    try:
        with open(MODEL_INFO_PATH, "r") as f:
            model_data = json.load(f)
        return JSONResponse(content=model_data)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

MODEL_METRICS_PATH = Path(__file__).parent / "model_metrics.json"

@app.get("/model_metrics")
def get_model_metrics():
    try:
        with open(MODEL_METRICS_PATH, "r") as f:
            metrics_data = json.load(f)
        return JSONResponse(content=metrics_data)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})