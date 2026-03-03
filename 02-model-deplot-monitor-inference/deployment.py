from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json
import os
import logging
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionInput(BaseModel):
    Gender: int
    Age: float
    Height: float
    Weight: float
    Duration: float
    Heart_Rate: float
    Body_Temp: float

class PredictionOutput(BaseModel):
    predicted_calories: float
    model_name: str
    prediction_id: str

class FinalModelService:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_name = None
        self.predictions_log = []
        
    def load_model(self):
        logger.info("Loading model and preprocessors...")
        
        try:
            with open('best_model_info.json', 'r') as f:
                model_info = json.load(f)
            self.model_name = model_info['model_name']
            
            model_path = f"models/{self.model_name}/model.pkl"
            self.model = joblib.load(model_path)
            
            self.scaler = joblib.load('models/scaler.pkl')
            
            logger.info(f"Model loaded: {self.model_name}")
            logger.info(f"Training R2 score: {model_info['r2_score']:.4f}")
            
            self.run_test()
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise e
    
    def run_test(self):
        test_input = PredictionInput(
            Gender=1, Age=30.0, Height=175.0, Weight=75.0,
            Duration=30.0, Heart_Rate=120.0, Body_Temp=37.5
        )
        
        prediction, _ = self.predict(test_input)
        logger.info(f"Test prediction: {prediction:.2f} calories")
        
        if 10 <= prediction <= 400:
            logger.info("Test prediction looks reasonable")
        else:
            logger.warning(f"Test prediction {prediction:.2f} may be unreasonable")
    
    def preprocess_input(self, input_data: PredictionInput):
        data = pd.DataFrame([[
            input_data.Gender,
            input_data.Age,
            input_data.Height,
            input_data.Weight,
            input_data.Duration,
            input_data.Heart_Rate,
            input_data.Body_Temp
        ]], columns=['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])
        
        if 10 <= input_data.Age <= 100:
            scaled_data = self.scaler.transform(data)
        else:
            scaled_data = data.values
        
        return pd.DataFrame(scaled_data, columns=data.columns)
    
    def predict(self, input_data: PredictionInput):
        if self.model is None:
            raise ValueError("Model not loaded")
        
        processed_data = self.preprocess_input(input_data)
        prediction = self.model.predict(processed_data)[0]
        prediction_id = f"pred_{len(self.predictions_log) + 1:06d}"
        
        log_entry = {
            'prediction_id': prediction_id,
            'input_data': input_data.model_dump(),
            'prediction': float(prediction),
            'model_name': self.model_name,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        self.predictions_log.append(log_entry)
        
        return prediction, prediction_id
    
    def get_prediction_logs(self):
        return self.predictions_log

model_service = FinalModelService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load_model()
    yield

app = FastAPI(title="Calories Prediction API", version="3.0.0", lifespan=lifespan)

@app.get("/")
async def root():
    return {
        "message": "Calories Prediction API",
        "status": "running",
        "model_name": model_service.model_name
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_service.model is not None,
        "model_name": model_service.model_name,
        "total_predictions": len(model_service.predictions_log)
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict_calories(input_data: PredictionInput):
    try:
        prediction, prediction_id = model_service.predict(input_data)
        
        return PredictionOutput(
            predicted_calories=float(prediction),
            model_name=model_service.model_name,
            prediction_id=prediction_id
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/predictions")
async def get_predictions():
    predictions = model_service.get_prediction_logs()
    
    if predictions:
        pred_values = [p['prediction'] for p in predictions]
        stats = {
            'count': len(pred_values),
            'mean': float(np.mean(pred_values)),
            'median': float(np.median(pred_values)),
            'std': float(np.std(pred_values)),
            'min': float(np.min(pred_values)),
            'max': float(np.max(pred_values))
        }
    else:
        stats = {}
    
    return {
        "total_predictions": len(predictions),
        "predictions": predictions,
        "statistics": stats
    }

@app.get("/model_info")
async def get_model_info():
    return {
        "model_name": model_service.model_name,
        "model_type": type(model_service.model).__name__,
        "total_predictions": len(model_service.predictions_log)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")