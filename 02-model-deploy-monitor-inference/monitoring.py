from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json
import os
import requests
from contextlib import asynccontextmanager
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

class PredictionInput(BaseModel):
    Gender: int
    Age: float
    Height: float
    Weight: float
    Duration: float
    Heart_Rate: float
    Body_Temp: float

class FinalMonitor:
    def __init__(self):
        self.baseline_stats = None
        
    def load_baseline_data(self):
        try:
            if os.path.exists('X_train.csv'):
                X_train = pd.read_csv('X_train.csv')
                self.baseline_stats = {
                    'mean': X_train.mean().to_dict(),
                    'std': X_train.std().to_dict(),
                    'count': len(X_train)
                }
                return True
        except:
            pass
        return False
    
    def load_model_info(self):
        try:
            if os.path.exists('best_model_info.json'):
                with open('best_model_info.json', 'r') as f:
                    return json.load(f)
        except:
            pass
        return {"model_name": "Unknown", "r2_score": 0.0}
    
    def get_api_predictions(self):
        try:
            response = requests.get("http://127.0.0.1:8000/predictions", timeout=3)
            if response.status_code == 200:
                return response.json().get('predictions', [])
        except:
            pass
        return []
    
    def check_system_health(self):
        status = {}
        try:
            response = requests.get("http://127.0.0.1:8000/health", timeout=3)
            status['api'] = 'Online' if response.status_code == 200 else 'Error'
        except:
            status['api'] = 'Offline'
        
        try:
            response = requests.get("http://127.0.0.1:5005", timeout=3)
            status['mlflow'] = 'Online' if response.status_code == 200 else 'Error'
        except:
            status['mlflow'] = 'Offline'
        
        status['data'] = 'Ready' if self.baseline_stats else 'Missing'
        return status
    
    def make_prediction(self, input_data):
        try:
            response = requests.post("http://127.0.0.1:8000/predict", 
                                   json=input_data.model_dump(), timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def run_validation_test(self, sample_size=30):
        try:
            if not os.path.exists('X_test.csv') or not os.path.exists('y_test.csv'):
                return {"error": "Test data not found"}
            
            X_test = pd.read_csv('X_test.csv')
            y_test = pd.read_csv('y_test.csv').values.ravel()
            
            indices = np.random.choice(len(X_test), min(sample_size, len(X_test)), replace=False)
            
            predictions = []
            actuals = []
            
            for idx in indices:
                input_data = PredictionInput(
                    Gender=int(X_test.iloc[idx]['Gender']),
                    Age=float(X_test.iloc[idx]['Age']),
                    Height=float(X_test.iloc[idx]['Height']),
                    Weight=float(X_test.iloc[idx]['Weight']),
                    Duration=float(X_test.iloc[idx]['Duration']),
                    Heart_Rate=float(X_test.iloc[idx]['Heart_Rate']),
                    Body_Temp=float(X_test.iloc[idx]['Body_Temp'])
                )
                
                result = self.make_prediction(input_data)
                if result:
                    predictions.append(result['predicted_calories'])
                    actuals.append(y_test[idx])
            
            if predictions and actuals:
                predictions = np.array(predictions)
                actuals = np.array(actuals)
                
                mse = np.mean((predictions - actuals) ** 2)
                mae = np.mean(np.abs(predictions - actuals))
                rmse = np.sqrt(mse)
                
                ss_res = np.sum((actuals - predictions) ** 2)
                ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                return {
                    "success": True,
                    "samples": len(predictions),
                    "metrics": {
                        "r2": float(r2),
                        "rmse": float(rmse),
                        "mae": float(mae),
                        "mse": float(mse)
                    },
                    "prediction_range": [float(predictions.min()), float(predictions.max())],
                    "actual_range": [float(actuals.min()), float(actuals.max())]
                }
            
        except Exception as e:
            return {"error": str(e)}
        
        return {"error": "Validation failed"}
    
    def modify_and_test(self, sample_size=25):
        try:
            if not os.path.exists('X_test.csv'):
                return {"error": "Test data not found"}
            
            X_test = pd.read_csv('X_test.csv')
            X_modified = X_test.copy()
            
            numeric_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
            features_to_modify = np.random.choice(numeric_features, 2, replace=False)
            modifications = []
            
            for feature in features_to_modify:
                noise_std = X_test[feature].std() * 0.3
                noise = np.random.normal(0, noise_std, len(X_test))
                X_modified[feature] = X_test[feature] + noise
                modifications.append(f"{feature}: noise (std={noise_std:.2f})")
            
            indices = np.random.choice(len(X_test), min(sample_size, len(X_test)), replace=False)
            
            original_preds = []
            modified_preds = []
            
            for idx in indices:
                orig_input = PredictionInput(
                    Gender=int(X_test.iloc[idx]['Gender']),
                    Age=float(X_test.iloc[idx]['Age']),
                    Height=float(X_test.iloc[idx]['Height']),
                    Weight=float(X_test.iloc[idx]['Weight']),
                    Duration=float(X_test.iloc[idx]['Duration']),
                    Heart_Rate=float(X_test.iloc[idx]['Heart_Rate']),
                    Body_Temp=float(X_test.iloc[idx]['Body_Temp'])
                )
                
                mod_input = PredictionInput(
                    Gender=int(X_modified.iloc[idx]['Gender']),
                    Age=float(X_modified.iloc[idx]['Age']),
                    Height=float(X_modified.iloc[idx]['Height']),
                    Weight=float(X_modified.iloc[idx]['Weight']),
                    Duration=float(X_modified.iloc[idx]['Duration']),
                    Heart_Rate=float(X_modified.iloc[idx]['Heart_Rate']),
                    Body_Temp=float(X_modified.iloc[idx]['Body_Temp'])
                )
                
                orig_result = self.make_prediction(orig_input)
                mod_result = self.make_prediction(mod_input)
                
                if orig_result and mod_result:
                    original_preds.append(orig_result['predicted_calories'])
                    modified_preds.append(mod_result['predicted_calories'])
            
            if original_preds and modified_preds:
                avg_orig = np.mean(original_preds)
                avg_mod = np.mean(modified_preds)
                diff_percent = ((avg_mod - avg_orig) / avg_orig) * 100 if avg_orig != 0 else 0
                
                return {
                    "success": True,
                    "modifications": modifications,
                    "samples": len(original_preds),
                    "original_avg": float(avg_orig),
                    "modified_avg": float(avg_mod),
                    "difference_percent": float(diff_percent)
                }
            
        except Exception as e:
            return {"error": str(e)}
        
        return {"error": "Modification test failed"}
    
    def detect_data_drift(self):
        try:
            predictions = self.get_api_predictions()
            
            if len(predictions) == 0:
                return {"message": "No prediction data"}
            
            input_data = [pred['input_data'] for pred in predictions]
            current_data = pd.DataFrame(input_data)
            
            drift_results = {}
            alerts = []
            
            if self.baseline_stats:
                for feature in current_data.columns:
                    if feature in self.baseline_stats['mean']:
                        baseline_mean = self.baseline_stats['mean'][feature]
                        baseline_std = self.baseline_stats['std'][feature]
                        current_mean = current_data[feature].mean()
                        
                        z_score = abs(current_mean - baseline_mean) / baseline_std if baseline_std > 0 else 0
                        
                        if z_score > 2:
                            drift_results[feature] = {
                                'drift_detected': True,
                                'z_score': z_score,
                                'baseline_mean': baseline_mean,
                                'current_mean': current_mean
                            }
                            alerts.append(f"{feature}: drift detected (Z={z_score:.2f})")
                        else:
                            drift_results[feature] = {
                                'drift_detected': False,
                                'z_score': z_score,
                                'baseline_mean': baseline_mean,
                                'current_mean': current_mean
                            }
            
            return {
                'drift_detected': len(alerts) > 0,
                'alerts': alerts,
                'drift_results': drift_results
            }
            
        except Exception as e:
            return {"error": f"Drift analysis failed: {str(e)}"}
    
    def analyze_predictions(self):
        try:
            predictions = self.get_api_predictions()
            
            if len(predictions) == 0:
                return {"message": "No prediction data"}
            
            pred_values = [p['prediction'] for p in predictions]
            
            return {
                'total_predictions': len(predictions),
                'prediction_stats': {
                    'mean': float(np.mean(pred_values)),
                    'std': float(np.std(pred_values)),
                    'min': float(np.min(pred_values)),
                    'max': float(np.max(pred_values)),
                    'median': float(np.median(pred_values))
                },
                'recent_predictions': predictions[-5:] if len(predictions) >= 5 else predictions
            }
            
        except Exception as e:
            return {"error": f"Prediction analysis failed: {str(e)}"}
    
    def generate_plots_base64(self):
        plots = {}
        
        try:
            predictions = self.get_api_predictions()
            
            if len(predictions) > 0:
                pred_values = [p['prediction'] for p in predictions]
                timestamps = [p['timestamp'][:10] for p in predictions]  # Get date part
                
                # Prediction distribution plot
                plt.figure(figsize=(8, 6))
                plt.hist(pred_values, bins=20, alpha=0.7, color='blue', edgecolor='black')
                plt.title('Prediction Distribution')
                plt.xlabel('Predicted Calories')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                plots['prediction_dist'] = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                # Predictions over time
                if len(predictions) > 1:
                    plt.figure(figsize=(10, 6))
                    plt.plot(range(len(pred_values)), pred_values, marker='o', alpha=0.7)
                    plt.title('Predictions Over Time')
                    plt.xlabel('Prediction Number')
                    plt.ylabel('Predicted Calories')
                    plt.grid(True, alpha=0.3)
                    
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                    buffer.seek(0)
                    plots['prediction_time'] = base64.b64encode(buffer.getvalue()).decode()
                    plt.close()
                
                # Feature distribution (if we have input data)
                input_data = [pred['input_data'] for pred in predictions]
                if input_data:
                    df = pd.DataFrame(input_data)
                    
                    plt.figure(figsize=(12, 8))
                    numeric_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
                    
                    for i, col in enumerate(numeric_cols, 1):
                        if col in df.columns:
                            plt.subplot(2, 3, i)
                            plt.hist(df[col], bins=15, alpha=0.7, color='green', edgecolor='black')
                            plt.title(f'{col} Distribution')
                            plt.xlabel(col)
                            plt.ylabel('Frequency')
                    
                    plt.tight_layout()
                    
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                    buffer.seek(0)
                    plots['feature_dist'] = base64.b64encode(buffer.getvalue()).decode()
                    plt.close()
            
            plot_files = [
                ('correlation_matrix', 'plots/correlation_matrix.png'),
                ('target_analysis', 'plots/target_analysis.png'),
                ('feature_importance', 'plots/feature_importance.png')
            ]
            
            for plot_name, plot_path in plot_files:
                if os.path.exists(plot_path):
                    with open(plot_path, 'rb') as f:
                        plots[plot_name] = base64.b64encode(f.read()).decode()
                        
        except Exception as e:
            print(f"Plot generation error: {str(e)}")
            
        return plots

monitor = FinalMonitor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    monitor.load_baseline_data()
    yield

monitoring_app = FastAPI(title="Model Monitoring", version="3.0.0", lifespan=lifespan)

@monitoring_app.get("/", response_class=HTMLResponse)
async def dashboard():
    model_info = monitor.load_model_info()
    predictions = monitor.get_api_predictions()
    health = monitor.check_system_health()
    plots = monitor.generate_plots_base64()
    
    total_preds = len(predictions)
    avg_pred = np.mean([p['prediction'] for p in predictions]) if predictions else 0
    r2_score = model_info.get('r2_score', 0)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Model Monitor</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, sans-serif;
                background: #f8f9fa;
                color: #2c3e50;
                margin: 0;
                padding: 20px;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            
            .header {{
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }}
            
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }}
            
            .metric-card {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            }}
            
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 5px;
            }}
            
            .metric-label {{
                color: #7f8c8d;
                font-size: 14px;
            }}
            
            .status-section {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }}
            
            .status-item {{
                display: flex;
                justify-content: space-between;
                padding: 10px 0;
                border-bottom: 1px solid #ecf0f1;
            }}
            
            .status-item:last-child {{ border-bottom: none; }}
            
            .status-online {{ color: #27ae60; }}
            .status-offline {{ color: #e74c3c; }}
            
            .tools-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }}
            
            .tool-card {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                cursor: pointer;
                transition: transform 0.2s;
            }}
            
            .tool-card:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            }}
            
            .tool-title {{
                font-weight: bold;
                margin-bottom: 8px;
            }}
            
            .tool-desc {{
                color: #7f8c8d;
                font-size: 14px;
                margin-bottom: 10px;
            }}
            
            .result {{
                margin-top: 10px;
                padding: 10px;
                background: #f8f9fa;
                border-radius: 4px;
                font-size: 12px;
            }}
            
            .predict-section {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            
            .input-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                margin-bottom: 15px;
            }}
            
            .input-group label {{
                display: block;
                margin-bottom: 5px;
                font-weight: 500;
            }}
            
            .input-group input, .input-group select {{
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            
            .predict-btn {{
                background: #3498db;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }}
            
            .predict-btn:hover {{
                background: #2980b9;
            }}
            
            .plots-section {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }}
            
            .plots-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            
            .plot-container {{
                text-align: center;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                background: #fafafa;
            }}
            
            .plot-container h4 {{
                margin-top: 0;
                color: #2c3e50;
            }}
            
            .plot-container img {{
                max-width: 100%;
                height: auto;
                border-radius: 4px;
            }}
        </style>
        <script>
            async function makePrediction() {{
                const form = document.getElementById('predictForm');
                const formData = new FormData(form);
                const data = {{
                    Gender: parseInt(formData.get('gender')),
                    Age: parseFloat(formData.get('age')),
                    Height: parseFloat(formData.get('height')),
                    Weight: parseFloat(formData.get('weight')),
                    Duration: parseFloat(formData.get('duration')),
                    Heart_Rate: parseFloat(formData.get('heart_rate')),
                    Body_Temp: parseFloat(formData.get('body_temp'))
                }};
                
                try {{
                    const response = await fetch('/predict', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify(data)
                    }});
                    
                    const result = await response.json();
                    document.getElementById('result').innerHTML = 
                        response.ok ? 
                        `Prediction: ${{result.predicted_calories.toFixed(1)}} calories` :
                        `Error: ${{result.detail}}`;
                }} catch (error) {{
                    document.getElementById('result').innerHTML = `Error: ${{error.message}}`;
                }}
            }}
            
            async function runValidation() {{
                document.getElementById('validationResult').innerHTML = 'Running...';
                try {{
                    const response = await fetch('/run_validation');
                    const result = await response.json();
                    if (result.success) {{
                        document.getElementById('validationResult').innerHTML = 
                            `Samples: ${{result.samples}}, R²: ${{result.metrics.r2.toFixed(4)}}, RMSE: ${{result.metrics.rmse.toFixed(2)}}`;
                    }} else {{
                        document.getElementById('validationResult').innerHTML = 
                            `Error: ${{result.error}}`;
                    }}
                }} catch (error) {{
                    document.getElementById('validationResult').innerHTML = 
                        `Error: ${{error.message}}`;
                }}
            }}
            
            async function runModificationTest() {{
                document.getElementById('modificationResult').innerHTML = 'Running...';
                try {{
                    const response = await fetch('/modify_test');
                    const result = await response.json();
                    if (result.success) {{
                        document.getElementById('modificationResult').innerHTML = 
                            `${{result.modifications.join(', ')}}<br>Original: ${{result.original_avg.toFixed(1)}}, Modified: ${{result.modified_avg.toFixed(1)}}, Change: ${{result.difference_percent.toFixed(1)}}%`;
                    }} else {{
                        document.getElementById('modificationResult').innerHTML = 
                            `Error: ${{result.error}}`;
                    }}
                }} catch (error) {{
                    document.getElementById('modificationResult').innerHTML = 
                        `Error: ${{error.message}}`;
                }}
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Model Monitoring Dashboard</h1>
                <p>Real-time monitoring for calories prediction model</p>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{total_preds}</div>
                    <div class="metric-label">Total Predictions</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{r2_score:.4f}</div>
                    <div class="metric-label">Training R² Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{avg_pred:.0f}</div>
                    <div class="metric-label">Avg Prediction</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{model_info.get('model_name', 'Unknown')}</div>
                    <div class="metric-label">Model Type</div>
                </div>
            </div>
            
            <div class="status-section">
                <h3>System Status</h3>
                <div class="status-item">
                    <span>API Service</span>
                    <span class="status-{('online' if health['api'] == 'Online' else 'offline')}">{health['api']}</span>
                </div>
                <div class="status-item">
                    <span>MLflow</span>
                    <span class="status-{('online' if health['mlflow'] == 'Online' else 'offline')}">{health['mlflow']}</span>
                </div>
                <div class="status-item">
                    <span>Data Status</span>
                    <span class="status-{('online' if health['data'] == 'Ready' else 'offline')}">{health['data']}</span>
                </div>
            </div>
            
            <div class="tools-grid">
                <div class="tool-card" onclick="runValidation()">
                    <div class="tool-title">Run Validation Test</div>
                    <div class="tool-desc">Test model with validation data</div>
                    <div id="validationResult" class="result"></div>
                </div>
                <div class="tool-card" onclick="runModificationTest()">
                    <div class="tool-title">Feature Modification Test</div>
                    <div class="tool-desc">Modify features and observe changes</div>
                    <div id="modificationResult" class="result"></div>
                </div>
            </div>
            
            <div class="plots-section">
                <h3>Analytics & Visualizations</h3>
                <div class="plots-grid">
                    {'<div class="plot-container"><h4>Prediction Distribution</h4><img src="data:image/png;base64,' + plots.get('prediction_dist', '') + '" alt="Prediction Distribution"></div>' if plots.get('prediction_dist') else '<div class="plot-container"><h4>Prediction Distribution</h4><p>No prediction data available</p></div>'}
                    
                    {'<div class="plot-container"><h4>Predictions Over Time</h4><img src="data:image/png;base64,' + plots.get('prediction_time', '') + '" alt="Predictions Over Time"></div>' if plots.get('prediction_time') else '<div class="plot-container"><h4>Predictions Over Time</h4><p>Need more predictions for time series</p></div>'}
                    
                    {'<div class="plot-container"><h4>Feature Distributions</h4><img src="data:image/png;base64,' + plots.get('feature_dist', '') + '" alt="Feature Distributions"></div>' if plots.get('feature_dist') else '<div class="plot-container"><h4>Feature Distributions</h4><p>No feature data available</p></div>'}
                    
                    {'<div class="plot-container"><h4>Correlation Matrix</h4><img src="data:image/png;base64,' + plots.get('correlation_matrix', '') + '" alt="Correlation Matrix"></div>' if plots.get('correlation_matrix') else '<div class="plot-container"><h4>Correlation Matrix</h4><p>EDA plot not found</p></div>'}
                    
                    {'<div class="plot-container"><h4>Target Analysis</h4><img src="data:image/png;base64,' + plots.get('target_analysis', '') + '" alt="Target Analysis"></div>' if plots.get('target_analysis') else '<div class="plot-container"><h4>Target Analysis</h4><p>EDA plot not found</p></div>'}
                    
                    {'<div class="plot-container"><h4>Feature Importance</h4><img src="data:image/png;base64,' + plots.get('feature_importance', '') + '" alt="Feature Importance"></div>' if plots.get('feature_importance') else '<div class="plot-container"><h4>Feature Importance</h4><p>EDA plot not found</p></div>'}
                </div>
            </div>
            
            <div class="predict-section">
                <h3>Make Prediction</h3>
                <form id="predictForm">
                    <div class="input-grid">
                        <div class="input-group">
                            <label>Gender</label>
                            <select name="gender" required>
                                <option value="0">Female</option>
                                <option value="1">Male</option>
                            </select>
                        </div>
                        <div class="input-group">
                            <label>Age</label>
                            <input type="number" name="age" value="30" required>
                        </div>
                        <div class="input-group">
                            <label>Height (cm)</label>
                            <input type="number" name="height" value="170" step="0.1" required>
                        </div>
                        <div class="input-group">
                            <label>Weight (kg)</label>
                            <input type="number" name="weight" value="70" step="0.1" required>
                        </div>
                        <div class="input-group">
                            <label>Duration (min)</label>
                            <input type="number" name="duration" value="30" step="0.1" required>
                        </div>
                        <div class="input-group">
                            <label>Heart Rate</label>
                            <input type="number" name="heart_rate" value="120" step="0.1" required>
                        </div>
                        <div class="input-group">
                            <label>Body Temp (°C)</label>
                            <input type="number" name="body_temp" value="37.5" step="0.1" required>
                        </div>
                    </div>
                    <button type="button" class="predict-btn" onclick="makePrediction()">Predict</button>
                </form>
                <div id="result" class="result"></div>
            </div>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html)

@monitoring_app.get("/drift_analysis")
async def drift_analysis():
    return monitor.detect_data_drift()

@monitoring_app.get("/prediction_analysis")
async def prediction_analysis():
    return monitor.analyze_predictions()

@monitoring_app.post("/predict")
async def predict_calories(input_data: PredictionInput):
    result = monitor.make_prediction(input_data)
    if result:
        return result
    else:
        raise HTTPException(status_code=500, detail="Prediction failed")

@monitoring_app.get("/run_validation")
async def run_validation():
    return monitor.run_validation_test()

@monitoring_app.get("/modify_test")
async def modify_test():
    return monitor.modify_and_test()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(monitoring_app, host="127.0.0.1", port=8001, log_level="warning")