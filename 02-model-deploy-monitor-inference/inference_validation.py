import pandas as pd
import numpy as np
import requests
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class InferenceValidator:
    def __init__(self, api_url="http://127.0.0.1:8000", monitoring_url="http://127.0.0.1:8001"):
        self.api_url = api_url
        self.monitoring_url = monitoring_url
        self.predictions = []
        self.actuals = []
        
    def load_test_data(self):
        try:
            X_test = pd.read_csv('X_test.csv')
            y_test = pd.read_csv('y_test.csv')
            
            print(f"Using test data - target range: [{y_test.min().iloc[0]:.1f}, {y_test.max().iloc[0]:.1f}]")
            print(f"Test feature sample: {X_test.iloc[0].to_dict()}")
            print(f"Test data shape: {X_test.shape}")
            
            return X_test, y_test.values.ravel()
            
        except Exception as e:
            print(f"Failed to load test data: {str(e)}")
            return None, None
    
    def prepare_prediction_input(self, row):
        return {
            "Gender": int(row['Gender']),
            "Age": float(row['Age']),
            "Height": float(row['Height']),
            "Weight": float(row['Weight']),
            "Duration": float(row['Duration']),
            "Heart_Rate": float(row['Heart_Rate']),
            "Body_Temp": float(row['Body_Temp'])
        }
    
    def make_prediction(self, input_data):
        try:
            response = requests.post(f"{self.api_url}/predict", json=input_data, timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def validate_model_performance(self, X_test, y_test, sample_size=50):
        print(f"Validating model performance, sample size: {sample_size}")
        
        indices = np.random.choice(len(X_test), min(sample_size, len(X_test)), replace=False)
        
        predictions = []
        actuals = []
        failed_count = 0
        
        for i, idx in enumerate(indices):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(indices)}")
            
            input_data = self.prepare_prediction_input(X_test.iloc[idx])
            prediction_result = self.make_prediction(input_data)
            
            if prediction_result:
                predictions.append(prediction_result['predicted_calories'])
                actuals.append(y_test[idx])
            else:
                failed_count += 1
        
        self.predictions = predictions
        self.actuals = actuals
        
        print(f"Successful predictions: {len(predictions)}, Failed: {failed_count}")
        
        if len(predictions) > 0:
            metrics = self.calculate_metrics(actuals, predictions)
            self.print_metrics(metrics)
            return metrics
        else:
            print("No successful prediction results")
            return None
    
    def calculate_metrics(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': rmse,
            'samples': len(y_true)
        }
    
    def print_metrics(self, metrics):
        print("Model performance metrics:")
        print(f"  Sample count: {metrics['samples']}")
        print(f"  R² score: {metrics['r2']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.2f}")
        print(f"  MAE: {metrics['mae']:.2f}")
        print(f"  MSE: {metrics['mse']:.2f}")
    
    def check_monitoring_system(self):
        print("Checking monitoring system...")
        
        try:
            response = requests.get(f"{self.monitoring_url}/prediction_analysis", timeout=5)
            if response.status_code == 200:
                analysis = response.json()
                print("Monitoring system running normally")
                print(f"  Total predictions: {analysis.get('total_predictions', 0)}")
                
                pred_stats = analysis.get('prediction_stats', {})
                if pred_stats:
                    print(f"  Prediction mean: {pred_stats.get('mean', 0):.2f}")
                    print(f"  Prediction std: {pred_stats.get('std', 0):.2f}")
            else:
                print(f"Monitoring system access failed: {response.status_code}")
        except Exception as e:
            print(f"Monitoring system check failed: {str(e)}")
        
        try:
            response = requests.get(f"{self.monitoring_url}/drift_analysis", timeout=5)
            if response.status_code == 200:
                drift_result = response.json()
                if drift_result.get('drift_detected'):
                    print("Data drift detected:")
                    for alert in drift_result.get('alerts', [])[:3]:
                        print(f"  {alert}")
                else:
                    print("No data drift detected")
            else:
                print("Data drift check failed")
        except Exception as e:
            print(f"Data drift check failed: {str(e)}")
    
    def save_validation_results(self, metrics):
        results = {
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'metrics': metrics,
            'predictions_count': len(self.predictions),
            'predictions_sample': self.predictions[:10],
            'actuals_sample': self.actuals[:10]
        }
        
        with open('validation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("Validation results saved to validation_results.json")

def main():
    print("Model Inference Validation")
    print("="*30)
    
    validator = InferenceValidator()
    X_test, y_test = validator.load_test_data()
    print(f"Test data shape: {X_test.shape}")
    
    metrics = validator.validate_model_performance(X_test, y_test, sample_size=50)
    
    if metrics:
        validator.check_monitoring_system()
        validator.save_validation_results(metrics)
    
    print("Inference validation completed")

if __name__ == "__main__":
    main()