import os
import sys
import subprocess
import time
import json
import signal
import requests

def print_banner():
    print("MLOps Calories Prediction System")
    print("=" * 40)

def check_dependencies():
    print("Checking dependencies...")
    try:
        import pandas, numpy, sklearn, mlflow, fastapi, uvicorn, matplotlib, seaborn
        print("All dependencies installed")
        return True
    except ImportError as e:
        print(f"Missing dependency: {str(e)}")
        return False

def run_eda():
    print("\nRunning exploratory data analysis...")
    try:
        from eda_analysis import main as eda_main
        eda_main()
        print("EDA completed")
        return True
    except Exception as e:
        print(f"EDA failed: {str(e)}")
        return False

def run_data_preprocessing():
    print("\nData preprocessing...")
    try:
        from data_preprocessing import prepare_data
        prepare_data()
        print("Data preprocessing completed")
        return True
    except Exception as e:
        print(f"Data preprocessing failed: {str(e)}")
        return False

def run_model_training():
    print("\nModel training...")
    try:
        from mlflow_training import main as train_main
        train_main()
        print("Model training completed")
        return True
    except Exception as e:
        print(f"Model training failed: {str(e)}")
        return False

def start_mlflow_ui():
    try:
        mlflow_process = subprocess.Popen([
            "mlflow", "ui", "--host", "127.0.0.1", "--port", "5005"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        print("MLflow UI started: http://127.0.0.1:5005")
        return mlflow_process
    except Exception as e:
        print(f"MLflow UI startup failed: {str(e)}")
        return None

def start_model_deployment():
    try:
        deployment_process = subprocess.Popen([
            "python", "-c", 
            """
import sys
sys.path.append('.')
from deployment import app
import uvicorn
uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
"""
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(12)
        print("Model API started: http://127.0.0.1:8000")
        return deployment_process
    except Exception as e:
        print(f"Model deployment failed: {str(e)}")
        return None

def start_monitoring_service():
    try:
        monitoring_process = subprocess.Popen([
            "python", "-c",
            """
import sys
sys.path.append('.')
from monitoring import monitoring_app
import uvicorn
uvicorn.run(monitoring_app, host="127.0.0.1", port=8001, log_level="warning")
"""
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)
        print("Monitoring service started: http://127.0.0.1:8001")
        return monitoring_process
    except Exception as e:
        print(f"Monitoring service failed: {str(e)}")
        return None

def test_api_endpoints():
    print("\nTesting API endpoints...")
    
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"Model API: OK ({health_data.get('model_name')})")
        else:
            print(f"Model API: Error ({response.status_code})")
            return False
    except Exception as e:
        print(f"Model API: Offline ({str(e)})")
        return False
    
    try:
        test_data = {
            "Gender": 1, "Age": 30.0, "Height": 175.0, "Weight": 75.0,
            "Duration": 30.0, "Heart_Rate": 120.0, "Body_Temp": 37.5
        }
        response = requests.post("http://127.0.0.1:8000/predict", json=test_data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            prediction = result['predicted_calories']
            print(f"Prediction test: {prediction:.1f} calories")
            if 20 <= prediction <= 400:
                print("Prediction value looks reasonable")
            else:
                print(f"Warning: Prediction value {prediction:.1f} may be unreasonable")
        else:
            print(f"Prediction test failed: {response.status_code}")
    except Exception as e:
        print(f"Prediction test failed: {str(e)}")
    
    try:
        response = requests.get("http://127.0.0.1:8001/", timeout=5)
        if response.status_code == 200:
            print("Monitoring service: OK")
        else:
            print(f"Monitoring service: Error ({response.status_code})")
    except Exception as e:
        print(f"Monitoring service: Offline ({str(e)})")
    
    return True

def run_inference_validation():
    print("\nRunning inference validation...")
    try:
        from inference_validation import main as inference_main
        inference_main()
        print("Inference validation completed")
        return True
    except Exception as e:
        print(f"Inference validation failed: {str(e)}")
        return False

def run_data_modification_inference():
    print("\nRunning data modification test...")
    try:
        from data_modification_inference import main as mod_main
        mod_main()
        print("Data modification test completed")
        return True
    except Exception as e:
        print(f"Data modification test failed: {str(e)}")
        return False

def validate_results():
    print("\nValidating results...")
    
    if os.path.exists('best_model_info.json'):
        try:
            with open('best_model_info.json', 'r') as f:
                model_info = json.load(f)
            training_r2 = model_info.get('r2_score', 0)
            print(f"Training R² score: {training_r2:.4f}")
            print(f"Best model: {model_info.get('model_name', 'Unknown')}")
        except:
            print("Could not read model info")
    
    if os.path.exists('validation_results.json'):
        try:
            with open('validation_results.json', 'r') as f:
                validation_data = json.load(f)
            metrics = validation_data.get('metrics', {})
            inference_r2 = metrics.get('r2', 0)
            print(f"Inference R² score: {inference_r2:.4f}")
            
            if os.path.exists('best_model_info.json'):
                with open('best_model_info.json', 'r') as f:
                    model_info = json.load(f)
                training_r2 = model_info.get('r2_score', 0)
                diff = abs(training_r2 - inference_r2)
                if diff < 0.1:
                    print("Training and inference results are consistent")
                else:
                    print(f"Training vs inference difference: {diff:.4f}")
        except:
            print("Could not read validation results")

def print_summary():
    print("\n" + "=" * 40)
    print("MLOps Pipeline Summary")
    print("=" * 40)
    
    print("\nService endpoints:")
    print("- MLflow UI: http://127.0.0.1:5005")
    print("- Model API: http://127.0.0.1:8000")
    print("- API Docs: http://127.0.0.1:8000/docs")
    print("- Monitoring: http://127.0.0.1:8001")
    
    
    if os.path.exists('best_model_info.json'):
        try:
            with open('best_model_info.json', 'r') as f:
                model_info = json.load(f)
            print(f"\nBest model: {model_info['model_name']}")
            print(f"Training R² score: {model_info['r2_score']:.4f}")
        except:
            print("\nCould not read model metrics")

def cleanup_processes(processes):
    print("\nShutting down services...")
    for process in processes:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    print("All services stopped")

def signal_handler(processes):
    print("\nReceived interrupt signal, cleaning up...")
    cleanup_processes(processes)
    sys.exit(0)

def main():
    print_banner()
    
    if not check_dependencies():
        return
    
    processes = []
    
    try:
        if not run_eda():
            print("EDA failed, but continuing...")
        
        if not run_data_preprocessing():
            print("Data preprocessing failed, exiting")
            return
        
        if not run_model_training():
            print("Model training failed, exiting")
            return
        
        mlflow_process = start_mlflow_ui()
        if mlflow_process:
            processes.append(mlflow_process)
        
        deployment_process = start_model_deployment()
        if deployment_process:
            processes.append(deployment_process)
        else:
            print("Model deployment failed, exiting")
            cleanup_processes(processes)
            return
        
        monitoring_process = start_monitoring_service()
        if monitoring_process:
            processes.append(monitoring_process)
        
        if not test_api_endpoints():
            print("API tests failed, but continuing...")
        
        if not run_inference_validation():
            print("Inference validation failed, but continuing...")
        
        if not run_data_modification_inference():
            print("Data modification test failed, but continuing...")
        
        validate_results()
        print_summary()
        
        print("\n" + "=" * 40)
        print("MLOps pipeline completed")
        print("Services are running, press Ctrl+C to exit")
        print("=" * 40)
        
        signal.signal(signal.SIGINT, lambda s, f: signal_handler(processes))
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
    
    finally:
        cleanup_processes(processes)

if __name__ == "__main__":
    main()