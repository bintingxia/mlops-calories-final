import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import json
from sklearn.dummy import DummyRegressor

class MLflowTrainer:
    def __init__(self, experiment_name="calories_prediction"):
        self.experiment_name = experiment_name
        self.models = {
            'DummyRegressor': DummyRegressor(strategy='mean'),
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(random_state=42),
            'Lasso': Lasso(random_state=42),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0)
        }
        
    def calculate_metrics(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        return {
            'mse': float(mse), 
            'mae': float(mae), 
            'r2': float(r2), 
            'rmse': float(rmse)
        }
    
    def validate_data(self, X_train, X_test, y_train, y_test):
        print("\n" + "="*50)
        print("Data Validation")
        print("="*50)
        
        print(f"Training features shape: {X_train.shape}")
        print(f"Test features shape: {X_test.shape}")
        print(f"Training targets shape: {y_train.shape}")
        print(f"Test targets shape: {y_test.shape}")
        
        print(f"\nTraining set target statistics:")
        print(f"  Mean: {y_train.mean():.2f}")
        print(f"  Median: {np.median(y_train):.2f}")
        print(f"  Std: {y_train.std():.2f}")
        print(f"  Min: {y_train.min():.2f}")
        print(f"  Max: {y_train.max():.2f}")
        
        print(f"\nTest set target statistics:")
        print(f"  Mean: {y_test.mean():.2f}")
        print(f"  Median: {np.median(y_test):.2f}")
        print(f"  Std: {y_test.std():.2f}")
        print(f"  Min: {y_test.min():.2f}")
        print(f"  Max: {y_test.max():.2f}")
        
        print(f"\nFeature names: {list(X_train.columns)}")
        
        if y_train.min() < 0 or y_test.min() < 0:
            print("Found negative calorie values, this is unreasonable!")
        
        if y_train.max() > 1000 or y_test.max() > 1000:
            print("Found excessively high calorie values, please check data!")
        
        print(f"\nFeature statistics:")
        for col in X_train.columns:
            train_mean = X_train[col].mean()
            train_std = X_train[col].std()
            print(f"  {col}: mean={train_mean:.4f}, std={train_std:.4f}")
            
            if abs(train_mean) > 0.1:
                print(f"    {col} mean not close to 0, standardization may have issues")
            if abs(train_std - 1.0) > 0.1:
                print(f"    {col} std not close to 1, standardization may have issues")
    
    def train_model(self, model_name, model, X_train, X_test, y_train, y_test):
        print(f"\n{'='*30}")
        print(f"Training model: {model_name}")
        print(f"{'='*30}")
        
        with mlflow.start_run(run_name=model_name):
            try:
                print("Starting training...")
                model.fit(X_train, y_train)
                print("Training completed")
                
                print("Making predictions...")
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                print(f"Training predictions range: [{y_train_pred.min():.2f}, {y_train_pred.max():.2f}]")
                print(f"Test predictions range: [{y_test_pred.min():.2f}, {y_test_pred.max():.2f}]")
                print(f"Training predictions mean: {y_train_pred.mean():.2f}")
                print(f"Test predictions mean: {y_test_pred.mean():.2f}")
                
                if y_test_pred.max() < 10 or y_test_pred.min() > 500:
                    print("Prediction value range abnormal, potential issues!")
                
                train_metrics = self.calculate_metrics(y_train, y_train_pred)
                test_metrics = self.calculate_metrics(y_test, y_test_pred)
                
                print(f"\nTraining performance:")
                for metric, value in train_metrics.items():
                    print(f"  {metric.upper()}: {value:.4f}")
                
                print(f"\nTest performance:")
                for metric, value in test_metrics.items():
                    print(f"  {metric.upper()}: {value:.4f}")
                
                if hasattr(model, 'get_params'):
                    params = model.get_params()
                    for param, value in params.items():
                        mlflow.log_param(param, value)
                
                for metric, value in train_metrics.items():
                    mlflow.log_metric(f"train_{metric}", value)
                for metric, value in test_metrics.items():
                    mlflow.log_metric(f"test_{metric}", value)
                
                mlflow.sklearn.log_model(model, "model")
                
                model_dir = f"models/{model_name}"
                os.makedirs(model_dir, exist_ok=True)
                joblib.dump(model, f"{model_dir}/model.pkl")
                
                validation_sample = {
                    'Gender': 1, 'Age': 30.0, 'Height': 175.0, 'Weight': 75.0,
                    'Duration': 30.0, 'Heart_Rate': 120.0, 'Body_Temp': 37.5
                }
                sample_prediction = model.predict(pd.DataFrame([validation_sample]))[0]
                
                with open(f"{model_dir}/validation_sample.json", 'w') as f:
                    json.dump({
                        'input': validation_sample,
                        'prediction': float(sample_prediction)
                    }, f, indent=2)
                
                print(f"{model_name} training completed")
                
                return {
                    'model_name': model_name,
                    'test_metrics': test_metrics,
                    'run_id': mlflow.active_run().info.run_id
                }
                
            except Exception as e:
                print(f"Training failed for {model_name}: {str(e)}")
                return None
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        print("="*50)
        print("Starting MLflow Model Training")
        print("="*50)
        print(f"MLflow experiment name: {self.experiment_name}")
        
        mlflow.set_experiment(self.experiment_name)
        
        print("\nLoading preprocessed data...")
        self.validate_data(X_train, X_test, y_train, y_test)
        
        print(f"\nStarting training for {len(self.models)} models...")
        
        results = []
        
        for model_name, model in self.models.items():
            result = self.train_model(model_name, model, X_train, X_test, y_train, y_test)
            if result:
                results.append(result)
        
        return results
    
    def select_best_model(self, results):
        print("\n" + "="*50)
        print("All Model Results Summary")
        print("="*50)
        
        results_sorted = sorted(results, key=lambda x: x['test_metrics']['r2'], reverse=True)
        
        for result in results_sorted:
            r2_score = result['test_metrics']['r2']
            model_name = result['model_name']
            print(f"{model_name:<20} R² = {r2_score:.4f}")
        
        best_result = results_sorted[0]
        best_model_name = best_result['model_name']
        best_score = best_result['test_metrics']['r2']
        best_run_id = best_result['run_id']
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best R² score: {best_score:.4f}")
        print(f"Run ID: {best_run_id}")
        
        try:
            client = mlflow.tracking.MlflowClient()
            model_version = mlflow.register_model(
                model_uri=f"runs:/{best_run_id}/model",
                name="calories_prediction_model"
            )
            if model_version:
                print("Best model registered to MLflow")
        except Exception as e:
            print(f"Model registration failed: {str(e)}")
        
        best_model_info = {
            'model_name': best_model_name,
            'run_id': best_run_id,
            'r2_score': best_score,
            'all_results': [
                {
                    'model_name': r['model_name'],
                    'r2_score': r['test_metrics']['r2'],
                    'run_id': r['run_id']
                }
                for r in results_sorted
            ]
        }
        
        with open('best_model_info.json', 'w') as f:
            json.dump(best_model_info, f, indent=2)
        
        print("Best model info saved to best_model_info.json")
        
        return best_model_info

def load_preprocessed_data():
    print("Checking data preprocessing...")
    
    required_files = ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"Missing required file: {file}")
            print("Please run data preprocessing first")
            return None, None, None, None
    
    print("Data files exist")
    
    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')
    y_train = pd.read_csv('y_train.csv').values.ravel()
    y_test = pd.read_csv('y_test.csv').values.ravel()
    
    return X_train, X_test, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    if X_train is None:
        print("Failed to load data, exiting")
        return
    
    trainer = MLflowTrainer()
    results = trainer.train_all_models(X_train, X_test, y_train, y_test)
    
    if results:
        best_model_info = trainer.select_best_model(results)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best model: {best_model_info['model_name']}")
        print(f"Run ID: {best_model_info['run_id']}")
        print(f"R² score: {best_model_info['r2_score']:.4f}")
        print("="*60)
    else:
        print("No models trained successfully")

if __name__ == "__main__":
    main()