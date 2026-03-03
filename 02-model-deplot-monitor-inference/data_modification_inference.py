import pandas as pd
import numpy as np
import requests
import json
import time

class DataModificationInference:
    def __init__(self, api_url="http://127.0.0.1:8000", monitoring_url="http://127.0.0.1:8001"):
        self.api_url = api_url
        self.monitoring_url = monitoring_url
        self.original_predictions = []
        self.modified_predictions = []
        self.original_actuals = []
        self.modified_actuals = []
        
    def load_test_data(self):
        X_test = pd.read_csv('X_test.csv')
        y_test = pd.read_csv('y_test.csv')
        return X_test, y_test.values.ravel()
    
    def modify_features(self, X_test, modification_type="random"):
        X_modified = X_test.copy()
        modifications_applied = []
        
        if modification_type == "random":
            features_to_modify = np.random.choice(X_test.columns, 2, replace=False)
            
            for feature in features_to_modify:
                noise_std = X_test[feature].std() * 0.5
                noise = np.random.normal(0, noise_std, len(X_test))
                X_modified[feature] = X_test[feature] + noise
                modifications_applied.append(f"Feature {feature}: added noise (std={noise_std:.2f})")
                
        elif modification_type == "swap":
            features_to_swap = np.random.choice(X_test.columns, 2, replace=False)
            feature1, feature2 = features_to_swap
            
            X_modified[feature1] = X_test[feature2].values
            X_modified[feature2] = X_test[feature1].values
            modifications_applied.append(f"Swapped features {feature1} and {feature2} values")
            
        elif modification_type == "shift":
            features_to_modify = np.random.choice(X_test.columns, 2, replace=False)
            
            for feature in features_to_modify:
                shift_amount = X_test[feature].mean() * 0.3
                X_modified[feature] = X_test[feature] + shift_amount
                modifications_applied.append(f"Feature {feature}: systematic shift +{shift_amount:.2f}")
        
        return X_modified, modifications_applied
    
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
    
    def run_inference_batch(self, X_data, y_data, batch_name, sample_size=25):
        print(f"Running {batch_name} data inference, sample size: {sample_size}")
        
        indices = np.random.choice(len(X_data), min(sample_size, len(X_data)), replace=False)
        
        predictions = []
        actuals = []
        failed_count = 0
        
        for i, idx in enumerate(indices):
            if i % 5 == 0:
                print(f"Progress: {i+1}/{len(indices)}")
            
            input_data = self.prepare_prediction_input(X_data.iloc[idx])
            prediction_result = self.make_prediction(input_data)
            
            if prediction_result:
                predictions.append(prediction_result['predicted_calories'])
                actuals.append(y_data[idx])
            else:
                failed_count += 1
        
        print(f"{batch_name} inference completed: {len(predictions)} successful, {failed_count} failed")
        
        return predictions, actuals
    
    def compare_predictions(self, original_preds, modified_preds):
        results = {
            'original_count': len(original_preds),
            'modified_count': len(modified_preds),
            'original_mean': float(np.mean(original_preds)) if original_preds else 0,
            'modified_mean': float(np.mean(modified_preds)) if modified_preds else 0,
            'original_std': float(np.std(original_preds)) if original_preds else 0,
            'modified_std': float(np.std(modified_preds)) if modified_preds else 0
        }
        
        if original_preds and modified_preds:
            mean_diff = results['modified_mean'] - results['original_mean']
            results['mean_difference'] = float(mean_diff)
            results['mean_difference_percent'] = float((mean_diff / results['original_mean']) * 100) if results['original_mean'] != 0 else 0
        
        return results
    
    def check_monitoring_system_changes(self):
        print("Waiting for monitoring system update...")
        time.sleep(2)
        
        print("Checking monitoring system changes:")
        try:
            response = requests.get(f"{self.monitoring_url}/", timeout=5)
            if response.status_code == 200:
                print("  Monitoring system accessible")
            else:
                print(f"  Monitoring system access failed: {response.status_code}")
        except Exception as e:
            print(f"  Monitoring system access failed: {str(e)}")
    
    def save_modification_results(self, modifications, comparison_results):
        results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'modifications_applied': modifications,
            'comparison_results': comparison_results,
            'original_predictions_sample': self.original_predictions[:10],
            'modified_predictions_sample': self.modified_predictions[:10]
        }
        
        with open('modification_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("Modification results saved to modification_results.json")

def main():
    print("Data Modification and Re-inference Validation")
    print("="*40)
    
    modifier = DataModificationInference()
    
    print("Loading original test data...")
    X_test, y_test = modifier.load_test_data()
    print(f"Original test data shape: {X_test.shape}")
    
    original_preds, original_actuals = modifier.run_inference_batch(X_test, y_test, "original", sample_size=25)
    
    print("Modifying test data features...")
    modification_types = ["random", "swap", "shift"]
    selected_modification = np.random.choice(modification_types)
    print(f"Selected modification type: {selected_modification}")
    
    X_modified, modifications = modifier.modify_features(X_test, modification_type=selected_modification)
    print("Applied modifications:")
    for mod in modifications:
        print(f"  {mod}")
    
    X_modified.to_csv('X_test.csv', index=False)
    print("Modified data saved as X_test.csv")
    
    modified_preds, modified_actuals = modifier.run_inference_batch(X_modified, y_test, "modified", sample_size=25)
    
    modifier.original_predictions = original_preds
    modifier.modified_predictions = modified_preds
    modifier.original_actuals = original_actuals
    modifier.modified_actuals = modified_actuals
    
    comparison_results = modifier.compare_predictions(original_preds, modified_preds)
    
    print("Prediction results comparison:")
    if comparison_results['original_count'] > 0 and comparison_results['modified_count'] > 0:
        print(f"  Original average: {comparison_results['original_mean']:.2f}")
        print(f"  Modified average: {comparison_results['modified_mean']:.2f}")
        print(f"  Difference: {comparison_results['mean_difference']:.2f} ({comparison_results['mean_difference_percent']:.1f}%)")
    else:
        print("  Insufficient data for comparison")
    
    modifier.check_monitoring_system_changes()
    modifier.save_modification_results(modifications, comparison_results)
    
    print("Data modification and re-inference validation completed")

if __name__ == "__main__":
    main()