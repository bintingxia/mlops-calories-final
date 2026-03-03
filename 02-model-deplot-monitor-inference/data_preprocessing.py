import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

class DataPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        print(f"Original data shape: {data.shape}")
        print(f"Original data columns: {list(data.columns)}")
        
        if 'Calories' in data.columns:
            calories = data['Calories']
            print(f"Calories statistics:")
            print(f"  Mean: {calories.mean():.2f}")
            print(f"  Median: {calories.median():.2f}")
            print(f"  Std: {calories.std():.2f}")
            print(f"  Min: {calories.min():.2f}")
            print(f"  Max: {calories.max():.2f}")
        
        return data
    
    def preprocess_data(self, df):
        print("\nStarting data preprocessing...")
        
        data = df.copy()
        
        if 'User_ID' in data.columns:
            data = data.drop('User_ID', axis=1)
            print("Removed User_ID column")
        
        print(f"Gender distribution: {data['Gender'].value_counts().to_dict()}")
        
        data['Gender'] = self.label_encoder.fit_transform(data['Gender'])
        print(f"Gender encoded: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        
        X = data.drop('Calories', axis=1)
        y = data['Calories']
        
        print(f"Feature columns: {list(X.columns)}")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target variable shape: {y.shape}")
        
        print("\nFeature statistics (before standardization):")
        for col in X.columns:
            print(f"  {col}: mean={X[col].mean():.2f}, std={X[col].std():.2f}")
        
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        print("\nFeature statistics (after standardization):")
        for col in X_scaled.columns:
            print(f"  {col}: mean={X_scaled[col].mean():.4f}, std={X_scaled[col].std():.4f}")
        
        print(f"\nTarget variable statistics (not standardized):")
        print(f"  Mean: {y.mean():.2f}")
        print(f"  Std: {y.std():.2f}")
        print(f"  Range: [{y.min():.2f}, {y.max():.2f}]")
        
        return X_scaled, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\nData split results:")
        print(f"Training features: {X_train.shape}")
        print(f"Test features: {X_test.shape}")
        print(f"Training targets: {y_train.shape}")
        print(f"Test targets: {y_test.shape}")
        
        print(f"\nTraining set target statistics:")
        print(f"  Mean: {y_train.mean():.2f}")
        print(f"  Std: {y_train.std():.2f}")
        print(f"  Range: [{y_train.min():.2f}, {y_train.max():.2f}]")
        
        print(f"\nTest set target statistics:")
        print(f"  Mean: {y_test.mean():.2f}")
        print(f"  Std: {y_test.std():.2f}")
        print(f"  Range: [{y_test.min():.2f}, {y_test.max():.2f}]")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessors(self, save_dir="models"):
        os.makedirs(save_dir, exist_ok=True)
        
        joblib.dump(self.label_encoder, os.path.join(save_dir, 'label_encoder.pkl'))
        joblib.dump(self.scaler, os.path.join(save_dir, 'scaler.pkl'))
        
        preprocessor_info = {
            'label_classes': self.label_encoder.classes_.tolist(),
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist(),
            'feature_names': ['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
        }
        
        import json
        with open(os.path.join(save_dir, 'preprocessor_info.json'), 'w') as f:
            json.dump(preprocessor_info, f, indent=2)
        
        print(f"\nPreprocessors saved to {save_dir}/")
        print(f"Label classes: {self.label_encoder.classes_}")
        print(f"Standardization mean: {self.scaler.mean_}")
        print(f"Standardization std: {self.scaler.scale_}")
    
    def load_preprocessors(self, save_dir="models"):
        self.label_encoder = joblib.load(os.path.join(save_dir, 'label_encoder.pkl'))
        self.scaler = joblib.load(os.path.join(save_dir, 'scaler.pkl'))
        
        print(f"Preprocessors loaded from {save_dir}/")

def prepare_data():
    print("="*60)
    print("Starting data preprocessing and splitting")
    print("="*60)
    
    preprocessor = DataPreprocessor()
    data = preprocessor.load_data('data.csv')
    
    print(f"\nData quality check:")
    print(f"Missing values: {data.isnull().sum().sum()}")
    print(f"Duplicate rows: {data.duplicated().sum()}")
    
    X, y = preprocessor.preprocess_data(data)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    preprocessor.save_preprocessors()
    
    print("\nSaving data files...")
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)
    
    print("\nValidating saved data:")
    X_train_loaded = pd.read_csv('X_train.csv')
    y_train_loaded = pd.read_csv('y_train.csv').values.ravel()
    
    print(f"Loaded training features shape: {X_train_loaded.shape}")
    print(f"Loaded training targets shape: {y_train_loaded.shape}")
    print(f"Loaded target variable range: [{y_train_loaded.min():.2f}, {y_train_loaded.max():.2f}]")
    
    print("\nData preprocessing completed!")
    
    return X_train, X_test, y_train, y_test

def main():
    prepare_data()

if __name__ == "__main__":
    main()