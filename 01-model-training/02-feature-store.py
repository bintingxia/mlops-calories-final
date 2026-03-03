# ============================================================================
# FEATURE STORE VERSIONING - CONFIGURATION
# ============================================================================

import pandas as pd
import numpy as np
import boto3
import json
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import pickle

# S3 Configuration
S3_BUCKET = "mlops-calories"
FEATURE_STORE_PREFIX = "feature_store"

# Feature version configurations
FEATURE_VERSIONS = {
    "v1": {
        "name": "raw_features",
        "description": "Raw features with one-hot encoded categorical variables",
        "features": {
            "categorical": ["Gender"],  # Will be encoded to Gender_Male
            "numerical": ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
        },
        "target": "Calories"
    },
    "v2": {
        "name": "standardized_features",
        "description": "Standardized numerical features with saved transformation parameters",
        "features": {
            "categorical": ["Gender_Male"],  # Already encoded in v1
            "numerical": ["Age_std", "Height_std", "Weight_std", "Duration_std",
                          "Heart_Rate_std", "Body_Temp_std"]
        },
        "base_numerical": ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"],
        "target": "Calories"
    }
}

print("=" * 80)
print("🔧 FEATURE STORE VERSIONING - INITIALIZED")
print("=" * 80)
print(f"\n📦 S3 Bucket: {S3_BUCKET}")
print(f"📁 Feature Store Prefix: {FEATURE_STORE_PREFIX}")
print(f"\n✅ Available Feature Versions: {list(FEATURE_VERSIONS.keys())}")

# ============================================================================
# FEATURE ENGINEERING PIPELINE
# ============================================================================

class FeatureEngineer:
    """
    Feature engineering pipeline with versioning support
    Supports v1 (raw) and v2 (standardized) features
    """

    def __init__(self, version="v1", s3_bucket=S3_BUCKET):
        """
        Initialize Feature Engineer

        Parameters:
        -----------
        version : str
            Feature version ('v1' or 'v2')
        s3_bucket : str
            S3 bucket for storing artifacts
        """
        self.version = version
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client('s3')
        self.config = FEATURE_VERSIONS[version]
        self.scaler = None
        self.scaler_params = None

        print(f"\n🔧 FeatureEngineer initialized: version={version}")
        print(f"   Description: {self.config['description']}")

    def create_v1_features(self, df, include_target=True):
        """
        Create v1 (raw) features

        Parameters:
        -----------
        df : pd.DataFrame
            Raw dataframe
        include_target : bool
            Whether to include target variable

        Returns:
        --------
        pd.DataFrame : v1 features
        """
        print("\n📊 Creating v1 (raw) features...")

        df_v1 = df.copy()

        # One-hot encode Gender
        if 'Gender' in df_v1.columns:
            df_v1['Gender_Male'] = (df_v1['Gender'] == 'male').astype(int)
            df_v1 = df_v1.drop('Gender', axis=1)

        # Select features
        feature_cols = ['Gender_Male'] + self.config['features']['numerical']

        if include_target and self.config['target'] in df_v1.columns:
            feature_cols.append(self.config['target'])

        df_v1 = df_v1[feature_cols]

        print(f"   ✅ Created v1 features with {len(feature_cols)} columns")
        print(f"   Columns: {list(df_v1.columns)}")

        return df_v1

    def create_v2_features(self, df, fit=True, include_target=True):
        """
        Create v2 (standardized) features

        Parameters:
        -----------
        df : pd.DataFrame
            Raw dataframe or v1 features
        fit : bool
            Whether to fit scaler (True for training, False for inference)
        include_target : bool
            Whether to include target variable

        Returns:
        --------
        pd.DataFrame : v2 features
        """
        print(f"\n📊 Creating v2 (standardized) features... (fit={fit})")

        # First create v1 features if not already done
        if 'Gender_Male' not in df.columns:
            df = self.create_v1_features(df, include_target=True)

        df_v2 = df.copy()

        # Get numerical columns to standardize
        numerical_cols = FEATURE_VERSIONS['v2']['base_numerical']

        if fit:
            # Fit scaler on training data
            self.scaler = StandardScaler()
            scaled_values = self.scaler.fit_transform(df_v2[numerical_cols])

            # Save scaler parameters
            self.scaler_params = {
                'mean': self.scaler.mean_.tolist(),
                'scale': self.scaler.scale_.tolist(),
                'var': self.scaler.var_.tolist(),
                'n_samples_seen': int(self.scaler.n_samples_seen_),
                'feature_names': numerical_cols
            }

            print(f"   ✅ Scaler fitted on {len(numerical_cols)} numerical features")
        else:
            # Use existing scaler or load from S3
            if self.scaler is None:
                print("   ⚠️  No scaler found. Loading from S3...")
                self.load_scaler()

            scaled_values = self.scaler.transform(df_v2[numerical_cols])
            print(f"   ✅ Applied existing scaler")

        # Create standardized columns
        for i, col in enumerate(numerical_cols):
            df_v2[f"{col}_std"] = scaled_values[:, i]

        # Select final features
        feature_cols = ['Gender_Male'] + [f"{col}_std" for col in numerical_cols]

        if include_target and self.config['target'] in df_v2.columns:
            feature_cols.append(self.config['target'])

        df_v2 = df_v2[feature_cols]

        print(f"   ✅ Created v2 features with {len(feature_cols)} columns")
        print(f"   Columns: {list(df_v2.columns)}")

        return df_v2

    def process_features(self, df, fit=True, include_target=True):
        """
        Process features based on version

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        fit : bool
            Whether to fit transformations (for v2)
        include_target : bool
            Whether to include target variable

        Returns:
        --------
        pd.DataFrame : Processed features
        """
        if self.version == "v1":
            return self.create_v1_features(df, include_target=include_target)
        elif self.version == "v2":
            return self.create_v2_features(df, fit=fit, include_target=include_target)
        else:
            raise ValueError(f"Unsupported version: {self.version}")

    def save_to_s3(self, df, split="train", data_version="v1.0"):
        """
        Save features to S3 with versioning

        Parameters:
        -----------
        df : pd.DataFrame
            Feature dataframe
        split : str
            Data split ('train', 'test', 'validation')
        data_version : str
            Data version (e.g., 'v1.0')
        """
        # Create S3 path
        s3_key = f"{FEATURE_STORE_PREFIX}/{self.version}/{data_version}/{split}.csv"
        s3_path = f"s3://{self.s3_bucket}/{s3_key}"

        print(f"\n💾 Saving {self.version} {split} features to S3...")

        # Save to S3
        csv_buffer = df.to_csv(index=False)
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=s3_key,
            Body=csv_buffer
        )

        print(f"   ✅ Saved to: {s3_path}")

        return s3_path

    def save_scaler(self, data_version="v1.0"):
        """
        Save scaler and parameters to S3

        Parameters:
        -----------
        data_version : str
            Data version (e.g., 'v1.0')
        """
        if self.version != "v2" or self.scaler is None:
            print("   ⚠️  No scaler to save (only for v2)")
            return

        print(f"\n💾 Saving scaler parameters to S3...")

        # Save scaler parameters as JSON
        params_key = f"{FEATURE_STORE_PREFIX}/{self.version}/{data_version}/scaler_params.json"
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=params_key,
            Body=json.dumps(self.scaler_params, indent=2)
        )

        print(f"   ✅ Saved scaler params to: s3://{self.s3_bucket}/{params_key}")

        # Save scaler object as pickle
        scaler_key = f"{FEATURE_STORE_PREFIX}/{self.version}/{data_version}/scaler.pkl"
        scaler_bytes = pickle.dumps(self.scaler)
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=scaler_key,
            Body=scaler_bytes
        )

        print(f"   ✅ Saved scaler object to: s3://{self.s3_bucket}/{scaler_key}")

    def load_scaler(self, data_version="v1.0"):
        """
        Load scaler from S3

        Parameters:
        -----------
        data_version : str
            Data version (e.g., 'v1.0')
        """
        if self.version != "v2":
            print("   ⚠️  Scaler only available for v2")
            return

        print(f"\n📥 Loading scaler from S3...")

        try:
            # Load scaler object
            scaler_key = f"{FEATURE_STORE_PREFIX}/{self.version}/{data_version}/scaler.pkl"
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=scaler_key)
            self.scaler = pickle.loads(response['Body'].read())

            # Load scaler parameters
            params_key = f"{FEATURE_STORE_PREFIX}/{self.version}/{data_version}/scaler_params.json"
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=params_key)
            self.scaler_params = json.loads(response['Body'].read())

            print(f"   ✅ Loaded scaler successfully")

        except Exception as e:
            print(f"   ❌ Error loading scaler: {str(e)}")
            raise

    def load_from_s3(self, split="train", data_version="v1.0"):
        """
        Load features from S3

        Parameters:
        -----------
        split : str
            Data split ('train', 'test', 'validation')
        data_version : str
            Data version (e.g., 'v1.0')

        Returns:
        --------
        pd.DataFrame : Feature dataframe
        """
        s3_key = f"{FEATURE_STORE_PREFIX}/{self.version}/{data_version}/{split}.csv"
        s3_path = f"s3://{self.s3_bucket}/{s3_key}"

        print(f"\n📥 Loading {self.version} {split} features from S3...")
        print(f"   Path: {s3_path}")

        # Load from S3
        df = pd.read_csv(s3_path)

        print(f"   ✅ Loaded {len(df)} rows, {len(df.columns)} columns")

        return df

    def get_feature_names(self):
        """Get feature names for the current version"""
        if self.version == "v1":
            return ['Gender_Male'] + self.config['features']['numerical']
        elif self.version == "v2":
            return self.config['features']['categorical'] + self.config['features']['numerical']

# Create instances
print("\n" + "=" * 80)
print("📦 CREATING FEATURE ENGINEER INSTANCES")
print("=" * 80)

fe_v1 = FeatureEngineer(version="v1")
fe_v2 = FeatureEngineer(version="v2")

print("\n✅ Feature engineers ready for use!")

# ============================================================================
# EXAMPLE USAGE - FEATURE STORE VERSIONING WORKFLOW
# ============================================================================

print("=" * 80)
print("📖 EXAMPLE USAGE GUIDE")
print("=" * 80)

example_code = """
# ============================================================================
# WORKFLOW 1: Create v1 Features (Raw Features)
# ============================================================================

# Load your raw data
raw_df = pd.read_csv('s3://mlops-calories/data_versions/v1.0/processed/train.csv')

# Initialize v1 feature engineer
fe_v1 = FeatureEngineer(version="v1")

# Process features (one-hot encode Gender, keep numerical features raw)
train_v1 = fe_v1.process_features(raw_df, include_target=True)

# Save to Feature Store
fe_v1.save_to_s3(train_v1, split="train", data_version="v1.0")

# ============================================================================
# WORKFLOW 2: Create v2 Features (Standardized Features)
# ============================================================================

# Load your raw data
raw_df = pd.read_csv('s3://mlops-calories/data_versions/v1.0/processed/train.csv')

# Initialize v2 feature engineer
fe_v2 = FeatureEngineer(version="v2")

# Process TRAINING features (fit=True to fit scaler)
train_v2 = fe_v2.process_features(raw_df, fit=True, include_target=True)

# Save training features
fe_v2.save_to_s3(train_v2, split="train", data_version="v1.0")

# Save scaler parameters (IMPORTANT for reproducibility!)
fe_v2.save_scaler(data_version="v1.0")

# Process TEST features (fit=False to use existing scaler)
test_df = pd.read_csv('s3://mlops-calories/data_versions/v1.0/processed/test.csv')
test_v2 = fe_v2.process_features(test_df, fit=False, include_target=True)

# Save test features
fe_v2.save_to_s3(test_v2, split="test", data_version="v1.0")

# ============================================================================
# WORKFLOW 3: Load Features for Training
# ============================================================================

# Load v1 features for training
fe_v1 = FeatureEngineer(version="v1")
train_v1 = fe_v1.load_from_s3(split="train", data_version="v1.0")

# OR load v2 features for training
fe_v2 = FeatureEngineer(version="v2")
train_v2 = fe_v2.load_from_s3(split="train", data_version="v1.0")

# Split features and target
X_train = train_v1.drop('Calories', axis=1)
y_train = train_v1['Calories']

# ============================================================================
# WORKFLOW 4: Inference with v2 Features
# ============================================================================

# Initialize v2 feature engineer
fe_v2 = FeatureEngineer(version="v2")

# Load scaler from S3
fe_v2.load_scaler(data_version="v1.0")

# New data for inference (raw format)
new_data = pd.DataFrame([{
    'Gender': 'male',
    'Age': 30,
    'Height': 180,
    'Weight': 75,
    'Duration': 30,
    'Heart_Rate': 120,
    'Body_Temp': 39.5
}])

# Transform to v2 features
new_features = fe_v2.process_features(new_data, fit=False, include_target=False)

# Now use for prediction...

# ============================================================================
# WORKFLOW 5: Compare Feature Versions
# ============================================================================

# Load both versions
train_v1 = fe_v1.load_from_s3(split="train", data_version="v1.0")
train_v2 = fe_v2.load_from_s3(split="train", data_version="v1.0")

print("v1 Features Sample:")
print(train_v1.head())

print("\\nv2 Features Sample:")
print(train_v2.head())

# Train models with both versions and compare performance!
"""

print("\n" + example_code)

print("\n" + "=" * 80)
print("💡 KEY BENEFITS")
print("=" * 80)
print("""
✅ Version Control: Easy switching between feature versions
✅ Reproducibility: Scaler parameters saved for exact reproduction
✅ Isolation: Each version stored separately in S3
✅ Flexibility: Can train/compare models with different feature sets
✅ Production Ready: Load scaler for inference with v2 features
✅ Audit Trail: Track which features were used for each model
""")

print("=" * 80)
print("🎯 NEXT STEPS")
print("=" * 80)
print("""
1. Load your existing raw data
2. Create v1 features (one-hot encoded)
3. Create v2 features (standardized)
4. Train Autopilot with v1 (your current approach)
5. Optionally train another model with v2 to compare
6. Use appropriate feature engineer for inference
""")
