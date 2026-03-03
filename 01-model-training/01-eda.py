# ============================================================================
# 1. SET ENVIRONMENT VARIABLES FOR SAGEMAKER STUDIO IMPORTS
# ============================================================================

import os
os.environ['DataZoneProjectId'] = 'ahny572cp1dr15'
os.environ['DataZoneDomainId'] = 'dzd-4ixhg0bzrkytw9'
os.environ['DataZoneEnvironmentId'] = '5l48miq9l3wu1l'
os.environ['DataZoneDomainRegion'] = 'ap-northeast-1'

# Create both a function and variable for metadata access
_resource_metadata = None

def _get_resource_metadata():
    global _resource_metadata
    if _resource_metadata is None:
        _resource_metadata = {
            "AdditionalMetadata": {
                "DataZoneProjectId": "ahny572cp1dr15",
                "DataZoneDomainId": "dzd-4ixhg0bzrkytw9",
                "DataZoneEnvironmentId": "5l48miq9l3wu1l",
                "DataZoneDomainRegion": "ap-northeast-1",
            }
        }
    return _resource_metadata

metadata = _get_resource_metadata()

# ============================================================================
# 2. LOGGING CONFIGURATION
# ============================================================================

from typing import Optional

def _set_logging(log_dir: str, log_file: str, log_name: Optional[str] = None):
    import os
    import logging
    from logging.handlers import RotatingFileHandler

    level = logging.INFO
    max_bytes = 5 * 1024 * 1024
    backup_count = 5

    # Fallback to /tmp dir on access, helpful for local dev setup
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        log_dir = "/tmp/kernels/"

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger() if not log_name else logging.getLogger(log_name)
    logger.handlers = []
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Rotating file handler
    fh = RotatingFileHandler(filename=log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f"Logging initialized for {log_name}.")

_set_logging("/var/log/computeEnvironments/kernel/", "kernel.log")
_set_logging("/var/log/studio/data-notebook-kernel-server/", "metrics.log", "metrics")

import logging
from sagemaker_studio import ClientConfig, sqlutils, sparkutils, dataframeutils

logger = logging.getLogger(__name__)
logger.info("Initializing sparkutils")
spark = sparkutils.init()
logger.info("Finished initializing sparkutils")

# ============================================================================
# 3. RESET WORKING DIRECTORY (FIX MOUNT TIMING ISSUE)
# ============================================================================

def _reset_os_path():
    """
    Reset the process's working directory to handle mount timing issues.
    """
    try:
        import os
        import logging
        logger = logging.getLogger(__name__)
        logger.info("---------Before------")
        logger.info("CWD: %s", os.getcwd())
        os.chdir("/home/sagemaker-user")
        logger.info("---------After------")
        logger.info("CWD: %s", os.getcwd())
    except Exception as e:
        logger.exception(f"Failed to reset working directory: {e}")

_reset_os_path()

# ============================================================================
# 4. PROJECT CONFIGURATION (MLOps COMPLIANCE)
# ============================================================================

DATA_VERSION = "v1.0"  # Data version identifier
S3_BUCKET = "mlops-calories"  # S3 bucket name
S3_RAW_FILE_PATH = "data/calories.csv"  # Original raw data path
RANDOM_STATE = 42  # Random seed for reproducibility
TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% test split

# Versioned S3 paths
S3_DATA_VERSIONS_PREFIX = f"data_versions/{DATA_VERSION}"
S3_RAW_PATH = f"{S3_DATA_VERSIONS_PREFIX}/raw"
S3_PROCESSED_PATH = f"{S3_DATA_VERSIONS_PREFIX}/processed"
S3_LATEST_PREFIX = "data_versions/latest"

# ============================================================================
# 5. IMPORT REQUIRED LIBRARIES
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sagemaker
import boto3
from datetime import datetime
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import io

warnings.filterwarnings('ignore')  # Suppress irrelevant warnings

# ============================================================================
# 6. CONFIGURE SAGEMAKER & S3 ENVIRONMENT
# ============================================================================

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
s3_client = boto3.client('s3', region_name='ap-northeast-1')  # Tokyo region

# ============================================================================
# 7. CONFIGURE VISUALIZATION STYLE (ACADEMIC-GRADE)
# ============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
sns.set_palette("Set2")

# ============================================================================
# 8. HELPER FUNCTIONS
# ============================================================================

def load_data_from_s3(s3_path):
    """Load CSV data directly from AWS S3 into a Pandas DataFrame"""
    try:
        if not s3_path.startswith('s3://'):
            raise ValueError("S3 path must start with 's3://'")
        print(f"📥 Loading data from S3: {s3_path}")
        df = pd.read_csv(s3_path)
        print(f"✅ Data loaded successfully - Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Failed to load data from S3: {str(e)}")
        raise

def upload_to_s3(df, s3_key, bucket=S3_BUCKET):
    """Upload DataFrame to S3 as CSV"""
    try:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        s3_client.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=csv_buffer.getvalue()
        )
        full_s3_path = f"s3://{bucket}/{s3_key}"
        print(f"✅ Uploaded to S3: {full_s3_path}")
        print(f"   - Shape: {df.shape}")
        print(f"   - Size: {len(csv_buffer.getvalue()) / 1024:.2f} KB")
        return full_s3_path
    except Exception as e:
        print(f"❌ Failed to upload to S3: {str(e)}")
        raise

def create_soft_link(source_key, link_key, bucket=S3_BUCKET):
    """Create a soft link in S3 to point to latest version"""
    try:
        s3_client.copy_object(
            CopySource={'Bucket': bucket, 'Key': source_key},
            Bucket=bucket,
            Key=link_key
        )
        print(f"✅ Created soft link: s3://{bucket}/{link_key} -> s3://{bucket}/{source_key}")
    except Exception as e:
        print(f"⚠️ Failed to create soft link: {str(e)}")

# ============================================================================
# 9. LOAD DATASET FROM S3
# ============================================================================

FULL_S3_PATH = f"s3://{S3_BUCKET}/{S3_RAW_FILE_PATH}"
df = load_data_from_s3(FULL_S3_PATH)


# ============================================================================
# CALORIES CONSUMPTION PREDICTION - EDA & PREPROCESSING PIPELINE
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================
DATA_VERSION = "v1.0"
RANDOM_STATE = 42
TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% test
TARGET_VARIABLE = "Calories"

# ============================================================================
# VISUALIZATION CONFIGURATION (ACADEMIC-GRADE)
# ============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
sns.set_palette("Set2")

print("="*80)
print("CALORIES CONSUMPTION PREDICTION - EDA & PREPROCESSING")
print("="*80)
print(f"Target Variable: {TARGET_VARIABLE}")
print(f"Data Version: {DATA_VERSION}")
print(f"Random State: {RANDOM_STATE}")


# ============================================================================
# 1. DATA CLEANING & FEATURE SELECTION
# ============================================================================
print("\n" + "="*80)
print("1. DATA CLEANING & FEATURE SELECTION")
print("="*80)

# 1.1 Remove unnecessary columns (User_ID is not useful for prediction)
initial_cols = df.columns.tolist()
df = df.drop('User_ID', axis=1)
print(f"\n✅ Removed User_ID column (non-predictive)")
print(f" - Initial columns: {len(initial_cols)}")
print(f" - Remaining columns: {len(df.columns)}")
print(f" - Columns: {df.columns.tolist()}")

# 1.2 Check for missing values
print(f"\n🔍 Missing Values Check:")
missing_vals = df.isnull().sum()
missing_percent = (missing_vals / len(df)) * 100
missing_df = pd.DataFrame({
    "Missing Count": missing_vals,
    "Missing %": missing_percent.round(2)
})
print(missing_df if missing_vals.sum() > 0 else "✅ No missing values in the dataset")

# 1.3 Check for duplicate rows
duplicate_count = df.duplicated().sum()
if duplicate_count > 0:
    print(f"\n⚠️ Found {duplicate_count} duplicate rows")
    df = df.drop_duplicates()
    print(f"✅ Removed duplicates - Dataset shape: {df.shape}")
else:
    print(f"\n✅ No duplicate rows found")


# ============================================================================
# 2. BASIC DATA INSPECTION
# ============================================================================
print("\n" + "="*80)
print("2. BASIC DATA INSPECTION")
print("="*80)

print(f"\n📏 Dataset Shape (Rows, Columns): {df.shape}")

print(f"\n📋 Data Types:")
print(df.dtypes)

print(f"\n📊 Descriptive Statistics:")
print(df.describe().round(2))

print(f"\n🔤 Unique Values:")
for col in df.columns:
    unique_count = df[col].nunique()
    if unique_count <= 10:  # Categorical or low-cardinality
        print(f" - {col}: {unique_count} unique values | {df[col].unique().tolist()}")
    else:  # Numerical
        print(f" - {col}: {unique_count} unique values | Range: [{df[col].min():.2f}, {df[col].max():.2f}]")


# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "="*80)
print("3. EXPLORATORY DATA ANALYSIS (ALL VARIABLES)")
print("="*80)

# --------------------------
# 3.1 Univariate Analysis
# --------------------------
print("\n3.1 Univariate Analysis")

# 3.1.1 Target Variable: Calories
plt.figure(figsize=(10, 6))
sns.histplot(df['Calories'], kde=True, bins=30, color='teal')
plt.title('Distribution of Calories Burned')
plt.xlabel('Calories')
plt.ylabel('Frequency')
plt.savefig('calories_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("- Generated: calories_distribution.png")

# 3.1.2 Gender (Categorical)
plt.figure(figsize=(8, 6))
gender_counts = df['Gender'].value_counts()
sns.barplot(x=gender_counts.index, y=gender_counts.values, palette='pastel')
plt.title('Distribution of Gender')
plt.xlabel('Gender')
plt.ylabel('Number of Samples')
plt.savefig('gender_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("- Generated: gender_distribution.png")

# 3.1.3 Age
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True, bins=20, color='coral')
plt.title('Distribution of Age')
plt.xlabel('Age (Years)')
plt.ylabel('Frequency')
plt.savefig('age_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("- Generated: age_distribution.png")

# 3.1.4 Height
plt.figure(figsize=(10, 6))
sns.histplot(df['Height'], kde=True, bins=20, color='skyblue')
plt.title('Distribution of Height')
plt.xlabel('Height (cm)')
plt.ylabel('Frequency')
plt.savefig('height_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("- Generated: height_distribution.png")

# 3.1.5 Weight
plt.figure(figsize=(10, 6))
sns.histplot(df['Weight'], kde=True, bins=20, color='lightgreen')
plt.title('Distribution of Weight')
plt.xlabel('Weight (kg)')
plt.ylabel('Frequency')
plt.savefig('weight_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("- Generated: weight_distribution.png")

# 3.1.6 Duration (Exercise Duration)
plt.figure(figsize=(10, 6))
sns.histplot(df['Duration'], kde=True, bins=20, color='orange')
plt.title('Distribution of Exercise Duration')
plt.xlabel('Duration (minutes)')
plt.ylabel('Frequency')
plt.savefig('duration_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("- Generated: duration_distribution.png")

# 3.1.7 Heart Rate
plt.figure(figsize=(10, 6))
sns.histplot(df['Heart_Rate'], kde=True, bins=20, color='red')
plt.title('Distribution of Heart Rate')
plt.xlabel('Heart Rate (bpm)')
plt.ylabel('Frequency')
plt.savefig('heart_rate_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("- Generated: heart_rate_distribution.png")

# 3.1.8 Body Temperature
plt.figure(figsize=(10, 6))
sns.histplot(df['Body_Temp'], kde=True, bins=20, color='purple')
plt.title('Distribution of Body Temperature')
plt.xlabel('Body Temperature (°C)')
plt.ylabel('Frequency')
plt.savefig('body_temp_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("- Generated: body_temp_distribution.png")

# --------------------------
# 3.2 Bivariate Analysis (vs Target Variable: Calories)
# --------------------------
print("\n3.2 Bivariate Analysis (All Variables vs Calories)")

# 3.2.1 Gender vs Calories
plt.figure(figsize=(8, 6))
sns.boxplot(x='Gender', y='Calories', data=df, palette='Set2')
plt.title('Gender vs Calories Burned')
plt.xlabel('Gender')
plt.ylabel('Calories')
plt.savefig('gender_vs_calories.png', dpi=300, bbox_inches='tight')
plt.show()
print("- Generated: gender_vs_calories.png")

# 3.2.2 Age vs Calories
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Calories', data=df, hue='Gender', alpha=0.6)
plt.title('Age vs Calories Burned')
plt.xlabel('Age (Years)')
plt.ylabel('Calories')
plt.legend(title='Gender')
plt.savefig('age_vs_calories.png', dpi=300, bbox_inches='tight')
plt.show()
print("- Generated: age_vs_calories.png")

# 3.2.3 Height vs Calories
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Height', y='Calories', data=df, hue='Gender', alpha=0.6)
plt.title('Height vs Calories Burned')
plt.xlabel('Height (cm)')
plt.ylabel('Calories')
plt.legend(title='Gender')
plt.savefig('height_vs_calories.png', dpi=300, bbox_inches='tight')
plt.show()
print("- Generated: height_vs_calories.png")

# 3.2.4 Weight vs Calories
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Weight', y='Calories', data=df, hue='Gender', alpha=0.6)
plt.title('Weight vs Calories Burned')
plt.xlabel('Weight (kg)')
plt.ylabel('Calories')
plt.legend(title='Gender')
plt.savefig('weight_vs_calories.png', dpi=300, bbox_inches='tight')
plt.show()
print("- Generated: weight_vs_calories.png")

# 3.2.5 Duration vs Calories
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Duration', y='Calories', data=df, hue='Gender', alpha=0.6)
plt.title('Exercise Duration vs Calories Burned')
plt.xlabel('Duration (minutes)')
plt.ylabel('Calories')
plt.legend(title='Gender')
plt.savefig('duration_vs_calories.png', dpi=300, bbox_inches='tight')
plt.show()
print("- Generated: duration_vs_calories.png")

# 3.2.6 Heart Rate vs Calories
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Heart_Rate', y='Calories', data=df, hue='Gender', alpha=0.6)
plt.title('Heart Rate vs Calories Burned')
plt.xlabel('Heart Rate (bpm)')
plt.ylabel('Calories')
plt.legend(title='Gender')
plt.savefig('heart_rate_vs_calories.png', dpi=300, bbox_inches='tight')
plt.show()
print("- Generated: heart_rate_vs_calories.png")

# 3.2.7 Body Temp vs Calories
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Body_Temp', y='Calories', data=df, hue='Gender', alpha=0.6)
plt.title('Body Temperature vs Calories Burned')
plt.xlabel('Body Temperature (°C)')
plt.ylabel('Calories')
plt.legend(title='Gender')
plt.savefig('body_temp_vs_calories.png', dpi=300, bbox_inches='tight')
plt.show()
print("- Generated: body_temp_vs_calories.png")

# 3.2.8 Duration vs Calories (with Heart Rate as size)
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Duration', y='Calories', hue='Gender', size='Heart_Rate',
                sizes=(50, 300), data=df, alpha=0.7, palette='Set1')
plt.title('Duration vs Calories (Colored by Gender, Sized by Heart Rate)')
plt.xlabel('Duration (minutes)')
plt.ylabel('Calories')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('duration_vs_calories_complex.png', dpi=300, bbox_inches='tight')
plt.show()
print("- Generated: duration_vs_calories_complex.png")

# --------------------------
# 3.3 Correlation Analysis
# --------------------------
print("\n3.3 Correlation Analysis (Numerical Features)")

# Select numerical columns for correlation matrix
numerical_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
corr_matrix = df[numerical_cols].corr().round(2)

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
            linewidths=0.5, fmt='.2f', center=0)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("- Generated: correlation_matrix.png")

# 3.4 Gender-specific analysis
print("\n3.4 Gender-specific Analysis")
gender_stats = df.groupby('Gender')['Calories'].agg(['mean', 'std', 'min', 'max'])
print(f"\nCalories Statistics by Gender:")
print(gender_stats.round(2))


# ============================================================================
# 4. KEY EDA INSIGHTS
# ============================================================================
print("\n" + "="*80)
print("4. KEY EDA INSIGHTS")
print("="*80)

insights = [
    f"1. Target Variable (Calories): Ranges from {df['Calories'].min():.0f} to {df['Calories'].max():.0f} "
    f"(mean: {df['Calories'].mean():.0f}, std: {df['Calories'].std():.0f}).",

    f"2. Gender Distribution: Male ({df[df['Gender']=='male'].shape[0]}) vs Female ({df[df['Gender']=='female'].shape[0]}).",

    f"3. Age Range: {df['Age'].min()} to {df['Age'].max()} years (mean: {df['Age'].mean():.1f} years).",

    f"4. Duration Impact: Strong positive correlation between exercise duration and calories burned "
    f"(r={df['Duration'].corr(df['Calories']):.2f}).",

    f"5. Heart Rate Impact: Strong positive correlation between heart rate and calories burned "
    f"(r={df['Heart_Rate'].corr(df['Calories']):.2f}).",

    f"6. Gender Difference: Average calories burned - "
    f"Male: {df[df['Gender']=='male']['Calories'].mean():.0f} kcal vs "
    f"Female: {df[df['Gender']=='female']['Calories'].mean():.0f} kcal.",

    f"7. Body Temperature Impact: Moderate positive correlation (r={df['Body_Temp'].corr(df['Calories']):.2f}).",

    f"8. Weight Impact: Moderate positive correlation (r={df['Weight'].corr(df['Calories']):.2f}).",

    f"9. Missing Values: {missing_vals.sum()} total missing values ({'None' if missing_vals.sum() == 0 else missing_vals.idxmax()}).",

    f"10. Data Quality: Dataset contains {len(df)} clean samples ready for modeling."
]

for insight in insights:
    print(f"- {insight}")


# ============================================================================
# 5. DATA PREPROCESSING
# ============================================================================
print("\n" + "="*80)
print("5. DATA PREPROCESSING")
print("="*80)

# 5.1 One-Hot Encoding for Gender (only categorical variable)
print(f"\n🔄 Step 1: One-Hot Encoding for Gender")
print(f" - Original Gender values: {df['Gender'].unique().tolist()}")

# Map gender: male -> 1, female -> 0 (or use one-hot encoding)
gender_mapping = {'male': 1, 'female': 0}
df['Gender_Encoded'] = df['Gender'].map(gender_mapping)
df = df.drop('Gender', axis=1)

print(f" ✅ Gender encoded: {gender_mapping}")
print(f" - Gender_Encoded distribution: {df['Gender_Encoded'].value_counts().to_dict()}")

# 5.2 Identify original numerical columns (for outlier detection)
print(f"\n🔍 Step 2: Identifying Numerical Features for Outlier Detection")

original_numerical_cols = [
    'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp',
    'Gender_Encoded'  # Binary (0/1), exclude from outlier removal
]

# Features to check for outliers (exclude binary and target)
numerical_cols_to_check = [col for col in original_numerical_cols
                           if col in df.columns and col != TARGET_VARIABLE and col != 'Gender_Encoded']

print(f"✅ Numerical columns to check for outliers:")
for col in numerical_cols_to_check:
    print(f"   - {col}")

print(f"\n⚠️  Binary columns (excluded from outlier removal):")
print(f"   - Gender_Encoded: [0, 1] (no outlier removal)")


# 5.3 Outlier Removal (> 2 standard deviations) - NO SCALING
print(f"\n🔥 Step 3: Outlier Removal (> 2 Standard Deviations)")
print(f"   - Method: Remove rows where value > mean ± 2*std")
print(f"   - Keep original values (NO standardization)")

initial_rows = len(df)

for col in numerical_cols_to_check:
    # Calculate mean and std for this column
    col_mean = df[col].mean()
    col_std = df[col].std()

    # Define outlier thresholds
    lower_threshold = col_mean - 2 * col_std
    upper_threshold = col_mean + 2 * col_std

    # Count outliers before removal
    outliers_count = ((df[col] < lower_threshold) | (df[col] > upper_threshold)).sum()
    outliers_pct = (outliers_count / len(df)) * 100

    # Print outlier statistics
    print(f"\n   Column: {col}")
    print(f"     - Mean: {col_mean:.2f}, Std: {col_std:.2f}")
    print(f"     - Thresholds: [{lower_threshold:.2f}, {upper_threshold:.2f}]")
    print(f"     - Outliers found: {outliers_count} ({outliers_pct:.2f}%)")

    # Remove rows where the column has outliers
    df = df[
        (df[col] >= lower_threshold) &
        (df[col] <= upper_threshold)
        ]

final_rows = len(df)
rows_removed = initial_rows - final_rows
removed_pct = (rows_removed / initial_rows) * 100

print(f"\n✅ Outlier Removal Summary:")
print(f"   - Initial rows: {initial_rows}")
print(f"   - Rows removed (outliers): {rows_removed} ({removed_pct:.2f}%)")
print(f"   - Final rows: {final_rows}")

# 5.4 Verify Gender_Encoded remains binary (0/1)
print(f"\n🔍 Step 4: Verifying Binary Feature Remains Intact")
gender_unique = df['Gender_Encoded'].unique()
print(f" - Gender_Encoded unique values: {sorted(gender_unique)}")
if set(gender_unique) == {0, 1}:
    print(f" ✅ Gender_Encoded correctly remains 0/1")
else:
    print(f" ⚠️  WARNING: Gender_Encoded has unexpected values!")

# 5.5 Final data statistics after outlier removal
print(f"\n📊 Step 5: Final Statistics After Outlier Removal")
print(f"   Target Variable (Calories):")
print(f"   - Mean: {df['Calories'].mean():.2f}")
print(f"   - Std: {df['Calories'].std():.2f}")
print(f"   - Min: {df['Calories'].min():.2f}")
print(f"   - Max: {df['Calories'].max():.2f}")


# ============================================================================
# 6. TRAIN-TEST SPLIT (80/20)
# ============================================================================
print("\n" + "="*80)
print("6. 80/20 TRAIN-TEST SPLIT")
print("="*80)

# Prepare features and target
X = df.drop(TARGET_VARIABLE, axis=1)
y = df[TARGET_VARIABLE]

print(f"\n📊 Feature Shape: {X.shape}")
print(f"📊 Target Shape: {y.shape}")
print(f"\nFeatures: {X.columns.tolist()}")

# Perform 80/20 split
train_df, test_df = train_test_split(
    df,
    test_size=(1 - TRAIN_TEST_SPLIT),
    random_state=RANDOM_STATE
)

print(f"\n✅ Train-Test Split Completed:")
print(f" - Training set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
print(f" - Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

# Verify target variable distribution
print(f"\n🎯 Target Variable (Calories) Distribution:")
print(f" - Training set: mean={train_df['Calories'].mean():.2f}, std={train_df['Calories'].std():.2f}")
print(f" - Test set: mean={test_df['Calories'].mean():.2f}, std={test_df['Calories'].std():.2f}")


# ============================================================================
# 7. SAVE PROCESSED DATA
# ============================================================================
print("\n" + "="*80)
print("7. SAVING PROCESSED DATA")
print("="*80)

# Save training data
train_df.to_csv(f'train.csv', index=False)
print(f"✅ Saved train.csv (shape: {train_df.shape})")

# Save test data
test_df.to_csv(f'test.csv', index=False)
print(f"✅ Saved test.csv (shape: {test_df.shape})")

# Save feature names for future reference
feature_names = [col for col in X.columns]
with open('feature_names.txt', 'w') as f:
    for feature in feature_names:
        f.write(f"{feature}\n")
print(f"✅ Saved feature_names.txt ({len(feature_names)} features)")


# ============================================================================
# 8. FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✅ EDA AND PREPROCESSING COMPLETED SUCCESSFULLY")
print("="*80)

summary = {
    "Data Version": DATA_VERSION,
    "Initial Samples": initial_rows,
    "After Outlier Removal": final_rows,
    "Outliers Removed": rows_removed,
    "Training Samples": len(train_df),
    "Test Samples": len(test_df),
    "Features Count": len(feature_names),
    "Target Variable": TARGET_VARIABLE,
    "Random State": RANDOM_STATE,
}

print("\n📋 Execution Summary:")
for key, value in summary.items():
    print(f" - {key}: {value}")

print(f"\n📊 Final Dataset Statistics:")
print(f" - Features: {feature_names}")
print(f" - Target range: {train_df['Calories'].min():.0f} - {train_df['Calories'].max():.0f} kcal")
print(f" - Average calories: {train_df['Calories'].mean():.2f} ± {train_df['Calories'].std():.2f} kcal")

print(f"\n🔥 Top Correlations with Calories:")
corr_with_target = df[numerical_cols].corr()['Calories'].abs().sort_values(ascending=False)[1:]
for feature, corr in corr_with_target.head(3).items():
    print(f" - {feature}: r={corr:.3f}")

print("\n" + "="*80)
print("🎯 READY FOR MACHINE LEARNING MODELING")
print("="*80)


# ============================================================================
# 15. UPLOAD EDA PLOTS TO S3
# ============================================================================

print("\n" + "="*80)
print("UPLOADING EDA PLOTS TO S3")
print("="*80)

import os

for file in os.listdir("."):
    if file.endswith(".png"):
        s3_key = f"eda-plots/{DATA_VERSION}/{file}"
        s3_client.upload_file(file, S3_BUCKET, s3_key)
        print(f"✅ Uploaded {file} to s3://{S3_BUCKET}/{s3_key}")

# ============================================================================
# 19. SAVE PROCESSED DATA TO S3 (VERSIONED)
# ============================================================================

print("\n" + "="*80)
print("8. SAVING PROCESSED DATA TO S3 (VERSIONED)")
print("="*80)

# 19.1 Save training data to versioned S3 path
train_s3_key = f"{S3_PROCESSED_PATH}/train.csv"
train_s3_path = upload_to_s3(train_df, train_s3_key)

# 19.2 Save test data to versioned S3 path
test_s3_key = f"{S3_PROCESSED_PATH}/test.csv"
test_s3_path = upload_to_s3(test_df, test_s3_key)

# 19.3 Create soft links to latest/
print("\n" + "="*80)
print("9. CREATING SOFT LINKS TO LATEST/")
print("="*80)

# Copy train.csv to latest/
create_soft_link(
    source_key=train_s3_key,
    link_key=f"{S3_LATEST_PREFIX}/processed/train.csv"
)

# Copy test.csv to latest/
create_soft_link(
    source_key=test_s3_key,
    link_key=f"{S3_LATEST_PREFIX}/processed/test.csv"
)

# 19.4 Save encoded data locally for reference
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
print(f"\n💾 Saved locally:")
print(f"   - train.csv (shape: {train_df.shape})")
print(f"   - test.csv (shape: {test_df.shape})")

# 19.5 Save preprocessing artifacts
import pickle


# Save feature names for inference
target_col = 'Calories'
feature_names = [col for col in train_df.columns if col != target_col]
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print(f"   - feature_names.pkl ({len(feature_names)} features)")

# ============================================================================
# 20. FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ EDA AND DATA PREPROCESSING COMPLETED SUCCESSFULLY")
print("="*80)

summary = {
    "Data Version": DATA_VERSION,
    "Training Samples": len(train_df),
    "Test Samples": len(test_df),
    "Features Count": len(train_df.columns) - 1,
    "Target Variable": target_col,
    "Random State": RANDOM_STATE,
    "Train S3 Path": train_s3_path,
    "Test S3 Path": test_s3_path,
}

print("\n📋 Execution Summary:")
for key, value in summary.items():
    print(f"   - {key}: {value}")

print("\n📁 S3 Structure Created:")
print(f"   s3://{S3_BUCKET}/")
print(f"   ├── {S3_RAW_PATH}/")
print(f"   │   └── (raw data from original source)")
print(f"   ├── {S3_PROCESSED_PATH}/")
print(f"   │   ├── train.csv ({len(train_df)} samples)")
print(f"   │   └── test.csv ({len(test_df)} samples)")
print(f"   └── {S3_LATEST_PREFIX}/processed/")
print(f"       ├── train.csv (→ {DATA_VERSION}/processed/train.csv)")
print(f"       └── test.csv (→ {DATA_VERSION}/processed/test.csv)")

print("\n🎯 Ready for SageMaker Autopilot Training:")
print(f"   - Use this S3 path for Autopilot: s3://{S3_BUCKET}/{S3_LATEST_PREFIX}/processed/train.csv")
print(f"   - Test data path for inference: s3://{S3_BUCKET}/{S3_LATEST_PREFIX}/processed/test.csv")

print("\n" + "="*80)