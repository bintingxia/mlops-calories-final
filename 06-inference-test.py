# ============================================================================
# INFERENCE TEST - DIRECT ENDPOINT INVOCATION
# ============================================================================

import pandas as pd
import numpy as np
import boto3
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

AWS_REGION = "ap-northeast-1"
ENDPOINT_NAME = "calories-0304-v1-0-endpoint"

print("=" * 80)
print("SAGEMAKER ENDPOINT INFERENCE TEST")
print("=" * 80)
print(f"\n📍 Endpoint Name: {ENDPOINT_NAME}")
print(f"📍 Region: {AWS_REGION}")

# ============================================================================
# INITIALIZE SAGEMAKER RUNTIME CLIENT
# ============================================================================

runtime_client = boto3.client('runtime.sagemaker', region_name=AWS_REGION)

print("\n✅ SageMaker Runtime client initialized")

# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

def predict_calories(gender, age, height, weight, duration, heart_rate, body_temp):
    """
    Predict calories burned using SageMaker endpoint

    Parameters:
    -----------
    gender : str
        'Male' or 'Female'
    age : int
        Age in years
    height : float
        Height in cm
    weight : float
        Weight in kg
    duration : float
        Exercise duration in minutes
    heart_rate : float
        Heart rate in bpm
    body_temp : float
        Body temperature in °C

    Returns:
    --------
    float : Predicted calories burned
    """
    try:
        # Encode gender (Male=1, Female=0)
        gender_encoded = 1.0 if gender.lower() == "male" else 0.0

        # Prepare input data in CSV format (matching training data order)
        # Order: Gender_Encoded, Age, Height, Weight, Duration, Heart_Rate, Body_Temp
        input_data = f"{gender_encoded},{age},{height},{weight},{duration},{heart_rate},{body_temp}"

        # Invoke endpoint
        response = runtime_client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='text/csv',
            Body=input_data
        )

        # Parse result
        result = json.loads(response['Body'].read().decode())

        # Handle different response formats
        if isinstance(result, dict):
            prediction = result.get('predictions', [result.get('prediction', [0])])[0]
            if isinstance(prediction, list):
                prediction = prediction[0]
        elif isinstance(result, list):
            prediction = result[0]
        else:
            prediction = float(result)

        return float(prediction)

    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")

# ============================================================================
# TEST CASES
# ============================================================================

print("\n" + "=" * 80)
print("RUNNING TEST PREDICTIONS")
print("=" * 80)

# Define test cases
test_cases = [
    {
        "name": "Light Exercise (Female)",
        "gender": "Female",
        "age": 25,
        "height": 165,
        "weight": 55,
        "duration": 15,
        "heart_rate": 90,
        "body_temp": 38.5
    },
    {
        "name": "Moderate Exercise (Male)",
        "gender": "Male",
        "age": 30,
        "height": 180,
        "weight": 75,
        "duration": 30,
        "heart_rate": 120,
        "body_temp": 39.5
    },
    {
        "name": "Intense Exercise (Male)",
        "gender": "Male",
        "age": 28,
        "height": 175,
        "weight": 70,
        "duration": 60,
        "heart_rate": 150,
        "body_temp": 40.0
    },
    {
        "name": "Long Duration (Female)",
        "gender": "Female",
        "age": 35,
        "height": 170,
        "weight": 65,
        "duration": 45,
        "heart_rate": 135,
        "body_temp": 39.8
    }
]

# Run predictions
results = []

for i, test_case in enumerate(test_cases, 1):
    print(f"\n{'─' * 80}")
    print(f"Test Case {i}: {test_case['name']}")
    print(f"{'─' * 80}")

    # Extract parameters
    name = test_case.pop('name')

    print("\n📝 Input Parameters:")
    for key, value in test_case.items():
        print(f"   - {key.replace('_', ' ').title()}: {value}")

    try:
        # Make prediction
        prediction = predict_calories(**test_case)

        print(f"\n🔥 Predicted Calories: {prediction:.2f} kcal")

        # Store result
        results.append({
            'Test Case': name,
            **test_case,
            'Predicted Calories': round(prediction, 2),
            'Status': 'Success'
        })

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        results.append({
            'Test Case': name,
            **test_case,
            'Predicted Calories': None,
            'Status': f'Failed: {str(e)}'
        })

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("TEST RESULTS SUMMARY")
print("=" * 80)

# Create results dataframe
results_df = pd.DataFrame(results)

# Display summary
print(f"\n✅ Tests Completed: {len(results)}")
print(f"✅ Successful Predictions: {results_df['Status'].eq('Success').sum()}")
print(f"❌ Failed Predictions: {results_df['Status'].ne('Success').sum()}")

# Display results table
print("\n" + "=" * 80)
print("DETAILED RESULTS")
print("=" * 80)
print()

# Select key columns for display
display_cols = ['Test Case', 'gender', 'age', 'duration', 'heart_rate', 'Predicted Calories', 'Status']
results_df[display_cols]

# ============================================================================
# INTERACTIVE PREDICTION FUNCTION
# ============================================================================

print("\n" + "=" * 80)
print("INTERACTIVE PREDICTION FUNCTION READY")
print("=" * 80)

print("""
\n📌 You can now make custom predictions using the predict_calories() function:

Example usage:
--------------
calories = predict_calories(
    gender="Male",
    age=30,
    height=180,
    weight=75,
    duration=45,
    heart_rate=130,
    body_temp=39.8
)
print(f"Predicted Calories: {calories:.2f} kcal")
""")

print("\n🎯 Endpoint is ready for inference!")
