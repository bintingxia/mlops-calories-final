# ============================================================================
# 1. SET ENVIRONMENT VARIABLES FOR SAGEMAKER STUDIO IMPORTS
# ============================================================================

import os
os.environ['DataZoneProjectId'] = 'ahny572cp1dr15'
os.environ['DataZoneDomainId'] = 'dzd-4ixhg0bzrkytw9'
os.environ['DataZoneEnvironmentId'] = '5l48miq9l3wu1l'
os.environ['DataZoneDomainRegion'] = 'ap-northeast-1'

# ============================================================================
# 2. PROJECT CONFIGURATION
# ============================================================================

DATA_VERSION = "v1.0"
S3_BUCKET = "mlops-calories"
S3_REGION = "ap-northeast-1"

# Paths
S3_MODEL_METADATA_PATH = f"s3://{S3_BUCKET}/model-artifacts/{DATA_VERSION}/best_model_metadata.json"

# AutoML Job Name
AUTOPILOT_JOB_NAME = f"calories-0304-{DATA_VERSION.replace('.', '-')}"

# Real-time Endpoint Configuration
ENDPOINT_NAME = f"{AUTOPILOT_JOB_NAME}-endpoint"
REALTIME_INSTANCE_TYPE = "ml.m5.xlarge"
REALTIME_INITIAL_INSTANCE_COUNT = 1

# ============================================================================
# 3. IMPORT LIBRARIES
# ============================================================================

import pandas as pd
import numpy as np
import sagemaker
import boto3
from botocore.exceptions import ClientError
import json
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 4. INITIALIZE SAGEMAKER SESSION
# ============================================================================

print("\n" + "="*80)
print("INITIALIZING DEPLOYMENT SESSION")
print("="*80)

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
region = sagemaker_session.boto_region_name

sm_client = boto3.client('sagemaker', region_name=region)
s3_client = boto3.client('s3', region_name=region)

print(f"✅ SageMaker Session Initialized")
print(f"   - Region: {region}")
print(f"   - Role: {role[:50]}...")

# ============================================================================
# 5. GET AUTOML JOB DETAILS
# ============================================================================

print("\n" + "="*80)
print("RETRIEVING AUTOML JOB DETAILS")
print("="*80)

try:
    job_response = sm_client.describe_auto_ml_job_v2(AutoMLJobName=AUTOPILOT_JOB_NAME)
    job_status = job_response['AutoMLJobStatus']

    if job_status != 'Completed':
        raise Exception(f"Job is not completed: {job_status}")

    best_candidate = job_response.get('BestCandidate', {})

    if not best_candidate:
        raise Exception("No best candidate found")

    candidate_name = best_candidate.get('CandidateName')
    inference_containers = best_candidate.get('InferenceContainers', [])

    if not inference_containers:
        raise Exception("No inference containers in best candidate")

    print(f"✅ Job retrieved successfully")
    print(f"   - Job Status: {job_status}")
    print(f"   - Best Candidate: {candidate_name}")
    print(f"   - Number of Containers: {len(inference_containers)}")

    for idx, container in enumerate(inference_containers):
        container_image = container.get('Image', 'N/A')
        print(f"   - Container {idx + 1}: {container_image.split('/')[-1] if container_image != 'N/A' else 'N/A'}")

    objective_metric = best_candidate.get('FinalAutoMLJobObjectiveMetric', {})
    validation_mse = objective_metric.get('Value', 'N/A')
    print(f"   - Validation MSE: {validation_mse}")

except Exception as e:
    print(f"❌ Failed to retrieve job details: {str(e)}")
    raise

# ============================================================================
# 6. CREATE SAGEMAKER MODEL
# ============================================================================

print("\n" + "="*80)
print("CREATING SAGEMAKER MODEL")
print("="*80)

model_name = f"{AUTOPILOT_JOB_NAME}-model"

try:
    # Check if model already exists
    try:
        sm_client.describe_model(ModelName=model_name)
        print(f"✅ Model already exists: {model_name}")
    except ClientError as e:
        if 'ValidationException' in str(e) or 'Could not find model' in str(e):
            # Create model
            if len(inference_containers) == 1:
                # Single container model
                print(f"📦 Creating single-container model...")
                create_model_params = {
                    "ModelName": model_name,
                    "PrimaryContainer": {
                        "Image": inference_containers[0].get('Image'),
                        "ModelDataUrl": inference_containers[0].get('ModelDataUrl'),
                        "Environment": inference_containers[0].get('Environment', {})
                    },
                    "ExecutionRoleArn": role
                }
            else:
                # Multi-container model (inference pipeline)
                print(f"📦 Creating multi-container inference pipeline model...")
                containers = []
                for container in inference_containers:
                    containers.append({
                        "Image": container.get('Image'),
                        "ModelDataUrl": container.get('ModelDataUrl'),
                        "Environment": container.get('Environment', {})
                    })

                create_model_params = {
                    "ModelName": model_name,
                    "Containers": containers,
                    "ExecutionRoleArn": role
                }

            # Add tags
            create_model_params["Tags"] = [
                {"Key": "Project", "Value": "MLOps-Calories-Prediction"},
                {"Key": "DataVersion", "Value": DATA_VERSION},
                {"Key": "AutoMLJobName", "Value": AUTOPILOT_JOB_NAME}
            ]

            # Create the model
            create_model_response = sm_client.create_model(**create_model_params)
            print(f"✅ Model created: {model_name}")
            print(f"   - ARN: {create_model_response['ModelArn']}")
        else:
            raise

except Exception as e:
    print(f"❌ Failed to create model: {str(e)}")
    raise

# ============================================================================
# 7. CREATE ENDPOINT CONFIGURATION
# ============================================================================

print("\n" + "="*80)
print("CREATING ENDPOINT CONFIGURATION")
print("="*80)

endpoint_config_name = f"{ENDPOINT_NAME}-config"

try:
    # Check if endpoint config already exists
    try:
        sm_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        print(f"✅ Endpoint config already exists: {endpoint_config_name}")
    except ClientError as e:
        if 'ValidationException' in str(e) or 'Could not find' in str(e):
            # Create endpoint config
            create_endpoint_config_response = sm_client.create_endpoint_config(
                EndpointConfigName=endpoint_config_name,
                ProductionVariants=[
                    {
                        "VariantName": "AllTraffic",
                        "ModelName": model_name,
                        "InitialInstanceCount": REALTIME_INITIAL_INSTANCE_COUNT,
                        "InstanceType": REALTIME_INSTANCE_TYPE
                    }
                ]
            )
            print(f"✅ Endpoint config created: {endpoint_config_name}")
        else:
            raise

except Exception as e:
    print(f"❌ Failed to create endpoint config: {str(e)}")
    raise

# ============================================================================
# 8. DEPLOY ENDPOINT
# ============================================================================

print("\n" + "="*80)
print("DEPLOYING REAL-TIME ENDPOINT")
print("="*80)

print(f"\n📦 Configuration:")
print(f"   - Instance Type: {REALTIME_INSTANCE_TYPE}")
print(f"   - Initial Instance Count: {REALTIME_INITIAL_INSTANCE_COUNT}")
print(f"   - Supports multi-container inference pipelines")

try:
    # Check if endpoint already exists
    endpoint_exists = False
    try:
        endpoint_response = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        endpoint_exists = True
        endpoint_status = endpoint_response['EndpointStatus']
        print(f"\n✅ Endpoint already exists: {ENDPOINT_NAME}")
        print(f"   - Status: {endpoint_status}")

        if endpoint_status in ['Creating', 'Updating']:
            print(f"\n⏳ Waiting for endpoint to be ready...")
            start_time = time.time()
            while endpoint_status in ['Creating', 'Updating']:
                time.sleep(30)
                endpoint_response = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
                endpoint_status = endpoint_response['EndpointStatus']
                elapsed = int(time.time() - start_time)
                print(f"   [{elapsed}s] Status: {endpoint_status}")

            if endpoint_status == 'InService':
                elapsed = int(time.time() - start_time)
                print(f"✅ Endpoint is ready! (Elapsed: {elapsed}s)")
            elif endpoint_status == 'Failed':
                failure_reason = endpoint_response.get('FailureReason', 'N/A')
                raise Exception(f"Endpoint creation failed: {failure_reason}")

    except ClientError as e:
        if 'ValidationException' in str(e) or 'Could not find' in str(e):
            endpoint_exists = False
        else:
            raise

    # Create endpoint if it doesn't exist
    if not endpoint_exists:
        create_endpoint_response = sm_client.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=endpoint_config_name
        )
        print(f"\n✅ Endpoint creation started: {ENDPOINT_NAME}")
        print(f"   - ARN: {create_endpoint_response['EndpointArn']}")

        # Wait for endpoint to be ready
        print(f"\n⏳ Waiting for endpoint to be ready (this may take 5-10 minutes)...")
        start_time = time.time()

        while True:
            endpoint_response = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
            endpoint_status = endpoint_response['EndpointStatus']

            if endpoint_status == 'InService':
                elapsed = int(time.time() - start_time)
                print(f"\n✅ Endpoint is ready! (Elapsed: {elapsed}s)")
                break
            elif endpoint_status == 'Failed':
                failure_reason = endpoint_response.get('FailureReason', 'N/A')
                raise Exception(f"Endpoint creation failed: {failure_reason}")

            elapsed = int(time.time() - start_time)
            print(f"   [{elapsed}s] Status: {endpoint_status}", end='\r')
            time.sleep(30)

except Exception as e:
    print(f"\n❌ Failed to deploy endpoint: {str(e)}")
    raise

# ============================================================================
# 9. DEPLOYMENT SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ DEPLOYMENT COMPLETED SUCCESSFULLY")
print("="*80)

print(f"\n📌 Endpoint Details:")
print(f"   - Endpoint Name: {ENDPOINT_NAME}")
print(f"   - Model Name: {model_name}")
print(f"   - Instance Type: {REALTIME_INSTANCE_TYPE}")
print(f"   - Instance Count: {REALTIME_INITIAL_INSTANCE_COUNT}")
print(f"   - Status: InService")

print(f"\n🎯 Ready for inference!")
