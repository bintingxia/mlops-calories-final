# ============================================================================
# 1. SET ENVIRONMENT VARIABLES FOR SAGEMAKER STUDIO IMPORTS
# ============================================================================

import os
os.environ['DataZoneProjectId'] = 'ahny572cp1dr15'
os.environ['DataZoneDomainId'] = 'dzd-4ixhg0bzrkytw9'
os.environ['DataZoneEnvironmentId'] = '5l48miq9l3wu1l'
os.environ['DataZoneDomainRegion'] = 'ap-northeast-1'

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
    import logging
    from logging.handlers import RotatingFileHandler

    level = logging.INFO
    max_bytes = 5 * 1024 * 1024
    backup_count = 5

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
# 3. RESET WORKING DIRECTORY
# ============================================================================

def _reset_os_path():
    try:
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
# 4. PROJECT CONFIGURATION
# ============================================================================

DATA_VERSION = "v1.0"
S3_BUCKET = "mlops-calories"
S3_TRAIN_DATA_PATH = f"s3://{S3_BUCKET}/data_versions/{DATA_VERSION}/processed/train.csv"
S3_MODEL_OUTPUT_PATH = f"s3://{S3_BUCKET}/model-artifacts/{DATA_VERSION}"
AUTOPILOT_JOB_NAME = f"calories-0304-{DATA_VERSION.replace('.', '-')}"
MAX_CANDIDATES = 30
MAX_RUNTIME_SECONDS = 7200
MAX_RUNTIME_PER_JOB_SECONDS = 600
TARGET_ATTRIBUTE_NAME = "Calories"
PROBLEM_TYPE = "Regression"

# ============================================================================
# 5. IMPORT LIBRARIES
# ============================================================================

import pandas as pd
import numpy as np
import sagemaker
from sagemaker import AutoML
import boto3
import time
from datetime import datetime
import warnings
import json

warnings.filterwarnings('ignore')

# ============================================================================
# 6. INITIALIZE SAGEMAKER
# ============================================================================

print("\n" + "="*80)
print("INITIALIZING SAGEMAKER SESSION")
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
# 7. HELPER FUNCTIONS
# ============================================================================

def check_s3_file_exists(s3_path):
    try:
        bucket_name = s3_path.split('/')[2]
        key = '/'.join(s3_path.split('/')[3:])
        s3_client.head_object(Bucket=bucket_name, Key=key)
        print(f"✅ S3 file exists: {s3_path}")
        return True
    except Exception as e:
        print(f"❌ S3 file NOT found: {s3_path}")
        return False

def check_job_status(job_name):
    try:
        response = sm_client.describe_auto_ml_job_v2(AutoMLJobName=job_name)
        status = response['AutoMLJobStatus']
        secondary_status = response.get('AutoMLJobSecondaryStatus', 'N/A')
        return status, secondary_status, response
    except Exception:
        return None, None, None

# ============================================================================
# 8. VALIDATE TRAINING DATA
# ============================================================================

print("\n" + "="*80)
print("VALIDATING TRAINING DATA")
print("="*80)

if not check_s3_file_exists(S3_TRAIN_DATA_PATH):
    raise Exception(f"Training data not found: {S3_TRAIN_DATA_PATH}")

print(f"✅ Training Data Validated: {S3_TRAIN_DATA_PATH}")

# ============================================================================
# 9. CHECK FOR EXISTING JOB
# ============================================================================

print("\n" + "="*80)
print("CHECKING FOR EXISTING JOB")
print("="*80)

job_status, secondary_status, job_response = check_job_status(AUTOPILOT_JOB_NAME)

if job_status:
    print(f"\n✅ Found existing job: '{AUTOPILOT_JOB_NAME}'")
    print(f"   - Status: {job_status}")
    print(f"   - Secondary Status: {secondary_status}")

    if job_status == 'Completed':
        print("\n🎉 Job already completed!")

        best_candidate = job_response.get('BestCandidate', {})
        if best_candidate:
            objective_metric = best_candidate.get('FinalAutoMLJobObjectiveMetric', {})
            metric_value = objective_metric.get('Value', 'N/A')

            print(f"\n🏆 Best Model:")
            print(f"   - Candidate: {best_candidate.get('CandidateName')}")
            print(f"   - Metric Value: {metric_value}")

            # Save model info
            best_model_info = {
                "auto_ml_job_name": AUTOPILOT_JOB_NAME,
                "best_candidate_name": best_candidate.get('CandidateName'),
                "validation_metric_value": metric_value,
                "data_version": DATA_VERSION,
                "timestamp": datetime.now().isoformat()
            }

            best_model_file = f"best_model_info_{DATA_VERSION}.json"
            with open(best_model_file, 'w') as f:
                json.dump(best_model_info, f, indent=2)

            print(f"\n💾 Model info saved: {best_model_file}")
        else:
            print("\n⚠️ No best candidate found")

    elif job_status in ['Failed', 'Stopped']:
        print(f"\n⚠️ Job finished with status: {job_status}")
        print("   Change AUTOPILOT_JOB_NAME to start a new job")

    else:
        print(f"\n⏳ Job is running (Status: {job_status})")
        print("   Monitor in SageMaker console")
        print("   Estimated time: 30-60 minutes")

else:
    print(f"\n🚀 Starting new AutoML job: {AUTOPILOT_JOB_NAME}")
    print(f"   - Max Candidates: {MAX_CANDIDATES}")
    print(f"   - Max Runtime: {MAX_RUNTIME_SECONDS}s ({MAX_RUNTIME_SECONDS//60}m)")

    # Initialize AutoML - ALL config goes HERE
    print(f"\n🔧 Initializing AutoML...")

    # Create job objective for Regression
    job_objective = {"MetricName": "RMSE"}

    # Pass ALL parameters during initialization
    auto_ml = AutoML(
        role=role,
        target_attribute_name=TARGET_ATTRIBUTE_NAME,
        problem_type=PROBLEM_TYPE,
        job_objective=job_objective,
        max_candidates=MAX_CANDIDATES,
        max_runtime_per_training_job_in_seconds=MAX_RUNTIME_PER_JOB_SECONDS,
        total_job_runtime_in_seconds=MAX_RUNTIME_SECONDS,
        sagemaker_session=sagemaker_session
    )

    print(f"✅ AutoML configured")
    print(f"   - Problem Type: {PROBLEM_TYPE}")
    print(f"   - Objective Metric: {job_objective['MetricName']}")
    print(f"   - Max Candidates: {MAX_CANDIDATES}")
    print(f"   - Max Runtime: {MAX_RUNTIME_SECONDS}s")

    try:
        # fit() only needs inputs and job_name
        auto_ml.fit(
            inputs=S3_TRAIN_DATA_PATH,
            job_name=AUTOPILOT_JOB_NAME,
            wait=False,
            logs=False
        )

        print(f"\n✅ Job started successfully!")
        print(f"   Job will run in the background")
        print(f"   Re-run this cell to check status")

    except Exception as e:
        print(f"\n❌ Failed to start job: {str(e)}")
        raise

print("\n" + "="*80)
print("✅ AUTOPILOT JOB CHECK COMPLETED")
print("="*80)
print("\n💡 Next Steps:")
print("   - If running: Wait and re-run cell to check status")
print("   - If completed: Proceed to deployment")
print("="*80)

