# Monitor AutoML Job Progress
import boto3
import time
from datetime import datetime

# Configuration
AUTOPILOT_JOB_NAME = "calories-0304-v1-0"
sm_client = boto3.client('sagemaker', region_name='ap-northeast-1')

print("="*80)
print("AUTOPILOT JOB PROGRESS MONITOR")
print("="*80)

try:
    # Get job details
    response = sm_client.describe_auto_ml_job_v2(AutoMLJobName=AUTOPILOT_JOB_NAME)

    # Extract key information
    job_status = response['AutoMLJobStatus']
    secondary_status = response.get('AutoMLJobSecondaryStatus', 'N/A')
    creation_time = response.get('CreationTime', 'N/A')
    end_time = response.get('EndTime', None)

    print(f"\n📋 Job Name: {AUTOPILOT_JOB_NAME}")
    print(f"📊 Status: {job_status}")
    print(f"🔄 Secondary Status: {secondary_status}")
    print(f"🕐 Started: {creation_time}")

    # Calculate elapsed time
    if creation_time != 'N/A':
        if end_time:
            duration = end_time - creation_time
        else:
            duration = datetime.now(creation_time.tzinfo) - creation_time

        minutes = int(duration.total_seconds() / 60)
        print(f"⏱️  Elapsed Time: {minutes} minutes")

    # Show progress based on status
    if job_status == 'InProgress':
        print(f"\n✅ Job is running!")
        print(f"   Current Stage: {secondary_status}")

        # Explain current stage
        stage_info = {
            'Starting': 'Initializing the AutoML job',
            'AnalyzingData': 'Analyzing your training data',
            'FeatureEngineering': 'Creating and selecting features',
            'ModelTuning': 'Training and tuning models',
            'MaxCandidatesReached': 'Maximum candidates reached, finishing up',
        }

        if secondary_status in stage_info:
            print(f"   📌 {stage_info[secondary_status]}")

        # Show candidates if available
        if 'AutoMLJobArtifacts' in response:
            artifacts = response['AutoMLJobArtifacts']
            if 'CandidateDefinitionNotebookLocation' in artifacts:
                print(f"\n📓 Candidate notebook: {artifacts['CandidateDefinitionNotebookLocation']}")

        print(f"\n💡 Tip: Re-run this cell every few minutes to check progress")

    elif job_status == 'Completed':
        print(f"\n🎉 Job completed successfully!")

        # Get best candidate
        best_candidate = response.get('BestCandidate', {})
        if best_candidate:
            candidate_name = best_candidate.get('CandidateName', 'N/A')
            objective_metric = best_candidate.get('FinalAutoMLJobObjectiveMetric', {})
            metric_name = objective_metric.get('MetricName', 'N/A')
            metric_value = objective_metric.get('Value', 'N/A')

            print(f"\n🏆 Best Model:")
            print(f"   Name: {candidate_name}")
            print(f"   {metric_name}: {metric_value}")

        if end_time:
            print(f"\n🕐 Completed: {end_time}")

    elif job_status == 'Failed':
        print(f"\n❌ Job failed!")
        failure_reason = response.get('FailureReason', 'No failure reason provided')
        print(f"   Reason: {failure_reason}")

    elif job_status == 'Stopped':
        print(f"\n⚠️ Job was stopped")

    print("\n" + "="*80)

except sm_client.exceptions.ResourceNotFound:
    print(f"\n❌ Job not found: {AUTOPILOT_JOB_NAME}")
    print("   Make sure the job has been started")
except Exception as e:
    print(f"\n❌ Error: {str(e)}")

# List All Candidates Generated So Far
import boto3
import pandas as pd

AUTOPILOT_JOB_NAME = "calories-0304-v1-0"
sm_client = boto3.client('sagemaker', region_name='ap-northeast-1')

print("="*80)
print("AUTOPILOT CANDIDATES LIST")
print("="*80)

try:
    # List all candidates
    response = sm_client.list_candidates_for_auto_ml_job(
        AutoMLJobName=AUTOPILOT_JOB_NAME,
        MaxResults=50,
        SortBy='FinalObjectiveMetricValue',
        SortOrder='Ascending'
    )

    candidates = response.get('Candidates', [])

    if not candidates:
        print("\n⏳ No candidates generated yet. Job is still in early stages.")
        print("   Candidates will appear once ModelTuning stage begins.")
    else:
        print(f"\n📊 Found {len(candidates)} candidate(s)\n")

        # Prepare data for display
        candidate_data = []
        for i, candidate in enumerate(candidates, 1):
            name = candidate.get('CandidateName', 'N/A')
            status = candidate.get('CandidateStatus', 'N/A')

            # Get objective metric
            objective = candidate.get('FinalAutoMLJobObjectiveMetric', {})
            metric_name = objective.get('MetricName', 'N/A')
            metric_value = objective.get('Value', 'N/A')

            # Format metric value
            if metric_value != 'N/A':
                metric_value = f"{metric_value:.4f}"

            candidate_data.append({
                'Rank': i,
                'Name': name,
                'Status': status,
                f'{metric_name}': metric_value
            })

        # Display as DataFrame
        df = pd.DataFrame(candidate_data)
        print(df.to_string(index=False))

        # Show best candidate
        if candidates:
            best = candidates[0]
            print(f"\n🏆 Best Candidate So Far:")
            print(f"   {best.get('CandidateName')}")

            best_metric = best.get('FinalAutoMLJobObjectiveMetric', {})
            if best_metric:
                print(f"   {best_metric.get('MetricName')}: {best_metric.get('Value'):.4f}")

    print("\n" + "="*80)
    print("💡 Tip: Re-run to see newly generated candidates")
    print("="*80)

except sm_client.exceptions.ResourceNotFound:
    print(f"\n❌ Job not found: {AUTOPILOT_JOB_NAME}")
except Exception as e:
    print(f"\n❌ Error: {str(e)}")
