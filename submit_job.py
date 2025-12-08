"""
Submit backtest job to Cloud Run Jobs
"""
import json
import os
import time
from google.cloud.run import JobsClient, Job, Container, ResourceRequirements, EnvVar, RunJobRequest, CreateJobRequest
from google.cloud import storage
from gcp_config import PROJECT_ID, REGION, BUCKET_NAME, get_storage_client

JOB_NAME = "polybench-worker"  # Reusable job name

def create_or_get_job():
    """Create or get Cloud Run Job"""
    jobs_client = JobsClient()
    parent = f"projects/{PROJECT_ID}/locations/{REGION}"
    job_path = f"{parent}/jobs/{JOB_NAME}"
    
    # Check if job exists
    try:
        job = jobs_client.get_job(name=job_path)
        return job
    except Exception:
        # Job doesn't exist, create it
        pass
    
    # Create job
    job = Job()
    job.template.template.containers = [Container()]
    job.template.template.containers[0].image = f"gcr.io/{PROJECT_ID}/polybench-worker:latest"
    job.template.template.containers[0].command = ["python3.11", "/app/backtest_worker.py"]
    
    # Environment variables (will be overridden per execution)
    job.template.template.containers[0].env = [
        EnvVar(name="GCP_PROJECT_ID", value=PROJECT_ID),
        EnvVar(name="GCP_REGION", value=REGION),
        EnvVar(name="GCS_BUCKET_NAME", value=BUCKET_NAME),
    ]
    
    # Resource limits
    job.template.template.containers[0].resources = ResourceRequirements()
    job.template.template.containers[0].resources.limits = {
        "cpu": "2",
        "memory": "4Gi"
    }
    
    # Timeout (Cloud Run Jobs max is 24 hours = 86400s)
    # Set to 4 hours (14400s) - adjust as needed, max is 86400s (24 hours)
    job.template.template.timeout = "14400s"  # 4 hours
    
    # Max retries
    job.template.template.max_retries = 0
    
    # Create the job
    request = CreateJobRequest(
        parent=parent,
        job=job,
        job_id=JOB_NAME
    )
    
    operation = jobs_client.create_job(request=request)
    operation.result()  # Wait for creation
    
    return jobs_client.get_job(name=job_path)

def submit_backtest_job(
    model_id,
    prompt_template,
    start_date,
    end_date,
    min_volume=0,
    categories=None,
    max_events=10,
    data_path="markets_final_df.feather",
    test_name=None
):
    """Submit a backtest job with given parameters"""
    
    # Generate execution ID - use test_name if provided, otherwise timestamp
    timestamp = int(time.time())
    if test_name:
        # Sanitize test_name for use in filename
        safe_name = "".join(c for c in test_name if c.isalnum() or c in (' ', '-', '_')).strip()[:50]
        safe_name = safe_name.replace(' ', '_')
        execution_id = f"{safe_name}-{timestamp}"
    else:
        execution_id = f"backtest-{timestamp}"
    
    # Prepare job config
    job_config = {
        'job_id': execution_id,
        'test_name': test_name,  # Store test name in config
        'model_id': model_id,
        'prompt_template': prompt_template,
        'start_date': str(start_date),
        'end_date': str(end_date),
        'min_volume': min_volume,
        'categories': categories or [],
        'max_events': max_events,
        'data_path': data_path,
        'bucket_name': BUCKET_NAME,
        'project_id': PROJECT_ID,
        'region': REGION,
    }
    
    # Upload config to GCS
    storage_client = get_storage_client()
    bucket = storage_client.bucket(BUCKET_NAME)
    config_blob = bucket.blob(f"jobs/{execution_id}/config.json")
    config_blob.upload_from_string(json.dumps(job_config), content_type='application/json')
    
    # Create or get the job (don't wait for creation if it's new)
    try:
        jobs_client = JobsClient()
        parent = f"projects/{PROJECT_ID}/locations/{REGION}"
        job_path = f"{parent}/jobs/{JOB_NAME}"
        job = jobs_client.get_job(name=job_path)
    except Exception:
        # Job doesn't exist, create it (this might take time on first run)
        job = create_or_get_job()
        jobs_client = JobsClient()
    
    # Execute the job with config path as argument
    execution_request = RunJobRequest(
        name=job.name,
        overrides=RunJobRequest.Overrides(
            container_overrides=[
                RunJobRequest.Overrides.ContainerOverride(
                    args=[f"gs://{BUCKET_NAME}/jobs/{execution_id}/config.json"]
                )
            ]
        )
    )
    
    # Start execution - get the execution immediately without waiting
    operation = jobs_client.run_job(request=execution_request)
    
    # Extract execution name from operation response
    # The operation response contains the execution name
    try:
        # Wait just long enough to get the execution name (usually instant)
        execution = operation.result(timeout=5)
        return execution
    except Exception:
        # If we can't get it immediately, return with the execution_id we generated
        # The execution will be created, we just don't have the full name yet
        class MockExecution:
            def __init__(self, name):
                self.name = name
        
        # Use our generated execution_id
        return MockExecution(f"{job.name}/executions/{execution_id}")

if __name__ == "__main__":
    # Example usage
    submit_backtest_job(
        model_id="gemini-1.5-pro",
        prompt_template="Predict: {question}",
        start_date="2024-01-01",
        end_date="2024-12-31",
        max_events=10
    )
