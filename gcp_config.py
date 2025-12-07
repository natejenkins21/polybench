"""
GCP Configuration and Authentication
"""
import os
from google.cloud import storage
from google.cloud import aiplatform
from google.auth import default

# GCP Project Configuration
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "coms6998llms")  # Set via environment variable or .env
REGION = os.getenv("GCP_REGION", "us-central1")  # Default region
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "polybench-data")  # Your GCS bucket name

def get_credentials():
    """Get GCP credentials - works with service account or gcloud auth"""
    credentials, project = default()
    if not PROJECT_ID:
        PROJECT_ID = project
    return credentials, PROJECT_ID

def get_storage_client():
    """Get GCS storage client"""
    return storage.Client(project=PROJECT_ID)

def get_vertex_ai_client():
    """Initialize Vertex AI"""
    aiplatform.init(project=PROJECT_ID, location=REGION)
    return aiplatform

def check_gcp_setup():
    """Check if GCP is properly configured"""
    try:
        credentials, project = get_credentials()
        print(f"✅ GCP Authentication successful")
        print(f"   Project: {project}")
        print(f"   Region: {REGION}")
        return True
    except Exception as e:
        print(f"❌ GCP Authentication failed: {e}")
        print("\nTo set up GCP authentication:")
        print("1. Install gcloud CLI: https://cloud.google.com/sdk/docs/install")
        print("2. Run: gcloud auth application-default login")
        print("3. Set GCP_PROJECT_ID environment variable")
        return False

if __name__ == "__main__":
    check_gcp_setup()

