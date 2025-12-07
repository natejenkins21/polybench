#!/bin/bash
# Check Cloud Run Job execution status

EXECUTION_ID=${1:-""}
PROJECT_ID=${GCP_PROJECT_ID:-coms6998llms}
REGION=${GCP_REGION:-us-central1}
JOB_NAME="polybench-worker"

if [ -z "$EXECUTION_ID" ]; then
    echo "Usage: ./check_job.sh <execution_id>"
    echo "Listing recent executions:"
    gcloud run jobs executions list --job=$JOB_NAME --region=$REGION --project=$PROJECT_ID \
        --format="table(name,status.conditions[0].type,status.completionTime)" --limit=10
else
    gcloud run jobs executions describe $EXECUTION_ID --region=$REGION --project=$PROJECT_ID \
        --format="yaml(status.conditions,status.logUri)" 2>/dev/null || \
    gcloud run jobs executions list --job=$JOB_NAME --region=$REGION --project=$PROJECT_ID \
        --filter="name:$EXECUTION_ID"
fi

