#!/bin/bash
# Build and push Docker image to Google Container Registry

set -e

# Load environment variables if .env exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

PROJECT_ID=${GCP_PROJECT_ID:-coms6998llms}
IMAGE_NAME="polybench-worker"
TAG="latest"
FULL_IMAGE="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"

echo "🔨 Building Docker image for project: ${PROJECT_ID}..."
echo "   Building for linux/amd64 platform (required by Cloud Run)..."
docker build --platform linux/amd64 -t ${IMAGE_NAME}:${TAG} .

echo "🏷️  Tagging image as ${FULL_IMAGE}..."
docker tag ${IMAGE_NAME}:${TAG} ${FULL_IMAGE}

echo "📤 Pushing to GCR..."
echo "   (Make sure you've run: gcloud auth configure-docker)"
docker push ${FULL_IMAGE}

echo "✅ Image pushed successfully: ${FULL_IMAGE}"
echo ""
echo "You can now use this image in Cloud Batch jobs."
echo "Verify with: gcloud container images list --repository=gcr.io/${PROJECT_ID}"

