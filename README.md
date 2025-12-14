# PolyBench: LLM Backtesting Research Tool

A backtesting framework for evaluating LLM-based prediction strategies on historical Polymarket data.

## Architecture

- **UI**: Streamlit app (`backtest_ui.py`) - User interface for configuring and launching backtests
- **API**: FastAPI server (`api.py`) - Receives requests and submits Cloud Run Jobs
- **Worker**: Docker container (`backtest_worker.py`) - Runs on Cloud Run Jobs, processes events in parallel
- **Storage**: GCS bucket for data and results

## Quick Start

### 1. Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Authenticate with GCP
gcloud auth application-default login
gcloud auth configure-docker
```

### 2. Configuration

Default GCP settings are in `gcp_config.py`:
- Project: `coms6998llms`
- Region: `us-central1`
- Bucket: `polybench-data`

Override with environment variables if needed:
```bash
export GCP_PROJECT_ID=your-project
export GCS_BUCKET_NAME=your-bucket
```

### 3. Setup GCP Resources

```bash
# Enable APIs
gcloud services enable run.googleapis.com aiplatform.googleapis.com containerregistry.googleapis.com --project=coms6998llms

# Create service account
gcloud iam service-accounts create polybench-worker --display-name="PolyBench Worker" --project=coms6998llms

# Grant permissions
gcloud projects add-iam-policy-binding coms6998llms \
    --member="serviceAccount:polybench-worker@coms6998llms.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding coms6998llms \
    --member="serviceAccount:polybench-worker@coms6998llms.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

# Upload data to GCS
gsutil cp markets_final_df.feather gs://polybench-data/
```

### 4. Build and Push Docker Image

```bash
./build_and_push.sh
```

### 5. Run the System

**Terminal 1 - Start API:**
```bash
./start_api.sh
# Or: python3.11 -m uvicorn api:app --host 0.0.0.0 --port 8000
```

**Terminal 2 - Start UI:**
```bash
streamlit run backtest_ui.py
```

Open `http://localhost:8501` in your browser.

### 6. Monitor Jobs

```bash
# List job executions
gcloud run jobs executions list --job=polybench-worker --region=us-central1 --project=coms6998llms

# Check specific execution
gcloud run jobs executions describe <execution_id> --region=us-central1 --project=coms6998llms

# View results
gsutil ls gs://polybench-data/results/
```

## Key Features

- **Parallel Processing**: Worker processes up to 10 LLM calls concurrently
- **Flexible Prompts**: Use placeholders like `{question}`, `{description}`, `{outcomes}`
- **Event Filtering**: Filter by date range, volume, categories
- **Multiple Models**: Support for Gemini models via Vertex AI
- **Remote Execution**: Runs on Cloud Batch, no local compute needed

## File Structure

```
.
├── build_polymarket_dataset.ipynb  # Pull data and preprocess
├── backtest_ui.py          # Streamlit UI
├── api.py                  # FastAPI backend
├── submit_job.py           # Cloud Batch job submission
├── backtest_worker.py      # Worker script (runs in Docker)
├── gcp_config.py           # GCP configuration
├── Dockerfile              # Docker image for worker
├── build_and_push.sh       # Build and push script
├── start_api.sh            # Start API server
├── check_job.sh            # Check job status
└── requirements.txt        # Python dependencies
```

## Usage

1. Configure backtest in UI:
   - Select model (e.g., `gemini-1.5-pro`)
   - Enter prompt template
   - Set filters (date range, max events, etc.)

2. Submit job - UI sends request to API → API submits to Cloud Batch

3. Monitor - Check job status in UI or via `check_job.sh`

4. Results - Saved to `gs://polybench-data/results/<job_id>.feather`

## Notes

- Worker processes events in parallel (10 concurrent by default)
- Job IDs use format: `backtest-{base26-timestamp}`
- Results include: event_id, question, prediction, outcomes, dates
