"""
FastAPI backend for PolyBench
Handles job submission and status checking
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import os
from submit_job import submit_backtest_job

app = FastAPI(title="PolyBench API")

# CORS middleware for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BacktestRequest(BaseModel):
    model_id: str
    prompt_template: str
    start_date: str
    end_date: str
    min_volume: int = 0
    categories: Optional[List[str]] = None
    max_events: int = 10
    test_name: Optional[str] = None

class BacktestResponse(BaseModel):
    job_id: str
    status: str
    message: str

@app.get("/")
def root():
    return {"message": "PolyBench API", "status": "running"}

@app.post("/api/submit-backtest", response_model=BacktestResponse)
async def submit_backtest(request: BacktestRequest):
    """Submit a new backtest job"""
    try:
        execution = submit_backtest_job(
            model_id=request.model_id,
            prompt_template=request.prompt_template,
            start_date=request.start_date,
            end_date=request.end_date,
            min_volume=request.min_volume,
            categories=request.categories,
            max_events=request.max_events,
            test_name=request.test_name
        )
        
        # Extract execution name from the execution
        # Execution name format: projects/.../locations/.../jobs/.../executions/...
        execution_name = execution.name.split('/')[-1] if '/' in execution.name else execution.name
        
        return BacktestResponse(
            job_id=execution_name,
            status="submitted",
            message="Backtest job submitted successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/job-status/{job_id}")
def get_job_status(job_id: str):
    """Get status of a backtest job"""
    # TODO: Implement job status checking from Cloud Batch
    return {
        "job_id": job_id,
        "status": "running",  # Placeholder
        "progress": 0.5
    }

@app.get("/api/jobs")
def list_jobs():
    """List all backtest jobs"""
    # TODO: Implement job listing from Cloud Batch
    return {"jobs": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

