#!/usr/bin/env python3
"""
Local test script for backtest_worker.py
Run this to test the worker logic before deploying to Cloud Run Jobs
"""
import json
import os
import tempfile
from backtest_worker import process_backtest

def create_test_config():
    """Create a test config file"""
    config = {
        'job_id': 'test-local-123',
        'model_id': 'gemini-1.5-flash',  # Use flash for faster/cheaper testing
        'prompt_template': 'Predict the probability that: {question}',
        'start_date': '2024-01-01',
        'end_date': '2024-12-31',
        'min_volume': 0,
        'categories': [],
        'max_events': 3,  # Small number for testing
        'data_path': 'markets_final_df.feather',
        'bucket_name': os.getenv('GCS_BUCKET_NAME', 'polybench-data'),
        'project_id': os.getenv('GCP_PROJECT_ID', 'coms6998llms'),
        'region': os.getenv('GCP_REGION', 'us-central1'),
    }
    
    # Create temp config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=2)
        return f.name

if __name__ == "__main__":
    print("🧪 Testing backtest_worker locally...")
    print("=" * 60)
    
    # Check if data file exists locally
    data_path = "markets_final_df.feather"
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found in current directory")
        print("   Make sure you have the data file locally or update the path")
        exit(1)
    
    # Create test config
    config_path = create_test_config()
    print(f"📝 Created test config: {config_path}")
    print(f"   Model: gemini-1.5-pro")
    print(f"   Max events: 3")
    print()
    
    try:
        # Run the worker
        print("🚀 Running backtest worker...")
        process_backtest(config_path)
        print()
        print("✅ Test completed successfully!")
        print(f"   Check the results in: results/test-local-123/")
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    finally:
        # Clean up
        if os.path.exists(config_path):
            os.unlink(config_path)
            print(f"🧹 Cleaned up test config file")

