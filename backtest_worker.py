#!/usr/bin/env python3.11
"""
Backtest Worker Script
This runs on Cloud Batch to process events and get LLM predictions
"""
import sys
import json
import pandas as pd
import ast
from google.cloud import storage
import os

def load_data_from_gcs(bucket_name, file_path):
    """Load feather file from GCS or local file system"""
    import os
    
    # Check if file exists locally (for testing)
    if os.path.exists(file_path):
        print(f"Loading data from local file: {file_path}")
        return pd.read_feather(file_path)
    
    # Otherwise load from GCS
    print(f"Loading data from GCS: gs://{bucket_name}/{file_path}")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    
    # Download to local temp file
    local_path = "/tmp/data.feather"
    blob.download_to_filename(local_path)
    
    return pd.read_feather(local_path)

def parse_outcomes(outcomes_str):
    """Parse outcomes string to list"""
    if isinstance(outcomes_str, str):
        return ast.literal_eval(outcomes_str)
    return outcomes_str

def construct_prompt(user_prompt_template, event_row):
    """Construct the full prompt by replacing placeholders"""
    prompt = user_prompt_template
    
    # Replace placeholders with actual values
    replacements = {
        '{event_ticker}': str(event_row.get('event_ticker', '')),
        '{question}': str(event_row.get('question', '')),
        '{description}': str(event_row.get('description', '')),
        '{outcomes}': str(parse_outcomes(event_row.get('outcomes', []))),
        '{event_startDate}': str(event_row.get('event_startDate', '')),
        '{event_endDate}': str(event_row.get('event_endDate', '')),
        '{category}': str(event_row.get('category', '')),
        '{event_volume}': str(event_row.get('event_volume', '')),
    }
    
    # Handle outcomePrices if present (optional)
    if '{outcomePrices}' in prompt and pd.notna(event_row.get('outcomePrices')):
        try:
            prices = ast.literal_eval(str(event_row.get('outcomePrices')))
            replacements['{outcomePrices}'] = str(prices)
        except:
            replacements['{outcomePrices}'] = 'N/A'
    
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)
    
    return prompt

def get_llm_prediction(prompt, model_id, project_id, location, max_retries=3):
    """Get prediction from Vertex AI with retry logic for rate limits"""
    from vertexai import init as vertex_init
    from vertexai.generative_models import GenerativeModel
    import time
    import random
    
    # Initialize Vertex AI
    vertex_init(project=project_id, location=location)
    
    # Create model instance
    # Vertex AI model IDs should be like "gemini-1.5-pro" or "gemini-1.5-pro-001"
    model = GenerativeModel(model_id)
    
    # Retry logic with exponential backoff
    for attempt in range(max_retries):
        try:
            # Log the request
            print(f"📤 Sending request to {model_id} (attempt {attempt + 1}/{max_retries})")
            print(f"   Prompt length: {len(prompt)} chars")
            print(f"   Prompt preview: {prompt[:200]}...")
            
            # Generate content
            response = model.generate_content(prompt)
            
            # Log the response
            text = response.text.strip()
            print(f"📥 Received response from {model_id}")
            print(f"   Response length: {len(text)} chars")
            print(f"   Response preview: {text[:200]}...")
            
            # Extract probability from response
            
            # Try to extract a number between 0 and 1
            import re
            # Match numbers like 0.5, .5, 1.0, 0.0, etc.
            numbers = re.findall(r'\b0?\.\d+\b|\b1\.0\b|\b0\.0\b', text)
            if numbers:
                prob = float(numbers[0])
                return max(0.0, min(1.0, prob))  # Clamp to [0, 1]
            
            # Fallback: look for percentage
            percentages = re.findall(r'(\d+(?:\.\d+)?)%', text)
            if percentages:
                prob = float(percentages[0]) / 100
                return max(0.0, min(1.0, prob))
            
            # Default: return 0.5 if can't parse
            print(f"⚠️  Could not parse probability from: {text[:100]}")
            return 0.5
            
        except Exception as e:
            error_str = str(e)
            # Check if it's a rate limit/quota error
            if "429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower():
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter (wait longer for each retry)
                    wait_time = (2 ** attempt) * 2 + random.uniform(0, 2)  # 2s, 4s, 8s with jitter
                    print(f"⚠️  Rate limit hit (attempt {attempt + 1}/{max_retries}), waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Rate limit error after {max_retries} attempts: {e}")
                    raise
            else:
                # For other errors, raise immediately
                print(f"❌ Error getting prediction: {e}")
                raise
    
    # Should not reach here, but just in case
    raise Exception("Failed to get prediction after retries")

def process_backtest(config_path):
    """Main function to process backtest"""
    import sys
    
    print("=" * 80)
    print("🚀 Starting PolyBench Backtest")
    print("=" * 80)
    
    # Handle GCS path or local path
    if config_path.startswith('gs://'):
        # Download config from GCS
        storage_client = storage.Client()
        bucket_name = config_path.split('/')[2]
        blob_path = '/'.join(config_path.split('/')[3:])
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        config_json = blob.download_as_text()
        config = json.loads(config_json)
    else:
        # Local file
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Log test info
    test_name = config.get('test_name', 'Unnamed Test')
    job_id = config.get('job_id', 'unknown')
    print(f"📋 Test Name: {test_name}")
    print(f"🆔 Job ID: {job_id}")
    print(f"🤖 Model: {config.get('model_id', 'unknown')}")
    print(f"📅 Date Range: {config.get('start_date', 'N/A')} to {config.get('end_date', 'N/A')}")
    print(f"📊 Max Events: {config.get('max_events', 'N/A')}")
    print("=" * 80)
    
    # Load data
    print(f"Loading data from GCS: {config['data_path']}")
    df = load_data_from_gcs(config['bucket_name'], config['data_path'])
    
    # Log initial data size
    print(f"📊 Loaded {len(df)} total events from dataset")
    
    # Apply filters
    if 'start_date' in config:
        start_dt = pd.to_datetime(config['start_date'])
        # Make timezone-naive if comparing with timezone-naive column
        if df['event_startDate'].dtype.tz is None and start_dt.tz is not None:
            start_dt = start_dt.tz_localize(None)
        elif df['event_startDate'].dtype.tz is not None and start_dt.tz is None:
            start_dt = start_dt.tz_localize('UTC')
        before_filter = len(df)
        df = df[df['event_startDate'] >= start_dt]
        print(f"   After start_date filter ({config['start_date']}): {len(df)} events (removed {before_filter - len(df)})")
    
    if 'end_date' in config:
        end_dt = pd.to_datetime(config['end_date'])
        # Make timezone-naive if comparing with timezone-naive column
        if df['event_startDate'].dtype.tz is None and end_dt.tz is not None:
            end_dt = end_dt.tz_localize(None)
        elif df['event_startDate'].dtype.tz is not None and end_dt.tz is None:
            end_dt = end_dt.tz_localize('UTC')
        before_filter = len(df)
        df = df[df['event_startDate'] <= end_dt]
        print(f"   After end_date filter ({config['end_date']}): {len(df)} events (removed {before_filter - len(df)})")
    
    if 'min_volume' in config:
        before_filter = len(df)
        df = df[df['event_volume'] >= config['min_volume']]
        print(f"   After min_volume filter (>{config['min_volume']}): {len(df)} events (removed {before_filter - len(df)})")
    
    if 'categories' in config and config['categories']:
        before_filter = len(df)
        df = df[df['category'].isin(config['categories'])]
        print(f"   After categories filter ({config['categories']}): {len(df)} events (removed {before_filter - len(df)})")
    
    # Limit to max_events (random sample if more events than requested)
    max_events = config.get('max_events', len(df))
    before_limit = len(df)
    if before_limit > max_events:
        # Random sample to get diverse events
        df = df.sample(n=max_events, random_state=42).reset_index(drop=True)
        print(f"   After max_events limit ({max_events}): {len(df)} events (randomly sampled from {before_limit})")
    else:
        df = df.reset_index(drop=True)
    
    if len(df) == 0:
        print("⚠️  WARNING: No events match the filter criteria!")
        print("   Check your date range, min_volume, and category filters.")
    else:
        print(f"✅ Processing {len(df)} events in parallel...")
    
    # Process events in parallel using asyncio
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    # Reduce concurrency to avoid rate limits (default was 10, reduce to 3 for quota limits)
    max_concurrent = min(3, config.get('max_concurrent_requests', 3))
    
    async def process_event_async(idx, row, semaphore, executor):
        """Process a single event asynchronously"""
        async with semaphore:  # Limit concurrent requests
            try:
                # Construct prompt
                prompt = construct_prompt(config['prompt_template'], row)
                
                # Get LLM prediction (run in executor since Vertex AI is sync)
                loop = asyncio.get_event_loop()
                prediction = await loop.run_in_executor(
                    executor,
                    get_llm_prediction,
                    prompt,
                    config['model_id'],
                    config['project_id'],
                    config['region']
                )
                
                # Add a small delay after each request to avoid hitting rate limits
                await asyncio.sleep(0.2)  # 200ms delay between requests
                
                # Get actual outcome
                outcomes = parse_outcomes(row.get('outcomes', []))
                
                # Determine actual outcome from outcomePrices if available
                # The outcome with price closest to 1.0 is likely the winner
                actual_outcome = None
                if pd.notna(row.get('outcomePrices')) and outcomes:
                    try:
                        prices = ast.literal_eval(str(row.get('outcomePrices')))
                        if isinstance(prices, list) and len(prices) == len(outcomes):
                            # Find outcome with highest price (closest to 1.0)
                            max_price_idx = max(range(len(prices)), key=lambda i: float(prices[i]))
                            actual_outcome = outcomes[max_price_idx] if max_price_idx < len(outcomes) else None
                    except:
                        pass
                
                return {
                    'event_id': row.get('id'),
                    'event_ticker': row.get('event_ticker'),
                    'question': row.get('question'),
                    'prediction': prediction,
                    'outcomes': outcomes,
                    'actual_outcome': actual_outcome,
                    'event_endDate': str(row.get('event_endDate')),
                }
            except Exception as e:
                print(f"Error processing event {idx}: {e}")
                return None
    
    async def process_all_events_async(df, config):
        """Process all events asynchronously"""
        # Create semaphore to limit concurrent LLM calls (avoid rate limits)
        # Reduced to 3 to avoid hitting quota limits (was 10)
        max_concurrent = config.get('max_concurrent_requests', 3)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Thread pool for running sync Vertex AI calls
        executor = ThreadPoolExecutor(max_workers=max_concurrent)
        
        # Create tasks for all events
        tasks = [
            process_event_async(idx, row, semaphore, executor)
            for idx, (_, row) in enumerate(df.iterrows())
        ]
        
        # Process in batches to show progress
        results = []
        batch_size = 50
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in batch_results:
                if result and not isinstance(result, Exception):
                    results.append(result)
                elif isinstance(result, Exception):
                    print(f"❌ Task error: {result}")
                    import traceback
                    print(f"   Traceback: {traceback.format_exc()}")
            
            print(f"Processed {min(i+batch_size, len(tasks))}/{len(tasks)} events...")
        
        executor.shutdown(wait=True)
        return results
    
    # Run async processing
    results = asyncio.run(process_all_events_async(df, config))
    
    # Log results summary
    print(f"\n{'=' * 80}")
    print(f"📊 Processing Complete")
    print(f"{'=' * 80}")
    print(f"Total events processed: {len(df)}")
    print(f"Results collected: {len(results)}")
    
    if not results:
        print("⚠️  WARNING: No results collected! Check for errors above.")
        # Create empty dataframe with expected columns
        results_df = pd.DataFrame(columns=['event_id', 'event_ticker', 'question', 'prediction', 'outcomes', 'actual_outcome', 'event_endDate'])
    else:
        # Save results
        results_df = pd.DataFrame(results)
    
    print(f"Results DataFrame shape: {results_df.shape}")
    print(f"{'=' * 80}\n")
    
    output_path = f"/tmp/results_{config['job_id']}.feather"
    results_df.to_feather(output_path)
    
    # Upload to GCS (or save locally for testing)
    try:
        storage_client = storage.Client(project=config.get('project_id'))
        bucket = storage_client.bucket(config['bucket_name'])
        blob = bucket.blob(f"results/{config['job_id']}.feather")
        blob.upload_from_filename(output_path)
        print(f"Results saved to gs://{config['bucket_name']}/results/{config['job_id']}.feather")
    except Exception as e:
        # For local testing, save to local results directory
        import os
        os.makedirs('results', exist_ok=True)
        local_results_path = f"results/{config['job_id']}.feather"
        results_df.to_feather(local_results_path)
        print(f"⚠️  Could not upload to GCS: {e}")
        print(f"   Results saved locally to: {local_results_path}")
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python backtest_worker.py <config_json_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    process_backtest(config_path)

