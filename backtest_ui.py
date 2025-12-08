import streamlit as st
from datetime import datetime
import requests
import os
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="PolyBench",
    page_icon="📊",
    layout="wide"
)

st.title("📊 PolyBench: LLM Backtesting Research Tool")

st.markdown("---")

# Configuration Section
st.header("⚙️ Configuration")

# Test name (mandatory)
test_name = st.text_input(
    "Test Name *",
    value="",
    placeholder="e.g., Baseline Gemini Flash, Experiment 1, etc.",
    help="Give your backtest a name to easily identify it later (required)"
)

# Model selection - Vertex AI models available on GCP
# Only including currently supported models (retired models return 404 errors)
models = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-image",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite-preview-09-2025",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite-001",
]
model_display_names = {
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-flash-image": "Gemini 2.5 Flash Image",
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
    "gemini-2.5-flash-lite-preview-09-2025": "Gemini 2.5 Flash Lite Preview (Sep 2025)",
    "gemini-2.0-flash-001": "Gemini 2.0 Flash",
    "gemini-2.0-flash-lite-001": "Gemini 2.0 Flash Lite",
}
selected_model = st.selectbox(
    "Select LLM Model (Vertex AI)",
    options=models,
    format_func=lambda x: model_display_names[x],
    index=0,
    help="All models are accessed through Vertex AI on GCP"
)

st.markdown("---")

# Prediction Prompt
st.subheader("Prediction Prompt")

# Available placeholders info
with st.expander("📋 Available Placeholders", expanded=False):
    st.markdown("""
    You can use these placeholders in your prompt template:
    
    **Core Fields (always available):**
    - `{question}` - The prediction question
    - `{description}` - Detailed market description
    - `{outcomes}` - List of possible outcomes (usually ["Yes", "No"])
    
    **Additional Fields:**
    - `{event_ticker}` - Event identifier (auto-generated from title)
    - `{event_title}` - Full event title
    - `{event_startDate}` - When the event starts
    - `{event_endDate}` - When the event resolves
    - `{category}` - Event category (if available)
    - `{event_volume}` - Trading volume
    - `{prediction_date}` - Date that prediction takes place and is evaluted at.
    
    **Price Fields (new dataset):**
    - `{Price_Start}` - Price at event start
    - `{Price_Mid}` - Price at mid-point
    - `{Price_End}` - Price at event end
    
    """)

default_prompt = """You are an expert forecaster. Analyze the following prediction market and provide a probability between 0 and 1 for the outcome.

Question: {question}

Description: {description}

Possible Outcomes: {outcomes}

Resolution Date: {event_endDate}

Provide your probability prediction as a single number between 0 and 1 for the first outcome listed above."""

user_prompt = st.text_area(
    "Enter your prompt template",
    value=default_prompt,
    height=250,
    help="Use placeholders like {question}, {description}, {outcomes}, etc. that will be filled with event data"
)

st.markdown("---")

# Event Filters
st.subheader("Event Filters")

# Data availability info
with st.expander("📊 Dataset Information", expanded=False):
    st.markdown("""
    **Total Events:** 24,639
    
    **Data Distribution by Year:**
    - 2025: 16,882 events (68.5%)
    - 2024: 6,329 events (25.7%)
    - 2023: 1,342 events (5.4%)
    - 2022: 34 events (0.1%)
    - 2021: 52 events (0.2%)
    
    **Date Range:** January 1, 2021 to November 29, 2025
    """)

filter_col1, filter_col2 = st.columns(2)

with filter_col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime(2024, 1, 1)
    )
    
    min_volume = st.number_input(
        "Minimum Volume",
        min_value=0,
        value=1000,
        step=1000
    )

with filter_col2:
    end_date = st.date_input(
        "End Date",
        value=datetime(2024, 12, 31)
    )
    
    # Categories available in the new dataset
    # Note: Only ~0.06% of events (15 out of 24,639) have category labels
    available_categories = [
        "US-current-affairs"
    ]
    categories = st.multiselect(
        "Categories (optional)",
        available_categories,
        default=[],
        help="⚠️ Warning: Only ~0.06% of events have category labels. Filtering by category will exclude almost all events."
    )

# Sample size configuration
st.markdown("---")
st.subheader("Backtest Sample Size")
max_events = st.number_input(
    "Number of events to backtest",
    min_value=1,
    value=10,
    step=1,
    help="Limit the number of events to process (useful for testing). Set to a large number to process all matching events."
)

st.markdown("---")

# Launch Section
st.header("🚀 Launch Backtest")

# Summary of configuration
with st.container():
    st.markdown("### Configuration Summary")
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    with summary_col1:
        st.write(f"**Model:** {model_display_names[selected_model]}")
    with summary_col2:
        st.write(f"**Date Range:** {start_date} to {end_date}")
    with summary_col3:
        st.write(f"**Min Volume:** {min_volume:,}")
    st.write(f"**Max Events:** {max_events:,}")
    if categories:
        st.write(f"**Categories:** {', '.join(categories)}")

if max_events >= 10000:
    st.info("💡 This will process up to all matching events in the dataset. Results will be saved and accessible below.")
else:
    st.info(f"💡 This will process up to {max_events:,} matching events. Results will be saved and accessible below.")

# Submit button - centered and prominent (disabled if no test name)
submit_button = st.button(
    "🚀 Run Backtest",
    type="primary",
    use_container_width=True,
    disabled=(not test_name or not test_name.strip())
)

# Show warning if test name is missing
if not test_name or not test_name.strip():
    st.warning("⚠️ Please enter a test name before submitting.")

# Sidebar with info
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    Polybench allows you to backtest LLM predictions on historical Polymarket events.
    
    **How it works:**
    1. Configure model, prompt, and filters
    2. Launch the backtest
    3. View results and metrics
    """)

# Results section (initially hidden, shown after submission)
if submit_button:
    # Validate test name is provided
    if not test_name or not test_name.strip():
        st.error("❌ Test name is required. Please enter a name for your backtest.")
    else:
        st.markdown("---")
        st.header("Running Backtest...")
        
        # Submit job to API (which submits to Cloud Run Jobs)
        api_url = os.getenv("API_URL", "http://localhost:8000")
        
        with st.spinner("Submitting backtest job to Cloud Run Jobs..."):
            try:
                # Prepare request
                request_data = {
                    "model_id": selected_model,
                    "prompt_template": user_prompt,
                    "start_date": str(start_date),
                    "end_date": str(end_date),
                    "min_volume": int(min_volume),
                    "categories": categories if categories else None,
                    "max_events": int(max_events),
                    "test_name": test_name.strip()
                }
                
                # Submit to API
                response = requests.post(
                    f"{api_url}/api/submit-backtest",
                    json=request_data,
                    timeout=120  # Increased timeout for Cloud Run Jobs creation
                )
                
                if response.status_code == 200:
                    result = response.json()
                    job_id = result['job_id']
                    
                    st.success(f"✅ Backtest job submitted! Job ID: `{job_id}`")
                    st.info("The backtest is running on Cloud Run Jobs. Check back in a few minutes for results.")
                    
                    # Store job info in session state
                    if 'jobs' not in st.session_state:
                        st.session_state.jobs = []
                    
                    st.session_state.jobs.append({
                        'job_id': job_id,
                        'model': model_display_names[selected_model],
                        'model_id': selected_model,
                        'date_range': f"{start_date} to {end_date}",
                        'max_events': max_events,
                        'timestamp': datetime.now().isoformat(),
                        'status': 'running'
                    })
                else:
                    st.error(f"❌ Error submitting job: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Could not connect to API at {api_url}")
                st.info("💡 Make sure the API server is running. Start it with: `python api.py`")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Results Viewer Section
st.markdown("---")
st.header("📊 Results Viewer")

try:
    from google.cloud import storage
    import pandas as pd
    import tempfile
    
    # Get config from environment
    bucket_name = os.getenv("GCS_BUCKET_NAME", "polybench-data")
    project_id = os.getenv("GCP_PROJECT_ID", "coms6998llms")
    
    # List available results
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    results_blobs = list(bucket.list_blobs(prefix="results/"))
    
    # Filter for .feather files and extract job IDs
    result_files = []
    for blob in results_blobs:
        if blob.name.endswith('.feather') and blob.name.startswith('results/'):
            job_id = blob.name.replace('results/', '').replace('.feather', '')
            
            # Try to load config to get test name
            test_name = None
            try:
                config_blob = bucket.blob(f"jobs/{job_id}/config.json")
                if config_blob.exists():
                    config_json = config_blob.download_as_text()
                    config = json.loads(config_json)
                    test_name = config.get('test_name')
            except:
                pass  # If config doesn't exist or can't be loaded, just use job_id
            
            result_files.append({
                'job_id': job_id,
                'blob_name': blob.name,
                'size': blob.size,
                'created': blob.time_created,
                'test_name': test_name
            })
    
    if result_files:
        # Sort by creation time (newest first)
        result_files.sort(key=lambda x: x['created'], reverse=True)
        
        # Select result to view - show test name if available
        result_options = []
        for r in result_files:
            display_name = r['test_name'] if r['test_name'] else r['job_id']
            result_options.append(f"{display_name} ({r['created'].strftime('%Y-%m-%d %H:%M')})")
        selected_result = st.selectbox(
            "Select a backtest result to view:",
            options=result_options,
            index=0
        )
        
        if selected_result:
            # Extract display name from selection (could be test_name or job_id)
            selected_display_name = selected_result.split(' (')[0]
            # Try to find by test_name first, then fall back to job_id
            selected_file = None
            for r in result_files:
                display_name = r['test_name'] if r['test_name'] else r['job_id']
                if display_name == selected_display_name:
                    selected_file = r
                    break
            
            if not selected_file:
                st.error(f"❌ Could not find result file for: {selected_display_name}")
                st.stop()
            
            selected_job_id = selected_file['job_id']
            
            # Load button
            if st.button("📥 Load Results", key="load_results"):
                with st.spinner("Loading results from GCS..."):
                    try:
                        # Download result file
                        blob = bucket.blob(selected_file['blob_name'])
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.feather') as tmp_file:
                            blob.download_to_filename(tmp_file.name)
                            df = pd.read_feather(tmp_file.name)
                            os.unlink(tmp_file.name)
                        
                        # Validate dataframe
                        if df.empty:
                            st.warning("⚠️ Results file is empty")
                        elif 'prediction' not in df.columns:
                            st.error(f"❌ Missing 'prediction' column. Available columns: {', '.join(df.columns)}")
                        else:
                            # Store in session state
                            st.session_state['current_results'] = df
                            st.session_state['current_job_id'] = selected_job_id
                            st.success(f"✅ Loaded {len(df)} predictions")
                            st.rerun()  # Refresh to show results
                    except Exception as e:
                        import traceback
                        st.error(f"❌ Error loading results: {e}")
                        with st.expander("Error details"):
                            st.code(traceback.format_exc())
            
            # Display results if loaded
            if 'current_results' in st.session_state and st.session_state.get('current_job_id') == selected_job_id:
                df = st.session_state['current_results']
                
                # Check if dataframe is empty or missing columns
                if df.empty:
                    st.warning("⚠️ Results file is empty")
                elif 'prediction' not in df.columns:
                    st.error(f"❌ Missing 'prediction' column. Available columns: {', '.join(df.columns)}")
                else:
                    # Simple results display
                    st.markdown("### 📊 Results")
                    
                    # Number of predictions
                    st.metric("Number of Predictions", len(df))
                    
                    # Prediction distribution plot
                    st.markdown("#### Prediction Distribution")
                    # Create histogram using matplotlib (compact size)
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(6, 2.5))
                    ax.hist(df['prediction'], bins=20, edgecolor='black', alpha=0.7)
                    ax.set_xlabel('Prediction Value', fontsize=9)
                    ax.set_ylabel('Frequency', fontsize=9)
                    ax.set_title('Distribution of Predictions', fontsize=10)
                    ax.tick_params(labelsize=8)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=False)
                    
                    # Accuracy based on cutoff
                    st.markdown("#### Accuracy by Cutoff")
                    cutoff = st.slider(
                        "Prediction Cutoff",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.05,
                        help="Predictions above this threshold are considered 'Yes'"
                    )
                    
                    # Show predictions above/below cutoff
                    predicted_yes_count = (df['prediction'] >= cutoff).sum()
                    predicted_no_count = (df['prediction'] < cutoff).sum()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predictions ≥ Cutoff", predicted_yes_count)
                    with col2:
                        st.metric("Predictions < Cutoff", predicted_no_count)
                    
                    # Calculate accuracy if we have actual outcomes
                    if 'actual_outcome' in df.columns:
                        df_with_outcomes = df[df['actual_outcome'].notna()].copy()
                        if len(df_with_outcomes) > 0:
                            st.markdown("#### Accuracy & Brier Score")
                            # Accuracy calculation using FirstOutcome (actual resolved outcome from new dataset)
                            # Logic: prediction >= cutoff means we predicted the first outcome will win
                            #        actual_outcome (FirstOutcome) == first outcome means first outcome won
                            correct = 0
                            total = 0
                            brier_scores = []  # Store individual Brier scores
                            errors = []
                            import ast
                            for idx, row in df_with_outcomes.iterrows():
                                try:
                                    outcomes_raw = row.get('outcomes', [])
                                    actual = row['actual_outcome']  # This is FirstOutcome from the new dataset
                                    
                                    # Skip if actual is null/na (avoid pd.isna on arrays - it returns arrays)
                                    if actual is None:
                                        continue
                                    # Check for numpy arrays first
                                    if isinstance(actual, np.ndarray):
                                        continue
                                    # Only check pd.isna for scalar types
                                    try:
                                        if pd.isna(actual):
                                            continue
                                    except (ValueError, TypeError):
                                        # pd.isna might fail on some types, just continue
                                        pass
                                    
                                    # Parse outcomes - handle different storage formats
                                    # Check for numpy array FIRST before any boolean operations (pd.isna on arrays causes errors)
                                    outcomes = []
                                    if outcomes_raw is None:
                                        continue
                                    elif isinstance(outcomes_raw, np.ndarray):
                                        # Convert numpy array to list immediately - this is the most common case
                                        try:
                                            # Convert to list first, then process
                                            outcomes_list = outcomes_raw.tolist()
                                            outcomes = [str(o).strip() for o in outcomes_list]
                                            # Ensure it's actually a list, not still an array
                                            if not isinstance(outcomes, list):
                                                outcomes = list(outcomes)
                                        except Exception as e:
                                            # Skip this row if conversion fails
                                            errors.append(f"Row {idx}: Failed to convert numpy array: {str(e)}")
                                            continue
                                    elif isinstance(outcomes_raw, str):
                                        # Remove quotes and brackets if present
                                        cleaned = outcomes_raw.strip().strip('[]').strip('"').strip("'")
                                        try:
                                            # Try parsing as string representation of list
                                            parsed = ast.literal_eval(outcomes_raw)
                                            if isinstance(parsed, list):
                                                outcomes = [str(o).strip() for o in parsed]
                                            else:
                                                outcomes = []
                                        except:
                                            # If parsing fails, try splitting by comma
                                            outcomes = [o.strip().strip('"').strip("'") for o in cleaned.split(',')]
                                    elif isinstance(outcomes_raw, (list, tuple)):
                                        outcomes = [str(o).strip() for o in outcomes_raw]
                                    else:
                                        # Try to convert to list
                                        try:
                                            outcomes = [str(o).strip() for o in list(outcomes_raw)]
                                        except:
                                            outcomes = []
                                    
                                    # Ensure we have at least 2 outcomes
                                    if len(outcomes) >= 2:
                                        # Prediction >= cutoff means we predicted the first outcome will win
                                        predicted_first_outcome = bool(float(row['prediction']) >= cutoff)
                                        
                                        # Check if the actual outcome (determined from Price_End) matches the first outcome
                                        # actual_outcome is determined from Price_End: if Price_End > 0.5, first outcome won
                                        actual_str = str(actual).strip().strip('"').strip("'")
                                        first_outcome_str = str(outcomes[0]).strip().strip('"').strip("'")
                                        first_outcome_won = (actual_str.lower() == first_outcome_str.lower())
                                        
                                        # We're correct if our prediction matches reality
                                        if predicted_first_outcome == first_outcome_won:
                                            correct += 1
                                        
                                        # Calculate Brier Score contribution
                                        # Brier Score = (prediction - outcome)²
                                        # outcome = 1 if first outcome won, 0 if second outcome won
                                        outcome_binary = 1.0 if first_outcome_won else 0.0
                                        prediction_value = float(row['prediction'])
                                        brier_contribution = (prediction_value - outcome_binary) ** 2
                                        brier_scores.append(brier_contribution)
                                        
                                        total += 1
                                    else:
                                        errors.append(f"Row {idx}: Only {len(outcomes)} outcomes found")
                                except Exception as e:
                                    # Track errors for debugging
                                    errors.append(f"Row {idx}: {str(e)}")
                                    continue
                            
                            if total > 0:
                                accuracy = correct / total
                                # Calculate Brier Score: mean of squared differences
                                brier_score = sum(brier_scores) / len(brier_scores) if brier_scores else None
                                
                                # Display metrics side by side
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Accuracy", f"{accuracy:.1%}", f"{correct}/{total} correct")
                                with col2:
                                    if brier_score is not None:
                                        st.metric("Brier Score", f"{brier_score:.4f}", 
                                                 help="Lower is better (0 = perfect, 1 = worst)")
                                    else:
                                        st.metric("Brier Score", "N/A")
                            else:
                                # Show debug info
                                with st.expander("🔍 Debug Info (click to see why accuracy couldn't be calculated)"):
                                    st.write(f"Rows with outcomes: {len(df_with_outcomes)}")
                                    if errors:
                                        st.write("Errors encountered:")
                                        for err in errors[:5]:  # Show first 5 errors
                                            st.code(err)
                                    # Show sample data
                                    sample = df_with_outcomes.iloc[0]
                                    st.write("Sample row data:")
                                    st.json({
                                        "outcomes": str(sample.get('outcomes')),
                                        "outcomes_type": str(type(sample.get('outcomes'))),
                                        "actual_outcome": str(sample.get('actual_outcome')),
                                        "actual_outcome_type": str(type(sample.get('actual_outcome')))
                                    })
                                st.warning(f"⚠️ Could not calculate accuracy from {len(df_with_outcomes)} rows with outcomes.")
                        else:
                            st.info("ℹ️ No events with resolved outcomes available for accuracy calculation.")
                    else:
                        st.info("ℹ️ Actual outcomes not available in results. Accuracy calculation requires resolved outcomes.")
                    
                    # Preview table with pagination
                    st.markdown("#### Preview")
                    
                    # Initialize page number in session state
                    if 'preview_page' not in st.session_state:
                        st.session_state.preview_page = 0
                    
                    rows_per_page = 10
                    total_rows = len(df)
                    total_pages = (total_rows + rows_per_page - 1) // rows_per_page  # Ceiling division
                    
                    # Pagination controls
                    col1, col2, col3, col4 = st.columns([1, 1, 2, 1])
                    with col1:
                        prev_clicked = st.button("◀ Previous", disabled=(st.session_state.preview_page == 0), key="prev_button")
                        if prev_clicked:
                            st.session_state.preview_page = max(0, st.session_state.preview_page - 1)
                            st.rerun()
                    with col2:
                        next_clicked = st.button("Next ▶", disabled=(st.session_state.preview_page >= total_pages - 1), key="next_button")
                        if next_clicked:
                            st.session_state.preview_page = min(total_pages - 1, st.session_state.preview_page + 1)
                            st.rerun()
                    with col3:
                        st.caption(f"Page {st.session_state.preview_page + 1} of {total_pages} (Showing rows {st.session_state.preview_page * rows_per_page + 1}-{min((st.session_state.preview_page + 1) * rows_per_page, total_rows)} of {total_rows})")
                    with col4:
                        # Jump to page
                        jump_page = st.number_input("Go to page", min_value=1, max_value=total_pages, value=st.session_state.preview_page + 1, key="jump_page_input")
                        if jump_page != st.session_state.preview_page + 1:
                            st.session_state.preview_page = jump_page - 1
                            st.rerun()
                    
                    # Get the current page of data
                    start_idx = st.session_state.preview_page * rows_per_page
                    end_idx = start_idx + rows_per_page
                    page_df = df.iloc[start_idx:end_idx].copy()
                    
                    preview_df = page_df[['event_ticker', 'question', 'prediction']].copy()
                    
                    # Add actual outcome if available
                    if 'actual_outcome' in df.columns:
                        preview_df['actual_outcome'] = page_df['actual_outcome']
                    
                    # Calculate if prediction was correct
                    def is_correct(row):
                        """Determine if prediction was correct based on cutoff"""
                        if 'actual_outcome' not in row.index or pd.isna(row.get('actual_outcome')):
                            return "N/A"
                        
                        # Get outcomes from original df using the index
                        original_idx = row.name
                        outcomes_raw = df.loc[original_idx, 'outcomes'] if 'outcomes' in df.columns else None
                        if outcomes_raw is None:
                            return "N/A"
                        
                        # Parse outcomes
                        outcomes = []
                        try:
                            if isinstance(outcomes_raw, np.ndarray):
                                outcomes = [str(o).strip() for o in outcomes_raw.tolist()]
                            elif isinstance(outcomes_raw, list):
                                outcomes = [str(o).strip() for o in outcomes_raw]
                            elif isinstance(outcomes_raw, str):
                                import ast
                                try:
                                    parsed = ast.literal_eval(outcomes_raw)
                                    if isinstance(parsed, list):
                                        outcomes = [str(o).strip() for o in parsed]
                                except:
                                    outcomes = [o.strip() for o in outcomes_raw.strip('[]').split(',')]
                        except:
                            return "N/A"
                        
                        if len(outcomes) < 2:
                            return "N/A"
                        
                        # Check if prediction matches actual (actual_outcome determined from Price_End)
                        # Prediction >= cutoff means we predicted the first outcome will win
                        # actual_outcome is determined from Price_End: if Price_End > 0.5, first outcome won
                        predicted_first_outcome = float(row['prediction']) >= cutoff
                        actual = str(row['actual_outcome']).strip()  # Determined from Price_End in worker
                        first_outcome = str(outcomes[0]).strip()
                        first_outcome_won = (actual.lower() == first_outcome.lower())
                        
                        return "✓" if (predicted_first_outcome == first_outcome_won) else "✗"
                    
                    # Apply correctness check
                    preview_df['correct'] = preview_df.apply(is_correct, axis=1)
                    
                    # Format prediction as percentage
                    preview_df['prediction'] = preview_df['prediction'].apply(lambda x: f"{x:.1%}")
                    
                    # Select and rename columns for display
                    display_cols = ['event_ticker', 'question', 'prediction']
                    col_names = ['Ticker', 'Question', 'Prediction']
                    
                    if 'actual_outcome' in preview_df.columns:
                        display_cols.append('actual_outcome')
                        col_names.append('Actual Outcome')
                    
                    display_cols.append('correct')
                    col_names.append('Correct')
                    
                    preview_display = preview_df[display_cols].copy()
                    preview_display.columns = col_names
                    
                    st.dataframe(preview_display, use_container_width=True, hide_index=True)
    else:
        st.info("No results found. Run a backtest to generate results.")
        
except ImportError:
    st.warning("⚠️ Google Cloud Storage not available. Results viewer requires GCS access.")
except Exception as e:
    st.error(f"❌ Error loading results: {e}")
    st.info("Make sure you have GCS access configured.")

