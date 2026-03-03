import streamlit as st
import pandas as pd
import altair as alt
import os

from main import profile_model, simulate_model_over_time, MODEL_PATH
from boards_db import get_eligible_boards, get_board_tier_summary
from model_converter import ModelConverter, get_models_temp_dir, cleanup_models_dir
from sample_models_db import get_available_samples, get_sample_model_path, all_samples_available

# ── Streamlit Configuration ────────────────────────────────────
st.set_page_config(page_title="TinyML RAM Profiler", layout="wide", initial_sidebar_state="expanded")

# ── Initialize Session State ────────────────────────────────────
if 'current_model_path' not in st.session_state:
    st.session_state.current_model_path = None
if 'current_model_name' not in st.session_state:
    st.session_state.current_model_name = None
if 'models_dir' not in st.session_state:
    st.session_state.models_dir = get_models_temp_dir()
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = 'light'

# ── Top Navigation Bar ────────────────────────────────────────
col_home, col_title, col_theme = st.columns([1, 8, 2])

with col_home:
    if st.button("🏠", help="Back to Home", use_container_width=True, key="home_btn"):
        st.session_state.current_model_path = None
        st.session_state.current_model_name = None
        st.rerun()

with col_title:
    st.title("🤖 TinyML RAM Profiler Dashboard")

with col_theme:
    st.caption(f"**Theme:** {st.session_state.theme_mode.title()}")

# ── Sidebar Configuration ────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings & Info")
    
    # Theme toggle in sidebar
    st.markdown("**Theme Toggle:**")
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.theme_mode == 'light':
            if st.button("🌙 Dark Mode", use_container_width=True, key="dark_toggle"):
                st.session_state.theme_mode = 'dark'
                st.rerun()
        else:
            if st.button("☀️ Light Mode", use_container_width=True, key="light_toggle"):
                st.session_state.theme_mode = 'light'
                st.rerun()
    
    with col2:
        st.caption(f"Current: {st.session_state.theme_mode.title()}")
    
    st.divider()
    st.markdown("### 📊 Project Info")
    st.caption("""
    **TinyML RAM Profiler**
    
    Test ML models for embedded systems
    
    Profile RAM usage across different arena sizes
    """)
    
    st.divider()
    
    if st.session_state.current_model_path:
        st.markdown("### 📦 Current Model")
        st.info(f"**{st.session_state.current_model_name}**")
    else:
        st.markdown("### 📦 No Model Loaded")
        st.caption("Load a model from home to get started")

# ── Apply Theme CSS ────────────────────────────────────────
if st.session_state.theme_mode == 'dark':
    theme_css = """
    <style>
    body {
        background-color: #0e1117;
        color: #c9d1d9;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    [data-testid="stSidebar"] {
        background-color: #010409;
    }
    .stMarkdown {
        color: #c9d1d9;
    }
    </style>
    """
else:
    theme_css = """
    <style>
    body {
        background-color: #ffffff;
        color: #24292f;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff;
        color: #24292f;
    }
    [data-testid="stSidebar"] {
        background-color: #f6f8fa;
    }
    .stMarkdown {
        color: #24292f;
    }
    </style>
    """

st.markdown(theme_css, unsafe_allow_html=True)

# SECTION 1: Model Upload & Management
# ──────────────────────────────────────────────────────────────
st.header("📤 Step 1: Upload & Convert Model")

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### Upload your ML model")
    st.markdown("Supported formats: `.h5` (Keras), `.pb` (TensorFlow), `.tflite` (TensorFlow Lite)")
    
    uploaded_file = st.file_uploader(
        "Choose a model file",
        type=['h5', 'pb', 'tflite', 'zip'],
        accept_multiple_files=False
    )

with col2:
    st.markdown("### Current Model")
    if st.session_state.current_model_path:
        st.success(f"✅ Loaded: {st.session_state.current_model_name}")
    else:
        st.info("No model loaded yet")

# Process uploaded file
if uploaded_file is not None:
    st.subheader("🔄 Converting Model...")
    
    # Create progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("📦 Processing file...")
    progress_bar.progress(20)
    
    # Process the file
    success, message, model_path = ModelConverter.process_model_file(
        uploaded_file, 
        st.session_state.models_dir
    )
    
    progress_bar.progress(100)
    
    if success:
        st.session_state.current_model_path = model_path
        st.session_state.current_model_name = os.path.basename(uploaded_file.name)
        status_text.text(f"✅ Conversion Complete!")
        st.success(message)
        st.balloons()
    else:
        st.error(message)
        st.session_state.current_model_path = None
        st.session_state.current_model_name = None

# Model management
if st.session_state.current_model_path:
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Remove Current Model", use_container_width=True):
            model_path = st.session_state.current_model_path
            is_sample_model = "sample_database" in model_path
            
            try:
                # Only delete from filesystem if it's a user-uploaded model (not a sample)
                if not is_sample_model and os.path.exists(model_path):
                    os.remove(model_path)
                    deletion_msg = "User model deleted."
                else:
                    deletion_msg = "Sample model unloaded."
                
                # Always clear session state
                st.session_state.current_model_path = None
                st.session_state.current_model_name = None
                st.success(f"✅ {deletion_msg} Back to home.")
                st.rerun()
            except Exception as e:
                st.error(f"Error removing model: {e}")
    
    with col2:
        st.info(f"📊 Model size: ~{os.path.getsize(st.session_state.current_model_path) / 1024:.1f} KB")
    
    with col3:
        st.caption(f"Stored in: Temporary directory")

else:
    # ──────────────────────────────────────────────────────────
    # Show Sample Models when no model is loaded
    st.divider()
    st.header("📚 Try Sample Models (No Upload Needed)")
    
    sample_models = get_available_samples()
    available_count = sum(1 for m in sample_models if m['exists'])
    
    if available_count > 0:
        st.info(f"✅ {available_count} sample models available • Different ML tasks for IoT")
        
        for model_info in sample_models:
            if model_info['exists']:
                with st.container(border=True):
                    col1, col2, col3 = st.columns([2, 3, 1])
                    
                    with col1:
                        st.subheader(model_info['name'])
                        st.caption(model_info['description'])
                    
                    with col2:
                        st.markdown(f"**Task:** {model_info['task']}")
                        st.caption(f"📥 Input: {model_info['input']}")
                        st.caption(f"📤 Output: {model_info['output']}")
                        st.caption(f"🎯 Use case: {model_info['use_case']}")
                    
                    with col3:
                        if st.button("▶️ Test", key=f"test_{model_info['filename']}", use_container_width=True):
                            # Load sample model
                            model_path = model_info['path']
                            st.session_state.current_model_path = model_path
                            st.session_state.current_model_name = model_info['filename']
                            st.success(f"✅ Loaded: {model_info['name']}")
                            st.rerun()
        
        # Expandable section with more details
        with st.expander("📊 Model Comparison & Details"):
            comparison_data = []
            for model_info in sample_models:
                if model_info['exists']:
                    comparison_data.append({
                        'Model': model_info['name'],
                        'Task': model_info['task'],
                        'Complexity': model_info['complexity'],
                        'Expected RAM': f"{model_info['expected_ram_kb']} KB",
                        'Use Case': model_info['use_case']
                    })
            
            if comparison_data:
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ No sample models found. Run `python generate_samples.py` to generate them.")
        
        with st.expander("📖 How to generate sample models"):
            st.code("""
# In the project directory, run:
python generate_samples.py

This will create 5 sample TFLite models in the 'sample_database' folder:
1. 🌸 Iris Classifier - Multi-class classification
2. 🏠 Housing Predictor - Linear regression
3. 🔢 Digit Recognizer - Image classification (MNIST)
4. 🚶 Activity Classifier - Time-series classification
5. ⚠️ Anomaly Detector - Binary classification
            """, language="bash")
    
    st.info("👆 Or upload your own custom ML model above!")


# ──────────────────────────────────────────────────────────────
# SECTION 2: Profiling Parameters (only show if model loaded)
# ──────────────────────────────────────────────────────────────
if st.session_state.current_model_path:
    st.divider()
    st.header("⚙️ Step 2: Configure Profiling Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        threshold_pct = st.slider("Improvement threshold (%)", 0.0, 20.0, 5.0)
    with col2:
        iterations = st.slider("Iterations per arena (for averaging)", 1, 20, 5)
    with col3:
        st.info(f"Arena sizes: 8, 16, 32, 48, 64, 96, 128 KB")
    
    threshold = threshold_pct / 100.0
    
    # ──────────────────────────────────────────────────────────
    # SECTION 3: Run Profiler
    # ──────────────────────────────────────────────────────────
    st.divider()
    st.header("🧪 Step 3: Run Profiler")
    
    if st.button("▶️ Run Profiler", key="run_profiler", use_container_width=True):
        with st.spinner("Running tests..."):
            results, min_ram, opt_ram = profile_model(
                st.session_state.current_model_path, 
                threshold, 
                iterations
            )

            if not results:
                st.error("No results returned; check the model or try again.")
            else:
                df = pd.DataFrame(results)
                df['status'] = df['success'].map({True: 'PASS', False: 'FAIL'})
                df['time_ms'] = df['time'] * 1000

                df_plot = df[['kb', 'status', 'time_ms', 'stddev']].copy()

                st.subheader("📊 Raw Results")
                st.dataframe(df_plot, use_container_width=True)

                st.subheader("📈 RAM vs Time")
                
                chart = alt.Chart(df_plot).mark_line(point=True, color='steelblue').encode(
                    x=alt.X('kb:Q', title='Arena size (KB)'),
                    y=alt.Y('time_ms:Q', title='Elapsed time (ms)'),
                    tooltip=['kb', 'status', 'time_ms', 'stddev']
                )
                error_bars = alt.Chart(df_plot).mark_errorbar(extent='stdev', color='steelblue').encode(
                    x='kb:Q',
                    y='time_ms:Q',
                    yError='stddev:Q'
                )
                chart = error_bars + chart
                
                if opt_ram is not None:
                    optimal_time = df_plot.loc[df_plot['kb'] == opt_ram, 'time_ms'].iloc[0]
                    optimal_df = pd.DataFrame([{'kb': opt_ram, 'time_ms': optimal_time, 'point_type': 'Optimal'}])
                    marker = alt.Chart(optimal_df).mark_point(color='red', size=200, filled=True).encode(
                        x='kb:Q',
                        y='time_ms:Q',
                        tooltip=['kb', 'time_ms', 'point_type']
                    )
                    chart = chart + marker
                
                chart = chart.properties(width=700, height=400).interactive()
                st.altair_chart(chart, use_container_width=True)

                # RAM Requirements Summary
                st.subheader("💾 RAM Requirements Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Minimum RAM Required", f"{min_ram} KB")
                with col2:
                    if opt_ram is not None:
                        st.metric(f"Optimal RAM (within {threshold_pct:.0f}%)", f"{opt_ram} KB")

                # ────────────────────────────────────────────────
                # Eligible MCU Boards
                st.divider()
                st.subheader("📱 Eligible MCU Boards")
                
                eligible_boards = get_eligible_boards(min_ram)
                tier_summary = get_board_tier_summary(min_ram)
                
                if eligible_boards:
                    st.success(f"✅ **{len(eligible_boards)} boards eligible** for this model")
                    
                    # Create tabs for each tier
                    tier_names = list(tier_summary.keys())
                    tabs = st.tabs(tier_names)
                    
                    for tab, tier_name in zip(tabs, tier_names):
                        with tab:
                            boards_in_tier = tier_summary[tier_name]
                            
                            board_data = []
                            for board in boards_in_tier:
                                board_data.append({
                                    'MCU Model': board['name'],
                                    'Family': board['family'],
                                    'RAM': board['ram_range'],
                                    'Price': board['price_usd'],
                                    'Features': board['features'],
                                    'Best For': board['use_case']
                                })
                            
                            df_boards = pd.DataFrame(board_data)
                            st.dataframe(df_boards, use_container_width=True, hide_index=True)
                            
                            if boards_in_tier:
                                cheapest = min(boards_in_tier, key=lambda x: x['name'])
                                smallest = min(boards_in_tier, key=lambda x: x['ram_kb'])
                                st.caption(f"💰 Most affordable: **{cheapest['name']}** | 🎯 Most efficient: **{smallest['name']}**")
                else:
                    st.warning("❌ No eligible boards found. Model requires more RAM than tested boards support.")

    # ──────────────────────────────────────────────────────────
    # SECTION 4: Long-run Simulation
    # ──────────────────────────────────────────────────────────
    st.divider()
    st.header("📊 Step 4: Long-run Simulation (Optional)")
    
    col1, col2 = st.columns(2)
    with col1:
        runs = st.slider("Cycles to simulate (long-run)", 10, 1000, 100, step=10)
    with col2:
        drift_pct = st.slider("Drift per cycle (%)", 0.0, 10.0, 0.5)
    
    if st.button("▶️ Run Long-run Simulation", key="run_simulation", use_container_width=True):
        with st.spinner("Simulating degradation..."):
            sim_results = simulate_model_over_time(
                st.session_state.current_model_path, 
                runs, 
                drift_pct/100
            )

            records = []
            for entry in sim_results:
                if not entry['success']:
                    continue
                for idx, t in enumerate(entry['times']):
                    records.append({'kb': entry['kb'], 'cycle': idx+1, 'time_ms': t*1000})
            
            if not records:
                st.error("Long-run simulation failed for all arenas.")
            else:
                df_sim = pd.DataFrame(records)
                
                # Define distinct colors for each arena size
                arena_colors = {
                    8: '#E74C3C',      # Red
                    16: '#E67E22',     # Orange
                    32: '#F39C12',     # Yellow-Orange
                    48: '#27AE60',     # Green
                    64: '#3498DB',     # Blue
                    96: '#9B59B6',     # Purple
                    128: '#1ABC9C'     # Teal
                }
                
                st.subheader("📈 Long-run Performance Over Cycles")
                
                # Convert kb to string for consistent encoding
                df_sim['kb_str'] = df_sim['kb'].astype(str)
                
                # Create color scale based on arena sizes
                arena_list = sorted(df_sim['kb'].unique())
                colors = ['#E74C3C', '#E67E22', '#F39C12', '#27AE60', '#3498DB', '#9B59B6', '#1ABC9C']
                
                # Base line chart with better visibility
                line_chart = alt.Chart(df_sim).mark_line(
                    point=True, 
                    size=3,
                    opacity=0.9,
                    tension=0.2
                ).encode(
                    x=alt.X('cycle:Q', title='Cycle #', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('time_ms:Q', title='Elapsed time (ms)', scale=alt.Scale(zero=False)),
                    color=alt.Color(
                        'kb_str:N', 
                        scale=alt.Scale(
                            domain=[str(kb) for kb in arena_list],
                            range=colors[:len(arena_list)]
                        ),
                        title='Arena Size (KB)',
                        legend=alt.Legend(orient='bottom', direction='horizontal')
                    ),
                    tooltip=['kb:N', 'cycle:Q', alt.Tooltip('time_ms:Q', format='.3f')]
                ).properties(
                    width=900, 
                    height=500,
                    title='Performance Degradation Over Long-run Cycles'
                ).interactive()
                
                st.altair_chart(line_chart, use_container_width=True)
                
                # Analysis & Inference
                st.subheader("📊 Long-run Performance Analysis")
                
                # Calculate metrics for each arena
                arena_stats = []
                for kb in df_sim['kb'].unique():
                    arena_data = df_sim[df_sim['kb'] == kb]['time_ms']
                    initial_time = arena_data.iloc[0]
                    final_time = arena_data.iloc[-1]
                    degradation = final_time - initial_time
                    degradation_pct = (degradation / initial_time) * 100
                    avg_time = arena_data.mean()
                    
                    arena_stats.append({
                        'Arena (KB)': int(kb),
                        'Initial (ms)': f"{initial_time:.3f}",
                        'Final (ms)': f"{final_time:.3f}",
                        'Degradation': f"{degradation:.3f}",
                        'Degradation %': f"{degradation_pct:.1f}%",
                        'Avg Time (ms)': f"{avg_time:.3f}"
                    })
                
                df_analysis = pd.DataFrame(arena_stats).sort_values('Arena (KB)')
                st.dataframe(df_analysis, use_container_width=True, hide_index=True)
                
                # Find best performer
                best_arena = None
                best_degradation = float('inf')
                
                for kb in df_sim['kb'].unique():
                    arena_data = df_sim[df_sim['kb'] == kb]['time_ms']
                    final_time = arena_data.iloc[-1]
                    if final_time < best_degradation:
                        best_degradation = final_time
                        best_arena = int(kb)
                
                # Recommendations
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.info(f"🎯 **Best Long-run:** {best_arena} KB arena\n\nSmallest final execution time over {runs} cycles")
                
                with col2:
                    # Find most stable (least degradation)
                    min_degradation = float('inf')
                    stable_arena = None
                    for kb in df_sim['kb'].unique():
                        arena_data = df_sim[df_sim['kb'] == kb]['time_ms']
                        initial = arena_data.iloc[0]
                        final = arena_data.iloc[-1]
                        degradation = final - initial
                        if degradation < min_degradation:
                            min_degradation = degradation
                            stable_arena = int(kb)
                    
                    st.success(f"📌 **Most Stable:** {stable_arena} KB arena\n\nLeast performance degradation")
                
                with col3:
                    # Average performance
                    avg_final_times = df_sim.groupby('kb')['time_ms'].apply(lambda x: x.iloc[-1]).mean()
                    st.warning(f"⚠️ **Avg Performance:** {avg_final_times:.3f} ms\n\nCross-arena average at cycle {runs}")
                
                st.info("💡 **Insight:** Larger arenas typically maintain better performance over long runs due to less memory fragmentation and cache effects.")


# ──────────────────────────────────────────────────────────────
# SECTION 5: Model Upload Section (shown when no model loaded)


