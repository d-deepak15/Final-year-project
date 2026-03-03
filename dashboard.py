import streamlit as st
import pandas as pd
import altair as alt

from main import profile_model, simulate_model_over_time, MODEL_PATH
from boards_db import get_eligible_boards, get_board_tier_summary

st.title("TinyML RAM Profiler Dashboard")

model_path = st.text_input("TFLite model path", MODEL_PATH)
threshold_pct = st.slider("Improvement threshold (%)", 0.0, 20.0, 5.0)
iterations = st.slider("Iterations per arena (for averaging)", 1, 20, 5)

# long-run simulation parameters
runs = st.slider("Cycles to simulate (long-run)", 10, 1000, 100, step=10)
drift_pct = st.slider("Drift per cycle (%)", 0.0, 10.0, 0.5)

threshold = threshold_pct / 100.0

if st.button("Run profiler"):
    with st.spinner("Running tests..."):
        results, min_ram, opt_ram = profile_model(model_path, threshold, iterations)

        if not results:
            st.error("No results returned; check the model path or try again.")
            st.stop()

        df = pd.DataFrame(results)
        df['status'] = df['success'].map({True: 'PASS', False: 'FAIL'})
        df['time_ms'] = df['time'] * 1000

        # drop columns that contain non-scalar objects (e.g. tensor arrays)
        df_plot = df[['kb', 'status', 'time_ms', 'stddev']].copy()

        st.subheader("Raw results")
        st.dataframe(df_plot)

        st.subheader("RAM vs Time")
        
        # main line chart
        chart = alt.Chart(df_plot).mark_line(point=True, color='steelblue').encode(
            x=alt.X('kb:Q', title='Arena size (KB)'),
            y=alt.Y('time_ms:Q', title='Elapsed time (ms)'),
            tooltip=['kb', 'status', 'time_ms', 'stddev']
        )
        # add error bars representing ±stddev
        error_bars = alt.Chart(df_plot).mark_errorbar(extent='stdev', color='steelblue').encode(
            x='kb:Q',
            y='time_ms:Q',
            yError='stddev:Q'
        )
        chart = error_bars + chart
        
        # add optimal point marker in red
        if opt_ram is not None:
            optimal_time = df_plot.loc[df_plot['kb'] == opt_ram, 'time_ms'].iloc[0]
            optimal_df = pd.DataFrame([{'kb': opt_ram, 'time_ms': optimal_time, 'point_type': 'Optimal'}])
            marker = alt.Chart(optimal_df).mark_point(color='red', size=200, filled=True).encode(
                x='kb:Q',
                y='time_ms:Q',
                tooltip=['kb', 'time_ms', 'point_type']
            )
            chart = chart + marker
        
        # add legend annotation
        chart = chart.properties(
            width=700,
            height=400
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

        st.write(f"**Minimum RAM required:** {min_ram} KB")
        if opt_ram is not None:
            st.write(f"**Optimal RAM (within {threshold_pct:.0f}%):** {opt_ram} KB")

        # ──────────────────────────────────────────────────────
        # Eligible boards section
        st.subheader("📱 Eligible MCU Boards")
        
        eligible_boards = get_eligible_boards(min_ram)
        tier_summary = get_board_tier_summary(min_ram)
        
        if eligible_boards:
            st.info(f"✅ **{len(eligible_boards)} boards eligible** for this model")
            
            # Create tabs for each tier
            tier_names = list(tier_summary.keys())
            tabs = st.tabs(tier_names)
            
            for tab, tier_name in zip(tabs, tier_names):
                with tab:
                    boards_in_tier = tier_summary[tier_name]
                    
                    # Create a nice dataframe view
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
                    
                    # Show cost-performance recommendation
                    if boards_in_tier:
                        cheapest = min(boards_in_tier, key=lambda x: x['name'])
                        smallest = min(boards_in_tier, key=lambda x: x['ram_kb'])
                        st.caption(f"💰 Most affordable: **{cheapest['name']}** | 🎯 Most efficient: **{smallest['name']}**")
        else:
            st.warning("❌ No eligible boards found. Model requires more RAM than tested boards support.")

    # long-run simulation
    if st.button("Run long-run simulation"):
        with st.spinner("Simulating degradation..."):
            sim_results = simulate_model_over_time(model_path, runs, drift_pct/100)

        # build dataframe for chart (long format)
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
            st.subheader("Long-run performance")
            # line per arena
            chart2 = alt.Chart(df_sim).mark_line().encode(
                x=alt.X('cycle:Q', title='Cycle #'),
                y=alt.Y('time_ms:Q', title='Elapsed time (ms)'),
                color='kb:O',
                tooltip=['kb', 'cycle', 'time_ms']
            ).properties(width=700, height=400).interactive()
            st.altair_chart(chart2, use_container_width=True)

