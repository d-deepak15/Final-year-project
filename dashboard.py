import streamlit as st
import pandas as pd
import altair as alt

from main import profile_model, MODEL_PATH

st.title("TinyML RAM Profiler Dashboard")

model_path = st.text_input("TFLite model path", MODEL_PATH)
threshold_pct = st.slider("Improvement threshold (%)", 0.0, 20.0, 5.0)
iterations = st.slider("Iterations per arena (for averaging)", 1, 20, 5)
threshold = threshold_pct / 100.0

if st.button("Run profiler"):
    with st.spinner("Running tests..."):
        results, min_ram, opt_ram = profile_model(model_path, threshold)

        if not results:
            st.error("No results returned; check the model path or try again.")
            st.stop()

        df = pd.DataFrame(results)
        df['status'] = df['success'].map({True: 'PASS', False: 'FAIL'})
        df['time_ms'] = df['time'] * 1000

        # drop columns that contain non-scalar objects (e.g. tensor arrays)
        df_plot = df[['kb', 'status', 'time_ms']].copy()

        st.subheader("Raw results")
        st.dataframe(df_plot)

        st.subheader("RAM vs Time")
        
        # main line chart
        chart = alt.Chart(df_plot).mark_line(point=True, color='steelblue').encode(
            x=alt.X('kb:Q', title='Arena size (KB)'),
            y=alt.Y('time_ms:Q', title='Elapsed time (ms)'),
            tooltip=['kb', 'status', 'time_ms']
        )
        
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
