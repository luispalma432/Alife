import glob

import orjson
import plotly.express as px
import polars as pl
import streamlit as st

# --- CONFIGURATION ---
st.set_page_config(page_title="Evolutionary Dashboard", layout="wide")
st.title("🧬 Genetic Algorithm Analysis")

# FILE SELECTOR ---
json_files = glob.glob("results_*.json")

if not json_files:
    st.warning("No simulation files found. Please run 'main.py' first!")
    st.stop()

selected_file = st.sidebar.selectbox("Select Simulation File", json_files)


# -LOAD & PROCESS DATA ---
@st.cache_data
def load_data(filepath):
    try:
        with open(filepath, "rb") as f:
            json_data = orjson.loads(f.read())

        df = pl.DataFrame(json_data)

        flat_df = (
            df.explode("stats")
            .unnest("stats")
            .rename(
                {
                    "node_count": "Count",
                    "avg_tourney_score": "Avg Score",
                    "strategy": "Strategy",
                    "total_links": "Total Links",
                    "generation": "Generation",
                }
            )
        )

        return flat_df.to_pandas()

    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


df = load_data(selected_file)

# ---  VISUALIZATION DASHBOARD ---
if df is not None:
    # --- GLOBAL METRICS (Latest Generation) ---
    max_gen = df["Generation"].max()
    latest_df = df[df["Generation"] == max_gen]

    # Find the winner
    winner = latest_df.loc[latest_df["Count"].idxmax()]

    # Calculate Global Network Stats for the final generation
    total_pop = latest_df["Count"].sum()
    total_stubs = latest_df["Total Links"].sum()
    # Note: Avg Degree = Total Stubs / Total Pop
    global_avg_degree = total_stubs / total_pop if total_pop > 0 else 0

    # Display Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Generations", max_gen)
    c2.metric("Population", f"{total_pop:,.0f}")
    c3.metric("Avg Degree (k)", f"{global_avg_degree:.2f}")
    c4.metric("Winner", winner["Strategy"])
    c5.metric("Winner Score", f"{winner['Avg Score']:.1f}")

    st.markdown("---")

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Population", "🏆 Fitness", "🔗 Network Topology", "💾 Raw Data"]
    )

    # TAB 1: POPULATION
    with tab1:
        st.subheader("Strategy Dominance")
        fig_pop = px.area(
            df,
            x="Generation",
            y="Count",
            color="Strategy",
            line_group="Strategy",
            template="plotly_dark",
            title="Population Share by Strategy",
        )
        st.plotly_chart(fig_pop, use_container_width=True)

    # TAB 2: FITNESS
    with tab2:
        st.subheader("Fitness Evolution")
        fig_score = px.line(
            df,
            x="Generation",
            y="Avg Score",
            color="Strategy",
            markers=True,
            template="plotly_dark",
            title="Average Tournament Score",
        )
        st.plotly_chart(fig_score, use_container_width=True)

    # TAB 3: NETWORK STATS (Simplified)
    with tab3:
        st.subheader("Total Connectivity")
        st.caption("Total number of connections (edges) controlled by each strategy.")

        # Now this graph uses the full width
        fig_links = px.line(
            df,
            x="Generation",
            y="Total Links",
            color="Strategy",
            markers=True,
            template="plotly_dark",
            title="Total Links Controlled",
        )
        st.plotly_chart(fig_links, use_container_width=True)

    # TAB 4: DATA
    with tab4:
        st.dataframe(df, use_container_width=True)
