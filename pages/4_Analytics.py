import streamlit as st
import pandas as pd
import plotly.express as px
from src.incidents.incident_manager import IncidentManager
from src.analytics.metrics import create_status_donut_chart, create_category_bar_chart

st.set_page_config(page_title="SafeSight AI | Analytics", page_icon="📊", layout="wide")

st.title("📊 Safety Audit Analytics & Trends")
st.write("Quantitative breakdown of workplace incidents, review resolutions, and compliance trends.")

mgr = IncidentManager()
stats = mgr.get_summary_metrics()
incidents = mgr.list_incidents()

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(create_category_bar_chart(stats["by_category"]), use_container_width=True)
with c2:
    st.plotly_chart(create_status_donut_chart(stats), use_container_width=True)

st.markdown("---")
st.subheader("📈 Incident Distribution Over Time")

if incidents:
    df = pd.DataFrame(incidents)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["date"] = df["created_at"].dt.date
    
    time_df = df.groupby(["date", "event_category"]).size().reset_index(name="count")
    fig_time = px.line(
        time_df,
        x="date",
        y="count",
        color="event_category",
        markers=True,
        title="Daily Incident Frequency by Category"
    )
    st.plotly_chart(fig_time, use_container_width=True)
else:
    st.info("No time-series data available yet. Run analysis to populate trends.")
