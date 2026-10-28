import streamlit as st
import pandas as pd
from pathlib import Path
from src.incidents.incident_manager import IncidentManager
from src.analytics.metrics import create_status_donut_chart, create_category_bar_chart

st.set_page_config(
    page_title="SafeSight AI | Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling according to Section 7.1 Visual Design
st.markdown("""
<style>
    .metric-card {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 16px;
        color: white;
        border-left: 4px solid #3B82F6;
    }
    .metric-title { font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.8rem; font-weight: 700; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/safety-hat.png", width=64)
st.sidebar.title("SafeSight AI")
st.sidebar.caption("Industrial Safety & PPE Monitoring System")
st.sidebar.markdown("---")
st.sidebar.info("Assistive monitoring tool for industrial environments. Use with trained safety personnel.")

# Initialize database manager
mgr = IncidentManager()
stats = mgr.get_summary_metrics()

# Header
st.title("🛡️ SafeSight AI — Safety Operations Dashboard")
st.write("Real-time automated workplace safety compliance and incident audit center.")

# Metric row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Flagged Incidents", stats["total_incidents"])
with col2:
    st.metric("Awaiting Review", stats["pending_review"], delta=None)
with col3:
    st.metric("Confirmed Violations", stats["confirmed"], delta=None)
with col4:
    st.metric("Dismissed / False Positives", stats["dismissed"])

st.markdown("---")

# Analytics charts
c_left, c_right = st.columns(2)
with c_left:
    fig_cat = create_category_bar_chart(stats["by_category"])
    st.plotly_chart(fig_cat, use_container_width=True)

with c_right:
    fig_donut = create_status_donut_chart(stats)
    st.plotly_chart(fig_donut, use_container_width=True)

# Recent incidents table
st.subheader("📋 Recent Safety Incidents")
incidents = mgr.list_incidents()

if not incidents:
    st.info("No incidents recorded yet. Analyze an image or video to start auditing.")
else:
    df = pd.DataFrame(incidents)
    display_cols = ["id", "created_at", "source_filename", "source_type", "event_category", "confidence", "review_status"]
    available_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available_cols], use_container_width=True, hide_index=True)
