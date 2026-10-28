import streamlit as st
import pandas as pd
from pathlib import Path
from src.incidents.incident_manager import IncidentManager

st.set_page_config(page_title="SafeSight AI | Incident Management", page_icon="📝", layout="wide")

st.title("📝 Incident Management & Human Review")
st.write("Review, confirm, dismiss, or document notes for AI-flagged workplace safety events.")

mgr = IncidentManager()

# Filter sidebar
st.sidebar.header("Filter Incidents")
status_filter = st.sidebar.selectbox("Review Status", ["All", "pending", "confirmed", "dismissed", "uncertain"])
source_filter = st.sidebar.selectbox("Source Media", ["All", "image", "video"])
category_filter = st.sidebar.selectbox("Category", ["All", "Missing Helmet", "Missing Safety Vest", "Restricted Zone"])

incidents = mgr.list_incidents(
    status=status_filter if status_filter != "All" else None,
    source_type=source_filter if source_filter != "All" else None,
    category=category_filter if category_filter != "All" else None
)

# Export option
if incidents:
    df = pd.DataFrame(incidents)
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Filtered Incidents to CSV",
        data=csv_bytes,
        file_name="safesight_incidents.csv",
        mime="text/csv"
    )

st.markdown("---")

if not incidents:
    st.info("No matching incidents found.")
else:
    for inc in incidents:
        with st.expander(f"Incident #{inc['id']} — {inc['event_category']} | Status: {inc['review_status'].upper()} | Date: {inc['created_at']}", expanded=(inc['review_status'] == 'pending')):
            col_info, col_img = st.columns([1, 1])

            with col_info:
                st.write(f"**Source File:** `{inc['source_filename']}` ({inc['source_type']})")
                if inc['timestamp_seconds'] is not None:
                    st.write(f"**Video Timestamp:** `{inc['timestamp_seconds']}s`")
                st.write(f"**Confidence:** `{inc['confidence']:.2f}`")
                st.write(f"**Rule Status:** `{inc['rule_status']}`")
                st.write(f"**Current Review Status:** `{inc['review_status']}`")

                new_notes = st.text_area("Reviewer Notes", value=inc['reviewer_notes'] or "", key=f"notes_{inc['id']}")

                b_col1, b_col2, b_col3, b_col4 = st.columns(4)
                with b_col1:
                    if st.button("Confirm", key=f"conf_{inc['id']}", type="primary"):
                        mgr.update_review(inc['id'], "confirmed", new_notes)
                        st.rerun()
                with b_col2:
                    if st.button("Dismiss", key=f"dism_{inc['id']}"):
                        mgr.update_review(inc['id'], "dismissed", new_notes)
                        st.rerun()
                with b_col3:
                    if st.button("Uncertain", key=f"unc_{inc['id']}"):
                        mgr.update_review(inc['id'], "uncertain", new_notes)
                        st.rerun()
                with b_col4:
                    if st.button("Delete", key=f"del_{inc['id']}"):
                        mgr.delete(inc['id'])
                        st.rerun()

            with col_img:
                if inc['evidence_path'] and Path(inc['evidence_path']).exists():
                    st.image(inc['evidence_path'], caption=f"Evidence for Incident #{inc['id']}", use_column_width=True)
                else:
                    st.info("No evidence image saved for this incident.")
