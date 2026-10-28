import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

from src.models.detector import SafeSightDetector
from src.safety.ppe_rules import PPERuleEngine
from src.incidents.incident_manager import IncidentManager
from src.utils.file_validation import validate_image_file

st.set_page_config(page_title="SafeSight AI | Image Analysis", page_icon="📷", layout="wide")

st.title("📷 Image Safety & PPE Analysis")
st.write("Upload a workplace image to identify personnel and detect PPE compliance violations.")

# Sidebar controls
st.sidebar.header("Inference Settings")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.10, 1.00, 0.35, 0.05)
req_helmet = st.sidebar.checkbox("Require Hard Hat / Helmet", value=True)
req_vest = st.sidebar.checkbox("Require Safety Vest", value=True)

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # 1. Validate file
    valid, msg = validate_image_file(uploaded_file.name, uploaded_file.size)
    if not valid:
        st.error(msg)
    else:
        # 2. Decode image
        image_bytes = uploaded_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        col_orig, col_annot = st.columns(2)
        with col_orig:
            st.subheader("Original Image")
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

        # 3. Run Detection & PPE Rule Engine
        with st.spinner("Analyzing workplace safety..."):
            detector = SafeSightDetector(conf=conf_thresh)
            detections = detector.detect(img_bgr)
            rule_engine = PPERuleEngine(require_helmet=req_helmet, require_vest=req_vest)
            assessments = rule_engine.evaluate_scene(detections)
            annotated_bgr = rule_engine.annotate_safety(img_bgr, assessments)
            # Also overlay object detection boxes for non-workers (e.g. helmets, vests)
            non_workers = [d for d in detections if d.class_name not in ("person", "worker")]
            annotated_bgr = detector.draw_detections(annotated_bgr, non_workers)

        with col_annot:
            st.subheader("Audited Safety Output")
            st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

            # Download button
            is_success, buffer = cv2.imencode(".jpg", annotated_bgr)
            if is_success:
                st.download_button(
                    label="💾 Download Annotated Image",
                    data=buffer.tobytes(),
                    file_name=f"safesight_{uploaded_file.name}",
                    mime="image/jpeg"
                )

        # 4. Results Breakdown
        st.markdown("---")
        st.subheader("🔍 Safety Assessment Summary")
        
        workers_found = len(assessments)
        violations = [a for a in assessments if a.is_violation]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Workers Detected", workers_found)
        c2.metric("Violations Flagged", len(violations), delta=f"-{len(violations)}" if violations else "0")
        c3.metric("Compliant Personnel", workers_found - len(violations))

        if violations:
            st.warning(f"⚠️ {len(violations)} safety violation(s) detected in the uploaded image.")
            mgr = IncidentManager()
            for idx, viol in enumerate(violations):
                v_box = st.container()
                v_box.markdown(f"**Worker #{idx+1} Violation:** {', '.join(viol.violation_reasons)} (Confidence: {viol.confidence:.2f})")
                
                if st.button(f"Log Worker #{idx+1} Violation to Incident Database", key=f"log_{idx}"):
                    inc_id = mgr.log_incident(
                        source_filename=uploaded_file.name,
                        source_type="image",
                        event_category=", ".join(viol.violation_reasons),
                        confidence=viol.confidence,
                        rule_status="violation",
                        evidence_image=annotated_bgr
                    )
                    st.success(f"Successfully recorded incident #{inc_id} into persistent database.")
        elif workers_found > 0:
            st.success("✅ All detected personnel are compliant with configured safety rules.")
        else:
            st.info("No personnel detected in the frame.")
