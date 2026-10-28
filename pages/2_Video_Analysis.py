import streamlit as st
import tempfile
import cv2
from pathlib import Path

from src.models.detector import SafeSightDetector
from src.safety.ppe_rules import PPERuleEngine
from src.video.video_processor import VideoProcessor
from src.incidents.incident_manager import IncidentManager
from src.utils.file_validation import validate_video_file

st.set_page_config(page_title="SafeSight AI | Video Analysis", page_icon="🎥", layout="wide")

st.title("🎥 Video Stream & Footage Analysis")
st.write("Inspect CCTV and recorded workplace video streams frame-by-frame with automated incident flagging.")

st.sidebar.header("Video Processing Configuration")
sample_rate = st.sidebar.slider("Sample Rate (Process every Nth frame)", min_value=1, max_value=30, value=5)
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.10, 1.00, 0.35, 0.05)
req_helmet = st.sidebar.checkbox("Require Hard Hat / Helmet", value=True)
req_vest = st.sidebar.checkbox("Require Safety Vest", value=True)

uploaded_video = st.file_uploader("Upload video file (MP4, AVI, MOV, MKV)", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video is not None:
    valid, msg = validate_video_file(uploaded_video.name, uploaded_video.size)
    if not valid:
        st.error(msg)
    else:
        # Save temp file for OpenCV reading
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix)
        tfile.write(uploaded_video.read())
        tfile.flush()
        tfile.close()

        st.video(tfile.name)

        if st.button("🚀 Start Video Safety Audit", type="primary"):
            processor = VideoProcessor(tfile.name, sample_rate=sample_rate)
            meta = processor.get_metadata()
            
            st.info(f"Video Info: {meta['width']}x{meta['height']} @ {meta['fps']} FPS | Total Frames: {meta['total_frames']} | Duration: {meta['duration_seconds']}s")

            detector = SafeSightDetector(conf=conf_thresh)
            rule_engine = PPERuleEngine(require_helmet=req_helmet, require_vest=req_vest)
            mgr = IncidentManager()

            progress_bar = st.progress(0.0)
            status_text = st.empty()
            preview_frame = st.empty()

            flagged_count = 0
            processed_frames = 0
            estimated_sample_frames = max(1, meta["total_frames"] // sample_rate)

            for frame_idx, timestamp_sec, frame in processor.frame_generator():
                processed_frames += 1
                progress = min(1.0, processed_frames / estimated_sample_frames)
                progress_bar.progress(progress)
                status_text.text(f"Auditing frame {frame_idx}/{meta['total_frames']} (Timestamp: {timestamp_sec:.2f}s)...")

                detections = detector.detect(frame)
                assessments = rule_engine.evaluate_scene(detections)
                violations = [a for a in assessments if a.is_violation]

                annotated = rule_engine.annotate_safety(frame, assessments)
                preview_frame.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Time: {timestamp_sec}s", use_column_width=True)

                if violations:
                    flagged_count += len(violations)
                    for viol in violations:
                        mgr.log_incident(
                            source_filename=uploaded_video.name,
                            source_type="video",
                            timestamp_seconds=timestamp_sec,
                            event_category=", ".join(viol.violation_reasons),
                            confidence=viol.confidence,
                            rule_status="violation",
                            evidence_image=annotated
                        )

            progress_bar.progress(1.0)
            status_text.success(f"Audit completed! Processed {processed_frames} sampled frames. Flagged {flagged_count} potential violations.")
            if flagged_count > 0:
                st.warning(f"Recorded {flagged_count} incidents into the database. You can review and confirm them in the Incident Management page.")
