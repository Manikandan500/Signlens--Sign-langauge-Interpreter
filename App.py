import streamlit as st
import tempfile
import os
import time
import cv2
import mediapipe as mp
import backend

st.set_page_config(
    page_title="Hybrid SignLens",
    page_icon="✨",
    layout="wide"
)

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

st.markdown("""
<style>
/* Reduce top whitespace */
.block-container { padding-top: 1.5rem !important; }

/* Base */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* App background */
.stApp { background-color: #0B0F19; }

/* Hide Streamlit defaults */
#MainMenu, footer, header { visibility: hidden; }

/* Headings */
h1 { font-weight: 800; color: #FFFFFF; margin-bottom: 4px; }
h2, h3 { color: #E5E7EB; }

/* Subtitle */
.subtitle { color: #9CA3AF; font-size: 15px; }

/* Card */
.card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 18px;
}

/* Accent Card */
.card-accent { border-left: 4px solid #6366F1; }

/* Buttons */
.stButton button {
    width: 100%;
    border-radius: 10px;
    padding: 12px 18px;
    font-weight: 600;
    border: none;
}

/* Primary Button Gradient */
div[data-testid="stButton"] > button:first-child {
    background: linear-gradient(90deg, #6366F1, #8B5CF6);
    color: white;
}
div[data-testid="stButton"] > button:hover {
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    transform: translateY(-1px);
}

/* Info Box */
.info-box {
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 16px;
    color: #CBD5E1;
    text-align: center;
}

/* Output text */
.output-text {
    font-size: 26px;
    font-weight: 700;
    color: #FFFFFF;
    text-align: center;
    background: #1F2937;
    padding: 20px;
    border-radius: 10px;
}

.file-meta { font-size: 14px; color: #9CA3AF; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if "sentence" not in st.session_state:
    st.session_state.sentence = ""
if "video_buffer" not in st.session_state:
    st.session_state.video_buffer = []

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## ✨ Hybrid SignLens ")
    st.caption("v2.0 | Hybrid Model")
    st.markdown("---")

    app_mode = st.radio(
        "Mode",
        ["📷 Live Interpreter", "📂 Video Upload"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.success("Backend: Connected")
    st.info("Model: CNN + LSTM")

# =========================
# HEADER
# =========================
st.markdown("""
<div style="margin-bottom: 12px;">
    <h1>👐 Sign Language Interpreter</h1>
    <div class="subtitle">Modern, real-time sign language translation interface</div>
</div>
<hr style="opacity:0.2;">
""", unsafe_allow_html=True)

# =========================
# MODE 1: VIDEO UPLOAD
# =========================
if app_mode == "📂 Video Upload":

    col1, col2 = st.columns([1.1, 0.9], gap="large")

    with col1:
        st.markdown("<div class='card'><h3>📂 Upload Video</h3></div>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "avi"], label_visibility="collapsed")

        # Placeholder for the video (either raw or processed)
        video_display = st.empty()

        if uploaded:
            # Save uploaded file to temp immediately for processing
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded.read())
            temp_filename = tfile.name

            # Show the raw video initially
            video_display.video(uploaded)
        else:
            st.markdown("<div class='info-box'>Drag & drop a video file here</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card card-accent'><h3>🧠 Translation Output</h3></div>", unsafe_allow_html=True)

        if st.button("✨ Generate Translation", type="primary"):
            if uploaded:
                st.session_state.sentence = ""  # Reset previous result

                # --- STEP 1: VISUAL PROCESSING (MEDIAPIPE HAND LANDMARKS) ---
                status_text = st.empty()
                status_text.caption("🔍 Scanning hand gestures with MediaPipe...")

                cap = cv2.VideoCapture(temp_filename)

                # Setup MediaPipe Hands for landmark detection
                with mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5
                ) as hands:

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Convert to RGB for MediaPipe
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = hands.process(frame_rgb)

                        # Draw Landmarks
                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                mp_drawing.draw_landmarks(
                                    frame,
                                    hand_landmarks,
                                    mp_hands.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                )

                        # Display Frame in Streamlit
                        frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_display.image(frame_display, channels="RGB", use_column_width=True)
                        time.sleep(0.03)

                    cap.release()

                # --- STEP 2: CNN FEATURE EXTRACTION + LSTM SEQUENCE PREDICTION ---
                status_text.caption("🤖 Running CNN feature extraction → LSTM sequence prediction...")

                # Force release the video capture immediately
                if 'cap' in locals() and cap.isOpened():
                    cap.release()

                # Slight pause to let Windows release the file lock
                time.sleep(1.0)

                with st.spinner("Classifying sign language sequence..."):
                    try:
                        # backend.analyze_video_sign() should:
                        #   1. Extract per-frame features using the CNN model
                        #   2. Feed the feature sequence into the LSTM model
                        #   3. Return the predicted sign/sentence as a string
                        result = backend.analyze_video_sign(temp_filename)
                        st.session_state.sentence = result
                    except Exception as e:
                        st.error(f"Model Error: {e}")

                # --- STEP 3: SAFE CLEANUP ---
                try:
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)
                except PermissionError:
                    pass

                status_text.empty()  # Clear status
            else:
                st.warning("Please upload a video first.")

        # --- OUTPUT DISPLAY ---
        if st.session_state.sentence:
            st.markdown(f"<div class='output-text'>{st.session_state.sentence}</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔊 Speak Output", key="speak_btn_upload"):
                backend.speak_text(st.session_state.sentence)
        else:
            st.markdown("<div class='info-box'>Waiting for processed output…</div>", unsafe_allow_html=True)


# =========================
# MODE 2: LIVE MODE (MANUAL START/STOP)
# =========================
else:
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown("<div class='card'><h3>🎥 Live Camera Feed</h3></div>", unsafe_allow_html=True)
        video_placeholder = st.empty()

        # CONTROLS
        c1, c2 = st.columns(2)
        with c1:
            start_cam = st.toggle("🔌 Activate Camera", value=False)
        with c2:
            is_recording = st.toggle("🔴 Record / Stop", value=False)

    with col2:
        st.markdown("<div style='margin-top:48px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='card card-accent'><h3>📝 Recognized Text</h3></div>", unsafe_allow_html=True)

        if st.session_state.sentence:
            st.markdown(f"<div class='output-text'>{st.session_state.sentence}</div>", unsafe_allow_html=True)
            if st.button("🔊 Speak Result", key="speak_btn_live"):
                backend.speak_text(st.session_state.sentence)
        else:
            st.markdown("<div class='info-box'>Waiting for input...</div>", unsafe_allow_html=True)

    # --- MAIN LOGIC ---

    # 1. PROCESS RECORDING (Triggered when user turns switch OFF)
    if not is_recording and len(st.session_state.video_buffer) > 0:
        with col2:
            with st.spinner("🤖 Running CNN + LSTM sign classification..."):
                # Save Buffer to Temp File
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    # Determine dimensions from first frame
                    height, width, _ = st.session_state.video_buffer[0].shape

                    # Write collected frames to temp video file
                    out = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))
                    for f in st.session_state.video_buffer:
                        out.write(f)
                    out.release()
                    vid_path = tmp.name

                # Send to CNN + LSTM backend pipeline
                try:
                    # backend.analyze_video_sign() should:
                    #   1. Extract per-frame features using the CNN model
                    #   2. Feed the feature sequence into the LSTM model
                    #   3. Return the predicted sign/sentence as a string
                    text = backend.analyze_video_sign(vid_path)
                    st.session_state.sentence = text
                except Exception as e:
                    st.error(f"Model Error: {e}")

                # Clean up buffer and temp file
                st.session_state.video_buffer = []

                try:
                    if os.path.exists(vid_path):
                        os.remove(vid_path)
                except PermissionError:
                    pass

                st.rerun()

    # 2. CAMERA LOOP
    if start_cam:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # MediaPipe for live hand landmark overlay
        with mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=2) as hands:
            while cap.isOpened() and start_cam:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)

                # Add MediaPipe hand landmark overlay to live feed
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                        )

                # IF RECORDING IS ON: Collect Frames for CNN+LSTM inference
                if is_recording:
                    # Visual recording indicator
                    cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
                    cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(frame, "RECORDING...", (50, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    st.session_state.video_buffer.append(frame)

                # Show live video frame
                video_placeholder.image(frame, channels="BGR", use_column_width=True)
                time.sleep(0.01)

        cap.release()
    else:
        video_placeholder.markdown(
            "<div class='info-box' style='height:360px; display:flex; align-items:center; justify-content:center;'>Camera is Inactive</div>",
            unsafe_allow_html=True
        )