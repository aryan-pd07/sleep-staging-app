"""
app.py — NeuroSleep AI
Main Streamlit application for automated sleep stage classification.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import tempfile
from fpdf import FPDF
from supabase import create_client

from utils import (
    CLASS_LABELS, HYPNOGRAM_MAP, STAGE_COLORS,
    preprocess_batch, smooth_predictions, run_inference,
    load_edf_file, load_ground_truth, get_dummy_data
)


# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="NeuroSleep AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# SUPABASE CLIENT
# ─────────────────────────────────────────
supabase = create_client(
    os.environ.get("SUPABASE_URL"),
    os.environ.get("SUPABASE_KEY")
)

# ─────────────────────────────────────────
# MODEL — cached so it loads only once
# ─────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "best_cnn_lstm_model.keras")
# Remove after confirming
print(f"[DEBUG] Model exists: {os.path.exists(MODEL_PATH)} | Path: {MODEL_PATH}")
@st.cache_resource
def load_model():
    import tensorflow as tf
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    st.error(f"❌ Model not found at {MODEL_PATH}")
    return None


# ─────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container { padding-top: 2rem; }
h1, h2, h3 { letter-spacing: -0.5px; }
div[data-testid="stMetric"] {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e0e0e0;
}
[data-testid="stMetricValue"] { color: #000000 !important; }
[data-testid="stMetricLabel"] { color: #000000 !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# AUTH FUNCTIONS
# ─────────────────────────────────────────
def login(email, password):
    return supabase.auth.sign_in_with_password({"email": email, "password": password})

def signup(email, password, full_name):
    return supabase.auth.sign_up({
        "email": email,
        "password": password,
        "options": {"data": {"full_name": full_name}}
    })


# ─────────────────────────────────────────
# PDF REPORT
# ─────────────────────────────────────────
def safe_text(text):
    return str(text).encode("latin-1", errors="replace").decode("latin-1")
def create_pdf_report(pred_labels, accuracy, confidence, time_axis, y_ai, name, duration_hours, sampling_rate=100):
    """
    Generates a professional sleep analysis report as a PDF.
    Fixes: 
    - 'fpdf.errors.FPDFException' via explicit effective_width.
    - 'StreamlitAPIException' (bytearray error) via strict bytes() casting.
    """
    import matplotlib.pyplot as plt
    from datetime import datetime
    import tempfile
    import os
    import pandas as pd
    import numpy as np
    from fpdf import FPDF

    # 1. Initialize PDF and Page Dimensions
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Pre-calculate usable width (Page width - left & right margins)
    effective_width = pdf.w - 2 * pdf.l_margin

    total_epochs = len(pred_labels)
    sleep_epochs = sum(1 for s in pred_labels if s != "Wake")
    efficiency = (sleep_epochs / total_epochs) * 100 if total_epochs else 0

    def add_header(pdf):
        logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
        if os.path.exists(logo_path):
            pdf.image(logo_path, x=10, y=8, w=60)
            pdf.set_y(35)
        else:
            pdf.ln(10)
        pdf.set_font("helvetica", "B", 18)
        pdf.cell(0, 15, "NeuroSleep AI Report", ln=True, align="C")
        pdf.ln(5)

    # ─────────────────────────────────────────
    # PAGE 1 — Summary & Metrics
    # ─────────────────────────────────────────
    pdf.add_page()
    add_header(pdf)
    pdf.set_font("helvetica", "", 12)
    
    pdf.cell(0, 8, safe_text(f"Patient Name: {name}"), ln=True)
    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%d %b %Y %H:%M')}", ln=True)
    pdf.cell(0, 8, f"Recording Duration: {duration_hours:.2f} hours", ln=True)
    pdf.cell(0, 8, f"Sampling Rate: {sampling_rate} Hz", ln=True)
    pdf.cell(0, 8, "Model Architecture: CNN-LSTM (EEG + EOG Channels)", ln=True)
    pdf.ln(10)
    
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 8, "Key Metrics", ln=True)
    pdf.set_font("helvetica", "", 12)
    
    dom_stage = pd.Series(pred_labels).mode()[0] if len(pred_labels) > 0 else "N/A"
    pdf.cell(0, 8, safe_text(f"Dominant Stage: {dom_stage}"), ln=True)
    pdf.cell(0, 8, safe_text(f"Average AI Confidence: {confidence:.2f}%"), ln=True)
    pdf.cell(0, 8, safe_text(f"Accuracy vs Ground Truth: {accuracy}"), ln=True)
    pdf.cell(0, 8, safe_text(f"Calculated Sleep Efficiency: {efficiency:.1f}%"), ln=True)
    
    pdf.set_y(-25)
    pdf.set_text_color(120, 120, 120)
    pdf.set_font("helvetica", "I", 9)
    pdf.multi_cell(effective_width, 5, safe_text("Automatically generated by NeuroSleep AI - For research support only - Not a medical diagnosis"), align="C")
    pdf.set_text_color(0, 0, 0)

    # ─────────────────────────────────────────
    # PAGE 2 — Visualizations
    # ─────────────────────────────────────────
    pdf.add_page()
    add_header(pdf)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Hypnogram Plot
        hypno_path = os.path.join(tmpdir, "hypno.png")
        plt.figure(figsize=(15, 5))
        plt.step(time_axis, y_ai, where="post", color='#2980b9', linewidth=1.5)
        plt.yticks(list(HYPNOGRAM_MAP.values()), list(HYPNOGRAM_MAP.keys()))
        plt.gca().invert_yaxis()
        plt.xlabel("Time (hours)")
        plt.ylabel("Stage")
        plt.title("Sleep Hypnogram (AI Staging)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(hypno_path, dpi=150)
        plt.close()
        
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 10, "Sleep Architecture", ln=True)
        pdf.image(hypno_path, x=10, w=190)
        pdf.ln(10)

        # Pie Chart
        pie_path = os.path.join(tmpdir, "pie.png")
        stage_counts = pd.Series(pred_labels).value_counts()
        plt.figure(figsize=(6, 6))
        plt.pie(stage_counts.values, labels=stage_counts.index, autopct="%1.1f%%", startangle=140)
        plt.title("Sleep Stage Distribution")
        plt.tight_layout()
        plt.savefig(pie_path, dpi=100)
        plt.close()
        pdf.image(pie_path, x=70, w=70)

    # ─────────────────────────────────────────
    # Clinical Interpretation
    # ─────────────────────────────────────────
    comments = []
    if efficiency < 75:
        comments.append("Sleep efficiency is reduced, suggesting fragmented or disturbed sleep patterns.")
    elif efficiency < 85:
        comments.append("Sleep efficiency is within a moderate range.")
    else:
        comments.append("Sleep efficiency is within the normal healthy range.")
    
    if total_epochs > 0:
        if (pred_labels.count("REM") / total_epochs) * 100 < 15:
            comments.append("REM proportion appears lower than the typical adult clinical baseline.")
        if (pred_labels.count("N3") / total_epochs) * 100 < 10:
            comments.append("Deep sleep (N3) duration is limited.")
        if (pred_labels.count("Wake") / total_epochs) > 0.2:
            comments.append("Frequent awakenings detected throughout the recording period.")

    pdf.ln(10)
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "AI Clinical Interpretation", ln=True)
    pdf.set_font("helvetica", "", 12)
    
    for c in comments:
        pdf.multi_cell(effective_width, 8, safe_text(f"- {c}"))

    # ─────────────────────────────────────────
    # RETURN STRICT BYTES (Final Fix)
    # ─────────────────────────────────────────
    output_data = pdf.output(dest="S")
    
    # Essential: Convert to immutable 'bytes' to satisfy Streamlit's marshaller
    if isinstance(output_data, (bytes, bytearray)):
        return bytes(output_data) 
    
    # Fallback for older versions of FPDF that return string
    return str(output_data).encode("latin-1", errors="replace")


# =====================================================
# AUTH ROUTER
# =====================================================
try:
    user_response = supabase.auth.get_user()
    if user_response and user_response.user:
        st.session_state["user"] = user_response.user
except Exception:
    pass


# =====================================================
# LOGIN PAGE
# =====================================================
if "user" not in st.session_state:
    st.title("🔐 Login to NeuroSleep AI")

    full_name = st.text_input("Full Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login", use_container_width=True):
            try:
                res = login(email, password)
                if res and getattr(res, "user", None):
                    st.session_state["user"] = res.user
                    st.success("Login successful")
                    st.rerun()
            except Exception as e:
                msg = str(e)
                if "Email not confirmed" in msg:
                    st.warning("📩 Please verify your email before logging in.")
                else:
                    st.error("Invalid email or password")

    with col2:
        if st.button("Signup", use_container_width=True):
            try:
                res = signup(email, password, full_name)
                if res and getattr(res, "user", None):
                    st.success("📩 Verification email sent. Please check your inbox.")
                else:
                    st.error("Signup failed")
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.stop()


# =====================================================
# DASHBOARD
# =====================================================
if "user" in st.session_state:
    user = st.session_state["user"]
    name = (user.user_metadata or {}).get("full_name", user.email)

    st.sidebar.success(f"Welcome back, {name} 👋")
    if st.sidebar.button("Logout"):
        supabase.auth.sign_out()
        del st.session_state["user"]
        st.rerun()

    # ── SIDEBAR ──
    with st.sidebar:
        st.title("🧠 NeuroSleep AI")
        st.caption("AI Powered Sleep Staging")
        st.divider()

        st.subheader("1. Signal Source")
        use_dummy = st.checkbox("Use Demo Data", value=False)

        full_eeg, full_eog = None, None
        eeg_channel, eog_channel = "EEG Fpz-Cz", "EOG horizontal"

        if use_dummy:
            full_eeg, full_eog = get_dummy_data()
            st.success("Demo Data Loaded")
        else:
            uploaded_file = st.file_uploader("Upload Recording (EDF)", type=['edf'])
            if uploaded_file:
                        file_id = uploaded_file.name + str(uploaded_file.size)
                        if st.session_state.get("file_id") != file_id:
                            with st.spinner("Reading EDF file..."):
                                raw, err = load_edf_file(uploaded_file)
                                if err:
                                    st.error(f"EDF Error: {err}")
                                elif raw:
                                    chans = raw.ch_names
                                    data, _ = raw.get_data(return_times=True)
                                    st.session_state["file_id"] = file_id
                                    st.session_state["raw_chans"] = chans
                                    st.session_state["raw_data"] = data
            
                        if st.session_state.get("raw_data") is not None:
                            chans = st.session_state["raw_chans"]
                            data = st.session_state["raw_data"]
                            eeg_channel = st.selectbox("EEG Channel", chans, index=0)
                            eog_channel = st.selectbox("EOG Channel", chans, index=1 if len(chans) > 1 else 0)
                            full_eeg = data[chans.index(eeg_channel)]
                            full_eog = data[chans.index(eog_channel)]
                            st.success(f"✅ Loaded {len(full_eeg)//100//3600:.1f}h recording")

        st.divider()
        st.subheader("2. Ground Truth")
        gt_file = st.file_uploader("Upload Hypnogram", type=['edf', 'csv', 'txt'])

        st.divider()
        apply_smoothing = st.toggle("Apply Prediction Smoothing", True)

    # ── MAIN DASHBOARD ──
    if full_eeg is not None:
        total_hours = (len(full_eeg) / 100) / 3600

        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("Sleep Analysis Dashboard")
            st.caption("For research use only - Not a medical diagnosis")
        with col2:
            if st.button("▶ Run Analysis", type="primary", use_container_width=True):
                st.session_state["run"] = True
                st.session_state["results_ready"] = False

        start_h, end_h = st.slider(
            "Analysis Window (Hours)", 0.0, total_hours, (0.0, min(1.0, total_hours))
        )

        # ── RUN INFERENCE (only once per click) ──
        if st.session_state.get("run") and not st.session_state.get("results_ready"):
            start_epoch_idx = int((start_h * 3600) // 30)
            end_epoch_idx = int((end_h * 3600) // 30)
            num_epochs = end_epoch_idx - start_epoch_idx

            if num_epochs > 0:
                batch_eeg, batch_eog = [], []
                for i in range(num_epochs):
                    idx = start_epoch_idx + i
                    s, e = idx * 3000, (idx + 1) * 3000
                    if e > len(full_eeg):
                        break
                    batch_eeg.append(full_eeg[s:e].tolist())
                    batch_eog.append(full_eog[s:e].tolist())

                with st.spinner("🧠 Running AI inference..."):
                    model = load_model()
                    pred_indices, confidences = run_inference(model, batch_eeg, batch_eog)

                if pred_indices is not None:
                    if apply_smoothing:
                        pred_indices = smooth_predictions(pred_indices)

                    pred_labels = [CLASS_LABELS[i] for i in pred_indices]
                    has_truth = False
                    gt_indices = None

                    if gt_file:
                        gt_indices = load_ground_truth(gt_file, num_epochs, start_epoch_idx)
                        if gt_indices is not None and len(gt_indices) == len(pred_indices):
                            has_truth = True

                    time_axis = [start_h + (i * 30 / 3600) for i in range(len(pred_labels))]
                    y_ai = [HYPNOGRAM_MAP[l] for l in pred_labels]
                    acc = "N/A"
                    if has_truth:
                        mask = gt_indices != -1
                        acc = f"{accuracy_score(gt_indices[mask], pred_indices[mask])*100:.1f}%"

                    # Store results — avoids re-running on PDF click
                    st.session_state["results"] = {
                        "pred_labels": pred_labels,
                        "pred_indices": pred_indices.tolist(),
                        "confidences": confidences.tolist(),
                        "time_axis": time_axis,
                        "y_ai": y_ai,
                        "has_truth": has_truth,
                        "gt_indices": gt_indices.tolist() if gt_indices is not None else None,
                        "num_epochs": len(pred_labels),
                        "acc": acc,
                        "total_hours": total_hours,
                    }
                    st.session_state["results_ready"] = True

        # ── DISPLAY RESULTS ──
        if st.session_state.get("results_ready"):
            r = st.session_state["results"]
            pred_labels   = r["pred_labels"]
            pred_indices  = np.array(r["pred_indices"])
            confidences   = np.array(r["confidences"])
            time_axis     = r["time_axis"]
            y_ai          = r["y_ai"]
            has_truth     = r["has_truth"]
            gt_indices    = np.array(r["gt_indices"]) if r["gt_indices"] is not None else None
            num_epochs    = r["num_epochs"]
            acc           = r["acc"]
            total_hours_result = r["total_hours"]

            if has_truth:
                mask = gt_indices != -1
                valid_gt   = gt_indices[mask]
                valid_pred = pred_indices[mask]

            st.divider()
            st.markdown("### 📌 Key Results")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Duration", f"{(num_epochs*30)/60:.1f} min")
            m2.metric("Dominant Stage", pd.Series(pred_labels).mode()[0])
            m3.metric("Mean Confidence", f"{np.mean(confidences):.1f}%")
            m4.metric("Accuracy", acc)
            st.caption(f"{len(pred_labels)} epochs analysed - 30s resolution")

            tab1, tab2 = st.tabs(["📊 Hypnogram", "📈 Statistics"])

            with tab1:
                overlay = True
                if has_truth:
                    overlay = st.toggle("Show AI & Ground Truth in same graph", value=True)

                fig = go.Figure()
                if overlay or not has_truth:
                    fig.add_trace(go.Scatter(
                        x=time_axis, y=y_ai,
                        mode='lines+markers',
                        name="AI Prediction",
                        line=dict(color='#2980b9', width=3, shape='hv'),
                        text=[f"Stage: {l}<br>Confidence: {c:.1f}%" for l, c in zip(pred_labels, confidences)],
                        hoverinfo="text+x"
                    ))
                    if has_truth:
                        gt_labels_str = [CLASS_LABELS.get(int(i), "Unknown") for i in gt_indices]
                        y_gt = [HYPNOGRAM_MAP.get(l, -1) for l in gt_labels_str]
                        fig.add_trace(go.Scatter(
                            x=time_axis, y=y_gt,
                            mode='lines', name="Doctor (GT)",
                            line=dict(color='#27ae60', width=2, dash='dash', shape='hv')
                        ))
                    fig.update_layout(height=450, template="plotly_white")
                else:
                    gt_labels_str = [CLASS_LABELS.get(int(i), "Unknown") for i in gt_indices]
                    y_gt = [HYPNOGRAM_MAP.get(l, -1) for l in gt_labels_str]
                    fig.add_trace(go.Scatter(x=time_axis, y=y_ai, name="AI Prediction",
                        line=dict(color='#2980b9', width=3, shape='hv'), yaxis='y1'))
                    fig.add_trace(go.Scatter(x=time_axis, y=y_gt, name="Doctor (GT)",
                        line=dict(color='#27ae60', width=2, shape='hv'), yaxis='y2'))
                    fig.update_layout(height=600, template="plotly_white",
                        yaxis=dict(domain=[0.55, 1], title="AI"),
                        yaxis2=dict(domain=[0, 0.45], title="GT"))

                fig.update_yaxes(
                    tickvals=list(HYPNOGRAM_MAP.values()),
                    ticktext=list(HYPNOGRAM_MAP.keys()),
                    autorange="reversed"
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                c1, c2 = st.columns(2)
                with c1:
                    df = pd.DataFrame(pred_labels, columns=["Stage"])
                    fig_pie = px.pie(df, names="Stage", color="Stage",
                                    color_discrete_map=STAGE_COLORS, hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                with c2:
                    if has_truth:
                        cm = confusion_matrix(valid_gt, valid_pred, labels=[0,1,2,3,4])
                        fig_cm = px.imshow(cm, text_auto=True,
                                          x=list(CLASS_LABELS.values()),
                                          y=list(CLASS_LABELS.values()),
                                          color_continuous_scale="Blues")
                        st.plotly_chart(fig_cm, use_container_width=True)
                    else:
                        st.info("Upload GT file for validation metrics.")

                st.divider()
                if st.button("Generate PDF Report", use_container_width=True):
                    uname = user.user_metadata.get("full_name", user.email)
                    pdf_bytes = create_pdf_report(
                        pred_labels, acc, float(np.mean(confidences)),
                        time_axis, y_ai, uname, total_hours_result
                    )
                    st.download_button(
                        "⬇️ Download Report",
                        data=pdf_bytes,
                        file_name="NeuroSleep_Report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

    # ── HERO PAGE ──
    else:
        st.markdown("""
        <style>
        .hero-title {
            font-size: 70px; font-weight: 800;
            background: linear-gradient(90deg, #6EE7B7, #3B82F6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .hero-subtitle { font-size: 22px; color: #9CA3AF; margin-bottom: 20px; }
        .hero-container { text-align: center; margin-top: 100px; }
        </style>
        <div class='hero-container'>
            <div class='hero-title'>Decoding Sleep with AI</div>
            <div class='hero-subtitle'>Clinical-grade automated staging</div>
        </div>
        """, unsafe_allow_html=True)
