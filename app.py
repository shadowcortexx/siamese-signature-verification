import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import os
import numpy as np

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SigVerify — Signature Forgery Detector",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #1e2736;
    }

    /* Hide default header */
    header[data-testid="stHeader"] { background: transparent; }

    /* Hero title */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #e2e8f0 0%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0;
        line-height: 1.1;
    }
    .hero-sub {
        font-size: 1rem;
        color: #475569;
        margin-top: 6px;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    .hero-accent {
        display: inline-block;
        background: #1e3a5f;
        color: #60a5fa;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 12px;
    }

    /* Upload zone card */
    .upload-card {
        background: #161b27;
        border: 1px solid #1e2736;
        border-radius: 16px;
        padding: 28px 24px;
        height: 100%;
    }
    .upload-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 8px;
    }
    .upload-role {
        font-size: 1rem;
        color: #e2e8f0;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .upload-desc {
        font-size: 0.82rem;
        color: #475569;
        margin-bottom: 16px;
    }

    /* Result cards */
    .result-genuine {
        background: linear-gradient(135deg, #052e16 0%, #14532d 100%);
        border: 1px solid #166534;
        border-radius: 20px;
        padding: 32px;
        text-align: center;
    }
    .result-forged {
        background: linear-gradient(135deg, #3b0000 0%, #7f1d1d 100%);
        border: 1px solid #991b1b;
        border-radius: 20px;
        padding: 32px;
        text-align: center;
    }
    .result-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
    }
    .result-title-genuine { color: #4ade80; }
    .result-title-forged  { color: #f87171; }
    .result-subtitle {
        font-size: 0.85rem;
        color: #6b7280;
        letter-spacing: 0.5px;
    }

    /* Metric pills */
    .metric-pill {
        background: #1e2736;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #e2e8f0;
        font-variant-numeric: tabular-nums;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #475569;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 2px;
    }

    /* Distance bar */
    .dist-bar-bg {
        background: #1e2736;
        border-radius: 8px;
        height: 10px;
        overflow: hidden;
        margin: 8px 0 4px;
    }
    .dist-bar-fill-genuine {
        height: 100%;
        background: linear-gradient(90deg, #22c55e, #4ade80);
        border-radius: 8px;
    }
    .dist-bar-fill-forged {
        height: 100%;
        background: linear-gradient(90deg, #ef4444, #f87171);
        border-radius: 8px;
    }

    /* Info panel */
    .info-panel {
        background: #161b27;
        border: 1px solid #1e2736;
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
    }
    .info-heading {
        font-size: 0.72rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #334155;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .info-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid #1e2736;
        font-size: 0.85rem;
    }
    .info-row:last-child { border-bottom: none; }
    .info-key   { color: #475569; }
    .info-val   { color: #94a3b8; font-weight: 500; font-family: monospace; }

    /* Sidebar model info */
    .sidebar-chip {
        background: #1e2736;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.82rem;
        color: #94a3b8;
    }
    .sidebar-chip strong { color: #e2e8f0; display: block; font-size: 0.78rem;
                           letter-spacing: 1px; text-transform: uppercase;
                           color: #334155; margin-bottom: 2px; }

    /* Divider */
    .section-divider {
        border: none;
        border-top: 1px solid #1e2736;
        margin: 28px 0;
    }

    /* Placeholder zone */
    .placeholder-zone {
        background: #161b27;
        border: 2px dashed #1e2736;
        border-radius: 16px;
        padding: 60px 24px;
        text-align: center;
        margin-top: 24px;
    }
    .placeholder-icon { font-size: 2.5rem; margin-bottom: 12px; }
    .placeholder-text { color: #334155; font-size: 0.9rem; }

    /* Override Streamlit file uploader */
    [data-testid="stFileUploader"] > div {
        background: #0f1117 !important;
        border: 1px dashed #2d3748 !important;
        border-radius: 10px !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 14px 0 !important;
        width: 100%;
        letter-spacing: 0.3px;
        transition: opacity 0.15s;
    }
    .stButton > button:hover { opacity: 0.88; }

    /* Slider */
    [data-testid="stSlider"] > div > div > div { background: #1d4ed8 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Model definition (must match training) ──────────────────────────────────
class SiameseNet(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        bb = models.resnet18(weights=None)
        bb.fc = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, embed_dim))
        self.bb = bb

    def forward_one(self, x): return self.bb(x)
    def forward(self, x1, x2): return self.forward_one(x1), self.forward_one(x2)


@st.cache_resource
def load_model(ckpt_path, embed_dim=128):
    model = SiameseNet(embed_dim)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    return model


def get_transform():
    return T.Compose([
        T.Grayscale(3),
        T.Resize((105, 105)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def predict(model, img1: Image.Image, img2: Image.Image, threshold: float):
    tf = get_transform()
    with torch.no_grad():
        t1 = tf(img1.convert("RGB")).unsqueeze(0)
        t2 = tf(img2.convert("RGB")).unsqueeze(0)
        e1, e2 = model(t1, t2)
        dist = F.pairwise_distance(e1, e2).item()
    genuine = dist < threshold
    confidence = max(0.0, min(100.0, (1.0 - dist / (threshold * 2)) * 100))
    if not genuine:
        confidence = 100.0 - confidence
    return genuine, round(dist, 4), round(confidence, 1)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ✍️ SigVerify")
    st.markdown('<div style="color:#334155;font-size:0.8rem;margin-bottom:20px;">Siamese Network · Forgery Detection</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Model settings**")
    ckpt = st.text_input("Checkpoint path", value="siamese_best.pth", label_visibility="collapsed",
                         placeholder="Path to .pth checkpoint...")
    threshold = st.slider("Decision threshold", 0.1, 2.0, 0.5, 0.05,
                          help="Distance below threshold → GENUINE. Raise to be more lenient, lower to be stricter.")
    embed_dim = st.selectbox("Embedding dimension", [64, 128, 256], index=1)

    st.markdown("---")
    st.markdown('<div class="info-heading">Architecture</div>', unsafe_allow_html=True)
    for k, v in [("Backbone", "ResNet-18"), ("Pretrained", "ImageNet"), ("Loss", "Contrastive"),
                 ("Frozen layers", "layer1, layer2"), ("Image size", "105 × 105")]:
        st.markdown(f'<div class="sidebar-chip"><strong>{k}</strong>{v}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("CEDAR Dataset · 55 signers · 24 genuine + 24 forgeries each")


# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown('<span class="hero-accent">Deep Learning · Signature Verification</span>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Signature Forgery<br>Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload a reference and a test signature — the Siamese network decides.</div>', unsafe_allow_html=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ─── Upload columns ──────────────────────────────────────────────────────────
col_a, col_b = st.columns(2, gap="large")

with col_a:
    st.markdown('<div class="upload-label">Input A</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-role">Reference signature</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-desc">A known genuine sample from the signer</div>', unsafe_allow_html=True)
    ref_file = st.file_uploader("Reference", type=["png","jpg","jpeg","bmp"],
                                label_visibility="collapsed", key="ref")
    if ref_file:
        st.image(Image.open(ref_file), use_column_width=True)

with col_b:
    st.markdown('<div class="upload-label">Input B</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-role">Test signature</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-desc">The signature you want to verify</div>', unsafe_allow_html=True)
    test_file = st.file_uploader("Test", type=["png","jpg","jpeg","bmp"],
                                 label_visibility="collapsed", key="test")
    if test_file:
        st.image(Image.open(test_file), use_column_width=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ─── Analysis button & results ───────────────────────────────────────────────
btn_col, _ = st.columns([1, 2])
with btn_col:
    run = st.button("Analyse signatures →", use_container_width=True)

if run:
    if not ref_file or not test_file:
        st.warning("Please upload both signatures before analysing.")
    elif not os.path.exists(ckpt):
        st.error(f"Model checkpoint not found at `{ckpt}`. Train first with `python train.py`.")
    else:
        model = load_model(ckpt, embed_dim)
        ref_img  = Image.open(ref_file)
        test_img = Image.open(test_file)

        with st.spinner("Running inference…"):
            genuine, dist, confidence = predict(model, ref_img, test_img, threshold)

        st.markdown("<br>", unsafe_allow_html=True)

        # Result card
        verdict_label = "GENUINE" if genuine else "FORGED"
        card_cls      = "result-genuine" if genuine else "result-forged"
        title_cls     = "result-title-genuine" if genuine else "result-title-forged"
        icon          = "✅" if genuine else "❌"

        st.markdown(f"""
        <div class="{card_cls}">
            <div class="result-title {title_cls}">{icon} {verdict_label}</div>
            <div class="result-subtitle">{confidence}% confidence · distance {dist}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Metric pills
        m1, m2, m3, m4 = st.columns(4)
        fill_pct = min(100, int((dist / (threshold * 2)) * 100))
        bar_cls  = "dist-bar-fill-genuine" if genuine else "dist-bar-fill-forged"

        with m1:
            st.markdown(f'<div class="metric-pill"><div class="metric-value">{dist}</div><div class="metric-label">Distance</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-pill"><div class="metric-value">{threshold}</div><div class="metric-label">Threshold</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-pill"><div class="metric-value">{confidence}%</div><div class="metric-label">Confidence</div></div>', unsafe_allow_html=True)
        with m4:
            gap = round(abs(dist - threshold), 4)
            st.markdown(f'<div class="metric-pill"><div class="metric-value">{gap}</div><div class="metric-label">Margin</div></div>', unsafe_allow_html=True)

        # Distance bar
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="color:#475569;font-size:0.78rem;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">
            Distance vs threshold
        </div>
        <div class="dist-bar-bg">
            <div class="{bar_cls}" style="width:{fill_pct}%;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#334155;">
            <span>0.0</span><span>threshold {threshold}</span><span>{threshold*2:.1f}</span>
        </div>
        """, unsafe_allow_html=True)

        # Technical breakdown
        with st.expander("Technical breakdown"):
            st.markdown(f"""
            <div class="info-panel">
                <div class="info-heading">Decision details</div>
                <div class="info-row"><span class="info-key">Euclidean distance</span><span class="info-val">{dist}</span></div>
                <div class="info-row"><span class="info-key">Threshold</span><span class="info-val">{threshold}</span></div>
                <div class="info-row"><span class="info-key">Rule</span><span class="info-val">dist {'<' if genuine else '>='} threshold → {verdict_label}</span></div>
                <div class="info-row"><span class="info-key">Embedding dim</span><span class="info-val">{embed_dim}</span></div>
                <div class="info-row"><span class="info-key">Confidence</span><span class="info-val">{confidence}%</span></div>
            </div>
            """, unsafe_allow_html=True)

elif not (ref_file and test_file):
    st.markdown("""
    <div class="placeholder-zone">
        <div class="placeholder-icon">✍️</div>
        <div class="placeholder-text">Upload a reference and a test signature above, then click <strong>Analyse</strong></div>
    </div>
    """, unsafe_allow_html=True)
