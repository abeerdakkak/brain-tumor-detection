import textwrap
import streamlit as st
import torch
import torch.nn as nn
import open_clip
import requests
from io import BytesIO
from PIL import Image
from streamlit_paste_button import paste_image_button as pbutton

# ------------------------------------------------------------------
# App config
# ------------------------------------------------------------------
st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="◆", layout="centered")

CLASS_NAMES = ["Meningioma", "Glioma", "Pituitary Tumor"]
CLASS_INFO = {
    "Meningioma": "Forms in the membranes covering the brain and spinal cord.",
    "Glioma": "Originates in the brain's glial (supportive) cells.",
    "Pituitary Tumor": "Forms in the pituitary gland at the base of the brain.",
}
BIOMEDCLIP_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
HEAD_WEIGHTS_PATH = "biomedclip_head.pth"


# ------------------------------------------------------------------
# Styling — clinical console aesthetic: dark reading-room background
# (radiology viewers run dark to preserve contrast on grayscale scans),
# monospace readouts for data, a restrained teal accent for the one
# thing that should stand out: the prediction.
# ------------------------------------------------------------------
st.markdown(
    textwrap.dedent(
        """\
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
    :root {
        --bg: #12151b;
        --panel: #181c24;
        --border: #262b35;
        --text: #e7e9ed;
        --text-dim: #8891a0;
        --accent: #4fd1c5;
        --accent-dim: #2c6b65;
        --warn: #e8a557;
    }
    .stApp {
        background: var(--bg);
        font-family: 'IBM Plex Sans', sans-serif;
    }
    * { font-family: 'IBM Plex Sans', sans-serif; }
    .mono { font-family: 'IBM Plex Mono', monospace; }

    .status-bar {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        letter-spacing: 0.02em;
        color: var(--text-dim);
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.9rem;
        margin-bottom: 1.6rem;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--accent);
        display: inline-block;
    }

    .page-title {
        font-size: 1.9rem;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 0.15rem;
        letter-spacing: -0.01em;
    }
    .page-sub {
        color: var(--text-dim);
        font-size: 0.95rem;
        margin-bottom: 1.8rem;
        max-width: 46ch;
    }

    .panel-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: var(--text-dim);
        letter-spacing: 0.04em;
        margin-bottom: 0.6rem;
    }

    .readout {
        background: var(--panel);
        border: 1px solid var(--border);
        border-left: 2px solid var(--accent);
        padding: 1.3rem 1.4rem;
        margin-top: 0.4rem;
    }
    .readout-pred {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 0.15rem;
    }
    .readout-conf {
        font-family: 'IBM Plex Mono', monospace;
        color: var(--accent);
        font-size: 0.95rem;
        margin-bottom: 0.8rem;
    }
    .readout-desc {
        color: var(--text-dim);
        font-size: 0.88rem;
        line-height: 1.5;
    }

    .prob-row {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin-bottom: 0.55rem;
    }
    .prob-name {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        color: var(--text-dim);
        width: 130px;
        flex-shrink: 0;
    }
    .prob-track {
        flex: 1;
        height: 6px;
        background: #21262f;
        position: relative;
    }
    .prob-fill {
        height: 100%;
        background: var(--accent);
    }
    .prob-val {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        color: var(--text-dim);
        width: 46px;
        text-align: right;
        flex-shrink: 0;
    }

    [data-testid="stFileUploader"] {
        border: 1px dashed var(--border);
        background: var(--panel);
        padding: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
    }
    footer {visibility: hidden;}
    </style>
    """
    ),
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# Model definition (must match training notebook exactly)
# ------------------------------------------------------------------
class BioMedCLIPLinearProbeClassifier(nn.Module):
    def __init__(self, visual_encoder, embed_dim, num_classes):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.head = nn.Linear(embed_dim, num_classes)
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

    def forward(self, pixel_values):
        image_features = self.visual_encoder(pixel_values)
        return self.head(image_features)


@st.cache_resource
def load_model():
    biomedclip_model, preprocess = open_clip.create_model_from_pretrained(BIOMEDCLIP_NAME)
    visual_encoder = biomedclip_model.visual

    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224)
        embed_dim = visual_encoder(dummy).shape[-1]

    model = BioMedCLIPLinearProbeClassifier(visual_encoder, embed_dim, num_classes=len(CLASS_NAMES))
    state_dict = torch.load(HEAD_WEIGHTS_PATH, map_location="cpu")
    model.head.load_state_dict(state_dict)
    model.eval()

    return model, preprocess


def run_prediction(image: Image.Image):
    image = image.convert("RGB")
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="panel-label">INPUT SCAN</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with st.spinner("Running inference..."):
        model, preprocess = load_model()
        pixel_values = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            logits = model(pixel_values)
            probs = torch.softmax(logits, dim=1)[0]

    pred_idx = int(probs.argmax())
    pred_name = CLASS_NAMES[pred_idx]

    with col2:
        st.markdown('<div class="panel-label">CLASSIFICATION</div>', unsafe_allow_html=True)
        st.markdown(
            textwrap.dedent(
                f"""\
            <div class="readout">
                <div class="readout-pred">{pred_name}</div>
                <div class="readout-conf mono">{probs[pred_idx]*100:.1f}% confidence</div>
                <div class="readout-desc">{CLASS_INFO[pred_name]}</div>
            </div>
            """
            ),
            unsafe_allow_html=True,
        )

    st.markdown('<div class="panel-label" style="margin-top:1.6rem;">PROBABILITY DISTRIBUTION</div>', unsafe_allow_html=True)
    rows = ""
    for name, p in zip(CLASS_NAMES, probs.tolist()):
        rows += textwrap.dedent(
            f"""\
        <div class="prob-row">
            <div class="prob-name">{name}</div>
            <div class="prob-track"><div class="prob-fill" style="width:{p*100:.1f}%;"></div></div>
            <div class="prob-val mono">{p*100:.1f}%</div>
        </div>
        """
        )
    st.markdown(rows, unsafe_allow_html=True)


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.markdown(
    textwrap.dedent(
        """\
    <div class="status-bar">
        <span class="status-dot"></span> MODEL READY &nbsp;·&nbsp; BIOMEDCLIP LINEAR PROBE &nbsp;·&nbsp; 3-CLASS
    </div>
    """
    ),
    unsafe_allow_html=True,
)

st.markdown('<div class="page-title">Brain Tumor MRI Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">Upload, paste, or link a T1-weighted contrast-enhanced MRI slice to classify it as meningioma, glioma, or pituitary tumor.</div>',
    unsafe_allow_html=True,
)

tab_upload, tab_paste, tab_url = st.tabs(["Upload", "Paste", "URL"])

image_to_classify = None

with tab_upload:
    uploaded_file = st.file_uploader("MRI image file", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    if uploaded_file is not None:
        image_to_classify = Image.open(uploaded_file)

with tab_paste:
    st.caption("Copy an image, then paste it in. Works in Chrome, Edge, and Safari.")
    paste_result = pbutton("Paste image from clipboard")
    if paste_result.image_data is not None:
        image_to_classify = paste_result.image_data

with tab_url:
    image_url = st.text_input("Image URL", placeholder="https://example.com/scan.jpg", label_visibility="collapsed")
    if image_url:
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image_to_classify = Image.open(BytesIO(response.content))
        except Exception:
            st.error("Couldn't load an image from that link. Check the URL and try again.")

if image_to_classify is not None:
    run_prediction(image_to_classify)
else:
    st.markdown(
        '<div class="panel-label" style="margin-top:0.5rem;">Provide a scan above to get a classification.</div>',
        unsafe_allow_html=True,
    )
