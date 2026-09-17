import streamlit as st
import torch
import torch.nn as nn
import open_clip
from PIL import Image
from streamlit_paste_button import paste_image_button as pbutton

# ------------------------------------------------------------------
# App config
# ------------------------------------------------------------------
st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="centered")

CLASS_NAMES = ["Meningioma", "Glioma", "Pituitary Tumor"]
CLASS_INFO = {
    "Meningioma": "Usually forms in the membranes covering the brain and spinal cord.",
    "Glioma": "Originates in the brain's glial (supportive) cells.",
    "Pituitary Tumor": "Forms in the pituitary gland at the base of the brain.",
}
BIOMEDCLIP_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
HEAD_WEIGHTS_PATH = "biomedclip_head.pth"


# ------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0f1620 0%, #0a0e14 100%);
    }
    .hero {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #7dd3fc, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero p {
        color: #9ca3af;
        font-size: 1.05rem;
    }
    .result-card {
        background: #131b26;
        border: 1px solid #23303f;
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .pred-label {
        font-size: 1.6rem;
        font-weight: 700;
        color: #7dd3fc;
    }
    .pred-sub {
        color: #9ca3af;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }
    [data-testid="stFileUploader"] {
        border: 1px dashed #334155;
        border-radius: 14px;
        padding: 0.5rem;
    }
    </style>
    """,
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
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="MRI slice", use_container_width=True)

    with st.spinner("Analyzing scan..."):
        model, preprocess = load_model()
        pixel_values = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            logits = model(pixel_values)
            probs = torch.softmax(logits, dim=1)[0]

    pred_idx = int(probs.argmax())
    pred_name = CLASS_NAMES[pred_idx]

    with col2:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="pred-sub">Prediction</div>
                <div class="pred-label">{pred_name}</div>
                <div class="pred-sub">Confidence: {probs[pred_idx]*100:.1f}%</div>
                <div style="color:#cbd5e1; font-size:0.9rem;">{CLASS_INFO[pred_name]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Class probabilities")
    for name, p in zip(CLASS_NAMES, probs.tolist()):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.progress(p)
        with c2:
            st.write(f"**{name}** — {p*100:.1f}%")


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🧠 Brain Tumor MRI Classifier</h1>
        <p>Upload or paste a brain MRI slice to get a prediction</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_upload, tab_paste = st.tabs(["📁 Upload", "📋 Paste"])

image_to_classify = None

with tab_upload:
    uploaded_file = st.file_uploader("Upload an MRI image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image_to_classify = Image.open(uploaded_file)

with tab_paste:
    st.caption("Copy an image, then click below to paste it in (works in Chrome, Edge, Safari).")
    paste_result = pbutton("📋 Paste image from clipboard")
    if paste_result.image_data is not None:
        image_to_classify = paste_result.image_data

if image_to_classify is not None:
    run_prediction(image_to_classify)
else:
    st.info("Upload or paste an MRI image above to get started.")
