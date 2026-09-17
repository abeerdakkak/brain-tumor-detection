import streamlit as st
import torch
import torch.nn as nn
import open_clip
from PIL import Image

# ------------------------------------------------------------------
# App config
# ------------------------------------------------------------------
st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠", layout="centered")

CLASS_NAMES = ["Meningioma", "Glioma", "Pituitary Tumor"]
BIOMEDCLIP_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
HEAD_WEIGHTS_PATH = "biomedclip_head.pth"  # tiny file, a few KB


# ------------------------------------------------------------------
# Model definition (must match training notebook exactly)
# ------------------------------------------------------------------
class BioMedCLIPLinearProbeClassifier(nn.Module):
    """Frozen BioMedCLIP visual encoder + a trainable Linear(embed_dim -> num_classes) head."""

    def __init__(self, visual_encoder, embed_dim, num_classes):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.head = nn.Linear(embed_dim, num_classes)
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

    def forward(self, pixel_values):
        image_features = self.visual_encoder(pixel_values)
        return self.head(image_features)


# ------------------------------------------------------------------
# Cached model loading — runs once per app session, not per request.
# The BiomedCLIP backbone downloads from Hugging Face Hub the first
# time the app runs (and Streamlit Cloud caches it on disk after that).
# Only the tiny trained head is loaded from the file you upload to GitHub.
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("🧠 Brain Tumor MRI Classifier")
st.caption("AIxMED Summer Series — BioMedCLIP linear-probe classifier")
st.write(
    "Upload a T1-weighted contrast-enhanced brain MRI slice. The model predicts "
    "one of three tumor types: **Meningioma**, **Glioma**, or **Pituitary Tumor**."
)

st.warning(
    "⚠️ This is a student research project for educational purposes only. "
    "It is **not** a diagnostic tool and must not be used for real medical decisions."
)

uploaded_file = st.file_uploader("Upload an MRI image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI slice", use_container_width=True)

    with st.spinner("Loading model and running inference..."):
        model, preprocess = load_model()
        pixel_values = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            logits = model(pixel_values)
            probs = torch.softmax(logits, dim=1)[0]

    pred_idx = int(probs.argmax())
    st.subheader(f"Prediction: **{CLASS_NAMES[pred_idx]}**")
    st.write(f"Confidence: {probs[pred_idx]*100:.1f}%")

    st.write("### Class probabilities")
    for name, p in zip(CLASS_NAMES, probs.tolist()):
        st.write(f"{name}")
        st.progress(p)
