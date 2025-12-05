# app.py
import os
import streamlit as st
import torch
import torch.nn as nn
import requests
from urllib.parse import urlparse

# =========================
# BASIC CONFIG
# =========================
st.set_page_config(page_title="WebPhish-Lite++", page_icon="🕸️", layout="centered")

# Your checkpoint file name
MODEL_PATH = "webphish_lite_complete.pth"
DEVICE = torch.device("cpu")


# =========================
# 1. EXACT CHECKPOINT MODEL
# =========================
class WebPhishLitePP(nn.Module):
    def __init__(self, url_vocab_size=52, html_vocab_size=5002, embed_dim=16):
        super().__init__()

        # EXACT embedding sizes from checkpoint
        self.url_embedding = nn.Embedding(url_vocab_size, embed_dim, padding_idx=0)
        self.html_embedding = nn.Embedding(html_vocab_size, embed_dim, padding_idx=0)

        # CONCAT URL + HTML embeddings on last dim → 16 + 16 = 32 channels
        in_channels = 32

        # EXACT conv shapes from checkpoint: (32 input → 32 output)
        self.conv3 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)

        self.gelu = nn.GELU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        # 32 filters × 3 convs = 96
        self.layer_norm = nn.LayerNorm(96)

        # Fully connected layers EXACT from checkpoint
        self.fc1 = nn.Linear(96, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, url_ids, html_ids):
        # embeddings → (B, L, 16)
        url_emb = self.url_embedding(url_ids)
        html_emb = self.html_embedding(html_ids)

        # concat embeddings → (B, L, 32)
        x = torch.cat([url_emb, html_emb], dim=-1)

        # conv expects (B, C, L)
        x = x.transpose(1, 2)

        x3 = self.gelu(self.conv3(x))
        x5 = self.gelu(self.conv5(x))
        x7 = self.gelu(self.conv7(x))

        # concat conv outputs → (B, 96, L)
        x = torch.cat([x3, x5, x7], dim=1)

        x = self.maxpool(x)

        # global average pooling → (B, 96)
        x = x.mean(dim=-1)

        x = self.layer_norm(x)
        x = self.dropout(self.gelu(self.fc1(x)))
        x = self.dropout(self.gelu(self.fc2(x)))

        out = self.fc3(x)
        return out


# =========================
# 2. LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found in repo.")
        return None

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

    # Restore EXACT hyperparams
    url_vocab = ckpt["char_vocab_size"]      # 52
    html_vocab = ckpt["word_vocab_size"]     # 5002
    embed_dim = ckpt["char_emb_dim"]         # 16

    model = WebPhishLitePP(
        url_vocab_size=url_vocab,
        html_vocab_size=html_vocab,
        embed_dim=embed_dim
    )

    try:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    except Exception as e:
        st.error(f"Failed to load checkpoint: {e}")
        return None

    model.to(DEVICE)
    model.eval()
    return model


model = load_model()


# =========================
# 3. PREPROCESSING
# =========================

def fetch_html(url: str) -> str:
    try:
        resp = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0"
        }, timeout=8)
        resp.raise_for_status()
        return resp.text[:30000]  # truncate
    except:
        return ""


def encode_url(url: str, max_len=128):
    parsed = urlparse(url)
    if not parsed.scheme:
        url = "http://" + url

    ids = [ord(c) % 52 for c in url]  # match vocab=52
    ids = ids[:max_len] + [0] * max(0, max_len - len(ids))

    return torch.tensor([ids], dtype=torch.long, device=DEVICE), url


def encode_html(html: str, max_len=1024):
    tokens = html.split()
    ids = [(hash(t) % 5002) for t in tokens]  # match vocab=5002
    ids = ids[:max_len] + [0] * max(0, max_len - len(ids))

    return torch.tensor([ids], dtype=torch.long, device=DEVICE)


def predict(url: str):
    html = fetch_html(url)
    url_ids, clean_url = encode_url(url)
    html_ids = encode_html(html)

    with torch.no_grad():
        logit = model(url_ids, html_ids)
        prob = torch.sigmoid(logit).item()

    label = "Phishing" if prob >= 0.5 else "Legitimate"
    return clean_url, prob, label, html


# =========================
# 4. STREAMLIT UI
# =========================
st.title("🕸️ WebPhish-Lite++")
st.subheader("Phishing Website Detection (URL + HTML)")

url_input = st.text_input("Enter a URL:")

if st.button("Analyze") and url_input:
    if model is None:
        st.error("Model not loaded.")
    else:
        clean_url, prob, label, html = predict(url_input)

        st.metric("Phishing Probability", f"{prob*100:.2f}%")
        st.metric("Prediction", label)

        with st.expander("Normalized URL"):
            st.code(clean_url)

        with st.expander("HTML Preview"):
            st.code(html[:2000], language="html")
