import os
import streamlit as st
import torch
import torch.nn as nn
import requests
from urllib.parse import urlparse

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="WebPhish-Lite++", page_icon="🕸️")

MODEL_PATH = "webphish_lite_complete.pth"
DEVICE = torch.device("cpu")

# Will be filled from checkpoint:
URL_MAX_LEN = 0
HTML_MAX_LEN = 0
FINAL_LEN = 0


# =========================
# MODEL (EXACT CHECKPOINT)
# =========================
class WebPhishLitePP(nn.Module):
    def __init__(self, url_vocab_size=52, html_vocab_size=5002, embed_dim=16):
        super().__init__()

        self.url_embedding = nn.Embedding(url_vocab_size, embed_dim, padding_idx=0)
        self.html_embedding = nn.Embedding(html_vocab_size, embed_dim, padding_idx=0)

        # 16 + 16 = 32 channels after concat
        in_channels = 32

        self.conv3 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)

        self.gelu = nn.GELU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        self.layer_norm = nn.LayerNorm(96)

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(96, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, url_ids, html_ids):
        # (B, L, 16)
        u = self.url_embedding(url_ids)
        h = self.html_embedding(html_ids)

        # concat embedding dimension → (B, L, 32)
        x = torch.cat([u, h], dim=-1)

        # Conv expects (B, C, L)
        x = x.transpose(1, 2)

        x3 = self.gelu(self.conv3(x))
        x5 = self.gelu(self.conv5(x))
        x7 = self.gelu(self.conv7(x))

        x = torch.cat([x3, x5, x7], dim=1)

        x = self.maxpool(x)
        x = x.mean(dim=-1)  # global avg → (B, 96)

        x = self.layer_norm(x)

        x = self.dropout(self.gelu(self.fc1(x)))
        x = self.dropout(self.gelu(self.fc2(x)))

        return self.fc3(x)


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    global URL_MAX_LEN, HTML_MAX_LEN, FINAL_LEN

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found.")
        return None

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

    URL_MAX_LEN = ckpt["url_len"]   # EXACT from checkpoint
    HTML_MAX_LEN = ckpt["html_len"] # EXACT from checkpoint
    FINAL_LEN = max(URL_MAX_LEN, HTML_MAX_LEN)

    model = WebPhishLitePP(
        url_vocab_size=ckpt["char_vocab_size"],
        html_vocab_size=ckpt["word_vocab_size"],
        embed_dim=ckpt["char_emb_dim"]
    )

    try:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    except Exception as e:
        st.error(f"Could not load weights: {e}")
        return None

    model.to(DEVICE)
    model.eval()
    return model


model = load_model()


# =========================
# PREPROCESSING
# =========================
def encode_url(url: str):
    parsed = urlparse(url)
    if not parsed.scheme:
        url = "http://" + url

    ids = [ord(c) % 52 for c in url]

    # pad to URL_MAX_LEN
    ids = ids[:URL_MAX_LEN] + [0] * max(0, URL_MAX_LEN - len(ids))

    # pad up to FINAL_LEN for concatenation
    ids = ids + [0] * (FINAL_LEN - len(ids))

    return torch.tensor([ids], dtype=torch.long, device=DEVICE), url


def encode_html(html: str):
    tokens = html.split()
    ids = [(hash(t) % 5002) for t in tokens]

    # pad to HTML_MAX_LEN
    ids = ids[:HTML_MAX_LEN] + [0] * max(0, HTML_MAX_LEN - len(ids))

    # pad to FINAL_LEN
    ids = ids[:FINAL_LEN] + [0] * max(0, FINAL_LEN - len(ids))

    return torch.tensor([ids], dtype=torch.long, device=DEVICE)


def fetch_html(url: str):
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        resp.raise_for_status()
        return resp.text[:30000]
    except:
        return ""


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
# STREAMLIT UI
# =========================
st.title("🕸️ WebPhish-Lite++")
st.subheader("Neural phishing detector (URL + HTML)")

url_input = st.text_input("Enter URL")

if st.button("Analyze") and url_input:
    if model is None:
        st.error("Model could not be loaded.")
    else:
        clean_url, prob, label, html = predict(url_input)

        st.metric("Phishing Probability", f"{prob*100:.2f}%")
        st.metric("Prediction", label)

        with st.expander("Normalized URL"):
            st.code(clean_url)

        with st.expander("HTML Preview"):
            st.code(html[:2000], language="html")
