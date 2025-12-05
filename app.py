# app.py
import streamlit as st
import torch
import torch.nn as nn
import requests
from urllib.parse import urlparse

# =========================
# 0. BASIC CONFIG
# =========================
st.set_page_config(
    page_title="WebPhish-Lite++",
    page_icon="🕸️",
    layout="centered",
)

# Change this to your actual filename in the repo
MODEL_PATH = "webphish_lite_pp.pth"  # e.g. "checkpoint_webphish.pth"

DEVICE = torch.device("cpu")

# =========================
# 1. MODEL DEFINITION
# =========================
# Paste the exact same architecture you used during training.
# This example follows your described architecture.

class WebPhishLitePP(nn.Module):
    def __init__(
        self,
        url_vocab_size: int = 256,
        html_vocab_size: int = 50000,
        embed_dim: int = 16,
    ):
        super().__init__()
        # URL + HTML embeddings
        self.url_emb = nn.Embedding(url_vocab_size, embed_dim, padding_idx=0)
        self.html_emb = nn.Embedding(html_vocab_size, embed_dim, padding_idx=0)

        # Multi-kernel Conv1D block over concatenated embeddings
        self.conv3 = nn.Conv1d(embed_dim, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, 32, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(embed_dim, 32, kernel_size=7, padding=3)

        self.gelu = nn.GELU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        # 32 filters * 3 kernels = 96
        self.layernorm = nn.LayerNorm(96)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(96, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, url_ids, html_ids):
        """
        url_ids:  (batch, url_seq_len)  LongTensor
        html_ids: (batch, html_seq_len) LongTensor
        """
        url_emb = self.url_emb(url_ids)    # (B, L_u, E)
        html_emb = self.html_emb(html_ids) # (B, L_h, E)

        # Concatenate along sequence dimension
        x = torch.cat([url_emb, html_emb], dim=1)  # (B, L_u + L_h, E)

        # Conv1d expects (B, C_in, L)
        x = x.transpose(1, 2)  # (B, E, L)

        x3 = self.gelu(self.conv3(x))
        x5 = self.gelu(self.conv5(x))
        x7 = self.gelu(self.conv7(x))

        x = torch.cat([x3, x5, x7], dim=1)  # (B, 96, L)

        x = self.maxpool(x)                 # (B, 96, L/2)

        # Global average over sequence
        x = x.mean(dim=-1)                  # (B, 96)

        x = self.layernorm(x)
        x = self.dropout(self.gelu(self.fc1(x)))
        x = self.dropout(self.gelu(self.fc2(x)))
        logits = self.out(x)                # (B, 1)
        return logits


# =========================
# 2. GLOBAL HYPERPARAMS (WILL BE OVERRIDDEN FROM CKPT)
# =========================

URL_MAX_LEN = 200
HTML_MAX_LEN = 2000
URL_VOCAB_SIZE = 256
HTML_VOCAB_SIZE = 50000
EMBED_DIM = 16


# =========================
# 3. MODEL LOADING
# =========================
@st.cache_resource
def load_model():
    """
    Handles both:
    - full model saved via torch.save(model, path)
    - checkpoint dict saved via torch.save({
          "model_state_dict": model.state_dict(),
          "char_vocab_size": ...,
          "word_vocab_size": ...,
          "url_len": ...,
          "html_len": ...,
          "char_emb_dim": ...,
          "word_emb_dim": ...
      }, path)
    """
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

    # Case 1: full model object (no "model_state_dict" key)
    if not (isinstance(ckpt, dict) and "model_state_dict" in ckpt):
        model = ckpt
        model.to(DEVICE)
        model.eval()
        return model

    # Case 2: checkpoint dict (your logs show this case)
    global URL_MAX_LEN, HTML_MAX_LEN, URL_VOCAB_SIZE, HTML_VOCAB_SIZE, EMBED_DIM

    URL_VOCAB_SIZE = ckpt.get("char_vocab_size", URL_VOCAB_SIZE)
    HTML_VOCAB_SIZE = ckpt.get("word_vocab_size", HTML_VOCAB_SIZE)
    URL_MAX_LEN = ckpt.get("url_len", URL_MAX_LEN)
    HTML_MAX_LEN = ckpt.get("html_len", HTML_MAX_LEN)
    EMBED_DIM = ckpt.get("char_emb_dim", EMBED_DIM)  # assuming char_emb_dim == word_emb_dim

    model = WebPhishLitePP(
        url_vocab_size=URL_VOCAB_SIZE,
        html_vocab_size=HTML_VOCAB_SIZE,
        embed_dim=EMBED_DIM,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()

# =========================
# 4. PREPROCESSING
# =========================

def fetch_html(url: str) -> str:
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=8)
        resp.raise_for_status()
        # Hard truncate to avoid extremely long pages
        return resp.text[:50_000]
    except Exception as e:
        st.warning(f"Could not fetch HTML: {e}")
        return ""


def encode_url(url: str):
    """
    Simple character-level encoding.
    Make sure this matches what you used during training.
    """
    url = url.strip()
    # Ensure protocol so requests.get doesn't break
    parsed = urlparse(url)
    if not parsed.scheme:
        url = "http://" + url

    ids = []
    for ch in url:
        idx = ord(ch) if ord(ch) < URL_VOCAB_SIZE else 0
        ids.append(idx)

    # pad / truncate to URL_MAX_LEN (will be set from ckpt)
    if len(ids) < URL_MAX_LEN:
        ids = ids + [0] * (URL_MAX_LEN - len(ids))
    else:
        ids = ids[:URL_MAX_LEN]

    return torch.tensor([ids], dtype=torch.long, device=DEVICE), url


def encode_html(html: str):
    """
    Very basic whitespace token -> hashed index.
    Adapt this to your actual HTML tokenizer if needed.
    """
    tokens = html.split()
    ids = []
    for tok in tokens:
        idx = hash(tok) % HTML_VOCAB_SIZE
        ids.append(idx)

    if len(ids) < HTML_MAX_LEN:
        ids = ids + [0] * (HTML_MAX_LEN - len(ids))
    else:
        ids = ids[:HTML_MAX_LEN]

    return torch.tensor([ids], dtype=torch.long, device=DEVICE)


def predict(url: str):
    html = fetch_html(url)
    url_ids, normalized_url = encode_url(url)
    html_ids = encode_html(html)

    with torch.no_grad():
        logits = model(url_ids, html_ids)
        prob = torch.sigmoid(logits).item()

    label = "Phishing" if prob >= 0.5 else "Legitimate"
    return normalized_url, prob, label, html


# =========================
# 5. STREAMLIT UI
# =========================

st.title("🕸️ WebPhish-Lite++")
st.subheader("Phishing Website Detection (URL + HTML)")

st.markdown(
    """
Enter a URL below and the model will:

1. Fetch the HTML of the page  
2. Encode the URL + HTML  
3. Run them through your trained WebPhish-Lite++ model  
4. Return a phishing probability and label
"""
)

url_input = st.text_input(
    "Website URL",
    placeholder="example: https://secure-login-example.com",
)

if st.button("Analyze URL") and url_input:
    with st.spinner("Analyzing website..."):
        normalized_url, prob, label, html = predict(url_input)

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Phishing probability",
            f"{prob*100:.2f} %",
        )
    with col2:
        st.metric("Prediction", label)

    st.markdown("---")
    with st.expander("Show normalized URL"):
        st.code(normalized_url, language="text")

    with st.expander("Show first 1500 characters of HTML"):
        st.code(html[:1500], language="html")
elif url_input:
    st.info("Click 'Analyze URL' to run the model.")
