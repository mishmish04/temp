# app.py
import os
import pickle
import hashlib

import requests
from bs4 import BeautifulSoup

import torch
import torch.nn as nn
import torch.nn.functional as F

import streamlit as st


# =========================
# 1. MODEL DEFINITION
# =========================

class WebPhishLite(nn.Module):
    """
    WebPhish-Lite++: URL (char) + HTML (word) dual-branch CNN
    """
    def __init__(
        self,
        char_vocab_size=52,
        word_vocab_size=5002,
        char_emb_dim=16,
        word_emb_dim=16,
        url_len=200,
        html_len=500,
    ):
        super().__init__()

        self.url_len = url_len
        self.html_len = html_len

        # Embeddings
        self.url_embedding = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        self.html_embedding = nn.Embedding(word_vocab_size, word_emb_dim, padding_idx=0)

        in_channels = char_emb_dim + word_emb_dim  # 16 + 16 = 32

        # Multi-kernel Conv1D block
        self.conv3 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)

        self.pool = nn.MaxPool1d(kernel_size=2)

        # After concatenating 3 conv outputs: channels = 32 * 3 = 96
        self.layernorm = nn.LayerNorm(96)

        # Global average pooling -> 96-dim
        self.fc1 = nn.Linear(96, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, url_ids, html_ids):
        """
        url_ids:  (B, url_len)
        html_ids: (B, html_len)
        """
        # Embedding lookup
        url_emb = self.url_embedding(url_ids)   # (B, L_u, 16)
        html_emb = self.html_embedding(html_ids)  # (B, L_h, 16)

        # Force both to html_len along sequence dimension
        if url_emb.size(1) < self.html_len:
            pad_len = self.html_len - url_emb.size(1)
            pad = url_emb.new_zeros(url_emb.size(0), pad_len, url_emb.size(2))
            url_emb = torch.cat([url_emb, pad], dim=1)
        else:
            url_emb = url_emb[:, :self.html_len, :]

        if html_emb.size(1) < self.html_len:
            pad_len = self.html_len - html_emb.size(1)
            pad = html_emb.new_zeros(html_emb.size(0), pad_len, html_emb.size(2))
            html_emb = torch.cat([html_emb, pad], dim=1)
        else:
            html_emb = html_emb[:, :self.html_len, :]

        # Concatenate along feature dimension → (B, L, 32)
        x = torch.cat([url_emb, html_emb], dim=2)
        x = x.permute(0, 2, 1)  # (B, C=32, L)

        # Multi-kernel conv
        x3 = F.gelu(self.conv3(x))
        x5 = F.gelu(self.conv5(x))
        x7 = F.gelu(self.conv7(x))

        # Concatenate conv outputs → (B, 96, L)
        x = torch.cat([x3, x5, x7], dim=1)

        # MaxPool1d → (B, 96, L/2)
        x = self.pool(x)

        # LayerNorm across channels
        x = x.permute(0, 2, 1)       # (B, L/2, 96)
        x = self.layernorm(x)
        x = x.mean(dim=1)            # Global average pooling → (B, 96)

        # MLP head
        x = F.gelu(self.fc1(x))
        x = self.dropout1(x)

        x = F.gelu(self.fc2(x))
        x = self.dropout2(x)

        x = self.fc3(x)              # (B, 1)
        return x.squeeze(-1)         # (B,)


# =========================
# 2. LOADING ARTIFACTS
# =========================

@st.cache_resource(show_spinner=True)
def load_artifacts():
    """
    Loads:
      - torch model_package from 'webphish_lite_complete.pth'
      - char_to_idx / word_to_idx from 'char_vocab.pkl' & 'word_vocab.pkl'
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocabularies
    with open("char_vocab.pkl", "rb") as f:
        char_vocab = pickle.load(f)
        char_to_idx = char_vocab["char_to_idx"]

    with open("word_vocab.pkl", "rb") as f:
        word_vocab = pickle.load(f)
        word_to_idx = word_vocab["word_to_idx"]

    # Load model package
    package = torch.load("webphish_lite_complete.pth", map_location=device)
    state_dict = package["model_state_dict"]

    char_vocab_size = package.get("char_vocab_size", len(char_to_idx))
    word_vocab_size = package.get("word_vocab_size", len(word_to_idx))
    url_len = package.get("url_len", 200)
    html_len = package.get("html_len", 500)
    char_emb_dim = package.get("char_emb_dim", 16)
    word_emb_dim = package.get("word_emb_dim", 16)

    model = WebPhishLite(
        char_vocab_size=char_vocab_size,
        word_vocab_size=word_vocab_size,
        char_emb_dim=char_emb_dim,
        word_emb_dim=word_emb_dim,
        url_len=url_len,
        html_len=html_len,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, char_to_idx, word_to_idx, device, url_len, html_len


# =========================
# 3. PREPROCESSING HELPERS
# =========================

def url_to_sequence(url: str, char_to_idx: dict, max_len: int = 200):
    seq = []
    url = url.lower()
    for ch in url[:max_len]:
        seq.append(char_to_idx.get(ch, char_to_idx["<UNK>"]))
    # Pad with 0 (PAD)
    if len(seq) < max_len:
        seq.extend([0] * (max_len - len(seq)))
    return seq


def html_to_sequence(text: str, word_to_idx: dict, max_len: int = 500):
    seq = []
    words = text.lower().split()[:max_len]
    for w in words:
        seq.append(word_to_idx.get(w, word_to_idx["<UNK>"]))
    if len(seq) < max_len:
        seq.extend([0] * (max_len - len(seq)))
    return seq


def fetch_and_clean_html(url: str) -> str:
    """
    Download HTML and strip scripts/styles, return visible text.
    """
    try:
        # Basic security: verify TLS and timeout
        resp = requests.get(url, timeout=8, verify=True)
        html = resp.text
    except Exception:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = " ".join(text.split())
    return text


def sha256_of_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def predict_single_url(
    url: str,
    model: nn.Module,
    char_to_idx: dict,
    word_to_idx: dict,
    device: torch.device,
    url_len: int,
    html_len: int,
):
    """
    Full prediction pipeline:
      URL → fetch HTML → preprocess → model → (prob, label)
    """
    html_text = fetch_and_clean_html(url)

    url_seq = url_to_sequence(url, char_to_idx, max_len=url_len)
    html_seq = html_to_sequence(html_text, word_to_idx, max_len=html_len)

    url_tensor = torch.tensor([url_seq], dtype=torch.long, device=device)
    html_tensor = torch.tensor([html_seq], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(url_tensor, html_tensor)
        prob = torch.sigmoid(logits).item()

    label = "Phishing" if prob >= 0.5 else "Legitimate"
    return prob, label, html_text


# =========================
# 4. STREAMLIT APP
# =========================

def main():
    st.set_page_config(
        page_title="WebPhish-Lite++",
        page_icon="🛡️",
        layout="centered",
    )

    st.title("🛡️ WebPhish-Lite++ – Phishing URL Detector")

    st.markdown(
        """
Enter a URL below. The app will:
1. Fetch the page HTML  
2. Convert the URL (chars) and HTML (words) into embeddings  
3. Run them through the WebPhish-Lite++ CNN  
4. Return a phishing probability and classification
"""
    )

    with st.spinner("Loading model and vocab..."):
        (
            model,
            char_to_idx,
            word_to_idx,
            device,
            url_len,
            html_len,
        ) = load_artifacts()

    url_input = st.text_input(
        "URL",
        placeholder="https://example.com/login",
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        check_button = st.button("Analyze URL")

    if check_button:
        if not url_input.strip():
            st.error("Please enter a URL.")
            return

        if not (url_input.startswith("http://") or url_input.startswith("https://")):
            st.info("No scheme detected, assuming https://")
            url_to_check = "https://" + url_input.strip()
        else:
            url_to_check = url_input.strip()

        with st.spinner("Running WebPhish-Lite++ inference..."):
            prob, label, html_text = predict_single_url(
                url_to_check,
                model,
                char_to_idx,
                word_to_idx,
                device,
                url_len,
                html_len,
            )

        st.subheader("Result")
        st.metric(
            label="Phishing probability",
            value=f"{prob*100:.2f}%",
            delta=None,
        )

        if label == "Phishing":
            st.error(f"Classification: {label}")
        else:
            st.success(f"Classification: {label}")

        with st.expander("Security metadata"):
            st.write(f"URL SHA-256: `{sha256_of_url(url_to_check)}`")
            st.write(f"HTML length (characters): {len(html_text)}")

        with st.expander("HTML text preview (sanitized)"):
            if html_text:
                st.text(html_text[:4000])
            else:
                st.write("No HTML content could be fetched or parsed.")

    st.markdown("---")
    st.caption(
        "Model: WebPhish-Lite++ (URL char + HTML word CNN). "
        "Artifacts expected in the same directory: "
        "`webphish_lite_complete.pth`, `char_vocab.pkl`, `word_vocab.pkl`."
    )


if __name__ == "__main__":
    main()
