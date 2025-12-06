import streamlit as st
import torch
import torch.nn as nn
import pickle
import requests
from bs4 import BeautifulSoup
import ssl
import socket
from urllib.parse import urlparse
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="WebPhish-Lite++ Detector",
    page_icon="🔒",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        margin: 1rem 0;
    }
    .safe-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


# Model definition - MATCHES YOUR ACTUAL TRAINED MODEL
class WebPhishLitePlusPlus(nn.Module):
    def __init__(self, char_vocab_size, word_vocab_size, embedding_dim=16):
        super().__init__()

        # Embeddings (16-dim each)
        self.url_embedding = nn.Embedding(char_vocab_size, embedding_dim)
        self.html_embedding = nn.Embedding(word_vocab_size, embedding_dim)

        # Shared multi-kernel CNN (processes concatenated embeddings = 32 channels)
        # After concatenating 16+16, we get 32 input channels
        self.conv3 = nn.Conv1d(embedding_dim * 2, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embedding_dim * 2, 32, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(embedding_dim * 2, 32, kernel_size=7, padding=3)

        self.gelu = nn.GELU()
        self.maxpool = nn.MaxPool1d(2)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(96)  # 32*3 filters from multi-kernel

        # Fully connected layers
        self.fc1 = nn.Linear(96, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, url_seq, html_seq):
        # Get embeddings
        url_emb = self.url_embedding(url_seq)  # (batch, url_len, 16)
        html_emb = self.html_embedding(html_seq)  # (batch, html_len, 16)

        # Concatenate URL and HTML sequences
        combined_emb = torch.cat([url_emb, html_emb], dim=1)  # (batch, url_len+html_len, 16)

        # Transpose for Conv1d (needs channels first)
        combined_emb = combined_emb.transpose(1, 2)  # (batch, 16, seq_len)

        # Multi-kernel CNN
        conv3_out = self.gelu(self.conv3(combined_emb))
        conv5_out = self.gelu(self.conv5(combined_emb))
        conv7_out = self.gelu(self.conv7(combined_emb))

        # Concatenate all kernel outputs
        conv_concat = torch.cat([conv3_out, conv5_out, conv7_out], dim=1)  # (batch, 96, seq_len)

        # Max pooling
        pooled = self.maxpool(conv_concat)  # (batch, 96, seq_len/2)

        # Global average pooling
        pooled = torch.mean(pooled, dim=2)  # (batch, 96)

        # Layer norm
        normalized = self.layer_norm(pooled)

        # Fully connected layers
        x = self.gelu(self.fc1(normalized))
        x = self.dropout(x)
        x = self.gelu(self.fc2(x))
        x = self.dropout(x)
        logit = self.fc3(x)

        return logit


# Load model and vocabularies
@st.cache_resource
def load_artifacts():
    import os

    # Debug: Show available files
    st.sidebar.write("📁 Available files:", os.listdir('.'))

    try:
        # Check if vocab files exist
        if not os.path.exists('char_vocab.pkl'):
            raise FileNotFoundError("char_vocab.pkl not found")
        if not os.path.exists('word_vocab.pkl'):
            raise FileNotFoundError("word_vocab.pkl not found")

        # Load vocabularies
        with open('char_vocab.pkl', 'rb') as f:
            char_vocab_data = pickle.load(f)
        with open('word_vocab.pkl', 'rb') as f:
            word_vocab_data = pickle.load(f)

        char_to_idx = char_vocab_data['char_to_idx']
        word_to_idx = word_vocab_data['word_to_idx']

        st.sidebar.write(f"✓ Vocabularies loaded")
        st.sidebar.write(f"  Char vocab size: {len(char_to_idx)}")
        st.sidebar.write(f"  Word vocab size: {len(word_to_idx)}")

        # Initialize model with correct dimensions
        model = WebPhishLitePlusPlus(len(char_to_idx), len(word_to_idx), embedding_dim=16)

        # Try loading different model files
        model_files = ['best_model.pth', 'webphish_lite_complete.pth']
        loaded_file = None
        last_error = None

        for model_file in model_files:
            if not os.path.exists(model_file):
                st.sidebar.write(f"⚠️ {model_file} not found")
                continue

            try:
                st.sidebar.write(f"Trying to load {model_file}...")
                checkpoint = torch.load(model_file, map_location=torch.device('cpu'))

                # Handle webphish_lite_complete.pth format (nested dict)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    st.sidebar.write(f"  Format: nested checkpoint")
                    state_dict = checkpoint['model_state_dict']
                else:
                    # Handle best_model.pth format (direct state_dict)
                    st.sidebar.write(f"  Format: direct state_dict")
                    state_dict = checkpoint

                model.load_state_dict(state_dict)
                model.eval()
                loaded_file = model_file
                st.sidebar.write(f"✓ Successfully loaded {model_file}")
                break
            except Exception as e:
                last_error = str(e)
                st.sidebar.write(f"❌ Failed to load {model_file}: {str(e)[:150]}")
                continue

        if loaded_file is None:
            error_msg = f"Could not load any model file. Last error: {last_error}"
            raise Exception(error_msg)

        return model, char_to_idx, word_to_idx, loaded_file

    except Exception as e:
        st.sidebar.error(f"Error details: {str(e)}")
        raise e


# Preprocessing functions
def url_to_sequence(url, char_to_idx, max_len=200):
    url = url.lower()
    sequence = []
    for char in url[:max_len]:
        sequence.append(char_to_idx.get(char, char_to_idx.get('<UNK>', 1)))
    sequence.extend([0] * (max_len - len(sequence)))
    return sequence


def fetch_html(url, timeout=10):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=timeout, verify=False)
        return response.text if response.status_code == 200 else ""
    except:
        return ""


def extract_text_from_html(html):
    try:
        soup = BeautifulSoup(html, 'html.parser')
        for script in soup(['script', 'style']):
            script.decompose()
        text = soup.get_text(separator=' ')
        text = ' '.join(text.split())
        return text.lower()
    except:
        return ""


def html_to_sequence(html, word_to_idx, max_len=500):
    text = extract_text_from_html(html)
    words = text.split()
    sequence = []
    for word in words[:max_len]:
        sequence.append(word_to_idx.get(word, word_to_idx.get('<UNK>', 1)))
    sequence.extend([0] * (max_len - len(sequence)))
    return sequence


def extract_ssl_features(url):
    """Extract SSL features for display purposes (not used in model)"""
    features = {
        'has_https': 0,
        'has_valid_ssl': 0,
        'cert_trusted_ca': 0,
        'cert_age_days': -1
    }

    try:
        parsed = urlparse(url)
        features['has_https'] = 1 if parsed.scheme == 'https' else 0

        if parsed.scheme == 'https':
            hostname = parsed.netloc
            context = ssl.create_default_context()

            with socket.create_connection((hostname, 443), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    features['has_valid_ssl'] = 1

                    issuer = dict(x[0] for x in cert['issuer'])
                    trusted_cas = ['DigiCert', "Let's Encrypt", 'GlobalSign', 'Comodo', 'GoDaddy', 'Sectigo']
                    features['cert_trusted_ca'] = 1 if any(
                        ca in issuer.get('organizationName', '') for ca in trusted_cas) else 0

                    not_before = datetime.strptime(cert['notBefore'], '%b %d %H:%M:%S %Y %Z')
                    cert_age = (datetime.now() - not_before).days
                    features['cert_age_days'] = cert_age
    except:
        pass

    return features


def predict_url(url, model, char_to_idx, word_to_idx):
    # Preprocess URL
    url_seq = url_to_sequence(url, char_to_idx)

    # Fetch and preprocess HTML
    html = fetch_html(url)
    html_seq = html_to_sequence(html, word_to_idx)

    # Extract crypto features (for display only)
    crypto_features = extract_ssl_features(url)

    # Convert to tensors
    url_tensor = torch.tensor([url_seq], dtype=torch.long)
    html_tensor = torch.tensor([html_seq], dtype=torch.long)

    # Predict (model only takes URL and HTML)
    with torch.no_grad():
        logit = model(url_tensor, html_tensor)
        probability = torch.sigmoid(logit).item()

    return probability, crypto_features


# Main app
def main():
    st.markdown('<h1 class="main-header">🔒 WebPhish-Lite++ Detector</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Phishing Website Detection")
    st.markdown("Enter a URL to check if it's a phishing website or legitimate.")

    # Load model
    try:
        model, char_to_idx, word_to_idx, loaded_file = load_artifacts()
        st.success(f"✓ Model loaded successfully: {loaded_file}")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info(
            "Make sure 'best_model.pth' (or 'webphish_lite_complete.pth'), 'char_vocab.pkl', and 'word_vocab.pkl' are in the repository.")
        return

    # Input
    url_input = st.text_input("Enter URL:", placeholder="https://example.com")

    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_button = st.button("🔍 Analyze", type="primary")

    if analyze_button and url_input:
        with st.spinner("Analyzing URL... This may take a few seconds."):
            try:
                # Predict
                phishing_prob, crypto_features = predict_url(url_input, model, char_to_idx, word_to_idx)

                # Display result
                st.markdown("---")

                if phishing_prob > 0.5:
                    st.markdown(f"""
                        <div class="warning-box">
                            <h2>⚠️ WARNING: Potential Phishing Site</h2>
                            <p><strong>Phishing Probability:</strong> {phishing_prob * 100:.1f}%</p>
                            <p>This URL shows characteristics of a phishing website. Do not enter sensitive information!</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="safe-box">
                            <h2>✅ Likely Safe</h2>
                            <p><strong>Legitimate Probability:</strong> {(1 - phishing_prob) * 100:.1f}%</p>
                            <p>This URL appears to be legitimate, but always exercise caution online.</p>
                        </div>
                    """, unsafe_allow_html=True)

                # Security details
                st.markdown("### 🔐 Security Information")
                col1, col2, col3 = st.columns(3)

                with col1:
                    if crypto_features['has_https']:
                        st.metric("HTTPS", "✅ Enabled")
                    else:
                        st.metric("HTTPS", "❌ Not Enabled")

                with col2:
                    if crypto_features['has_valid_ssl']:
                        st.metric("SSL Certificate", "✅ Valid")
                    else:
                        st.metric("SSL Certificate", "❌ Invalid/Missing")

                with col3:
                    if crypto_features['cert_trusted_ca']:
                        st.metric("Certificate Authority", "✅ Trusted")
                    else:
                        st.metric("Certificate Authority", "❌ Untrusted")

                if crypto_features['cert_age_days'] >= 0:
                    st.info(
                        f"📅 Certificate Age: {crypto_features['cert_age_days']} days ({crypto_features['cert_age_days'] / 365:.1f} years)")

            except Exception as e:
                st.error(f"❌ Error analyzing URL: {str(e)}")
                st.info("Make sure the URL is accessible and properly formatted (include http:// or https://).")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem 0;">
            <p><strong>WebPhish-Lite++</strong> uses deep learning to detect phishing websites based on URL patterns and HTML content.</p>
            <p style="font-size: 0.9rem;">⚠️ This is a detection tool. Always verify URLs before entering sensitive information.</p>
            <p style="font-size: 0.8rem; margin-top: 1rem;">Note: SSL/Certificate information is shown for reference but not used in the prediction model.</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()