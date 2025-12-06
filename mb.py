import streamlit as st
import re
from urllib.parse import urlparse
import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def extract_features(url):
    """Extract comprehensive features from URL for phishing detection"""
    features = {}

    try:
        # Add scheme if missing
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url

        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path

        # URL length features
        features['url_length'] = len(url)
        features['domain_length'] = len(domain)
        features['path_length'] = len(path)

        # Special character counts
        features['dots_count'] = url.count('.')
        features['hyphens_count'] = url.count('-')
        features['underscores_count'] = url.count('_')
        features['slashes_count'] = url.count('/')
        features['at_symbol'] = 1 if '@' in url else 0
        features['double_slash_redirecting'] = 1 if url.count('//') > 1 else 0

        # Domain features
        features['subdomain_count'] = len(domain.split('.')) - 2 if len(domain.split('.')) > 2 else 0
        features['has_ip'] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', domain) else 0

        # Suspicious keywords
        suspicious_words = ['login', 'signin', 'account', 'verify', 'secure', 'update',
                            'bank', 'paypal', 'ebay', 'amazon', 'password', 'confirm',
                            'credential', 'suspended', 'locked', 'unusual', 'banking']
        features['suspicious_keywords'] = sum(1 for word in suspicious_words if word in url.lower())

        # Protocol
        features['https'] = 1 if parsed.scheme == 'https' else 0

        # Digit and letter ratios
        digits = sum(c.isdigit() for c in url)
        letters = sum(c.isalpha() for c in url)
        features['digit_letter_ratio'] = digits / letters if letters > 0 else 0

        # Entropy (randomness measure)
        features['entropy'] = calculate_entropy(url)

        # Port number
        features['has_port'] = 1 if parsed.port else 0

        # Query parameters
        features['query_length'] = len(parsed.query)
        features['num_query_params'] = len(parsed.query.split('&')) if parsed.query else 0

        # TLD check
        common_tlds = ['.com', '.org', '.net', '.edu', '.gov', '.uk', '.ca', '.au']
        features['common_tld'] = 1 if any(url.endswith(tld) for tld in common_tlds) else 0

        # Shortening service
        shorteners = ['bit.ly', 'goo.gl', 'tinyurl', 't.co', 'ow.ly', 'buff.ly', 'adf.ly']
        features['is_shortened'] = 1 if any(short in domain for short in shorteners) else 0

        # Suspicious patterns
        features['prefix_suffix'] = 1 if '-' in domain else 0
        features['has_hex_chars'] = 1 if re.search(r'%[0-9a-fA-F]{2}', url) else 0

    except Exception as e:
        st.error(f"Feature extraction error: {str(e)}")
        return None

    return features


def calculate_entropy(text):
    """Calculate Shannon entropy of text"""
    if not text:
        return 0
    entropy = 0
    char_count = {}
    for char in text:
        char_count[char] = char_count.get(char, 0) + 1

    for count in char_count.values():
        p = count / len(text)
        if p > 0:
            entropy += -p * math.log2(p)
    return entropy


@st.cache_resource
def get_trained_model():
    """Train and return a Random Forest classifier with realistic synthetic data"""
    np.random.seed(42)

    # Generate realistic benign URLs
    benign_samples = 800
    benign_features = []

    for i in range(benign_samples):
        # Legitimate sites: google.com, github.com, wikipedia.org, etc.
        url_len = np.random.normal(30, 10)
        domain_len = np.random.normal(12, 4)
        path_len = np.random.normal(15, 8)
        dots = np.random.randint(1, 3)
        hyphens = np.random.randint(0, 2)
        underscores = np.random.randint(0, 1)
        slashes = np.random.randint(1, 4)
        at_symbol = 0
        double_slash = 0
        subdomains = np.random.randint(0, 2)
        has_ip = 0
        suspicious_kw = 0  # Legitimate sites rarely have suspicious keywords
        https = 1 if np.random.random() > 0.2 else 0  # 80% HTTPS
        digit_ratio = np.random.uniform(0, 0.15)
        entropy = np.random.uniform(3.0, 4.0)
        has_port = 0
        query_len = np.random.normal(10, 15)
        num_params = np.random.randint(0, 3)
        common_tld = 1
        shortened = 0
        prefix_suffix = 0 if np.random.random() > 0.3 else 1
        hex_chars = 0 if np.random.random() > 0.2 else 1

        benign_features.append([
            url_len, domain_len, path_len, dots, hyphens, underscores,
            slashes, at_symbol, double_slash, subdomains, has_ip,
            suspicious_kw, https, digit_ratio, entropy, has_port,
            query_len, num_params, common_tld, shortened, prefix_suffix, hex_chars
        ])

    # Generate realistic phishing URLs
    phishing_samples = 800
    phishing_features = []

    for i in range(phishing_samples):
        # Phishing: paypal-verify.tk, amazon-account.xyz, etc.
        url_len = np.random.normal(70, 20)  # Much longer
        domain_len = np.random.normal(30, 10)  # Longer domains
        path_len = np.random.normal(20, 10)
        dots = np.random.randint(2, 6)  # More dots
        hyphens = np.random.randint(1, 5)  # More hyphens
        underscores = np.random.randint(0, 3)
        slashes = np.random.randint(2, 8)
        at_symbol = 1 if np.random.random() > 0.7 else 0  # 30% have @
        double_slash = 1 if np.random.random() > 0.8 else 0
        subdomains = np.random.randint(2, 6)  # Many subdomains
        has_ip = 1 if np.random.random() > 0.7 else 0  # 30% use IP
        suspicious_kw = np.random.randint(2, 6)  # Multiple suspicious keywords
        https = 1 if np.random.random() > 0.6 else 0  # Only 40% HTTPS
        digit_ratio = np.random.uniform(0.2, 0.5)  # More digits
        entropy = np.random.uniform(4.2, 5.5)  # Higher entropy
        has_port = 1 if np.random.random() > 0.85 else 0
        query_len = np.random.normal(25, 20)
        num_params = np.random.randint(1, 8)
        common_tld = 0 if np.random.random() > 0.6 else 1  # Often weird TLDs
        shortened = 1 if np.random.random() > 0.85 else 0
        prefix_suffix = 1 if np.random.random() > 0.3 else 0  # Often have hyphens
        hex_chars = 1 if np.random.random() > 0.5 else 0

        phishing_features.append([
            url_len, domain_len, path_len, dots, hyphens, underscores,
            slashes, at_symbol, double_slash, subdomains, has_ip,
            suspicious_kw, https, digit_ratio, entropy, has_port,
            query_len, num_params, common_tld, shortened, prefix_suffix, hex_chars
        ])

    # Combine data
    X_train = np.vstack([
        np.array(benign_features),
        np.array(phishing_features)
    ])
    y_train = np.hstack([
        np.zeros(benign_samples),
        np.ones(phishing_samples)
    ])

    # Shuffle
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train powerful Random Forest
    model = RandomForestClassifier(
        n_estimators=200,  # More trees
        max_depth=25,  # Deeper trees
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        bootstrap=True,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    return model, scaler


def predict_phishing(features, model, scaler):
    """Pure ML prediction - no hardcoded rules"""
    if features is None:
        return None, None, None

    # Convert to array with all 22 features
    feature_list = [
        features['url_length'],
        features['domain_length'],
        features['path_length'],
        features['dots_count'],
        features['hyphens_count'],
        features['underscores_count'],
        features['slashes_count'],
        features['at_symbol'],
        features['double_slash_redirecting'],
        features['subdomain_count'],
        features['has_ip'],
        features['suspicious_keywords'],
        features['https'],
        features['digit_letter_ratio'],
        features['entropy'],
        features['has_port'],
        features['query_length'],
        features['num_query_params'],
        features['common_tld'],
        features['is_shortened'],
        features['prefix_suffix'],
        features['has_hex_chars']
    ]

    feature_vector = np.array(feature_list).reshape(1, -1)

    # Scale features
    feature_vector_scaled = scaler.transform(feature_vector)

    # Get PURE ML prediction - NO RULES
    prediction = model.predict(feature_vector_scaled)[0]
    probability = model.predict_proba(feature_vector_scaled)[0]
    feature_importance = model.feature_importances_

    return int(prediction), probability, feature_importance


# Streamlit UI
st.set_page_config(page_title="Phishing URL Detector", page_icon="🔒", layout="wide")

st.title("🔒 Phishing URL Detector")
st.markdown("### Pure Machine Learning Detection System")
st.markdown("**No Hardcoded Rules - 100% ML-Driven Predictions**")
st.markdown("---")

# Input section
col1, col2 = st.columns([3, 1])

with col1:
    url_input = st.text_input(
        "Enter URL to analyze:",
        placeholder="https://example.com",
        help="Enter the full URL including http:// or https://"
    )

with col2:
    st.write("")
    st.write("")
    analyze_button = st.button("🔍 Analyze URL", type="primary", use_container_width=True)

# Example URLs
with st.expander("📋 Try Example URLs"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Legitimate URLs:**")
        st.code("https://www.google.com")
        st.code("https://www.github.com")
        st.code("https://www.wikipedia.org")
    with col2:
        st.markdown("**Suspicious URLs:**")
        st.code("http://paypal-secure-login.tk/verify")
        st.code("https://192.168.1.1/banking")
        st.code("http://amazon-account-verify.xyz")

# Analysis
if analyze_button and url_input:
    with st.spinner("🤖 Running ML analysis..."):
        try:
            features = extract_features(url_input)

            if features is None:
                st.error("❌ Invalid URL format.")
            else:
                model, scaler = get_trained_model()
                prediction, probability, feature_importance = predict_phishing(features, model, scaler)

                st.markdown("---")
                st.subheader("🤖 Pure ML Prediction (No Hardcoded Rules)")

                col1, col2, col3 = st.columns([1, 2, 1])

                with col2:
                    if prediction == 0:
                        st.success("✅ **BENIGN** - This URL appears safe")
                        confidence = probability[0] * 100
                        st.metric("ML Confidence", f"{confidence:.2f}%")
                        st.progress(confidence / 100)
                    else:
                        st.error("⚠️ **PHISHING DETECTED** - High threat probability")
                        confidence = probability[1] * 100
                        st.metric("ML Threat Score", f"{confidence:.2f}%")
                        st.progress(confidence / 100)

                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🟢 Benign Probability", f"{probability[0] * 100:.2f}%",
                              help="ML model's confidence this is legitimate")
                with col2:
                    st.metric("🔴 Phishing Probability", f"{probability[1] * 100:.2f}%",
                              help="ML model's confidence this is phishing")

                st.markdown("---")
                st.subheader("Detailed Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**URL Characteristics:**")
                    st.write(f"• URL Length: {features['url_length']} chars")
                    st.write(f"• Domain Length: {features['domain_length']} chars")
                    st.write(f"• HTTPS: {'✅ Yes' if features['https'] else '❌ No'}")
                    st.write(f"• Subdomains: {features['subdomain_count']}")
                    st.write(f"• IP Address: {'⚠️ Yes' if features['has_ip'] else '✅ No'}")
                    st.write(f"• Shortener: {'⚠️ Yes' if features['is_shortened'] else '✅ No'}")
                    st.write(f"• @ Symbol: {'⚠️ Yes' if features['at_symbol'] else '✅ No'}")

                with col2:
                    st.markdown("**Security Indicators:**")
                    st.write(f"• Suspicious Keywords: {features['suspicious_keywords']}")
                    st.write(f"• Dots: {features['dots_count']}")
                    st.write(f"• Hyphens: {features['hyphens_count']}")
                    st.write(f"• Entropy: {features['entropy']:.2f}")
                    st.write(f"• Common TLD: {'✅ Yes' if features['common_tld'] else '⚠️ No'}")
                    st.write(f"• Query Params: {features['num_query_params']}")
                    st.write(f"• URL Encoding: {'⚠️ Yes' if features['has_hex_chars'] else '✅ No'}")

                st.markdown("---")
                st.subheader("ML Model Insights")

                # Show top contributing features
                feature_names = [
                    'url_length', 'domain_length', 'path_length', 'dots_count',
                    'hyphens_count', 'underscores_count', 'slashes_count', 'at_symbol',
                    'double_slash', 'subdomain_count', 'has_ip', 'suspicious_keywords',
                    'https', 'digit_ratio', 'entropy', 'has_port', 'query_length',
                    'num_params', 'common_tld', 'is_shortened', 'prefix_suffix', 'has_hex_chars'
                ]

                top_features = sorted(zip(feature_names, feature_importance),
                                      key=lambda x: x[1], reverse=True)[:5]

                st.markdown("**Top 5 Most Important Features (Globally):**")
                for i, (fname, importance) in enumerate(top_features, 1):
                    st.write(f"{i}. **{fname.replace('_', ' ').title()}**: {importance:.4f}")

                st.markdown("---")
                st.subheader("Security Recommendations")

                if prediction == 1:
                    st.error("""
                    **🛡️ HIGH RISK - Avoid This URL:**
                    - ⛔ Do NOT enter credentials or personal data
                    - ⛔ Do NOT download any files
                    - ⛔ Do NOT proceed to this website
                    - 📧 Report suspicious emails containing this link
                    - 🔍 Verify the sender through official channels
                    """)
                else:
                    st.info("""
                    **🔒 Security Best Practices:**
                    - ✅ Always verify domains match expected websites
                    - 🔐 Look for HTTPS and valid certificates
                    - 📧 Be skeptical of urgent or threatening messages
                    - 🔑 Use password managers and 2FA
                    - 🔄 Keep browsers and security software updated
                    """)

                with st.expander("🔬 Technical Details & All Features"):
                    st.json(features)
                    st.markdown("**All Feature Importances:**")
                    for fname, importance in zip(feature_names, feature_importance):
                        st.write(f"• {fname}: {importance:.6f}")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Ensure URL is properly formatted with http:// or https://")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About This System")
    st.markdown("""
    **Pure Machine Learning Detection**

    This system uses ONLY machine learning - 
    zero hardcoded rules or thresholds.

    **Model Details:**
    - Algorithm: Random Forest
    - Trees: 200
    - Features: 22
    - Training: 1,600 samples
    - Feature Scaling: StandardScaler

    **How It Works:**
    1. Extracts 22 features from URL
    2. Scales features using StandardScaler
    3. Random Forest predicts probability
    4. Returns pure ML confidence scores

    **No Hardcoding:**
    - No fixed threat percentages
    - No manual rule boosting
    - Pure data-driven predictions
    """)

    st.markdown("---")
    st.header("📊 Model Stats")
    st.metric("Algorithm", "Random Forest")
    st.metric("Trees", "200")
    st.metric("Features", "22")
    st.metric("Training Samples", "1,600")

    st.markdown("---")
    st.caption("⚠️ Demo model. For production, train on datasets like PhishTank.")

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col2:
    st.caption("🔒 Pure ML Detection | No Hardcoded Rules")