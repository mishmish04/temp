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
        st.warning(f"Feature extraction warning: {str(e)}")
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
    """Train and return a Random Forest classifier"""
    # Create synthetic training data based on phishing patterns
    np.random.seed(42)

    # Generate benign URLs features (lower risk scores)
    benign_samples = 500
    benign_features = np.random.randn(benign_samples, 20)
    benign_features[:, 0] = np.random.uniform(10, 50, benign_samples)  # shorter URLs
    benign_features[:, 1] = np.random.uniform(5, 20, benign_samples)  # normal domain length
    benign_features[:, 4] = np.random.choice([0, 1], benign_samples, p=[0.3, 0.7])  # mostly HTTPS
    benign_features[:, 5] = 0  # no @ symbol
    benign_features[:, 6] = np.random.uniform(0, 2, benign_samples)  # few suspicious keywords
    benign_features[:, 7] = np.random.uniform(0, 2, benign_samples)  # few subdomains

    # Generate phishing URLs features (higher risk scores)
    phishing_samples = 500
    phishing_features = np.random.randn(phishing_samples, 20)
    phishing_features[:, 0] = np.random.uniform(60, 150, phishing_samples)  # longer URLs
    phishing_features[:, 1] = np.random.uniform(15, 40, phishing_samples)  # longer domains
    phishing_features[:, 4] = np.random.choice([0, 1], phishing_samples, p=[0.6, 0.4])  # less HTTPS
    phishing_features[:, 5] = np.random.choice([0, 1], phishing_samples, p=[0.7, 0.3])  # sometimes @ symbol
    phishing_features[:, 6] = np.random.uniform(2, 5, phishing_samples)  # more suspicious keywords
    phishing_features[:, 7] = np.random.uniform(2, 6, phishing_samples)  # more subdomains

    # Combine data
    X_train = np.vstack([benign_features, phishing_features])
    y_train = np.hstack([np.zeros(benign_samples), np.ones(phishing_samples)])

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)

    return model


def predict_phishing(features, model):
    """Predict if URL is phishing using Random Forest"""
    if features is None:
        return None, None, None

    # Convert features to array
    feature_vector = np.array(list(features.values())).reshape(1, -1)

    # Ensure correct number of features (pad or trim)
    if feature_vector.shape[1] < 20:
        padding = np.zeros((1, 20 - feature_vector.shape[1]))
        feature_vector = np.hstack([feature_vector, padding])
    elif feature_vector.shape[1] > 20:
        feature_vector = feature_vector[:, :20]

    # Get prediction and probability
    prediction = model.predict(feature_vector)[0]
    probability = model.predict_proba(feature_vector)[0]

    # Get feature importance for this prediction
    feature_importance = model.feature_importances_

    return prediction, probability, feature_importance


# Streamlit UI
st.set_page_config(page_title="Phishing URL Detector", page_icon="🔒", layout="wide")

st.title("🔒 Phishing URL Detector")
st.markdown("### Advanced ML-based URL Security Analysis")
st.markdown("Built with Random Forest Classifier")
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
        st.code("http://www.amazon-account-verify.xyz")

# Analysis section
if analyze_button and url_input:
    with st.spinner("Analyzing URL with ML model..."):
        try:
            # Extract features
            features = extract_features(url_input)

            if features is None:
                st.error("Invalid URL format. Please enter a valid URL.")
            else:
                # Load model
                model = get_trained_model()

                # Make prediction
                prediction, probability, feature_importance = predict_phishing(features, model)

                # Display results
                st.markdown("---")
                st.subheader("🤖 ML Model Prediction Results")

                # Main result
                col1, col2, col3 = st.columns([1, 2, 1])

                with col2:
                    if prediction == 0:  # Benign
                        st.success("✅ **BENIGN** - This URL appears safe")
                        confidence = probability[0] * 100
                        st.metric("Confidence Score", f"{confidence:.1f}%")
                        st.progress(confidence / 100)
                    else:  # Phishing
                        st.error("⚠️ **PHISHING DETECTED** - This URL appears suspicious")
                        confidence = probability[1] * 100
                        st.metric("Threat Level", f"{confidence:.1f}%")
                        st.progress(confidence / 100)

                # Model confidence breakdown
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Benign Probability", f"{probability[0] * 100:.1f}%")
                with col2:
                    st.metric("Phishing Probability", f"{probability[1] * 100:.1f}%")

                # Detailed analysis
                st.markdown("---")
                st.subheader("Detailed Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**URL Characteristics:**")
                    st.write(f"• URL Length: {features['url_length']} characters")
                    st.write(f"• Domain Length: {features['domain_length']} characters")
                    st.write(f"• HTTPS Enabled: {'✅ Yes' if features['https'] else '❌ No'}")
                    st.write(f"• Subdomain Count: {features['subdomain_count']}")
                    st.write(f"• IP Address in URL: {'⚠️ Yes' if features['has_ip'] else '✅ No'}")
                    st.write(f"• URL Shortener: {'⚠️ Yes' if features['is_shortened'] else '✅ No'}")
                    st.write(f"• Has @ Symbol: {'⚠️ Yes' if features['at_symbol'] else '✅ No'}")

                with col2:
                    st.markdown("**Security Indicators:**")
                    st.write(f"• Suspicious Keywords: {features['suspicious_keywords']}")
                    st.write(f"• Dots in URL: {features['dots_count']}")
                    st.write(f"• Hyphens: {features['hyphens_count']}")
                    st.write(f"• Entropy Score: {features['entropy']:.2f}")
                    st.write(f"• Common TLD: {'✅ Yes' if features['common_tld'] else '⚠️ No'}")
                    st.write(f"• Query Parameters: {features['num_query_params']}")
                    st.write(f"• Encoded Characters: {'⚠️ Yes' if features['has_hex_chars'] else '✅ No'}")

                # Risk factors
                st.markdown("---")
                st.subheader("Risk Factors Identified by ML Model")

                risk_factors = []
                if features['url_length'] > 75:
                    risk_factors.append("• Very long URL (>75 characters)")
                if features['https'] == 0:
                    risk_factors.append("• No HTTPS encryption")
                if features['has_ip'] == 1:
                    risk_factors.append("• IP address used instead of domain name")
                if features['suspicious_keywords'] >= 2:
                    risk_factors.append(f"• Multiple suspicious keywords detected ({features['suspicious_keywords']})")
                if features['subdomain_count'] > 3:
                    risk_factors.append("• Excessive subdomains")
                if features['is_shortened'] == 1:
                    risk_factors.append("• URL shortening service detected")
                if features['at_symbol'] == 1:
                    risk_factors.append("• @ symbol in URL (possible redirect)")
                if features['common_tld'] == 0:
                    risk_factors.append("• Uncommon or suspicious TLD")
                if features['entropy'] > 4.5:
                    risk_factors.append("• High randomness in URL structure")
                if features['has_port'] == 1:
                    risk_factors.append("• Non-standard port specified")
                if features['digit_letter_ratio'] > 0.3:
                    risk_factors.append("• High proportion of digits in URL")
                if features['has_hex_chars'] == 1:
                    risk_factors.append("• URL-encoded characters present")

                if risk_factors:
                    st.warning(f"⚠️ **{len(risk_factors)} Risk Factor(s) Identified:**")
                    for factor in risk_factors:
                        st.write(factor)
                else:
                    st.success("✅ No significant risk factors identified")

                # Recommendations
                st.markdown("---")
                st.subheader("Security Recommendations")
                if prediction == 1:
                    st.error("""
                    **🛡️ CRITICAL - Do Not Proceed:**
                    - ⛔ Do NOT enter personal information on this website
                    - ⛔ Do NOT download files from this URL
                    - ⛔ Do NOT click on links from this domain
                    - 📧 Report this URL to your organization's security team
                    - 🔍 Verify the sender if you received this link via email
                    - 🚫 Consider blocking this domain in your browser
                    """)
                else:
                    st.info("""
                    **🔒 General Security Best Practices:**
                    - ✅ Always verify the domain matches the expected website
                    - 🔐 Check for HTTPS and valid SSL certificates (look for the padlock icon)
                    - 📧 Be cautious of emails asking you to click links urgently
                    - 🔑 Enable two-factor authentication where possible
                    - 🔄 Keep your browser and security software updated
                    - 🤔 When in doubt, navigate directly to the website instead of clicking links
                    """)

                # Technical details expander
                with st.expander("🔬 View Technical Details & Feature Importance"):
                    st.markdown("**Extracted Features:**")
                    st.json(features)

                    st.markdown("**Top 5 Most Important Features for This Model:**")
                    feature_names = list(features.keys())[:20]
                    top_features = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)[:5]
                    for i, (fname, importance) in enumerate(top_features, 1):
                        st.write(f"{i}. **{fname}**: {importance:.4f}")

        except Exception as e:
            st.error(f"Error analyzing URL: {str(e)}")
            st.info("Please ensure you've entered a valid URL (e.g., https://example.com)")

# Sidebar information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This tool uses **Random Forest Machine Learning** to analyze URLs and detect potential phishing attempts.

    **ML Model:**
    - Algorithm: Random Forest Classifier
    - Trees: 100
    - Features: 20+
    - Training: Synthetic phishing dataset

    **Features Analyzed:**
    - 📏 URL structure and length
    - 🌐 Domain characteristics
    - 🔤 Special characters and patterns
    - 🔒 Security indicators (HTTPS)
    - 🚨 Suspicious keywords
    - 🎲 Entropy and randomness
    - 🌍 TLD verification
    - 🔢 IP address detection
    - 🔗 URL shorteners

    **How it works:**
    1. Extracts 20+ features from URL
    2. Feeds features to trained Random Forest
    3. Model predicts probability of phishing
    4. Provides confidence scores and recommendations
    """)

    st.markdown("---")

    st.header("📊 Model Statistics")
    st.metric("ML Algorithm", "Random Forest")
    st.metric("Number of Trees", "100")
    st.metric("Features", "20+")
    st.metric("Processing Time", "< 1 sec")

    st.markdown("---")
    st.markdown("**⚠️ Disclaimer:**")
    st.caption(
        "This tool uses ML models trained on synthetic data for demonstration. For production use, train on real phishing datasets like PhishTank or OpenPhish.")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col2:
    st.caption("🔒 ML Phishing Detector | Powered by scikit-learn")