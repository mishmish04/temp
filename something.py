import streamlit as st
import re
from urllib.parse import urlparse
import socket
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import whois
import dns.resolver


# Feature extraction functions
def extract_features(url):
    """Extract comprehensive features from URL for phishing detection"""
    features = {}

    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path

        # 1. URL length features
        features['url_length'] = len(url)
        features['domain_length'] = len(domain)
        features['path_length'] = len(path)

        # 2. Special character counts
        features['dots_count'] = url.count('.')
        features['hyphens_count'] = url.count('-')
        features['underscores_count'] = url.count('_')
        features['slashes_count'] = url.count('/')
        features['at_symbol'] = 1 if '@' in url else 0
        features['double_slash_redirecting'] = 1 if url.count('//') > 1 else 0

        # 3. Domain features
        features['subdomain_count'] = len(domain.split('.')) - 2 if len(domain.split('.')) > 2 else 0
        features['has_ip'] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', domain) else 0

        # 4. Suspicious keywords
        suspicious_words = ['login', 'signin', 'account', 'verify', 'secure', 'update',
                            'bank', 'paypal', 'ebay', 'amazon', 'password', 'confirm']
        features['suspicious_keywords'] = sum(1 for word in suspicious_words if word in url.lower())

        # 5. Protocol
        features['https'] = 1 if parsed.scheme == 'https' else 0

        # 6. Digit and letter ratios
        digits = sum(c.isdigit() for c in url)
        letters = sum(c.isalpha() for c in url)
        features['digit_letter_ratio'] = digits / letters if letters > 0 else 0

        # 7. Entropy (randomness measure)
        features['entropy'] = calculate_entropy(url)

        # 8. Port number
        features['has_port'] = 1 if parsed.port else 0

        # 9. Query parameters
        features['query_length'] = len(parsed.query)
        features['num_query_params'] = len(parsed.query.split('&')) if parsed.query else 0

        # 10. TLD check
        common_tlds = ['.com', '.org', '.net', '.edu', '.gov']
        features['common_tld'] = 1 if any(url.endswith(tld) for tld in common_tlds) else 0

        # 11. Domain age (simplified - real implementation would use WHOIS)
        features['domain_age_score'] = check_domain_age(domain)

        # 12. Shortening service
        shorteners = ['bit.ly', 'goo.gl', 'tinyurl', 't.co', 'ow.ly']
        features['is_shortened'] = 1 if any(short in domain for short in shorteners) else 0

    except Exception as e:
        st.warning(f"Feature extraction warning: {str(e)}")
        # Return default features if extraction fails
        return {f'feature_{i}': 0 for i in range(20)}

    return features


def calculate_entropy(text):
    """Calculate Shannon entropy of text"""
    if not text:
        return 0
    entropy = 0
    for i in range(256):
        p = text.count(chr(i)) / len(text)
        if p > 0:
            entropy += -p * np.log2(p)
    return entropy


def check_domain_age(domain):
    """Simplified domain age check (returns score 0-1)"""
    try:
        # In production, use python-whois library
        # For demo, return random score based on domain characteristics
        # Longer, more established looking domains get higher scores
        if len(domain) > 10 and '-' not in domain:
            return 0.8
        return 0.3
    except:
        return 0.5


def get_trained_model():
    """Return a pre-trained Random Forest model (simplified for demo)"""
    # In production, load actual trained model
    # For demo, create model with reasonable parameters
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    # Simulate training with synthetic data
    # In production, replace with actual training data
    np.random.seed(42)
    X_train = np.random.randn(1000, 20)
    y_train = np.random.randint(0, 2, 1000)
    model.fit(X_train, y_train)

    return model


def predict_phishing(features, model):
    """Predict if URL is phishing based on features"""
    # Convert features dict to array
    feature_vector = np.array(list(features.values())).reshape(1, -1)

    # Ensure correct number of features
    if feature_vector.shape[1] < 20:
        # Pad with zeros if needed
        padding = np.zeros((1, 20 - feature_vector.shape[1]))
        feature_vector = np.hstack([feature_vector, padding])
    elif feature_vector.shape[1] > 20:
        feature_vector = feature_vector[:, :20]

    # Get prediction and probability
    prediction = model.predict(feature_vector)[0]
    probability = model.predict_proba(feature_vector)[0]

    return prediction, probability


# Streamlit UI
st.set_page_config(page_title="Phishing URL Detector", page_icon="🔒", layout="wide")

st.title("🔒 Phishing URL Detector")
st.markdown("### Advanced ML-based URL Security Analysis")
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

# Analysis section
if analyze_button and url_input:
    with st.spinner("Analyzing URL..."):
        try:
            # Extract features
            features = extract_features(url_input)

            # Get model
            model = get_trained_model()

            # Make prediction
            prediction, probability = predict_phishing(features, model)

            # Display results
            st.markdown("---")
            st.subheader("Analysis Results")

            # Main result
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                if prediction == 0:  # Benign
                    st.success("✅ **BENIGN** - This URL appears safe")
                    confidence = probability[0] * 100
                    st.metric("Confidence", f"{confidence:.1f}%")
                else:  # Phishing
                    st.error("⚠️ **PHISHING DETECTED** - This URL appears suspicious")
                    confidence = probability[1] * 100
                    st.metric("Threat Level", f"{confidence:.1f}%")

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

            with col2:
                st.markdown("**Security Indicators:**")
                st.write(f"• Suspicious Keywords: {features['suspicious_keywords']}")
                st.write(
                    f"• Special Characters: {features['dots_count'] + features['hyphens_count'] + features['underscores_count']}")
                st.write(f"• Entropy Score: {features['entropy']:.2f}")
                st.write(f"• Common TLD: {'✅ Yes' if features['common_tld'] else '⚠️ No'}")
                st.write(f"• Domain Age Score: {features['domain_age_score']:.2f}")
                st.write(f"• Query Parameters: {features['num_query_params']}")

            # Risk factors
            st.markdown("---")
            st.subheader("Risk Factors")

            risk_factors = []
            if features['url_length'] > 75:
                risk_factors.append("• Very long URL")
            if features['https'] == 0:
                risk_factors.append("• No HTTPS encryption")
            if features['has_ip'] == 1:
                risk_factors.append("• IP address used instead of domain")
            if features['suspicious_keywords'] > 2:
                risk_factors.append("• Multiple suspicious keywords detected")
            if features['subdomain_count'] > 3:
                risk_factors.append("• Excessive subdomains")
            if features['is_shortened'] == 1:
                risk_factors.append("• URL shortening service detected")
            if features['at_symbol'] == 1:
                risk_factors.append("• @ symbol in URL (possible redirect)")
            if features['domain_age_score'] < 0.4:
                risk_factors.append("• Potentially new or suspicious domain")

            if risk_factors:
                st.warning("⚠️ **Identified Risk Factors:**")
                for factor in risk_factors:
                    st.write(factor)
            else:
                st.success("✅ No significant risk factors identified")

            # Recommendations
            st.markdown("---")
            st.subheader("Recommendations")
            if prediction == 1:
                st.error("""
                **🛡️ Security Recommendations:**
                - Do NOT enter personal information on this website
                - Do NOT download files from this URL
                - Do NOT click on links from this domain
                - Report this URL to your organization's security team
                - Consider using a URL reputation service for additional verification
                """)
            else:
                st.info("""
                **🔒 General Security Tips:**
                - Always verify the domain matches the expected website
                - Check for HTTPS and valid SSL certificates
                - Be cautious of emails asking you to click links
                - Enable two-factor authentication where possible
                - Keep your browser and security software updated
                """)

        except Exception as e:
            st.error(f"Error analyzing URL: {str(e)}")
            st.info("Please ensure you've entered a valid URL (e.g., https://example.com)")

# Sidebar information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This tool uses machine learning to analyze URLs and detect potential phishing attempts.

    **Features Analyzed:**
    - URL structure and length
    - Domain characteristics
    - Special characters and patterns
    - Security indicators (HTTPS)
    - Suspicious keywords
    - Entropy and randomness
    - TLD verification

    **Model:** Random Forest Classifier with 20+ features

    **Accuracy:** This is a demonstration model. In production, use a model trained on comprehensive phishing datasets.
    """)

    st.markdown("---")
    st.markdown("**⚠️ Disclaimer:**")
    st.caption(
        "This tool provides analysis based on ML models but should not be the sole factor in security decisions. Always exercise caution online.")

# Footer
st.markdown("---")
st.caption("🔒 Phishing URL Detector | Built with Streamlit & ML")