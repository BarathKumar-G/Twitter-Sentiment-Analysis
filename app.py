import streamlit as st
import os

from src.inference import predict_sentiment
from src.preprocessing import clean_text

st.set_page_config(page_title="Twitter Sentiment Analysis", layout="wide")

st.title("Twitter Sentiment Analysis Dashboard")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Prediction", "EDA & Insights", "Preprocessing Demo"])

with tab1:
    st.header("Sentiment Prediction")
    user_input = st.text_area("Enter a tweet or statement to analyze:")
    
    if st.button("Predict"):
        if not user_input.strip():
            st.warning("Please enter some text to predict.")
        else:
            try:
                result = predict_sentiment(user_input)
                sentiment = result["sentiment"]
                prob = result["probability"]
                
                if sentiment == "Positive":
                    st.success(f"**Sentiment:** {sentiment}")
                else:
                    st.error(f"**Sentiment:** {sentiment}")
                    
                st.info(f"**Probability (Positive):** {prob:.4f}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

with tab2:
    st.header("EDA & Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Summary")
        st.text("Best Model       : Logistic Regression + TF-IDF")
        st.text("Accuracy Approx  : ~0.8265")
        st.text("ROC-AUC Approx   : ~0.8647")
        
    st.markdown("---")
    st.subheader("Visualizations")
    
    OUTPUTS_DIR = "outputs"
    
    files = [
        ("Class Distribution", os.path.join(OUTPUTS_DIR, "class_distribution.png")),
        ("Tweet Length Distribution", os.path.join(OUTPUTS_DIR, "tweet_length_distribution.png")),
        ("Top Words Frequency", os.path.join(OUTPUTS_DIR, "top_words_visualization.png")),
    ]
    
    plot_cols = st.columns(3)
    for index, (title, img_path) in enumerate(files):
        with plot_cols[index]:
            if os.path.exists(img_path):
                st.image(img_path, caption=title, use_container_width=True)
            else:
                st.warning(f"Plot not found: {img_path}")
                
    st.markdown("---")
    st.subheader("Top Correlated Words")
    
    words_file = os.path.join(OUTPUTS_DIR, "top_words.txt")
    if os.path.exists(words_file):
        with open(words_file, "r", encoding="utf-8") as f:
            content = f.read()
            
        parts = content.split("Top 20 Negative Words:")
        
        pos_part = parts[0].replace("Top 20 Positive Words:\n", "").strip()
        neg_part = parts[1].strip() if len(parts) > 1 else ""
        
        word_col1, word_col2 = st.columns(2)
        with word_col1:
            st.markdown("### Positive Indicators")
            st.text(pos_part)
        with word_col2:
            st.markdown("### Negative Indicators")
            st.text(neg_part)
    else:
        st.warning(f"Top words file not found at {words_file}")

with tab3:
    st.header("Preprocessing Demo")
    demo_input = st.text_area("Enter text to see how it is cleaned and lemmatized:", key="demo_input")
    
    if demo_input:
        try:
            cleaned_text = clean_text(demo_input)
            st.markdown("**Original Text:**")
            st.info(demo_input)
            st.markdown("**Cleaned & Lemmatized Text:**")
            st.success(cleaned_text)
        except Exception as e:
            st.error(f"Preprocessing encountered an error: {e}")
