import streamlit as st
import joblib
import os
import numpy as np
import re

# ------------------ Helper Functions ------------------ #
def safe_load(path):
    if os.path.exists(path):
        return joblib.load(path)
    else:
        st.error(f"❌ Missing file: {path}")
        return None

def is_valid_input(text):
    # Checks for minimum 5 words and at least one alphabet character in a sentence
    return bool(re.search(r'[a-zA-Z]{3,}', text)) and len(text.strip().split()) >= 5

# ------------------ Load Resources ------------------ #

# Load vectorizer
vectorizer = safe_load("vectorizer.pkl")

# Define model files
models = {
    "Naive Bayes": {
        "I/E": safe_load("naivebayes_ie.pkl"),
        "N/S": safe_load("naivebayes_ns.pkl"),
        "F/T": safe_load("naivebayes_ft.pkl"),
        "P/J": safe_load("naivebayes_pj.pkl"),
    },
    "SVM": {
        "I/E": safe_load("svm_ie.pkl"),
        "N/S": safe_load("svm_ns.pkl"),
        "F/T": safe_load("svm_ft.pkl"),
        "P/J": safe_load("svm_pj.pkl"),
    },
    "Random Forest": {
        "I/E": safe_load("randomforest_ie.pkl"),
        "N/S": safe_load("randomforest_ns.pkl"),
        "F/T": safe_load("randomforest_ft.pkl"),
        "P/J": safe_load("randomforest_pj.pkl"),
    },
}

# Static accuracy values (example — update with your real model scores)
model_accuracies = {
    "Naive Bayes": 0.71,
    "SVM": 0.82,
    "Random Forest": 0.77,
}

label_map = {
    "I/E": {0: "I", 1: "E"},
    "N/S": {0: "N", 1: "S"},
    "F/T": {0: "F", 1: "T"},
    "P/J": {0: "P", 1: "J"}
}

# ------------------ UI ------------------ #
st.set_page_config(page_title="MBTI Predictor", page_icon="🔮", layout="centered")

st.title("🔮 MBTI Personality Predictor")
st.markdown("Enter a meaningful sentence or paragraph, and choose a model to predict your MBTI personality type.")

with st.expander("ℹ️ Model Accuracy Info"):
    for model_name, acc in model_accuracies.items():
        st.write(f"**{model_name}**: {acc*100:.1f}% accuracy")

# Input and selection
user_input = st.text_area("📝 Your Text:", height=200, placeholder="e.g. I enjoy spending time alone reflecting on ideas and possibilities.")
model_choice = st.selectbox("🧠 Choose a model:", list(models.keys()), index=1)

# ------------------ Prediction ------------------ #
if st.button("Predict"):
    if not is_valid_input(user_input):
        st.warning("⚠️ Please enter a complete sentence or paragraph with at least 5 words.")
    elif vectorizer is None:
        st.error("❌ Vectorizer file missing.")
    else:
        st.info("🔄 Analyzing input and predicting MBTI...")

        X_input = vectorizer.transform([user_input])
        preds = []
        probs_out = {}

        for dim in ["I/E", "N/S", "F/T", "P/J"]:
            model = models[model_choice][dim]
            if model:
                raw_pred = model.predict(X_input)[0]
                letter = label_map[dim][raw_pred]
                preds.append(letter)

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_input)[0]
                    probs_out[dim] = {
                        label_map[dim][0]: f"{proba[0]*100:.1f}%",
                        label_map[dim][1]: f"{proba[1]*100:.1f}%"
                    }

        if preds:
            mbti = "".join(preds)
            st.success(f"🎯 **Predicted MBTI type: `{mbti}`**")

            if probs_out:
                st.subheader("📊 Prediction Confidence by Dimension")
                for dim, scores in probs_out.items():
                    st.markdown(f"**{dim}** — {list(scores.items())[0][0]}: {list(scores.items())[0][1]}, "
                                f"{list(scores.items())[1][0]}: {list(scores.items())[1][1]}")

# ------------------ Sidebar ------------------ #
with st.sidebar:
    st.header("🔍 About This App")
    st.markdown("""
    This tool uses machine learning models (Naive Bayes, SVM, Random Forest) to predict your **MBTI personality type** from your writing.
    
    Models predict the four personality dimensions:
    - **I/E**: Introversion / Extraversion  
    - **N/S**: Intuition / Sensing  
    - **F/T**: Feeling / Thinking  
    - **P/J**: Perceiving / Judging  
    """)

    best_model = max(model_accuracies, key=model_accuracies.get)
    st.markdown(f"📈 **Most Accurate Model:** `{best_model}` ({model_accuracies[best_model]*100:.1f}%)")
