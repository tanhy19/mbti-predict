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
    return bool(re.search(r'[a-zA-Z]{3,}', text)) and len(text.strip().split()) >= 5

# ------------------ Load Resources ------------------ #
vectorizer = safe_load("vectorizer.pkl")

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

mbti_descriptions = {
    "I": "Introverted: energized by solitude, introspective.",
    "E": "Extraverted: energized by social interaction, outgoing.",
    "N": "Intuitive: focuses on ideas and concepts.",
    "S": "Sensing: focuses on facts and details.",
    "F": "Feeling: makes decisions based on emotion and values.",
    "T": "Thinking: makes decisions based on logic and reason.",
    "P": "Perceiving: spontaneous and flexible.",
    "J": "Judging: organized and prefers structure."
}

# ------------------ UI Layout ------------------ #
st.set_page_config(page_title="MBTI Predictor", page_icon="🔮", layout="centered")

st.title("🔮 MBTI Personality Predictor")
st.markdown("Enter a **meaningful paragraph or sentence** to predict your MBTI personality type using machine learning.")

# Accuracy display
with st.expander("ℹ️ Model Accuracy Info"):
    for model_name, acc in model_accuracies.items():
        st.write(f"**{model_name}**: {acc*100:.1f}% accuracy")

# Inputs
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
        low_conf_warnings = []
        missing_confidence_dims = []

        for dim in ["I/E", "N/S", "F/T", "P/J"]:
            model = models[model_choice][dim]
            if model:
                raw_pred = model.predict(X_input)[0]
                letter = label_map[dim][raw_pred]
                preds.append(letter)

                # Check confidence support
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_input)[0]
                    probs_out[dim] = {
                        label_map[dim][0]: f"{proba[0]*100:.1f}%",
                        label_map[dim][1]: f"{proba[1]*100:.1f}%"
                    }

                    if max(proba) < 0.6:
                        low_conf_warnings.append(f"🔍 Low confidence in **{dim}** — try providing more text.")
                else:
                    missing_confidence_dims.append(dim)

        if preds:
            mbti = "".join(preds)
            acc = model_accuracies.get(model_choice, None)

            col1, col2 = st.columns(2)
            with col1:
                st.success(f"🎯 **Predicted MBTI type: `{mbti}`**")
            with col2:
                st.metric("🔧 Model Accuracy", f"{acc * 100:.1f}%", label_visibility="visible")

            # Show MBTI breakdown
            st.subheader("🧩 Personality Breakdown")
            col1, col2, col3, col4 = st.columns(4)
            for i, col in enumerate([col1, col2, col3, col4]):
                letter = preds[i]
                col.metric(f"{list(label_map.keys())[i]}", letter, mbti_descriptions[letter])

            # Tabs
            tab1, tab2 = st.tabs(["📊 Confidence", "🔎 Details"])
            with tab1:
                if probs_out:
                    for dim, scores in probs_out.items():
                        st.markdown(f"**{dim}** — {list(scores.items())[0][0]}: {list(scores.items())[0][1]}, "
                                    f"{list(scores.items())[1][0]}: {list(scores.items())[1][1]}")
                if missing_confidence_dims:
                    st.info("ℹ️ This model does not provide confidence scores for:")
                    st.markdown(", ".join(missing_confidence_dims))

            with tab2:
                st.markdown("**Full MBTI Type Description:**")
                for letter in mbti:
                    st.markdown(f"- **{letter}**: {mbti_descriptions[letter]}")

            if low_conf_warnings:
                st.warning("⚠️ Some predictions have low confidence:")
                for msg in low_conf_warnings:
                    st.write(msg)

# ------------------ Sidebar ------------------ #
with st.sidebar:
    st.header("🔍 About This App")
    st.markdown("""
This tool uses ML models to predict your **MBTI personality** based on text.

🧬 **4 Dimensions**:
- **I/E**: Introversion / Extraversion  
- **N/S**: Intuition / Sensing  
- **F/T**: Feeling / Thinking  
- **P/J**: Perceiving / Judging  

📈 **Model Accuracies**:
""")
    for name, acc in model_accuracies.items():
        st.write(f"- {name}: {acc*100:.1f}%")

    best_model = max(model_accuracies, key=model_accuracies.get)
    st.markdown(f"✅ **Most Accurate Model:** `{best_model}` ({model_accuracies[best_model]*100:.1f}%)")
