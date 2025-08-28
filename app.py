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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_single_model(X_input, model, dim):
    pred = model.predict(X_input)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_input)[0]
        confidence = max(proba)
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X_input)
        confidence = sigmoid(decision[0])
    else:
        confidence = 0.5  # neutral confidence if unknown
    return pred, confidence

def ensemble_predict(X_input, dim):
    votes = []
    confidences = []

    for model_name in models.keys():
        model = models[model_name][dim]
        if model:
            pred, confidence = predict_single_model(X_input, model, dim)
            votes.append(pred)
            confidences.append(confidence)

    # Majority vote (most common prediction)
    final_pred = max(set(votes), key=votes.count)
    avg_confidence = np.mean(confidences)

    return final_pred, avg_confidence

def confidence_color(value):
    if value < 0.5:
        return "#ff4b4b"  # red-ish low confidence
    elif value < 0.7:
        return "#f5a623"  # orange medium
    else:
        return "#2ecc71"  # green high confidence

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
st.markdown(
    "Enter a paragraph or meaningful sentence to predict your MBTI type."
)

# Sidebar with About and Model choice
with st.sidebar:
    st.header("📚 About This App")
    st.markdown("""
This app uses **Machine Learning** models to predict your **MBTI personality** from text.

### 🧬 Personality Dimensions:
- **I / E**: Introversion / Extraversion  
- **N / S**: Intuition / Sensing  
- **F / T**: Feeling / Thinking  
- **P / J**: Perceiving / Judging  

### 📈 Model Accuracies:
""")
    for name, acc in model_accuracies.items():
        st.write(f"- {name}: {acc*100:.1f}%")

    best_model = max(model_accuracies, key=model_accuracies.get)
    st.markdown(f"✅ **Best Model**: `{best_model}` ({model_accuracies[best_model]*100:.1f}%)")

    st.markdown("---")
    st.markdown("### ⚙️ Prediction Mode")
    mode = st.radio("Choose prediction mode:", ["Ensemble", "Single Model"])

    selected_model = None
    if mode == "Single Model":
        selected_model = st.selectbox("Select model:", list(models.keys()))

# Input section
st.markdown("### 🧠 Your Input")
user_input = st.text_area(
    "Write about yourself (at least 5 words):", 
    height=200, 
    placeholder="e.g. I enjoy spending quiet time reflecting on my thoughts and feelings."
)
word_count = len(user_input.strip().split())
st.markdown(f"📝 Word count: **{word_count}**")

if st.button("🚀 Predict"):
    if not is_valid_input(user_input):
        st.warning("⚠️ Please enter a complete sentence or paragraph with at least 5 words.")
    elif vectorizer is None:
        st.error("❌ Vectorizer file missing.")
    else:
        st.info("🔄 Processing input and predicting MBTI type...")

        X_input = vectorizer.transform([user_input])
        preds = []
        confidences = {}
        low_conf_warnings = []

        if mode == "Ensemble":
            for dim in ["I/E", "N/S", "F/T", "P/J"]:
                pred, conf = ensemble_predict(X_input, dim)
                preds.append(label_map[dim][pred])
                confidences[dim] = conf
                if conf < 0.6:
                    low_conf_warnings.append(f"🔍 Low confidence in **{dim}** — try providing more text.")

        else:  # Single model selected
            for dim in ["I/E", "N/S", "F/T", "P/J"]:
                model = models[selected_model][dim]
                pred, conf = predict_single_model(X_input, model, dim)
                preds.append(label_map[dim][pred])
                confidences[dim] = conf
                if conf < 0.6:
                    low_conf_warnings.append(f"🔍 Low confidence in **{dim}** — try providing more text.")

        mbti = "".join(preds)

        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.success(f"🎯 **Predicted MBTI Type: `{mbti}`**")
        with col2:
            if mode == "Ensemble":
                avg_acc = np.mean(list(model_accuracies.values()))
                st.metric("📊 Average Model Accuracy", f"{avg_acc*100:.1f}%")
            else:
                acc = model_accuracies.get(selected_model, None)
                st.metric("📊 Model Accuracy", f"{acc*100:.1f}%" if acc else "N/A")

        # MBTI Letter Breakdown
        st.subheader("🧩 MBTI Breakdown")
        col1, col2, col3, col4 = st.columns(4)
        for i, col in enumerate([col1, col2, col3, col4]):
            letter = preds[i]
            col.metric(list(label_map.keys())[i], letter, mbti_descriptions[letter])

        # Confidence Display
        st.subheader("📊 Confidence per Dimension")
        for dim in ["I/E", "N/S", "F/T", "P/J"]:
            conf = confidences[dim]
            color = confidence_color(conf)
            st.markdown(f"**{dim}:** {label_map[dim][int(conf > 0.5)]} with confidence {conf*100:.1f}%")
            st.progress(int(conf * 100))

        if low_conf_warnings:
            st.warning("⚠️ Some predictions have low confidence:")
            for msg in low_conf_warnings:
                st.markdown(msg)

        # MBTI Letter Descriptions
        st.subheader("🔎 MBTI Letters Explained")
        for letter in mbti:
            st.markdown(f"- **{letter}**: {mbti_descriptions[letter]}")

# Footer
st.markdown("---")
st.markdown(
    "<small>Powered by an ensemble of Naive Bayes, SVM, and Random Forest models.</small>", 
    unsafe_allow_html=True
)
