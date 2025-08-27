import streamlit as st
import joblib
import os
import numpy as np

def safe_load(path):
    if os.path.exists(path):
        return joblib.load(path)
    else:
        st.error(f"❌ Missing file: {path}")
        return None

# Load vectorizer (TF-IDF)
vectorizer = safe_load("vectorizer.pkl")

# Load models
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

# Mapping for numeric predictions -> MBTI letters
label_map = {
    "I/E": {0: "I", 1: "E"},
    "N/S": {0: "N", 1: "S"},
    "F/T": {0: "F", 1: "T"},
    "P/J": {0: "P", 1: "J"}
}

st.title("🔮 MBTI Personality Predictor")

user_input = st.text_area("Enter some text:")

model_choice = st.selectbox("Choose a model:", list(models.keys()))

if st.button("Predict"):
    if not user_input.strip():
        st.warning("⚠️ Please enter text first.")
    elif vectorizer is None:
        st.error("❌ Vectorizer file missing.")
    else:
        # Vectorize input
        X_input = vectorizer.transform([user_input])
        preds = []
        probs_out = {}

        for dim in ["I/E", "N/S", "F/T", "P/J"]:
            model = models[model_choice][dim]
            if model:
                raw_pred = model.predict(X_input)[0]
                letter = label_map[dim][raw_pred]
                preds.append(letter)

                # If model supports predict_proba, show probabilities
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_input)[0]
                    probs_out[dim] = {
                        label_map[dim][0]: f"{proba[0]*100:.1f}%",
                        label_map[dim][1]: f"{proba[1]*100:.1f}%"
                    }

        if preds:
            mbti = "".join(preds)
            st.success(f"🎯 Predicted MBTI type: **{mbti}**")

            # Show confidence per dimension
            if probs_out:
                st.subheader("📊 Confidence per Dimension")
                for dim, scores in probs_out.items():
                    st.write(f"**{dim}** → {scores}")
