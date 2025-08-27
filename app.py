
# Force install dependencies at runtime (quick fix for Streamlit Cloud)
os.system("pip install joblib scikit-learn nltk pandas numpy matplotlib seaborn")

import streamlit as st
import joblib
import os

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
        for dim in ["I/E", "N/S", "F/T", "P/J"]:
            model = models[model_choice][dim]
            if model:
                preds.append(model.predict(X_input)[0])
        if preds:
            st.success(f"🎯 Predicted MBTI type: **{''.join(preds)}**")
