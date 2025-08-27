import streamlit as st
import joblib
import os

def safe_load(filename):
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        st.error(f"❌ File not found: {filename}")
        return None

# Load vectorizer
vectorizer = safe_load("vectorizer.pkl")

# Load models
models = {
    "Naive Bayes": {
        'I/E': safe_load('naivebayes_ie.pkl'),
        'N/S': safe_load('naivebayes_ns.pkl'),
        'F/T': safe_load('naivebayes_ft.pkl'),
        'P/J': safe_load('naivebayes_pj.pkl')
    },
    "SVM": {
        'I/E': safe_load('svm_ie.pkl'),
        'N/S': safe_load('svm_ns.pkl'),
        'F/T': safe_load('svm_ft.pkl'),
        'P/J': safe_load('svm_pj.pkl')
    },
    "Random Forest": {
        'I/E': safe_load('randomforest_ie.pkl'),
        'N/S': safe_load('randomforest_ns.pkl'),
        'F/T': safe_load('randomforest_ft.pkl'),
        'P/J': safe_load('randomforest_pj.pkl')
    }
}

st.title("MBTI Personality Predictor")
st.write("Enter text and get MBTI predictions using Naive Bayes, SVM, or Random Forest!")

user_input = st.text_area("📝 Enter your text here:")
selected_model = st.selectbox("🔍 Choose a model:", list(models.keys()))

if st.button("Predict"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text first.")
    elif not vectorizer:
        st.error("❌ Vectorizer is missing. Please upload vectorizer.pkl.")
    else:
        # Transform user input with vectorizer
        X_input = vectorizer.transform([user_input])

        preds = []
        for dim in ['I/E', 'N/S', 'F/T', 'P/J']:
            model = models[selected_model][dim]
            if model:
                pred = model.predict(X_input)[0]
                preds.append(pred)
                st.write(f"**{dim} Prediction:** {pred}")
        if preds:
            st.success(f"🎯 Final MBTI Type: {''.join(preds)}")
