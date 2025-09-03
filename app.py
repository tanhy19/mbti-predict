import streamlit as st
import joblib
import os
import numpy as np
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ------------------ Helper Functions ------------------ #

def safe_load(path):
    """
    Helper function to load a model from a file.
    If the model file does not exist, an error message will be shown.
    :param path: The path to the model file.
    :return: The loaded model or None if the file does not exist.
    """
    if os.path.exists(path):
        return joblib.load(path)
    else:
        st.error(f"❌ Missing file: {path}")  # Display an error if the file is not found.
        return None

def is_valid_input(text):
    """
    Validates the user input text to ensure it contains at least 5 words and 
    consists of valid alphanumeric characters.
    :param text: The input text from the user.
    :return: True if valid input, False otherwise.
    """
    return bool(re.search(r'[a-zA-Z]{3,}', text)) and len(text.strip().split()) >= 5

def sigmoid(x):
    """
    Sigmoid activation function used for converting output into a probability (confidence).
    :param x: The input value from model prediction.
    :return: A value between 0 and 1 representing confidence.
    """
    return 1 / (1 + np.exp(-x))

def predict_single_model(X_input, model, dim):
    """
    Makes a prediction using a single model and calculates confidence.
    The model used can be Naive Bayes, SVM, or Random Forest.
    :param X_input: The transformed input text for prediction.
    :param model: The machine learning model to use for prediction.
    :param dim: The MBTI dimension (I/E, N/S, F/T, P/J) for prediction.
    :return: The predicted class and its confidence score.
    """
    pred = model.predict(X_input)[0]  # Make the prediction.
    
    # Calculate confidence based on the model's methods (predict_proba or decision_function).
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_input)[0]  # Get probability of the classes.
        confidence = max(proba)  # Maximum probability is used as the confidence.
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X_input)
        confidence = sigmoid(decision[0])  # Use sigmoid on the decision function for confidence.
    else:
        confidence = 0.5  # Default confidence if no specific method is available.
    
    return pred, confidence

def ensemble_predict(X_input, dim):
    """
    Aggregates predictions from multiple models and applies a majority vote.
    The final prediction is based on the majority class predicted across all models.
    :param X_input: The transformed input text for prediction.
    :param dim: The MBTI dimension (I/E, N/S, F/T, P/J).
    :return: The final prediction and the average confidence from all models.
    """
    votes = []  # Store predictions from all models.
    confidences = []  # Store confidence values from all models.

    # Loop through each model (Naive Bayes, SVM, and Random Forest) for the given dimension.
    for model_name in models.keys():
        model = models[model_name][dim]
        if model:  # Ensure the model exists before prediction.
            pred, confidence = predict_single_model(X_input, model, dim)
            votes.append(pred)  # Store the prediction.
            confidences.append(confidence)  # Store the confidence.

    # Majority vote: Choose the most frequent prediction.
    final_pred = max(set(votes), key=votes.count)
    avg_confidence = np.mean(confidences)  # Calculate the average confidence of all models.

    return final_pred, avg_confidence

def confidence_color(value):
    """
    Assigns a color based on the confidence level for visual representation.
    Low confidence (below 0.5) is marked with red, medium with orange, and high with green.
    :param value: The confidence score (between 0 and 1).
    :return: The color associated with the confidence level.
    """
    if value < 0.5:
        return "#ff4b4b"  # Red color for low confidence.
    elif value < 0.7:
        return "#f5a623"  # Orange color for medium confidence.
    else:
        return "#2ecc71"  # Green color for high confidence.

# ------------------ Load Resources ------------------ #

vectorizer = safe_load("vectorizer.pkl")  # Load the vectorizer to convert input text into numerical features.

# Load the trained models for different MBTI dimensions (I/E, N/S, F/T, P/J) from disk.
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

# Descriptions of each MBTI letter for later explanation in the app.
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

# CAA styling
st.markdown("""
    <style>
        /* Global Background */
        body {
            background-color: #1e1e1e;
            color: #f5f5f5;
        }

        /* Title and Subtitle */
        .title {
            color: #00bcd4;  /* Cool cyan blue */
        }
        
        .subtitle {
            color: #cfd8dc;  /* Light grey with a cool vibe */
        }

        /* Sidebar */
        .sidebar .sidebar-content {
            background-color: #263238; /* Dark slate blue */
            color: #80deea; /* Light cyan for sidebar text */
        }
        .sidebar .sidebar-header {
            background-color: #37474f; /* Slightly lighter background for header */
            color: #ffffff;
        }
        
        /* Buttons */
        .stButton > button {
            background-color: #00bcd4; /* Cool cyan blue */
            color: white;
            border-radius: 5px;
            font-weight: bold;
        }

        .stButton > button:hover {
            background-color: #0288d1; /* Darker blue when hovered */
        }

        /* Input Fields */
        .stTextInput > div > input {
            background-color: #37474f;
            color: #ffffff;
            border: 1px solid #00bcd4;
        }

        .stTextArea > div > textarea {
            background-color: #37474f;
            color: #ffffff;
            border: 1px solid #00bcd4;
        }

        /* Metrics */
        .stMetric > div {
            background-color: #37474f;
            color: #00bcd4;
            border: 1px solid #00bcd4;
        }

        /* Expander */
        .stExpanderHeader {
            background-color: #263238; /* Dark slate */
            color: #ffffff;
        }

        .stExpanderContent {
            background-color: #37474f;
            color: #ffffff;
        }

        /* Progress Bar */
        .stProgressBar div {
            background-color: #00bcd4;
        }
        
    </style>
""", unsafe_allow_html=True)

# Set the page configuration for the Streamlit app
st.set_page_config(page_title="MBTI Predictor", page_icon="🔮", layout="centered")

# Title of the app
st.title("🔮 MBTI Personality Predictor")
st.markdown(
    "Enter a paragraph or meaningful sentence to predict your MBTI type."
)

# Sidebar for About section and model selection
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

# Input section for user to write about themselves
st.markdown("### 🧠 Your Input")
user_input = st.text_area(
    "Write about yourself (at least 5 words):", 
    height=200, 
    placeholder="e.g. I enjoy spending quiet time reflecting on my thoughts and feelings."
)

# Display the word count of the input text
word_count = len(user_input.strip().split())
st.markdown(f"📝 Word count: **{word_count}**")

# Show a wordcloud of the input text
if user_input:
    wordcloud = WordCloud().generate(user_input)
    
    # Create a figure and axis explicitly
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Display the wordcloud
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')  # Hide axis
    st.pyplot(fig)  # Pass the figure to st.pyplot

# Button to trigger prediction
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

        with st.spinner("Analyzing your text..."):
            if mode == "Ensemble":
                # Ensemble method: Predict for all dimensions (I/E, N/S, F/T, P/J) using all models
                for dim in ["I/E", "N/S", "F/T", "P/J"]:
                    pred, conf = ensemble_predict(X_input, dim)
                    preds.append(label_map[dim][pred])
                    confidences[dim] = conf
                    if conf < 0.6:
                        low_conf_warnings.append(f"🔍 Low confidence in **{dim}** — try providing more text.")

            else:  # Single model selected
                # Single model method: Predict for all dimensions using the selected model
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
        with st.expander("🧩 MBTI Breakdown"):
            col1, col2, col3, col4 = st.columns(4)
            for i, col in enumerate([col1, col2, col3, col4]):
                letter = preds[i]
                col.metric(list(label_map.keys())[i], letter, mbti_descriptions[letter])
                st.progress(int(confidences[list(label_map.keys())[i]] * 100))

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
        with st.expander("🔎 MBTI Letters Explained"):
            for letter in mbti:
                st.markdown(f"- **{letter}**: {mbti_descriptions[letter]}")

# Footer
st.markdown("---")
st.markdown(
    "<small>Powered by an ensemble of Naive Bayes, SVM, and Random Forest models.</small>", 
    unsafe_allow_html=True
)
