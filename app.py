import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import io
from sklearn.metrics import brier_score_loss
import matplotlib as plt

# -------------------------
# Page Configuration & Style
# -------------------------
st.set_page_config(page_title="Career Path Predictor")
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎓 Career Path Predictor for Students")

# -------------------------
# Storytelling Section
# -------------------------
st.markdown("""
## 🧭 Your Career Path Journey Starts Here
Choosing a career can feel overwhelming. This tool helps you reflect on your academic strengths and interests, and then connects those insights to real-world tech careers.

You can begin by entering your scores manually or uploading a template file. Once your data is in, our AI model will help you explore which broad career group you might fit best in—and why.

Along the way, you'll see an explanation of what influenced the prediction and suggestions for roles in that category. You’ll even get a downloadable report to take to your counselor or keep for future planning.
""")

# -------------------------
# Load trained artifacts
# -------------------------
import gdown
import os

model_path = 'rf_group_model.pkl'
gdrive_file_id = '1qsUR7b3bSdy2a1ep75cYClCxl8b9bj9v'
gdrive_url = f'https://drive.google.com/uc?id={gdrive_file_id}'

# First check: is the file already downloaded?
if not os.path.exists(model_path):
    try:
        gdown.download(gdrive_url, model_path, quiet=False)
    except Exception as e:
        st.error(f"❌ Failed to download model from Google Drive: {e}")
        st.stop()

# Now load the model
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Load the rest
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('label_encoder.pkl', 'rb'))
explainer = shap.TreeExplainer(model)


# -------------------------
# Feature list
# -------------------------
FEATURES = [
    'Percentage in Computer Architecture',
    'Percentage in Programming Concepts',
    'Percentage in Electronics Subjects',
    'Percentage in Communication skills',
    'Percentage in Computer Networks',
    'Percentage in Software Engineering',
    'percentage in Algorithms',
    'Acedamic percentage in Operating Systems',
    'Percentage in Mathematics',
    'public speaking points',
    'Logical quotient rating',
    'Hours working per day',
    'coding skills rating',
    'hackathons'
]

# -------------------------
# Career category map
# -------------------------
career_options = {
    'Security / Systems': [
        'Network Security Administrator', 'Network Security Engineer',
        'Information Security Analyst', 'Systems Security Administrator',
        'Information Technology Auditor', 'Systems Analyst'
    ],
    'Technical / Development': [
        'Software Engineer', 'Software Developer', 'Mobile Applications Developer',
        'Applications Developer', 'Web Developer', 'Programmer Analyst',
        'Software Systems Engineer', 'Software Quality Assurance (QA) / Testing',
        'Database Administrator', 'Database Developer', 'Database Manager', 'Data Architect'
    ],
    'Business / Analytics / Management': [
        'CRM Business Analyst', 'Business Systems Analyst',
        'E-Commerce Analyst', 'Business Intelligence Analyst',
        'Project Manager', 'Solutions Architect', 'Information Technology Manager',
        'CRM Technical Developer', 'Portal Administrator'
    ],
    'Support / Design / Other': [
        'UX Designer', 'Design & UX', 'Technical Support',
        'Technical Services/Help Desk/Tech Support', 'Technical Engineer'
    ]
}

# -------------------------
# Sidebar: Help & Sample Template
# -------------------------
st.sidebar.header("🔍 Data Loading Options")
st.sidebar.markdown("## ℹ️ Help & How It Works")
with st.sidebar.expander("📖 User Guide"):
    st.write("""
    - Upload a CSV file or enter values manually with the sliders.
    - For a CSV file upload the Sample Template downloaded below: Replace the sample numbers with your scores and ratings, Columns must match the template, All percentages should be between 0-100, All ratings (points, hours, coding, etc.) should be between 0-10.
    - Once data is entered, click "Predict Career Group" Button to see your results.
    """)

sample_data = [
    {
        'Percentage in Computer Architecture': '0-100 (%)',
        'Percentage in Programming Concepts': '0-100 (%)',
        'Percentage in Electronics Subjects': '0-100 (%)',
        'Percentage in Communication skills': '0-100 (%)',
        'Percentage in Computer Networks': '0-100 (%)',
        'Percentage in Software Engineering': '0-100 (%)',
        'percentage in Algorithms': '0-100 (%)',
        'Acedamic percentage in Operating Systems': '0-100 (%)',
        'Percentage in Mathematics': '0-100 (%)',
        'public speaking points': '0-10',
        'Logical quotient rating': '0-10',
        'Hours working per day': '0-10',
        'coding skills rating': '0-10',
        'hackathons': '0-10'
    },
    {
        'Percentage in Computer Architecture': 85,
        'Percentage in Programming Concepts': 90,
        'Percentage in Electronics Subjects': 80,
        'Percentage in Communication skills': 75,
        'Percentage in Computer Networks': 88,
        'Percentage in Software Engineering': 92,
        'percentage in Algorithms': 86,
        'Acedamic percentage in Operating Systems': 89,
        'Percentage in Mathematics': 91,
        'public speaking points': 8,
        'Logical quotient rating': 9,
        'Hours working per day': 6,
        'coding skills rating': 9,
        'hackathons': 3
    }
]
sample_csv_df = pd.DataFrame(sample_data)
csv_bytes = sample_csv_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="📥 Download Sample Template",
    data=csv_bytes,
    file_name="sample_input_template.csv",
    mime="text/csv"
)

# -------------------------
# Upload Data
# -------------------------
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=['csv'])
data_url = st.sidebar.text_input("...or enter a CSV URL to explore")

data = None
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Data loaded from uploaded file!")
elif data_url:
    try:
        data = pd.read_csv(data_url)
        st.sidebar.success("✅ Data loaded from URL!")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load data from URL. Error: {e}")

# -------------------------
# Display data if loaded
# -------------------------
if data is not None:
    st.subheader("📊 Loaded Data Preview")
    st.dataframe(data.head())
    st.write("Shape:", data.shape)
else:
    st.info("No data file loaded. You can still make manual predictions below.")

# -------------------------
# Manual prediction sliders
# -------------------------
st.subheader("🎯 Manual Career Group Prediction")
input_data = {}
for feature in FEATURES:
    if 'points' in feature or 'rating' in feature or 'Hours' in feature or 'hackathons' in feature:
        input_data[feature] = st.slider(f"{feature}:", 0, 10, 5)
    else:
        input_data[feature] = st.slider(f"{feature} (%):", 0, 100, 50)

# -------------------------
# Predict
# -------------------------
from sklearn.metrics import brier_score_loss

if st.button("🔮 Predict Career Group", key="predict_button"):
    df_input = pd.DataFrame([input_data])
    scaled_input = scaler.transform(df_input)

    pred = model.predict(scaled_input)
    pred_proba = model.predict_proba(scaled_input)
    predicted_group = encoder.inverse_transform(pred)[0]
    confidence = np.max(pred_proba)

    st.markdown("## 📘 Your Career Story Begins Here")
    st.write("You've entered scores across technical and soft skill areas. Based on those, our AI model analyzed your profile to suggest a matching career group.")

    # -------------------------
    # Show Primary Prediction
    # -------------------------
    st.markdown(f"### 🏷️ Predicted Career Group: <span style='color:darkblue'>{predicted_group}</span>", unsafe_allow_html=True)
    st.metric(label="Prediction Confidence", value=f"{confidence:.2f}")

    if predicted_group in career_options:
        st.write("### 🚀 Suggested Career Paths in this Category:")
        for career in career_options[predicted_group]:
            st.markdown(f"- {career}")

    # -------------------------
    # Suggest Backup Option if Confidence is Low
    # -------------------------
    if confidence < 0.50:
        second_best_index = np.argsort(pred_proba[0])[-2]
        second_best_label = encoder.inverse_transform([second_best_index])[0]
        st.markdown(f"🔁 Since confidence is under 50%, you may also want to explore: **{second_best_label}** as a possible fit.")

    # -------------------------
    # Brier Score Explanation
    # -------------------------
    true_label = np.zeros(pred_proba.shape[1])
    true_label[np.argmax(pred_proba)] = 1  # simulate correct class for demo
    brier = brier_score_loss(true_label, pred_proba.flatten())

    st.subheader("📏 Prediction Reliability: Brier Score")
    st.markdown(f"**Brier Score:** `{brier:.3f}`")
    st.markdown("""
    🔍 **What does this mean?**  
    The Brier Score tells us how reliable the model was when assigning probabilities to the possible outcomes.

    - A score **closer to 0** means the prediction was well-calibrated and thoughtful  
    - A score **closer to 1** means the model may have been uncertain or overly confident  

    This helps us measure trustworthiness beyond just raw confidence. Even if your match had low confidence, a good Brier Score suggests the model is being cautious and accurate.
    """)

    # -------------------------
    # SHAP Explanation
    # -------------------------
    st.subheader("📉 Why This Result? SHAP Explanation")
    try:
        shap_values = explainer.shap_values(scaled_input)

        if isinstance(shap_values, list):
            class_index = list(model.classes_).index(pred[0])
            if shap_values[class_index][0].ndim == 2:
                shap_input = shap_values[class_index][0][:, 0]
            else:
                shap_input = shap_values[class_index][0]
            expected_value = explainer.expected_value[class_index]
        else:
            if shap_values[0].ndim == 2:
                shap_input = shap_values[0][:, 0]
            else:
                shap_input = shap_values[0]
            expected_value = explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[0]

        shap_input = np.array(shap_input).flatten()

        st.pyplot(shap.plots._waterfall.waterfall_legacy(
            expected_value,
            shap_input,
            feature_names=FEATURES,
            show=False
        ))
    except Exception as e:
        st.warning(f"⚠️ SHAP explanation could not be generated. Error: {e}")


    # -------------------------
    # SHAP Local Explanation
    # -------------------------
    st.subheader("📉 Why This Result? SHAP Explanation")
    st.write("""
    You've just seen a prediction based on a blend of your technical and soft skill scores.  
    This chart breaks down which of those inputs most strongly influenced the model's decision.

    - Red bars = features that increased the prediction for this career group  
    - Blue bars = features that decreased the prediction  
    - Longer bars = stronger influence  
    """)

    try:
        shap_values = explainer.shap_values(scaled_input)

        # Handle multi-class classification
        if isinstance(shap_values, list):
            class_index = list(model.classes_).index(pred[0])
            # Get SHAP values for the first sample and first output dimension
            if shap_values[class_index][0].ndim == 2:
                shap_input = shap_values[class_index][0][:, 0]  # Take first column
            else:
                shap_input = shap_values[class_index][0]
            expected_value = explainer.expected_value[class_index]
        else:
            # For binary or single-output models
            if shap_values[0].ndim == 2:
                shap_input = shap_values[0][:, 0]
            else:
                shap_input = shap_values[0]
            expected_value = explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[0]

        # Final sanity check: shap_input must be 1D
        shap_input = np.array(shap_input).flatten()

        # Plot waterfall
        st.pyplot(shap.plots._waterfall.waterfall_legacy(
            expected_value,
            shap_input,
            feature_names=FEATURES,
            show=False
        ))

    except Exception as e:
        st.warning(f"⚠️ SHAP explanation could not be generated. Error: {e}")
    

    # -------------------------
    # Download prediction report
    # -------------------------
    report_df = pd.DataFrame({
        "Feature": FEATURES,
        "Value": df_input.values[0]
    })
    report_df.loc[len(report_df)] = ["Predicted Group", predicted_group]
    report_df.loc[len(report_df)] = ["Confidence", confidence]

    csv = report_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Prediction Report",
        data=csv,
        file_name='prediction_report.csv',
        mime='text/csv',
        key="download_button"
    )
