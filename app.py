import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os
import matplotlib.pyplot as plt

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Hide Streamlit Default UI ────────────────────────────────────────────────
hide_streamlit_style = """
<style>

/* Hide Streamlit header */
header {
    visibility: hidden;
}

/* Hide hamburger menu */
#MainMenu {
    visibility: hidden;
}

/* Hide footer */
footer {
    visibility: hidden;
}

/* Hide top toolbar */
[data-testid="stToolbar"] {
    display: none;
}

/* Hide deploy/manage app button */
.stDeployButton {
    display: none;
}

/* Hide top-right icons */
[data-testid="stDecoration"] {
    display: none;
}

/* Hide status widget */
[data-testid="stStatusWidget"] {
    display: none;
}

/* Hide bottom-right manage app button */
[data-testid="manage-app-button"] {
    display: none;
}

/* Remove extra top padding */
.block-container {
    padding-top: 1rem;
}

</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2rem 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 { font-size: 2.2rem; margin: 0; }
    .main-header p  { font-size: 1rem; margin: 0.4rem 0 0; opacity: 0.9; }

    .result-box {
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
        color: white;
        font-size: 1.1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.92; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #4a4a8a;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.3rem;
        margin: 1.2rem 0 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Model (files in repo root, not models/ subfolder) ───────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    with open('metadata.json') as f:
        meta = json.load(f)
    return model, scaler, meta

try:
    model, scaler, meta = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎓 Student Performance Predictor</h1>
    <p>Enter student details to predict exam score using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Model not found! Make sure `best_model.pkl`, `scaler.pkl`, and `metadata.json` are in the root of your repository.")
    st.stop()

# ─── Main Form ────────────────────────────────────────────────────────────────
st.markdown("## 📋 Student Profile Input")

with st.form("prediction_form"):
    # Row 1 — Demographics
    st.markdown('<div class="section-header">👤 Demographics</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    age           = c1.slider("Age", 17, 24, 20)
    gender        = c2.selectbox("Gender", ["Male", "Female", "Other"])
    part_time_job = c3.selectbox("Part-time Job?", ["No", "Yes"])

    # Row 2 — Academic Habits
    st.markdown('<div class="section-header">📚 Academic Habits</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    study_hours     = c1.slider("Study Hours / Day", 0.0, 9.0, 3.5, 0.1)
    attendance_pct  = c2.slider("Attendance %", 60.0, 100.0, 85.0, 0.5)
    extracurricular = c3.selectbox("Extracurricular Activities?", ["No", "Yes"])

    # Row 3 — Lifestyle
    st.markdown('<div class="section-header">🌿 Lifestyle & Wellbeing</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    sleep_hours        = c1.slider("Sleep Hours / Night", 4.0, 10.0, 7.0, 0.5)
    social_media_hours = c2.slider("Social Media Hours / Day", 0.0, 8.0, 2.0, 0.5)
    diet_quality       = c3.selectbox("Diet Quality", ["Poor", "Fair", "Good"])
    exercise_frequency = c4.slider("Exercise Days / Week", 0, 7, 3)

    # Row 4 — External Factors
    st.markdown('<div class="section-header">🏠 External Factors</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    mental_health_rating = c1.slider("Mental Health Rating (1–10)", 1, 10, 6)
    parental_education   = c2.selectbox("Parental Education Level", ["High School", "Bachelor", "Master"])
    internet_quality     = c3.selectbox("Internet Quality", ["Poor", "Average", "Good"])

    submitted = st.form_submit_button("🔮 Predict Exam Score")

# ─── Prediction ───────────────────────────────────────────────────────────────
if submitted:
    gender_enc = {'Male': 1, 'Female': 0, 'Other': 2}[gender]
    diet_enc   = {'Poor': 0, 'Fair': 1, 'Good': 2}[diet_quality]
    inet_enc   = {'Poor': 0, 'Average': 1, 'Good': 2}[internet_quality]
    edu_enc    = {'High School': 0, 'Bachelor': 1, 'Master': 2}[parental_education]
    pt_enc     = 1 if part_time_job == "Yes" else 0
    ec_enc     = 1 if extracurricular == "Yes" else 0

    feature_order = meta['features']

    input_dict = {
        'age': age,
        'gender': gender_enc,
        'study_hours_per_day': study_hours,
        'social_media_hours': social_media_hours,
        'part_time_job': pt_enc,
        'attendance_percentage': attendance_pct,
        'sleep_hours': sleep_hours,
        'diet_quality': diet_enc,
        'exercise_frequency': exercise_frequency,
        'parental_education_level': edu_enc,
        'internet_quality': inet_enc,
        'mental_health_rating': mental_health_rating,
        'extracurricular_participation': ec_enc,
    }

    input_df = pd.DataFrame([input_dict])[feature_order]

    if meta['use_scaler']:
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
    else:
        pred = model.predict(input_df)[0]

    pred = float(np.clip(pred, 0, 100))

    # ─── Result Display ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🎯 Prediction Result")

    if pred >= 75:
        grade, grade_label, color, emoji = "A", "Excellent",        "#2ecc71", "🌟"
    elif pred >= 60:
        grade, grade_label, color, emoji = "B", "Good",             "#3498db", "👍"
    elif pred >= 45:
        grade, grade_label, color, emoji = "C", "Average",          "#f39c12", "📝"
    else:
        grade, grade_label, color, emoji = "D", "Needs Improvement","#e74c3c", "⚠️"

    col_a, col_b, col_c = st.columns([1, 1.5, 1])
    with col_b:
        st.markdown(f"""
        <div class="result-box" style="background: linear-gradient(135deg, {color}aa, {color});">
            <div style="font-size: 3.5rem; font-weight: 900;">{pred:.1f} / 100</div>
            <div style="font-size: 1.3rem; margin-top: 0.5rem;">{emoji} Grade {grade} — {grade_label}</div>
            <div style="font-size: 0.9rem; margin-top: 0.3rem; opacity: 0.9;">
                Predicted by: {meta['model_name']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ─── Score Gauge + Input Summary ──────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📈 Score Gauge")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        for start, end, bg_color in [(0, 45, '#fadbd8'), (45, 60, '#fdebd0'),
                                      (60, 75, '#d4e6f1'), (75, 100, '#d5f5e3')]:
            ax.barh(0.5, end - start, left=start, height=0.4,
                    color=bg_color, edgecolor='white', linewidth=2)
        ax.barh(0.5, pred, height=0.4, color=color, alpha=0.9, edgecolor='white', linewidth=2)
        ax.axvline(pred, color='black', linewidth=2, linestyle='--', ymin=0.1, ymax=0.9)
        ax.text(pred, 0.9, f'{pred:.1f}', ha='center', va='center',
                fontsize=13, fontweight='bold', color='black')
        for val, label in [(22, 'D\n<45'), (52, 'C\n45-60'), (67, 'B\n60-75'), (87, 'A\n75+')]:
            ax.text(val, 0.1, label, ha='center', va='center', fontsize=8, color='#555')
        ax.set_yticks([])
        ax.set_xlabel('Exam Score')
        ax.set_title('Score Gauge')
        ax.spines[['top', 'right', 'left']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### 🔍 Input Summary")
        summary = {
            'Study Hours/Day': f'{study_hours}h',
            'Attendance': f'{attendance_pct}%',
            'Sleep Hours': f'{sleep_hours}h',
            'Mental Health': f'{mental_health_rating}/10',
            'Social Media': f'{social_media_hours}h/day',
            'Exercise': f'{exercise_frequency}x/week',
            'Diet Quality': diet_quality,
            'Internet Quality': internet_quality,
            'Extracurricular': extracurricular,
            'Part-time Job': part_time_job,
            'Parental Education': parental_education,
        }
        for k, v in summary.items():
            st.write(f"**{k}:** {v}")

    # ─── Insights ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💡 Key Factors Influencing This Prediction")

    insights = []
    if study_hours >= 5:
        insights.append(("✅", "High study hours (≥5h/day) strongly boost exam scores."))
    elif study_hours < 2:
        insights.append(("⚠️", "Low study hours (<2h/day). Increasing study time may improve scores significantly."))
    if attendance_pct >= 85:
        insights.append(("✅", f"Good attendance ({attendance_pct}%) positively impacts performance."))
    elif attendance_pct < 70:
        insights.append(("⚠️", f"Low attendance ({attendance_pct}%) is a risk factor. Try to attend more classes."))
    if mental_health_rating <= 3:
        insights.append(("❗", "Low mental health score. Consider seeking support — it greatly affects academic performance."))
    elif mental_health_rating >= 8:
        insights.append(("✅", "Excellent mental health supports focused studying."))
    if sleep_hours < 6:
        insights.append(("⚠️", f"Low sleep ({sleep_hours}h). Aim for 7–9 hours for optimal cognitive performance."))
    if social_media_hours > 4:
        insights.append(("⚠️", f"High social media usage ({social_media_hours}h/day) can distract from studying."))
    if not insights:
        insights.append(("ℹ️", "Profile looks balanced! Keep up the good habits."))

    for icon, text in insights:
        st.info(f"{icon} {text}")
