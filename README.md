# Student_performance_Prediction07
End-to-end ML project — data cleaning, EDA, model comparison (Linear Regression, Random Forest, Gradient Boosting, SVR), and a Streamlit web app to predict student exam performance from 13 input features.
# 🎓 Student Performance Predictor

A end-to-end Machine Learning web app that predicts student exam scores based on study habits, lifestyle, and personal factors.

## 🔗 Live Demo
[Click here to open the app](your-streamlit-url-here)

---

## 📌 Project Overview
This project follows a complete ML pipeline:
- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Encoding & Correlation Analysis
- Model Training & Comparison
- Streamlit Web App Deployment

---

## 📊 Dataset Features
| Feature | Description |
|---|---|
| Age | Student's age |
| Gender | Male / Female / Other |
| Study Hours/Day | Daily study time |
| Attendance % | Class attendance percentage |
| Sleep Hours | Hours of sleep per night |
| Social Media Hours | Daily social media usage |
| Diet Quality | Poor / Fair / Good |
| Exercise Frequency | Days per week |
| Mental Health Rating | Score out of 10 |
| Parental Education | High School / Bachelor / Master |
| Internet Quality | Poor / Average / Good |
| Part-time Job | Yes / No |
| Extracurricular | Yes / No |

---

## 🤖 Models Compared
| Model | R² | RMSE |
|---|---|---|
| Linear Regression 🏆 | 0.8807 | 5.53 |
| Ridge Regression | 0.8806 | 5.53 |
| Gradient Boosting | 0.8574 | 6.05 |
| Random Forest | 0.8439 | 6.33 |
| SVR | 0.7835 | 7.45 |

**Best Model: Linear Regression with R² = 0.88**

---

## 🛠️ Tech Stack
- **Language:** Python
- **ML:** Scikit-learn, Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **UI:** Streamlit
- **Notebooks:** Jupyter

---

## 🚀 Run Locally
```bash
# Clone the repo
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

├── app.py                        # Streamlit web app
├── requirements.txt
├── 01_data_cleaning_eda.ipynb    # Data cleaning & EDA
├── 02_model_training.ipynb       # Model training & selection
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── metadata.json
└── student_habits_performance.csv

---

## 📁 Project Structure
