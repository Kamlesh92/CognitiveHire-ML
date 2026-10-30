# CognitiveHire ML 🚀🧠

**CognitiveHire ML** is an advanced AI-powered Resume Screening & Fitment Intelligence System designed to parse, analyze, and evaluate resumes against diverse professional domains. It provides deep insights into candidate fitment, skill matching, and hireability metrics using machine learning and natural language processing.

---

## ✨ Key Features

* **Multi-Format Document Support:** Seamlessly extracts text from PDF, TXT, and Image (`.png`, `.jpg`, `.jpeg` via Tesseract OCR) formats.
* **Machine Learning Classification:** Utilizes Scikit-Learn (`TfidfVectorizer` & `LogisticRegression`) to accurately classify candidate resumes across multiple core technical domains.
* **Real-Time Fitment Radar & Analytics:** Evaluates technical skills, project impact, experience tenure, and keyword density to calculate comprehensive shortlist probabilities and scores.
* **Dynamic Role Fallback:** Automatically detects custom or unlisted domains in real-time through intelligent text parsing if a resume doesn't fit standard profiles.
* **Interactive Web Dashboard:** Built with a Flask backend and a clean, responsive interface to display matched/missing skills, suitable work purposes, incompatibilities, and cross-role compatibility metrics.

---

## 🛠️ Tech Stack

* **Backend:** Python, Flask
* **ML & NLP Libraries:** Scikit-Learn, PyPDF2, Pytesseract, Pillow (PIL), NumPy
* **Frontend & Visualization:** HTML, CSS, JavaScript, Chart.js

---

## ⚙️ Installation & Running

Follow these steps to set up and run the project locally on your system:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/CognitiveHire-ML.git](https://github.com/your-username/CognitiveHire-ML.git)
   cd CognitiveHire-ML
