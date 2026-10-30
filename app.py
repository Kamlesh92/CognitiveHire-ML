import os
import re
from collections import Counter
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import PyPDF2
from PIL import Image

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Safe folder check
upload_dir = app.config['UPLOAD_FOLDER']
if os.path.exists(upload_dir) and not os.path.isdir(upload_dir):
    os.remove(upload_dir)
os.makedirs(upload_dir, exist_ok=True)

# 8 Base Professional Domains
DOMAIN_DEFINITIONS = {
    'Machine Learning & AI Engineer': {
        'required_skills': ['python', 'tensorflow', 'pytorch', 'scikit-learn', 'deep learning', 'nlp', 'computer vision', 'opencv', 'keras', 'pandas', 'numpy', 'neural networks', 'transformers', 'data science'],
        'purpose_fit': 'AI model training, Computer Vision/NLP pipeline architecture, deep learning solutions, predictive modeling.',
        'not_suitable_for': 'Pure CSS/HTML frontend slicing or enterprise relational database administration.'
    },
    'Data Analyst / BI Specialist': {
        'required_skills': ['sql', 'power bi', 'tableau', 'excel', 'data visualization', 'statistics', 'python', 'business intelligence', 'dashboards', 'etl', 'eda', 'mysql'],
        'purpose_fit': 'Executive reporting, SQL data modeling, stakeholder dashboards, trend analysis, statistical testing.',
        'not_suitable_for': 'Full-stack application logic, infrastructure setup, low-level system design.'
    },
    'Full Stack Web Developer': {
        'required_skills': ['javascript', 'react', 'node.js', 'html', 'css', 'typescript', 'mongodb', 'express', 'rest api', 'next.js', 'sql', 'frontend', 'backend', 'vue', 'angular'],
        'purpose_fit': 'End-to-end web apps, responsive frontend dashboards, microservices, RESTful API integrations.',
        'not_suitable_for': 'Heavy statistical research, embedded firmware programming.'
    },
    'Python Backend Developer': {
        'required_skills': ['python', 'django', 'flask', 'fastapi', 'rest api', 'postgresql', 'redis', 'celery', 'docker', 'microservices', 'git', 'orm'],
        'purpose_fit': 'Backend web services, scalable REST APIs, database queries, asynchronous task pipelines.',
        'not_suitable_for': 'UI/UX interface crafting or deep computer vision model deployment.'
    },
    'DevOps & Cloud Engineer': {
        'required_skills': ['docker', 'kubernetes', 'aws', 'linux', 'ci/cd', 'jenkins', 'terraform', 'git', 'bash', 'ansible', 'cloud', 'azure', 'gcp'],
        'purpose_fit': 'Infrastructure as code, container orchestration, server automation, deployment pipelines, system security.',
        'not_suitable_for': 'Product UI prototyping, client-side JavaScript.'
    },
    'Cyber Security Analyst': {
        'required_skills': ['network security', 'penetration testing', 'wireshark', 'vulnerability assessment', 'kali linux', 'firewall', 'cryptography', 'owasp', 'soc', 'siem', 'python'],
        'purpose_fit': 'Security auditing, ethical hacking, vulnerability scanning, network traffic monitoring, threat prevention.',
        'not_suitable_for': 'Standard CRUD application development or non-technical business reporting.'
    },
    'Android / Mobile App Developer': {
        'required_skills': ['flutter', 'react native', 'android', 'kotlin', 'java', 'swift', 'mobile development', 'firebase', 'xml', 'ios', 'sqlite'],
        'purpose_fit': 'Cross-platform/native mobile app development, state management, mobile device optimization, app publishing.',
        'not_suitable_for': 'High-performance cloud server orchestration or enterprise data warehousing.'
    },
    'QA & Automation Test Engineer': {
        'required_skills': ['selenium', 'testing', 'automation testing', 'unit testing', 'pytest', 'qa', 'postman', 'jira', 'manual testing', 'cucumber', 'test cases'],
        'purpose_fit': 'Test suite automation, regression testing, bug reporting, API validation, release quality assurance.',
        'not_suitable_for': 'Production infrastructure provisioning or full-lifecycle application feature authoring.'
    }
}

# Domain Training Corpus
TRAINING_SAMPLES = [
    ("deep learning neural networks cnn rnn pytorch tensorflow scikit-learn computer vision nlp opencv transformers machine learning model training model evaluation data preprocessing", "Machine Learning & AI Engineer"),
    ("machine learning engineer building generative ai models nlp spacy huggingface classification regression predictive modeling feature engineering pandas numpy", "Machine Learning & AI Engineer"),
    ("data analysis power bi dashboards tableau sql queries excel spreadsheets data cleaning business intelligence statistics eda reporting visualization data insights", "Data Analyst / BI Specialist"),
    ("business intelligence specialist sql server power bi dax measures data modeling statistical analysis metrics tracking", "Data Analyst / BI Specialist"),
    ("full stack developer react nodejs express javascript typescript mongodb rest api web applications frontend html5 css3 tailwind responsive ui", "Full Stack Web Developer"),
    ("web developer building single page applications javascript vue node express html css websockets git web development", "Full Stack Web Developer"),
    ("python developer flask django fastapi postgresql redis microservices rest api unit testing orm docker celery background tasks backend architecture", "Python Backend Developer"),
    ("devops engineer aws cloud kubernetes docker containers ci cd pipelines jenkins terraform infrastructure as code linux bash scripting git ansible monitoring", "DevOps & Cloud Engineer"),
    ("cyber security analyst penetration testing ethical hacking kali linux wireshark vulnerability assessment owasp top 10 network security siem soc cryptography", "Cyber Security Analyst"),
    ("android developer kotlin java flutter dart cross platform mobile application development react native firebase android studio ui layouts sqlite rest api", "Android / Mobile App Developer"),
    ("qa automation engineer selenium webdriver pytest test automation test cases execution bug tracking jira manual testing postman api testing regression testing", "QA & Automation Test Engineer")
]

ml_model = None

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.\+#]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def train_engine():
    global ml_model
    corpus = [clean_text(item[0]) for item in TRAINING_SAMPLES]
    labels = [item[1] for item in TRAINING_SAMPLES]

    ml_model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1200, ngram_range=(1, 2), stop_words='english')),
        ('classifier', LogisticRegression(C=1.0, max_iter=300, random_state=42))
    ])
    ml_model.fit(corpus, labels)
    print("✓ Model Engine Trained Across 8 Domains!")

def extract_text_pdf(file_path):
    extracted = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted += page.extract_text() or ""
    except Exception as e:
        print(f"PDF extract error: {e}")
    return extracted

def extract_text_image(file_path):
    if not OCR_AVAILABLE:
        return ""
    try:
        img = Image.open(file_path)
        return pytesseract.image_to_string(img)
    except Exception as e:
        print(f"Image OCR error: {e}")
        return ""

def extract_dynamic_realtime_role(raw_text):
    """Fallback extractor for unlisted/open domains like UI/UX, Product Manager, Mechanical, etc."""
    lines = [line.strip() for line in raw_text.split('\n') if len(line.strip()) > 2]
    
    title_pattern = r'(?i)\b(senior|junior|lead|associate|principal|intern)?\s*([a-z\/\s\-]+(?:developer|engineer|designer|analyst|manager|specialist|consultant|architect|scientist|officer|executive|technician))\b'
    
    found_title = None
    for line in lines[:15]:
        match = re.search(title_pattern, line)
        if match:
            found_title = match.group(0).strip().title()
            break
            
    if not found_title:
        tfidf = TfidfVectorizer(ngram_range=(2, 3), stop_words='english', max_features=10)
        try:
            tfidf.fit_transform([raw_text])
            feature_names = tfidf.get_feature_names_out()
            found_title = feature_names[0].title() + " Specialist" if len(feature_names) > 0 else "Domain Professional"
        except:
            found_title = "Domain Specialist"

    words = re.findall(r'\b[A-Z][a-zA-Z0-9\+#\.]*\b', raw_text)
    common_stops = {'The', 'This', 'With', 'And', 'From', 'For', 'India', 'Resume', 'Curriculum', 'Vitae', 'Education', 'Projects', 'Experience', 'Summary'}
    extracted_skills = [w for w in words if w not in common_stops and len(w) > 2]
    top_skills = [item[0] for item in Counter(extracted_skills).most_common(8)]

    return found_title, top_skills

def calculate_candidate_metrics(cleaned_text, matched_skills_count, total_req_skills):
    skill_score = min(100, int((matched_skills_count / max(total_req_skills * 0.6, 1)) * 100))

    project_keywords = ['project', 'built', 'developed', 'created', 'designed', 'implemented', 'github', 'deployed', 'api', 'architecture', 'database', 'system', 'application']
    project_matches = sum(1 for kw in project_keywords if kw in cleaned_text)
    project_score = min(100, int((project_matches / len(project_keywords)) * 140))
    if project_score < 30: project_score = 35

    exp_matches = re.findall(r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?experience', cleaned_text)
    exp_years = 0
    if exp_matches:
        exp_years = max([int(x) for x in exp_matches if int(x) < 35])
        exp_score = min(100, int((exp_years / 5) * 100))
    elif any(term in cleaned_text for term in ['intern', 'internship', 'trainee', 'worked as']):
        exp_years = 1
        exp_score = 55
    else:
        exp_score = 40

    hireability_chance = round((skill_score * 0.40) + (project_score * 0.35) + (exp_score * 0.25), 1)

    return {
        'skill_score': skill_score,
        'project_score': project_score,
        'exp_score': exp_score,
        'exp_years': exp_years,
        'hireability_chance': hireability_chance
    }

def analyze_resume(resume_text):
    cleaned = clean_text(resume_text)
    if not cleaned:
        return None

    classes = ml_model.classes_
    probabilities = ml_model.predict_proba([cleaned])[0]

    scored_roles = []
    for idx, role in enumerate(classes):
        prob = probabilities[idx] * 100
        req_skills = DOMAIN_DEFINITIONS[role]['required_skills']
        
        found = [s for s in req_skills if re.search(r'\b' + re.escape(s) + r'\b', cleaned)]
        missing = [s for s in req_skills if s not in found]
        skill_coverage = (len(found) / max(len(req_skills), 1)) * 100
        
        composite_score = round((prob * 0.55) + (skill_coverage * 0.45), 1)

        scored_roles.append({
            'role': role,
            'match_score': composite_score,
            'probability': round(prob, 1),
            'skills_matched': found,
            'skills_missing': missing[:6],
            'total_req_len': len(req_skills),
            'purpose_fit': DOMAIN_DEFINITIONS[role]['purpose_fit'],
            'not_suitable_for': DOMAIN_DEFINITIONS[role]['not_suitable_for']
        })

    scored_roles.sort(key=lambda x: x['match_score'], reverse=True)
    primary_fit = scored_roles[0]

    # REAL-TIME DYNAMIC DETECTION FOR UNLISTED DOMAINS
    # Agar predefined 8 domains me score bahut low ho (< 32%), toh dynamic extraction activate hogi
    if primary_fit['match_score'] < 32:
        dynamic_role, dynamic_skills = extract_dynamic_realtime_role(resume_text)
        primary_fit['role'] = f"{dynamic_role} (Custom Domain)"
        primary_fit['skills_matched'] = dynamic_skills
        primary_fit['skills_missing'] = ['Industry Certification', 'Specialized Frameworks']
        primary_fit['purpose_fit'] = f"Directly matched for {dynamic_role} projects and domain-specific operations."
        primary_fit['not_suitable_for'] = "Cross-domain tech operations without dedicated training."
        primary_fit['match_score'] = 75.0  # Dynamic benchmark score

    metrics = calculate_candidate_metrics(
        cleaned, 
        len(primary_fit['skills_matched']), 
        primary_fit['total_req_len']
    )

    exp_level = "Fresher / Entry-Level"
    if metrics['exp_years'] >= 5: exp_level = f"Senior Level ({metrics['exp_years']}+ Yrs)"
    elif metrics['exp_years'] >= 2: exp_level = f"Mid Level ({metrics['exp_years']} Yrs)"
    elif metrics['exp_years'] == 1: exp_level = "Junior / Associate"

    all_tech_pool = set()
    for d in DOMAIN_DEFINITIONS.values():
        all_tech_pool.update(d['required_skills'])
    detected_all = [s for s in all_tech_pool if re.search(r'\b' + re.escape(s) + r'\b', cleaned)]

    return {
        'primary_role': primary_fit['role'],
        'confidence': primary_fit['match_score'],
        'experience_estimation': exp_level,
        'purpose_fit': primary_fit['purpose_fit'],
        'not_suitable_for': primary_fit['not_suitable_for'],
        'matched_skills': primary_fit['skills_matched'],
        'missing_skills': primary_fit['skills_missing'],
        'all_detected_skills': list(set(detected_all)),
        'alternate_roles': scored_roles[1:4],
        'metrics': metrics
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def process_upload():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file submitted'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename'}), 400

    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    allowed = {'pdf', 'txt', 'png', 'jpg', 'jpeg'}

    if ext not in allowed:
        return jsonify({'success': False, 'error': f'Unsupported format. Use {", ".join(allowed)}'}), 400

    saved_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(saved_path)

    resume_text = ""
    if ext == 'pdf':
        resume_text = extract_text_pdf(saved_path)
    elif ext == 'txt':
        with open(saved_path, 'r', encoding='utf-8', errors='ignore') as f:
            resume_text = f.read()
    elif ext in {'png', 'jpg', 'jpeg'}:
        resume_text = extract_text_image(saved_path)

    if os.path.exists(saved_path):
        os.remove(saved_path)

    if not resume_text.strip():
        return jsonify({'success': False, 'error': 'Could not extract readable text from document.'}), 400

    analysis = analyze_resume(resume_text)
    return jsonify({'success': True, 'filename': filename, 'data': analysis})

if __name__ == '__main__':
    train_engine()
    print("🚀 Server running on http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)