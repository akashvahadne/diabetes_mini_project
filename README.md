<h1>🩺 HealthPredict AI</h1>
<h2>Clinical Diabetes Risk Prediction System</h2>

<p>
HealthPredict AI is a Machine Learning-powered web application designed to assess
the probability of diabetes risk using standard clinical health metrics.
The system provides both a categorical risk classification and a probability percentage.
</p>

<hr>

<h2>🚀 Live Application</h2>
<p>
🔗 <b>Live Demo:</b>https://diabetes-mini-project.onrender.com
</p>

<hr>

<h2>📌 Features</h2>
<ul>
    <li>✔ AI-based Diabetes Risk Prediction</li>
    <li>✔ Probability Percentage Output</li>
    <li>✔ Professional Medical-Grade UI</li>
    <li>✔ Fully Responsive Landing Page</li>
    <li>✔ Real-Time Prediction Processing</li>
    <li>✔ Secure – No Data Stored</li>
</ul>

<hr>

<h2>📊 Input Parameters</h2>
<ul>
    <li>Pregnancies</li>
    <li>Glucose Level</li>
    <li>Blood Pressure</li>
    <li>Skin Thickness</li>
    <li>Insulin Level</li>
    <li>BMI</li>
    <li>Diabetes Pedigree Function</li>
    <li>Age</li>
</ul>

<hr>

<h2>🧠 Machine Learning Model</h2>
<ul>
    <li>Supervised Classification Model</li>
    <li>Trained on Clinical Dataset</li>
    <li>Uses <code>predict()</code> and <code>predict_proba()</code></li>
    <li>Returns both classification and probability</li>
</ul>

<hr>

<h2>🛠 Tech Stack</h2>
<ul>
    <li>Python</li>
    <li>Flask</li>
    <li>NumPy</li>
    <li>Scikit-Learn</li>
    <li>Gunicorn</li>
    <li>HTML + CSS (Custom Professional UI)</li>
    <li>Render (Deployment)</li>
</ul>

<hr>

<h2>📁 Project Structure</h2>

<pre>
diabetes_mini_project/
│
├── app.py
├── final_diabetes_model.pkl
├── requirements.txt
│
├── templates/
│   ├── landing.html
│   └── index.html
│
└── static/
</pre>

<hr>

<h2>⚙ Installation (Local Setup)</h2>

<pre>
git clone https://github.com/your-username/diabetes_mini_project.git
cd diabetes_mini_project
pip install -r requirements.txt
python app.py
</pre>

Then open:
<pre>http://127.0.0.1:5000</pre>

<hr>

<h2>🌍 Deployment</h2>
<p>
Deployed using <b>Render Web Service</b> with:
</p>

<pre>
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app
</pre>

<hr>

<h2>⚠ Disclaimer</h2>
<p>
This application is for educational and demonstration purposes only.
It is not intended to replace professional medical diagnosis or treatment.
Always consult a qualified healthcare provider for medical decisions.
</p>

<hr>

<h2>👨‍💻 Author</h2>
<p>
Developed by <b>Akash Vahadne</b><br>
B.E. Computer Engineering<br>
Machine Learning & Web Development Enthusiast
</p>

<hr>

<p align="center">
⭐ If you found this project useful, consider giving it a star!
</p>
