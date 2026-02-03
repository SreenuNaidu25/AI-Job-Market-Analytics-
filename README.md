# AI-Job-Market-Analytics-
AI Job Market Analytics – Interactive Streamlit Dashboard
📌 Project Overview

This project is an AI-driven interactive web application developed using Streamlit to analyze global AI job market trends and predict salaries.
It provides data visualization dashboards, skill demand analysis, and a machine learning–based salary prediction system, along with secure user authentication.

The application is designed for students, job seekers, recruiters, and organizations to gain insights into the AI job market.

🎯 Key Features

🔐 User Authentication

User Registration

Login & Logout functionality using Streamlit session state

📊 Interactive Dashboards

Dataset preview

Top AI job roles

Experience-level distribution

Salary analysis & trends

Country-wise salary comparison

In-demand AI skills

🤖 Machine Learning Model

Linear Regression for salary prediction

Performance metrics: MAE, MSE, R² Score

🎨 Modern UI

AI-themed background image

Glassmorphism effect

Sidebar navigation

🛠️ Technologies Used

Frontend / UI: Streamlit, HTML, CSS

Data Analysis: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-learn

Programming Language: Python

Dataset Format: CSV (Excel-compatible)

📂 Project Structure
AI_Job_Market_Analytics/
│
├── tejaapp.py                  # Main Streamlit application
├── ai_job_market_dataset.csv   # Dataset file
├── README.md                   # Project documentation
└── requirements.txt            # Required libraries

📊 Dataset Description

The dataset contains information about AI-related jobs, including:

Job Title

Required AI Skills

Experience Level

Company Location

Employment Type

Annual Salary (USD)

⚙️ Installation & Setup
1️⃣ Create Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate

2️⃣ Install Required Libraries
pip install streamlit pandas matplotlib seaborn scikit-learn

3️⃣ Run the Application
streamlit run tejaapp.py

4️⃣ Open in Browser
http://localhost:8501

🧠 Machine Learning Workflow

Data Cleaning (remove duplicates & null values)

Label Encoding for categorical features

Train-test split (80% / 20%)

Model training using Linear Regression

Performance evaluation using:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

R² Score

User-based salary prediction via form input

📈 Dashboards Included

📄 Dataset Preview

🔥 Top AI Job Roles

👨‍💼 Experience Level Distribution

💰 Salary Distribution

🌍 Country-wise Average Salary

🧠 Top AI Skills Demand

🔐 Authentication Flow

New users register using username/email & password

Existing users log in

Session-based authentication using Streamlit

Secure logout option via sidebar

🎓 Use Cases

Academic mini / major project

Data Science portfolio

Job market analysis

Resume project for AI / Data Science roles

Interview & viva demonstrations

🚀 Future Enhancements

Password encryption (bcrypt)

Database integration (MySQL / MongoDB)

Advanced ML models (Random Forest, XGBoost)

Role-based access (Admin/User)

Deployment on Streamlit Cloud / AWS

Downloadable PDF & Excel reports

🏁 Conclusion

This project demonstrates the practical application of data analytics, machine learning, and web development to solve real-world problems related to the AI job market.
