import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://tse1.mm.bing.net/th/id/OIP.u11WP0XhS-SYpN-Cj-XfpwHaEP?pid=Api&P=0&h=180");
            background-attachment: fixed;
            background-size: cover;
            background-position: center;
        }}

        /* Glassmorphism effect for containers */
        section[data-testid="stSidebar"] {{
            background-color: rgba(0, 0, 0, 0.7);
        }}

        div[data-testid="stVerticalBlock"] {{
            background: rgba(255, 255, 255, 0.85);
            padding: 20px;
            border-radius: 15px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
add_bg_from_url()



# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="AI Job Market Analytics", layout="wide")

# ------------------ SESSION INIT ------------------
if "users" not in st.session_state:
    st.session_state.users = {}   # store registered users

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

# ------------------ AUTH PAGE ------------------
if not st.session_state.logged_in:

    st.title("🔐 Authentication")

    tab1, tab2 = st.tabs(["📝 Register", "🔑 Login"])

    # ---------- REGISTER ----------
    with tab1:
        st.subheader("Create New Account")

        reg_user = st.text_input("Username or Email", key="reg_user")
        reg_pass = st.text_input("Password", type="password", key="reg_pass")
        reg_confirm = st.text_input("Confirm Password", type="password")

        if st.button("Register"):
            if not reg_user or not reg_pass:
                st.warning("Please fill all fields")
            elif reg_pass != reg_confirm:
                st.error("Passwords do not match ❌")
            elif reg_user in st.session_state.users:
                st.error("User already exists ❌")
            else:
                st.session_state.users[reg_user] = reg_pass
                st.success("Registration successful ✅ Please login")

    # ---------- LOGIN ----------
    with tab2:
        st.subheader("Login to Your Account")

        login_user = st.text_input("Username or Email", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            if login_user in st.session_state.users and \
               st.session_state.users[login_user] == login_pass:
                st.session_state.logged_in = True
                st.session_state.username = login_user
                st.success("Login successful ✅")
                st.rerun()
            else:
                st.error("Invalid credentials ❌")

    st.stop()

# ------------------ SIDEBAR LOGOUT ------------------
st.sidebar.success(f"👤 Logged in as {st.session_state.username}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

# ==================================================
# =============== MAIN APPLICATION =================
# ==================================================

# ------------------ LOAD DATA ------------------
# ==================================================
# =============== MAIN APPLICATION =================
# ==================================================

st.title("🤖 AI Job Market Analytics – Interactive Dashboard")

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\sreen\OneDrive\Desktop\ai_job_market_dataset.csv")

df = load_data().drop_duplicates().dropna()

# ------------------ SIDEBAR CONTROLS ------------------
st.sidebar.header("📌 Select What You Want")

feature = st.sidebar.selectbox(
    "Choose Feature",
    [
        "Dataset Preview",
        "Job Role Dashboard",
        "Salary Analysis",
        "Country-wise Salary",
        "Skill Demand",
        "Salary Prediction"
    ]
)

# ------------------ DATASET PREVIEW ------------------
if feature == "Dataset Preview":
    st.subheader("📄 Dataset Preview")
    st.dataframe(df)

# ------------------ JOB ROLE DASHBOARD ------------------
elif feature == "Job Role Dashboard":
    st.subheader("🔥 Top AI Job Roles")
    st.bar_chart(df["Job_Title"].value_counts())

    st.subheader("👨‍💼 Experience Level Distribution")
    st.bar_chart(df["Experience_Level"].value_counts())

# ------------------ SALARY ANALYSIS ------------------
elif feature == "Salary Analysis":
    st.subheader("💰 Salary Distribution")

    fig, ax = plt.subplots()
    sns.histplot(df["Salary_USD(Annual)"], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("💼 Salary by Experience Level")
    fig, ax = plt.subplots()
    sns.boxplot(
        x="Experience_Level",
        y="Salary_USD(Annual)",
        data=df,
        ax=ax
    )
    st.pyplot(fig)

# ------------------ COUNTRY-WISE SALARY ------------------
elif feature == "Country-wise Salary":
    st.subheader("🌍 Average Salary by Country")

    country_salary = (
        df.groupby("Company_Location")["Salary_USD(Annual)"]
        .mean()
        .sort_values(ascending=False)
    )
    st.bar_chart(country_salary)

# ------------------ SKILL DEMAND ------------------
elif feature == "Skill Demand":
    st.subheader("🧠 Most In-Demand AI Skills")

    skills = df["Required_AI_Skills"].str.split(",").explode()
    st.bar_chart(skills.value_counts().head(10))

# ------------------ SALARY PREDICTION ------------------
elif feature == "Salary Prediction":
    st.subheader("🤖 Predict Salary Based on Inputs")

    # Encode data
    le = LabelEncoder()
    df_enc = df.copy()

    for col in ["Job_Title", "Experience_Level", "Company_Location", "Employment_Type"]:
        df_enc[col] = le.fit_transform(df_enc[col])

    X = df_enc[["Job_Title", "Experience_Level", "Company_Location", "Employment_Type"]]
    y = df_enc["Salary_USD(Annual)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
    c2.metric("MSE", f"{mean_squared_error(y_test, y_pred):.2f}")
    c3.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")

    # User Input
    with st.form("predict_form"):
        job = st.selectbox("Job Title", df["Job_Title"].unique())
        exp = st.selectbox("Experience Level", df["Experience_Level"].unique())
        loc = st.selectbox("Company Location", df["Company_Location"].unique())
        emp = st.selectbox("Employment Type", df["Employment_Type"].unique())
        submit = st.form_submit_button("Predict Salary")

    if submit:
        input_df = pd.DataFrame(
            [[job, exp, loc, emp]],
            columns=["Job_Title", "Experience_Level", "Company_Location", "Employment_Type"]
        )

        for col in input_df.columns:
            input_df[col] = le.fit_transform(
                pd.concat([df[col], input_df[col]])
            )[-1:]

        salary = model.predict(input_df)[0]
        st.success(f"💰 Predicted Salary: ${salary:,.2f}")
