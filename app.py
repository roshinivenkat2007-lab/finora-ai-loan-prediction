import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Finora AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "loan_model.pkl"
DATASET_PATH = BASE_DIR / "loan_approval_dataset (1).csv"

# ============================================================
# LOAD EXTERNAL UI STYLE
# ============================================================

CSS_PATH = BASE_DIR / "ui_style.css"

if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )
else:
    st.warning("ui_style.css file not found. Please keep it in the same folder as app.py.")


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ============================================================
# FIND LOGO
# ============================================================

logo_candidates = [
    BASE_DIR / "finora_logo.png.png",
    BASE_DIR / "finora_logo.png",
    BASE_DIR / "Finora_logo.png",
    BASE_DIR / "Finora.png"
]

LOGO_PATH = None

for path in logo_candidates:
    if path.exists():
        LOGO_PATH = path
        break

if LOGO_PATH is None:
    png_files = list(BASE_DIR.glob("*.png"))
    for path in png_files:
        if "finora" in path.name.lower():
            LOGO_PATH = path
            break


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    df = pd.read_csv(DATASET_PATH)

    # Remove unwanted spaces
    df.columns = df.columns.str.strip()

    # Strip text columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Encode categorical columns
    if "education" in df.columns:
        df["education"] = df["education"].map({
            "Graduate": 0,
            "Not Graduate": 1
        })

    if "self_employed" in df.columns:
        df["self_employed"] = df["self_employed"].map({
            "No": 0,
            "Yes": 1
        })

    if "loan_status" in df.columns:
        df["loan_status"] = df["loan_status"].map({
            "Approved": 0,
            "Rejected": 1
        })

    # Remove rows with missing values
    df = df.dropna()

    return df


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============================================================
# LOAD MODEL + DATA
# ============================================================

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

try:
    df = load_dataset()
except Exception as e:
    st.error(f"Dataset loading error: {e}")
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="hero">', unsafe_allow_html=True)

header_col1, header_col2 = st.columns([1, 4])

with header_col1:

    if LOGO_PATH:
        st.image(str(LOGO_PATH), width=210)
    else:
        st.markdown(
            "<div style='font-size:42px;font-weight:800;'>FINORA</div>",
            unsafe_allow_html=True
        )

with header_col2:

    st.markdown(
        '<div class="hero-title">Finora AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-subtitle">'
        'Intelligent Credit Risk Assessment & Smart Loan Recommendation Framework'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="ai-active">● AI CREDIT ANALYSIS SYSTEM ACTIVE</div>',
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">Customer Information</div>',
    unsafe_allow_html=True
)

with st.container():

    col1, col2, col3 = st.columns(3)

    with col1:

        loan_id = st.number_input(
            "Loan ID",
            min_value=0,
            value=10001,
            step=1
        )

        dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            max_value=10,
            value=2
        )

        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )

        self_employed = st.selectbox(
            "Self Employed",
            ["No", "Yes"]
        )

    with col2:

        income = st.number_input(
            "Annual Income (₹)",
            min_value=0.0,
            value=500000.0,
            step=10000.0
        )

        loan_amount = st.number_input(
            "Loan Amount (₹)",
            min_value=0.0,
            value=1500000.0,
            step=50000.0
        )

        loan_term = st.number_input(
            "Loan Term (Years)",
            min_value=1,
            max_value=40,
            value=10
        )

        cibil = st.slider(
            "CIBIL Score",
            min_value=300,
            max_value=900,
            value=650
        )

    with col3:

        residential_assets = st.number_input(
            "Residential Assets (₹)",
            min_value=0.0,
            value=2000000.0,
            step=50000.0
        )

        commercial_assets = st.number_input(
            "Commercial Assets (₹)",
            min_value=0.0,
            value=500000.0,
            step=50000.0
        )

        luxury_assets = st.number_input(
            "Luxury Assets (₹)",
            min_value=0.0,
            value=500000.0,
            step=50000.0
        )

        bank_assets = st.number_input(
            "Bank Assets (₹)",
            min_value=0.0,
            value=500000.0,
            step=50000.0
        )


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_input():

    input_data = {
        "loan_id": loan_id,
        " no_of_dependents": dependents,
        " education": 0 if education == "Graduate" else 1,
        " self_employed": 0 if self_employed == "No" else 1,
        " income_annum": income,
        " loan_amount": loan_amount,
        " loan_term": loan_term,
        " cibil_score": cibil,
        " residential_assets_value": residential_assets,
        " commercial_assets_value": commercial_assets,
        " luxury_assets_value": luxury_assets,
        " bank_asset_value": bank_assets
    }

    input_df = pd.DataFrame([input_data])

    input_df.columns = input_df.columns.str.strip()

    # Align model features
    if hasattr(model, "feature_names_in_"):

        model_features = list(model.feature_names_in_)

        aligned_df = pd.DataFrame(
            columns=model_features
        )

        for feature in model_features:

            clean_feature = feature.strip()

            if clean_feature in input_df.columns:
                aligned_df[feature] = input_df[clean_feature].values

            elif feature in input_df.columns:
                aligned_df[feature] = input_df[feature].values

            else:
                aligned_df[feature] = 0

        input_df = aligned_df

    return input_df


# ============================================================
# CALCULATE EMI
# ============================================================

def calculate_emi(principal, years, annual_rate=8.5):

    if principal <= 0 or years <= 0:
        return 0

    monthly_rate = annual_rate / 12 / 100
    months = years * 12

    emi = (
        principal
        * monthly_rate
        * (1 + monthly_rate) ** months
    ) / (
        (1 + monthly_rate) ** months - 1
    )

    return emi


# ============================================================
# RISK SCORE
# ============================================================

def calculate_risk_score():

    score = 0

    # CIBIL
    if cibil >= 750:
        score += 40
    elif cibil >= 650:
        score += 30
    elif cibil >= 600:
        score += 20
    else:
        score += 10

    # Loan / Income
    loan_income_ratio = (
        loan_amount / income
        if income > 0
        else 999
    )

    if loan_income_ratio <= 1:
        score += 25
    elif loan_income_ratio <= 2:
        score += 20
    elif loan_income_ratio <= 3:
        score += 12
    else:
        score += 5

    # Assets
    total_assets = (
        residential_assets
        + commercial_assets
        + luxury_assets
        + bank_assets
    )

    if total_assets >= loan_amount:
        score += 20
    elif total_assets >= loan_amount * 0.5:
        score += 15
    elif total_assets >= loan_amount * 0.25:
        score += 10
    else:
        score += 5

    # Employment
    score += 10 if self_employed == "No" else 7

    # Dependents
    if dependents <= 2:
        score += 5
    elif dependents <= 4:
        score += 3
    else:
        score += 1

    score = max(0, min(100, score))

    if score >= 75:
        level = "Low Risk"
    elif score >= 50:
        level = "Medium Risk"
    else:
        level = "High Risk"

    return score, level


# ============================================================
# FINANCIAL HEALTH
# ============================================================

def calculate_health_score(emi, total_assets):

    score = 0

    # CIBIL
    cibil_score = min(
        35,
        max(0, (cibil - 300) / 600 * 35)
    )

    score += cibil_score

    # EMI / Income
    monthly_income = income / 12 if income > 0 else 0

    if monthly_income > 0:

        emi_ratio = emi / monthly_income

        if emi_ratio <= 0.25:
            score += 30
        elif emi_ratio <= 0.35:
            score += 24
        elif emi_ratio <= 0.45:
            score += 15
        else:
            score += 5

    # Assets
    if total_assets >= loan_amount:
        score += 25
    elif total_assets >= loan_amount * 0.5:
        score += 20
    elif total_assets >= loan_amount * 0.25:
        score += 12
    else:
        score += 5

    # Loan / income
    ratio = loan_amount / income if income > 0 else 999

    if ratio <= 1:
        score += 10
    elif ratio <= 2:
        score += 7
    elif ratio <= 3:
        score += 4
    else:
        score += 1

    score = round(min(100, score), 1)

    if score >= 75:
        level = "Strong"
    elif score >= 50:
        level = "Moderate"
    else:
        level = "Needs Improvement"

    return score, level


# ============================================================
# PREDICTION
# ============================================================

def calculate_prediction():

    input_df = prepare_input()

    prediction = model.predict(input_df)[0]

    status = "Approved" if prediction == 0 else "Rejected"

    probability = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_df)[0]

        classes = list(model.classes_)

        if 0 in classes:
            approved_index = classes.index(0)
            probability = probabilities[approved_index] * 100
        else:
            probability = probabilities[0] * 100

    risk_score, risk_level = calculate_risk_score()

    loan_income_ratio = (
        loan_amount / income
        if income > 0
        else 0
    )

    total_assets = (
        residential_assets
        + commercial_assets
        + luxury_assets
        + bank_assets
    )

    emi = calculate_emi(
        loan_amount,
        loan_term
    )

    monthly_income = income / 12 if income > 0 else 0

    emi_income_ratio = (
        emi / monthly_income * 100
        if monthly_income > 0
        else 0
    )

    health_score, health_level = calculate_health_score(
        emi,
        total_assets
    )

    return {
        "status": status,
        "probability": probability,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "loan_income_ratio": loan_income_ratio,
        "total_assets": total_assets,
        "emi": emi,
        "emi_income_ratio": emi_income_ratio,
        "health_score": health_score,
        "health_level": health_level
    }


# ============================================================
# PREDICT BUTTON
# ============================================================

st.markdown("")

predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])

with predict_col2:

    predict_button = st.button(
        "Analyze with Finora AI",
        width="stretch"
    )


if predict_button:

    result = calculate_prediction()

    st.session_state.prediction_result = result

    # Save history
    history_item = {
        "Loan ID": int(loan_id),
        "Prediction": result["status"],
        "Approval Probability": round(result["probability"], 2),
        "Risk Score": result["risk_score"],
        "Risk Level": result["risk_level"],
        "CIBIL": cibil,
        "Loan Amount": loan_amount
    }

    st.session_state.prediction_history.append(
        history_item
    )


# ============================================================
# SHOW RESULTS
# ============================================================

result = st.session_state.prediction_result


if result:

    st.markdown("---")

    st.markdown(
        '<div class="section-title">AI Credit Decision</div>',
        unsafe_allow_html=True
    )

    # ========================================================
    # MAIN RESULT
    # ========================================================

    result_col1, result_col2 = st.columns([1.25, 1])

    with result_col1:

        if result["status"] == "Approved":

            st.markdown(
                f"""
                <div class="result-approved">
                    <div class="small-label">MODEL DECISION</div>
                    <h2 style="color:#86EFAC !important;">
                        APPROVED
                    </h2>
                    <p style="color:#CBD5E1;">
                        Finora AI predicts that this loan application
                        satisfies the model's approval pattern.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result-rejected">
                    <div class="small-label">MODEL DECISION</div>
                    <h2 style="color:#FCA5A5 !important;">
                        REJECTED
                    </h2>
                    <p style="color:#CBD5E1;">
                        Finora AI predicts that this application
                        does not satisfy the model's approval pattern.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    with result_col2:

        st.markdown(
            f"""
            <div class="ai-card">
                <div class="ai-label">AI FINANCIAL ADVISOR</div>
                <div class="ai-title">
                    Finora AI Insight
                </div>
            """,
            unsafe_allow_html=True
        )

        if result["status"] == "Approved":

            if result["risk_level"] == "Low Risk":

                ai_message = (
                    f"Your application shows a relatively strong "
                    f"financial profile. The current CIBIL score of "
                    f"{cibil} and asset position support the application. "
                    f"Your estimated risk score is {result['risk_score']}/100."
                )

            else:

                ai_message = (
                    f"Your application is predicted as approved, but "
                    f"there are some financial factors that should be monitored. "
                    f"Your current risk level is {result['risk_level']}."
                )

        else:

            ai_message = (
                f"The model predicts rejection. The major areas to review "
                f"are your CIBIL score, loan-to-income ratio and repayment "
                f"affordability. Your current risk score is "
                f"{result['risk_score']}/100."
            )

        st.markdown(
            f'<div class="ai-text">{ai_message}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # ========================================================
    # METRICS
    # ========================================================

    st.markdown("")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "Approval Probability",
            f"{result['probability']:.1f}%"
        )

    with m2:
        st.metric(
            "Risk Score",
            f"{result['risk_score']}/100"
        )

    with m3:
        st.metric(
            "Financial Health",
            f"{result['health_score']}/100"
        )

    with m4:
        st.metric(
            "Estimated EMI",
            f"₹{result['emi']:,.0f}"
        )


    # ========================================================
    # PROGRESS BARS
    # ========================================================

    bar1, bar2 = st.columns(2)

    with bar1:

        st.write("Approval Confidence")

        st.progress(
            int(min(100, max(0, result["probability"])))
        )

    with bar2:

        st.write("Financial Risk Score")

        st.progress(
            int(result["risk_score"])
        )


    # ========================================================
    # FINANCIAL SNAPSHOT
    # ========================================================

    st.markdown(
        '<div class="section-title">Financial Snapshot</div>',
        unsafe_allow_html=True
    )

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.metric(
            "CIBIL Score",
            cibil
        )

    with s2:
        st.metric(
            "Loan / Income",
            f"{result['loan_income_ratio']:.2f}x"
        )

    with s3:
        st.metric(
            "Total Assets",
            f"₹{result['total_assets']:,.0f}"
        )

    with s4:
        st.metric(
            "EMI / Monthly Income",
            f"{result['emi_income_ratio']:.1f}%"
        )


    # ========================================================
    # SMART RECOMMENDATION
    # ========================================================

    st.markdown(
        '<div class="section-title">Smart Loan Recommendation</div>',
        unsafe_allow_html=True
    )

    recommendations = []

    if cibil >= 750:
        recommendations.append(
            "Your CIBIL score is strong. Continue maintaining good repayment behaviour."
        )
    elif cibil >= 650:
        recommendations.append(
            "Your CIBIL score is moderate. Maintaining timely repayments can strengthen your profile."
        )
    else:
        recommendations.append(
            "Improving your CIBIL score could strengthen future loan eligibility."
        )

    if result["loan_income_ratio"] > 3:
        recommendations.append(
            "The requested loan is high compared with annual income. Consider reducing the loan amount."
        )
    elif result["loan_income_ratio"] > 2:
        recommendations.append(
            "Your loan-to-income ratio is relatively high. Review the requested amount and repayment period."
        )
    else:
        recommendations.append(
            "Your requested loan amount is relatively reasonable compared with annual income."
        )

    if result["emi_income_ratio"] > 45:
        recommendations.append(
            "Estimated EMI is high compared with monthly income. A longer tenure or lower loan amount may improve affordability."
        )
    elif result["emi_income_ratio"] <= 35:
        recommendations.append(
            "Estimated EMI appears manageable compared with monthly income."
        )

    if result["total_assets"] >= loan_amount:
        recommendations.append(
            "Your asset position provides a relatively strong financial backing."
        )
    else:
        recommendations.append(
            "Building additional savings or assets can improve financial resilience."
        )

    st.markdown(
        '<div class="ai-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="ai-label">AI-POWERED RECOMMENDATION</div>',
        unsafe_allow_html=True
    )

    for item in recommendations:
        st.markdown(
            f'<div class="recommendation">✓ {item}</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # WHAT-IF ANALYSIS
    # ========================================================

    with st.expander("What-If Financial Simulator"):

        st.write(
            "Change the financial values below to explore how "
            "your estimated risk score may change."
        )

        w1, w2, w3, w4 = st.columns(4)

        with w1:

            what_cibil = st.slider(
                "What-If CIBIL",
                300,
                900,
                int(cibil)
            )

        with w2:

            what_income = st.number_input(
                "What-If Annual Income",
                min_value=0.0,
                value=float(income),
                step=10000.0
            )

        with w3:

            what_loan = st.number_input(
                "What-If Loan Amount",
                min_value=0.0,
                value=float(loan_amount),
                step=50000.0
            )

        with w4:

            what_term = st.number_input(
                "What-If Tenure",
                min_value=1,
                max_value=40,
                value=int(loan_term)
            )

        # Calculate scenario
        original_cibil = cibil
        original_income = income
        original_loan = loan_amount
        original_term = loan_term

        # CIBIL score component
        if what_cibil >= 750:
            cibil_points = 40
        elif what_cibil >= 650:
            cibil_points = 30
        elif what_cibil >= 600:
            cibil_points = 20
        else:
            cibil_points = 10

        what_ratio = (
            what_loan / what_income
            if what_income > 0
            else 999
        )

        if what_ratio <= 1:
            ratio_points = 25
        elif what_ratio <= 2:
            ratio_points = 20
        elif what_ratio <= 3:
            ratio_points = 12
        else:
            ratio_points = 5

        what_assets = result["total_assets"]

        if what_assets >= what_loan:
            asset_points = 20
        elif what_assets >= what_loan * 0.5:
            asset_points = 15
        elif what_assets >= what_loan * 0.25:
            asset_points = 10
        else:
            asset_points = 5

        employment_points = 10 if self_employed == "No" else 7

        dependent_points = (
            5 if dependents <= 2
            else 3 if dependents <= 4
            else 1
        )

        what_risk_score = (
            cibil_points
            + ratio_points
            + asset_points
            + employment_points
            + dependent_points
        )

        what_risk_score = max(
            0,
            min(100, what_risk_score)
        )

        if what_risk_score >= 75:
            what_level = "Low Risk"
        elif what_risk_score >= 50:
            what_level = "Medium Risk"
        else:
            what_level = "High Risk"

        scenario_df = pd.DataFrame({
            "Metric": [
                "CIBIL Score",
                "Annual Income",
                "Loan Amount",
                "Loan Tenure",
                "Risk Score",
                "Risk Level"
            ],
            "Current": [
                original_cibil,
                f"₹{original_income:,.0f}",
                f"₹{original_loan:,.0f}",
                f"{original_term} years",
                result["risk_score"],
                result["risk_level"]
            ],
            "What-If": [
                what_cibil,
                f"₹{what_income:,.0f}",
                f"₹{what_loan:,.0f}",
                f"{what_term} years",
                what_risk_score,
                what_level
            ]
        })

        st.dataframe(
            scenario_df,
            width="stretch",
            hide_index=True
        )

        difference = what_risk_score - result["risk_score"]

        if difference > 0:
            st.success(
                f"AI Scenario Insight: Estimated risk score improves "
                f"by {difference} points."
            )

        elif difference < 0:
            st.warning(
                f"AI Scenario Insight: Estimated risk score decreases "
                f"by {abs(difference)} points."
            )

        else:
            st.info(
                "AI Scenario Insight: The estimated risk score remains unchanged."
            )


    # ========================================================
    # EXPLAINABLE AI
    # ========================================================

    with st.expander("Explainable AI — Why this result?"):

        positive_factors = []
        risk_factors = []

        if cibil >= 750:
            positive_factors.append(
                "Strong CIBIL score"
            )
        elif cibil >= 650:
            positive_factors.append(
                "Moderate CIBIL score"
            )
        else:
            risk_factors.append(
                "Low CIBIL score"
            )

        if result["loan_income_ratio"] <= 2:
            positive_factors.append(
                "Reasonable loan-to-income ratio"
            )
        else:
            risk_factors.append(
                "High loan-to-income ratio"
            )

        if result["total_assets"] >= loan_amount:
            positive_factors.append(
                "Assets provide good financial backing"
            )
        else:
            risk_factors.append(
                "Assets are lower than requested loan amount"
            )

        if result["emi_income_ratio"] <= 35:
            positive_factors.append(
                "Estimated EMI is within a manageable range"
            )
        else:
            risk_factors.append(
                "Estimated EMI is relatively high compared with income"
            )

        ex1, ex2 = st.columns(2)

        with ex1:

            st.markdown("### Positive Factors")

            if positive_factors:

                for factor in positive_factors:
                    st.success(f"✓ {factor}")

            else:
                st.info("No strong positive factors identified.")

        with ex2:

            st.markdown("### Risk Factors")

            if risk_factors:

                for factor in risk_factors:
                    st.warning(f"• {factor}")

            else:
                st.success("No major risk factors identified.")


    # ========================================================
    # FINANCIAL ANALYTICS
    # ========================================================

    with st.expander("Financial Analytics"):

        chart1, chart2 = st.columns(2)

        with chart1:

            income_loan_df = pd.DataFrame({
                "Category": [
                    "Annual Income",
                    "Loan Amount"
                ],
                "Value": [
                    income,
                    loan_amount
                ]
            })

            st.write("Income vs Loan Amount")

            st.bar_chart(
                income_loan_df.set_index("Category")
            )

        with chart2:

            assets_df = pd.DataFrame({
                "Asset Type": [
                    "Residential",
                    "Commercial",
                    "Luxury",
                    "Bank"
                ],
                "Value": [
                    residential_assets,
                    commercial_assets,
                    luxury_assets,
                    bank_assets
                ]
            })

            st.write("Asset Distribution")

            st.bar_chart(
                assets_df.set_index("Asset Type")
            )

        chart3, chart4 = st.columns(2)

        with chart3:

            monthly_income = income / 12

            emi_df = pd.DataFrame({
                "Category": [
                    "Monthly Income",
                    "Estimated EMI"
                ],
                "Value": [
                    monthly_income,
                    result["emi"]
                ]
            })

            st.write("EMI vs Monthly Income")

            st.bar_chart(
                emi_df.set_index("Category")
            )

        with chart4:

            risk_df = pd.DataFrame({
                "Metric": [
                    "Risk Score",
                    "Approval Probability"
                ],
                "Value": [
                    result["risk_score"],
                    result["probability"]
                ]
            })

            st.write("Risk & Approval")

            st.bar_chart(
                risk_df.set_index("Metric")
            )


# ============================================================
# AI CHAT ASSISTANT
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">Finora AI Assistant</div>',
    unsafe_allow_html=True
)

chat_left, chat_right = st.columns([1.35, 1])

with chat_left:

    st.markdown(
        """
        <div class="ai-card">
            <div class="ai-label">AI CONVERSATIONAL ASSISTANT</div>
            <div class="ai-title">Ask Finora AI</div>
            <div class="ai-text">
                Ask questions about your loan prediction,
                risk level, CIBIL score, affordability or recommendations.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    user_question = st.text_input(
        "Ask your question",
        placeholder="Example: Why is my risk level medium?"
    )

    if st.button("Ask Finora AI") and user_question:

        q = user_question.lower()

        if result is None:

            answer = (
                "Please run the loan analysis first. "
                "Then I can explain your prediction and financial profile."
            )

        elif "risk" in q:

            answer = (
                f"Your current risk level is {result['risk_level']} "
                f"with an estimated risk score of "
                f"{result['risk_score']}/100. "
                f"The score considers CIBIL, loan-to-income ratio, "
                f"assets, employment status and dependents."
            )

        elif "cibil" in q or "credit" in q:

            if cibil >= 750:
                cibil_comment = "strong"
            elif cibil >= 650:
                cibil_comment = "moderate"
            else:
                cibil_comment = "relatively low"

            answer = (
                f"Your CIBIL score is {cibil}, which is considered "
                f"{cibil_comment} within this application's scoring logic. "
                f"A higher CIBIL contributes positively to the estimated risk score."
            )

        elif "reject" in q or "rejected" in q:

            answer = (
                f"The model's prediction is {result['status']}. "
                f"To understand the decision, review your CIBIL score, "
                f"loan-to-income ratio, assets and estimated EMI."
            )

        elif "approve" in q or "approved" in q:

            answer = (
                f"The model predicts {result['status']} with an estimated "
                f"approval probability of {result['probability']:.1f}%. "
                f"This probability comes from the Random Forest model."
            )

        elif "emi" in q:

            answer = (
                f"Your estimated monthly EMI is approximately "
                f"₹{result['emi']:,.0f}. "
                f"This is around {result['emi_income_ratio']:.1f}% "
                f"of your estimated monthly income."
            )

        elif "loan amount" in q or "loan" in q:

            answer = (
                f"Your requested loan amount is ₹{loan_amount:,.0f}. "
                f"Your loan-to-income ratio is "
                f"{result['loan_income_ratio']:.2f}x. "
                f"Reviewing the requested amount can improve affordability "
                f"when this ratio becomes high."
            )

        elif "income" in q:

            answer = (
                f"Your annual income is ₹{income:,.0f}. "
                f"The application uses income along with loan amount, "
                f"CIBIL, assets and other features for prediction."
            )

        elif "recommend" in q or "advice" in q or "improve" in q:

            answer = "Here are the main areas to focus on: "

            if cibil < 750:
                answer += "improving and maintaining CIBIL; "

            if result["loan_income_ratio"] > 2:
                answer += "reviewing the requested loan amount; "

            if result["emi_income_ratio"] > 35:
                answer += "improving EMI affordability; "

            if result["total_assets"] < loan_amount:
                answer += "building financial reserves/assets; "

            if (
                cibil >= 750
                and result["loan_income_ratio"] <= 2
                and result["emi_income_ratio"] <= 35
            ):
                answer += (
                    "your current profile already shows several positive indicators."
                )

        else:

            answer = (
                f"Based on your current analysis, your loan is "
                f"{result['status']} with a "
                f"{result['risk_level']} profile. "
                f"You can ask me about risk, CIBIL, EMI, loan amount, "
                f"income, approval or recommendations."
            )

        st.session_state.chat_history.append(
            {
                "user": user_question,
                "ai": answer
            }
        )


    # Chat history
    for chat in st.session_state.chat_history[-5:]:

        st.markdown(
            f"""
            <div class="chat-user">
                <b>You</b><br>
                {chat["user"]}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="chat-ai">
                <b>Finora AI</b><br>
                {chat["ai"]}
            </div>
            """,
            unsafe_allow_html=True
        )


with chat_right:

    st.markdown(
        """
        <div class="card">
            <div class="ai-label">SUGGESTED QUESTIONS</div>
            <br>
            <b>Try asking:</b>
            <br><br>
            • Why is my risk level high?
            <br><br>
            • Explain my CIBIL score
            <br><br>
            • How much is my EMI?
            <br><br>
            • Why was my loan rejected?
            <br><br>
            • How can I improve my profile?
            <br><br>
            • Is my loan amount affordable?
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DATASET ANALYSIS
# ============================================================

with st.expander("Dataset Analysis"):

    total_records = len(df)

    if "loan_status" in df.columns:

        approved_count = int(
            (df["loan_status"] == 0).sum()
        )

        rejected_count = int(
            (df["loan_status"] == 1).sum()
        )

        approval_rate = (
            approved_count / total_records * 100
            if total_records > 0
            else 0
        )

    else:

        approved_count = 0
        rejected_count = 0
        approval_rate = 0

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.metric(
            "Total Records",
            total_records
        )

    with d2:
        st.metric(
            "Approved",
            approved_count
        )

    with d3:
        st.metric(
            "Rejected",
            rejected_count
        )

    with d4:
        st.metric(
            "Approval Rate",
            f"{approval_rate:.1f}%"
        )

    st.write("Dataset Preview")

    st.dataframe(
        df.head(10),
        width="stretch"
    )

    st.caption(
        "The displayed dataset is the cleaned/preprocessed dataframe used by the application. "
        "Rows containing missing values are removed during preprocessing."
    )


# ============================================================
# PREDICTION HISTORY
# ============================================================

with st.expander("Prediction History"):

    if st.session_state.prediction_history:

        history_df = pd.DataFrame(
            st.session_state.prediction_history
        )

        st.dataframe(
            history_df,
            width="stretch",
            hide_index=True
        )

        if st.button("Clear Prediction History"):

            st.session_state.prediction_history = []

            st.rerun()

    else:

        st.info(
            "No prediction history available yet."
        )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

with st.expander("Feature Importance — ML Model"):

    if hasattr(model, "feature_importances_"):

        feature_names = list(
            getattr(
                model,
                "feature_names_in_",
                [f"Feature {i+1}" for i in range(len(model.feature_importances_))]
            )
        )

        importance_df = pd.DataFrame({
            "Feature": [
                str(x).strip()
                for x in feature_names
            ],
            "Importance": model.feature_importances_
        }).sort_values(
            "Importance",
            ascending=False
        )

        st.bar_chart(
            importance_df.set_index("Feature")
        )

        st.dataframe(
            importance_df,
            width="stretch",
            hide_index=True
        )

    else:

        st.info(
            "Feature importance is not available for this model."
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

with st.expander("Model Performance"):

    try:

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix
        )

        target_col = "loan_status"

        feature_cols = [
            col
            for col in df.columns
            if col != target_col
        ]

        X_eval = df[feature_cols]
        y_eval = df[target_col]

        # Align with model features
        if hasattr(model, "feature_names_in_"):

            model_features = list(model.feature_names_in_)

            aligned_eval = pd.DataFrame(
                columns=model_features,
                index=X_eval.index
            )

            for feature in model_features:

                clean_feature = feature.strip()

                if clean_feature in X_eval.columns:
                    aligned_eval[feature] = X_eval[clean_feature]

                elif feature in X_eval.columns:
                    aligned_eval[feature] = X_eval[feature]

                else:
                    aligned_eval[feature] = 0

            X_eval = aligned_eval

        y_pred = model.predict(X_eval)

        accuracy = accuracy_score(
            y_eval,
            y_pred
        )

        precision = precision_score(
            y_eval,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_eval,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_eval,
            y_pred,
            zero_division=0
        )

        p1, p2, p3, p4 = st.columns(4)

        with p1:
            st.metric(
                "Accuracy",
                f"{accuracy * 100:.2f}%"
            )

        with p2:
            st.metric(
                "Precision",
                f"{precision * 100:.2f}%"
            )

        with p3:
            st.metric(
                "Recall",
                f"{recall * 100:.2f}%"
            )

        with p4:
            st.metric(
                "F1 Score",
                f"{f1 * 100:.2f}%"
            )

        cm = confusion_matrix(
            y_eval,
            y_pred
        )

        st.write("Confusion Matrix")

        cm_df = pd.DataFrame(
            cm,
            index=["Actual Approved", "Actual Rejected"],
            columns=["Predicted Approved", "Predicted Rejected"]
        )

        st.dataframe(
            cm_df,
            width="stretch"
        )

        st.caption(
            "These metrics are calculated on the loaded dataset used for evaluation, "
            "not on a separate held-out test set."
        )

    except Exception as e:

        st.warning(
            f"Model performance calculation could not be completed: {e}"
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("Model Information"):

    st.write(
        "Algorithm: Random Forest Classifier"
    )

    st.write(
        f"Dataset Records: {len(df)}"
    )

    if hasattr(model, "feature_names_in_"):

        st.write(
            f"Model Features: {len(model.feature_names_in_)}"
        )

        st.write(
            "Features used:"
        )

        st.write(
            [
                str(feature).strip()
                for feature in model.feature_names_in_
            ]
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <b>Finora AI</b> · Intelligent Credit Risk Assessment Platform
        <br>
        Machine Learning + Explainable AI + Smart Financial Recommendations
    </div>
    """,
    unsafe_allow_html=True
)