import pickle as pkl
import streamlit as st
import pandas as pd
import numpy as np
import io

# --- 1. LOAD MODELS (cached — loads once, reused on every interaction) ---
@st.cache_resource
def load_models():
    return (
        pkl.load(open('model.pkl',       'rb')),
        pkl.load(open('Transformer.pkl', 'rb')),
        pkl.load(open('colText.pkl',     'rb')),
        pkl.load(open('imputer.pkl',     'rb')),
        pkl.load(open('colNumber.pkl',   'rb')),
        pkl.load(open('bounds.pkl',      'rb')),
    )

model, Transformer, colText, imputer, colNumber, bounds = load_models()

# --- 2. PAGE CONFIG ---
st.set_page_config(page_title="Fraud Detector", page_icon="🛡️", layout="centered")

# --- 3. CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    * { box-sizing: border-box; }

    .stApp {
        background-color: #f0f0f0 !important;
        font-family: 'DM Sans', sans-serif;
    }

    #MainMenu, footer, header { visibility: hidden; }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 700px !important;
        margin: 0 auto !important;
    }

    .app-header {
        text-align: center;
        margin-bottom: 1.2rem;
        padding: 0.5rem 0 0.5rem;
    }
    .app-header .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(220,60,60,0.08);
        border: 1px solid rgba(220,60,60,0.2);
        border-radius: 100px;
        padding: 4px 12px;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #cc3333;
        margin-bottom: 10px;
    }
    .app-header h1 {
        font-family: 'Syne', sans-serif;
        font-size: 34px;
        font-weight: 800;
        line-height: 1.1;
        margin: 0 0 6px;
        background: linear-gradient(135deg, #222 0%, #cc3333 50%, #e07000 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .app-header p {
        color: #999;
        font-size: 13px;
        margin: 0;
        font-weight: 300;
        letter-spacing: 0.02em;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: rgba(0,0,0,0.05) !important;
        border: 1px solid rgba(0,0,0,0.08) !important;
        border-radius: 12px !important;
        padding: 4px !important;
        gap: 4px !important;
        margin-bottom: 1rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px !important;
        color: #888 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        padding: 8px 20px !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(220,60,60,0.1) !important;
        color: #e03030 !important;
        border: 1px solid rgba(220,60,60,0.2) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }
    .stTabs [data-baseweb="tab-border"]    { display: none !important; }

    .form-section-title {
        font-family: 'Syne', sans-serif;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #bbb;
        margin-bottom: 0.6rem;
        padding-bottom: 6px;
        border-bottom: 1px solid rgba(0,0,0,0.06);
    }

    .stTextInput label,
    .stNumberInput label,
    .stSelectbox label {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        color: #999 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        margin-bottom: 2px !important;
    }
    .stTextInput input,
    .stNumberInput input {
        background: #f7f7f7 !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 10px !important;
        color: #222 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 14px !important;
        padding: 9px 12px !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    .stTextInput input:focus,
    .stNumberInput input:focus {
        border-color: rgba(220,60,60,0.5) !important;
        box-shadow: 0 0 0 3px rgba(220,60,60,0.08) !important;
        background: #fff !important;
    }
    .stSelectbox > div > div {
        background: #f7f7f7 !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 10px !important;
        color: #222 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 14px !important;
    }
    .stSelectbox > div > div:focus-within {
        border-color: rgba(220,60,60,0.5) !important;
        box-shadow: 0 0 0 3px rgba(220,60,60,0.08) !important;
    }
    .stTextInput, .stNumberInput, .stSelectbox {
        margin-bottom: 0 !important;
    }
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.4rem !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #ff4444 0%, #ff6b00 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        color: #fff !important;
        font-family: 'Syne', sans-serif !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em !important;
        padding: 13px 28px !important;
        transition: all 0.2s ease !important;
        text-transform: uppercase !important;
        box-shadow: 0 4px 20px rgba(255,68,68,0.25) !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 28px rgba(255,68,68,0.38) !important;
        filter: brightness(1.07) !important;
    }
    .stButton > button:active { transform: translateY(0) !important; }

    .dismiss-col .stButton > button {
        background: #e8e8e8 !important;
        border: 1px solid #ddd !important;
        border-radius: 8px !important;
        color: #666 !important;
        font-size: 13px !important;
        padding: 6px 10px !important;
        box-shadow: none !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 400 !important;
        width: auto !important;
    }

    .result-box {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 14px 18px;
        border-radius: 12px;
        font-family: 'Syne', sans-serif;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 1rem;
        letter-spacing: 0.02em;
        animation: slideDown 0.3s ease;
    }
    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-6px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .result-fraud {
        background: rgba(220,60,60,0.07);
        border: 1px solid rgba(220,60,60,0.22);
        color: #cc2222;
    }
    .result-safe {
        background: rgba(30,160,80,0.07);
        border: 1px solid rgba(30,160,80,0.22);
        color: #1a7a40;
    }
    .result-box .result-icon { font-size: 20px; }
    .result-box .result-text span {
        color: rgba(0,0,0,0.38);
        font-weight: 400;
        font-family: 'DM Sans', sans-serif;
        font-size: 12px;
        display: block;
        margin-bottom: 1px;
    }

    .error-box {
        background: rgba(255,60,60,0.07);
        border-left: 3px solid #ff4444;
        border-radius: 0 8px 8px 0;
        padding: 6px 12px;
        margin-top: 2px;
        margin-bottom: 4px;
        color: #cc2222;
        font-size: 12px;
        font-weight: 400;
    }

    .divider {
        border: none;
        border-top: 1px solid rgba(0,0,0,0.07);
        margin: 0.8rem 0;
    }

    [data-testid="stFileUploader"] {
        background: #fff !important;
        border: 2px dashed #d0d0d0 !important;
        border-radius: 14px !important;
        padding: 8px !important;
        transition: border-color 0.25s !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(220,60,60,0.35) !important;
        background: rgba(220,60,60,0.01) !important;
    }
    [data-testid="stFileUploadDropzone"] {
        background: transparent !important;
        border: none !important;
        padding: 20px 12px !important;
    }
    [data-testid="stFileUploadDropzone"] > div > div {
        color: #aaa !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
    }
    [data-testid="stFileUploadDropzone"] button {
        background: rgba(220,60,60,0.08) !important;
        border: 1px solid rgba(220,60,60,0.25) !important;
        border-radius: 8px !important;
        color: #cc3333 !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        padding: 5px 14px !important;
        box-shadow: none !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
    }
    [data-testid="stFileUploadDropzone"] button:hover {
        background: rgba(220,60,60,0.14) !important;
        transform: none !important;
    }

    .success-badge {
        background: rgba(30,160,80,0.07);
        border: 1px solid rgba(30,160,80,0.18);
        border-radius: 10px;
        padding: 11px 16px;
        color: #1a7a40;
        font-size: 14px;
        font-weight: 500;
        font-family: 'DM Sans', sans-serif;
    }

    .stat-card {
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin-bottom: 0.8rem;
    }
    .stat-card.fraud {
        background: rgba(220,60,60,0.07);
        border: 1px solid rgba(220,60,60,0.14);
    }
    .stat-card.safe {
        background: rgba(30,160,80,0.07);
        border: 1px solid rgba(30,160,80,0.14);
    }
    .stat-card .stat-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .stat-card.fraud .stat-label { color: #cc2222; }
    .stat-card.safe  .stat-label { color: #1a7a40; }
    .stat-card .stat-value {
        font-family: 'Syne', sans-serif;
        font-size: 28px;
        font-weight: 800;
        line-height: 1;
    }
    .stat-card.fraud .stat-value { color: #cc2222; }
    .stat-card.safe  .stat-value { color: #1a7a40; }

    .stDataFrame { border-radius: 12px; overflow: hidden; }
    iframe { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# --- 4. HEADER ---
st.markdown("""
    <div class="app-header">
        <div class="badge">&nbsp; AI-Powered Security</div>
        <h1>Fraud Detection</h1>
        <p>Real-time transaction analysis powered by machine learning</p>
    </div>
""", unsafe_allow_html=True)

# --- 5. SESSION STATE ---
if "show_result"       not in st.session_state: st.session_state.show_result       = False
if "result_label"      not in st.session_state: st.session_state.result_label      = ""
if "result_fraud"      not in st.session_state: st.session_state.result_fraud      = False
if "predicted_file"    not in st.session_state: st.session_state.predicted_file    = None
if "show_success"      not in st.session_state: st.session_state.show_success      = False
if "uploaded_bytes"    not in st.session_state: st.session_state.uploaded_bytes    = None
if "uploaded_filename" not in st.session_state: st.session_state.uploaded_filename = None

# --- 6. BATCH PREDICTION (cached — same file won't be reprocessed) ---
@st.cache_data
def run_batch(_imputer, _transformer, _model, colNumber, bounds, file_bytes):
    file = pd.read_csv(io.BytesIO(file_bytes))
    for col in file.columns:
        if file[col].dtype == "object":
            file[col] = file[col].str.replace("'",  "", regex=False).str.strip()
            file[col] = file[col].str.replace("es_", "", regex=False).str.strip()
    file["account age"] = pd.to_numeric(file["account age"], errors="coerce")
    file[colNumber]     = _imputer.transform(file[colNumber])
    for col in colNumber:
        file[col] = file[col].clip(lower=bounds[col]["lower"], upper=bounds[col]["upper"])
    transformed        = _transformer.transform(file)
    preds              = _model.predict(transformed)
    file["Prediction"] = ["Fraud" if x == 1 else "Not Fraud" for x in preds]
    return file

tab1, tab2 = st.tabs(["  Single Transaction  ", "  Batch Upload  "])

# ════════════════════════════════════════════════════════
with tab1:

    def showError(placeholder, msg):
        placeholder.markdown(f'<div class="error-box">⚠ &nbsp;{msg}</div>', unsafe_allow_html=True)

    def clearAll():
        errorAge.empty(); errorGender.empty(); errorMerchant.empty()
        errorMerchantCode.empty(); errorCategory.empty(); errorAmount.empty()

    # ── Result banner (rendered without rerun) ────────
    result_placeholder = st.empty()

    if st.session_state.show_result:
        css_class = "result-fraud" if st.session_state.result_fraud else "result-safe"
        icon      = "🚨" if st.session_state.result_fraud else "✅"
        label     = st.session_state.result_label
        col1, col2 = st.columns([11, 1])
        with col1:
            st.markdown(f"""
                <div class="result-box {css_class}">
                    <span class="result-icon">{icon}</span>
                    <div class="result-text">
                        <span>Prediction Result</span>
                        {label}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="dismiss-col">', unsafe_allow_html=True)
            if st.button("✕", key="dismiss_result"):
                st.session_state.show_result  = False
                st.session_state.age          = 0
                st.session_state.merchant     = ""
                st.session_state.merchantcode = ""
                st.session_state.amount       = 0
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Account Information ───────────────────────────
    st.markdown('<div class="form-section-title">Account Information</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        errorAge = st.empty()
        age      = st.number_input("Account Age (months)", min_value=0, key="age")
    with col_b:
        errorGender = st.empty()
        gender      = st.selectbox("Gender", options=["Male", "Female"], key="gender")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Merchant Details ─────────────────────────────
    st.markdown('<div class="form-section-title">Merchant Details</div>', unsafe_allow_html=True)
    col_c, col_d = st.columns(2)
    with col_c:
        errorMerchant = st.empty()
        merchant      = st.text_input("Merchant Name", placeholder="e.g. Amazon, Shopify", key="merchant")
    with col_d:
        errorMerchantCode = st.empty()
        merchantcode      = st.text_input("Merchant Zip Code", placeholder="e.g. 10001", key="merchantcode")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Transaction Details ──────────────────────────
    st.markdown('<div class="form-section-title">Transaction Details</div>', unsafe_allow_html=True)

    categories = [
        "transportation", "food", "health", "wellnessandbeauty",
        "fashion", "barsandrestaurants", "hyper", "sportsandtoys",
        "tech", "home", "hotelservices", "otherservices",
        "contents", "travel", "leisure"
    ]

    col_e, col_f = st.columns(2)
    with col_e:
        errorCategory = st.empty()
        category      = st.selectbox(
            "Category",
            options=["-- Select Category --"] + categories,
            key="category"
        )
    with col_f:
        errorAmount = st.empty()
        amount      = st.number_input("Amount ($)", min_value=0, key="amount")

    st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)
    buttonForm = st.button("🔍  Analyze Transaction", use_container_width=True)

    if buttonForm:
        clearAll()
        st.session_state.show_result = False
        has_error = False

        if age <= 0:
            showError(errorAge, "Please enter a valid Account Age")
            has_error = True
        if merchant == "" or merchant.isnumeric():
            showError(errorMerchant, "Please enter a valid Merchant Name")
            has_error = True
        if merchantcode == "" or not merchantcode.isnumeric():
            showError(errorMerchantCode, "Zip Code must be numeric")
            has_error = True
        if category == "-- Select Category --":
            showError(errorCategory, "Please select a Transaction Category")
            has_error = True
        if amount <= 0:
            showError(errorAmount, "Please enter a valid Amount")
            has_error = True

        if not has_error:
            data = pd.DataFrame({
                "account age" : [age],
                "gender"      : [gender],
                "merchant"    : [merchant],
                "zipMerchant" : [merchantcode],
                "category"    : [category],
                "amount"      : [amount]
            })
            data[colNumber] = imputer.transform(data[colNumber])
            for col in colNumber:
                data[col] = data[col].clip(
                    lower=bounds[col]["lower"],
                    upper=bounds[col]["upper"]
                )
            transformData = Transformer.transform(data)
            predict       = model.predict(transformData)

            # FIX: store result and rerun only once (to show banner at top)
            st.session_state.show_result  = True
            st.session_state.result_label = "Fraud" if predict[0] == 1 else "Not Fraud"
            st.session_state.result_fraud = (predict[0] == 1)
            st.rerun()

# ════════════════════════════════════════════════════════
with tab2:

    st.markdown("""
        <div style="text-align:center; margin-bottom:1rem;">
            <p style="font-family:'Syne',sans-serif;font-size:17px;font-weight:700;color:#333;margin-bottom:4px;">
                Batch Transaction Analysis
            </p>
            <p style="font-size:13px;color:#999;margin:0;">
                Upload a CSV to scan multiple transactions at once
            </p>
        </div>
    """, unsafe_allow_html=True)

    fileUpload = st.file_uploader("Upload CSV file", type="csv", key="fraud_uploader")

    if fileUpload is not None:
        st.session_state.uploaded_bytes    = fileUpload.read()
        st.session_state.uploaded_filename = fileUpload.name

    st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)
    fileButton = st.button("Run Batch Analysis", use_container_width=True, key="predict_file_btn")

    if fileButton:
        if st.session_state.uploaded_bytes is None:
            st.markdown('<div class="error-box">⚠ &nbsp;Please upload a CSV file before running analysis</div>', unsafe_allow_html=True)
        else:
            try:
                # FIX: cached — won't reprocess same file on re-click
                file = run_batch(imputer, Transformer, model, colNumber, bounds, st.session_state.uploaded_bytes)
                st.session_state.predicted_file = file
                st.session_state.show_success   = True
                st.rerun()
            except Exception as e:
                st.error(f"Error processing file: {e}")

    # ── Results ──────────────────────────────────────
    if st.session_state.predicted_file is not None and st.session_state.show_success:
        fraud    = (st.session_state.predicted_file["Prediction"] == "Fraud").sum()
        notFraud = (st.session_state.predicted_file["Prediction"] == "Not Fraud").sum()
        total    = fraud + notFraud
        pct      = round((fraud / total * 100), 1) if total > 0 else 0

        st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)

        c1, c2 = st.columns([0.88, 0.12])
        with c1:
            st.markdown('<div class="success-badge">✅ &nbsp; Analysis complete — predictions ready</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="dismiss-col">', unsafe_allow_html=True)
            if st.button("✖", key="clear_success"):
                st.session_state.show_success   = False
                st.session_state.predicted_file = None
                st.session_state.uploaded_bytes = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
                <div class="stat-card fraud">
                    <div class="stat-label">🚨 Fraud</div>
                    <div class="stat-value">{fraud:,}</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="stat-card safe">
                    <div class="stat-label">Legitimate</div>
                    <div class="stat-value">{notFraud:,}</div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="stat-card fraud">
                    <div class="stat-label">Fraud Rate</div>
                    <div class="stat-value">{pct}%</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)

        MAX_STYLED_ROWS = 10000

        # FIX: applymap on single column instead of apply across all rows (~10x faster)
        def highlight_prediction(val):
            return 'background-color: #fde8e8' if val == 'Fraud' else 'background-color: #e8f8ee'

        if len(st.session_state.predicted_file) <= MAX_STYLED_ROWS:
            st.dataframe(
                st.session_state.predicted_file.style.applymap(
                    highlight_prediction, subset=["Prediction"]
                ),
                use_container_width=True
            )
        else:
            st.info(f"File has {len(st.session_state.predicted_file):,} rows — row highlighting disabled for performance.")
            st.dataframe(st.session_state.predicted_file, use_container_width=True)