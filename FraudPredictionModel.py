import pickle as pkl
import streamlit as st
import pandas as pd
import numpy as np

# --- 1. LOAD MODELS ---
model       = pkl.load(open('model.pkl', 'rb'))
Transformer = pkl.load(open('Transformer.pkl', 'rb'))
colText     = pkl.load(open('colText.pkl', 'rb'))

# --- 2. CSS ---
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #fff5f5, #ffe0e0, #fff0e0) !important;
        }
        .upload-card {
            border: 2px dashed #ccc;
            border-radius: 12px;
            padding: 40px 20px;
            text-align: center;
            background-color: #fafafa;
            transition: border-color 0.2s, background-color 0.2s;
            cursor: pointer;
            margin-bottom: 0px;
        }
        .upload-card:hover, .upload-card.active {
            border-color: #ff4b4b;
            background-color: #fff5f5;
        }
        .upload-icon  { font-size: 40px; margin-bottom: 10px; }
        .upload-title { font-size: 16px; font-weight: 600; color: #333; margin-bottom: 6px; }
        .upload-sub   { font-size: 13px; color: #888; }
        .success-badge {
            background-color: #f0fff4;
            border-left: 4px solid #28a745;
            border-radius: 6px;
            padding: 12px 16px;
            color: #155724;
            font-size: 15px;
            font-weight: 500;
        }
        .error-box {
            background-color: #fff0f0;
            border-left: 4px solid #ff4b4b;
            border-radius: 4px;
            padding: 8px 12px;
            margin-bottom: 10px;
            color: #cc0000;
            font-size: 14px;
        }
        .result-box {
            display: flex;
            align-items: center;
            padding: 12px 16px;
            border-radius: 6px;
            font-size: 15px;
            font-weight: 500;
            margin-bottom: 16px;
        }
        .result-fraud {
            background-color: #fff0f0;
            border-left: 4px solid #ff4b4b;
            color: #cc0000;
        }
        .result-safe {
            background-color: #f0fff4;
            border-left: 4px solid #28a745;
            color: #155724;
        }
    </style>
""", unsafe_allow_html=True)

# --- 3. TITLE ---
st.markdown("""
    <h1 style="
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(to right, #ff4b4b, #f9a825);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    ">Fraud Prediction Model</h1>
""", unsafe_allow_html=True)

# --- 4. SESSION STATE ---
if "show_result"     not in st.session_state: st.session_state.show_result     = False
if "result_label"    not in st.session_state: st.session_state.result_label    = ""
if "result_fraud"    not in st.session_state: st.session_state.result_fraud    = False
if "predicted_file"  not in st.session_state: st.session_state.predicted_file  = None
if "show_success"    not in st.session_state: st.session_state.show_success    = False

tab1, tab2 = st.tabs(["Fraud Prediction Form", "Fraud Prediction Via Upload"])

# ════════════════════════════════════════════════
with tab1:

    def showError(placeholder, msg):
        placeholder.markdown(f'<div class="error-box">⚠️ {msg}</div>', unsafe_allow_html=True)

    def clearAll():
        errorAge.empty()
        errorMerchant.empty()
        errorAmount.empty()

    # ── Result at top ─────────────────────────
    if st.session_state.show_result:
        css_class = "result-fraud" if st.session_state.result_fraud else "result-safe"
        icon      = "🚨" if st.session_state.result_fraud else "✅"
        label     = st.session_state.result_label
        col1, col2 = st.columns([11, 1])
        with col1:
            st.markdown(f"""
                <div class="result-box {css_class}">
                    {icon} &nbsp; Prediction: <strong>{label}</strong>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("✕", key="dismiss_result"):
                st.session_state.show_result = False
                st.session_state.age         = 0
                st.session_state.merchant    = ""
                st.session_state.amount      = 0
                st.rerun()

    # ── Fields ────────────────────────────────
    errorAge = st.empty()
    age      = st.number_input("Enter Account Age", min_value=0, key="age")

    errorMerchant = st.empty()
    merchant      = st.text_input("Enter Merchant Name", key="merchant")

    errorAmount = st.empty()
    amount      = st.number_input("Enter Amount", min_value=0, key="amount")

    buttonForm = st.button("Predict", use_container_width=True)

    if buttonForm:
        clearAll()
        st.session_state.show_result = False

        if age <= 0:
            showError(errorAge, "Please enter valid Age")
        elif merchant == "" or merchant.isnumeric():
            showError(errorMerchant, "Please Enter Valid Value | Merchant")
        elif amount <= 0:
            showError(errorAmount, "Please Enter Transaction Amount")
        else:
            data              = pd.DataFrame({"age": [age], "merchant": [merchant], "amount": [amount]})
            transformData     = Transformer.transform(data)
            frameWithFeatures = pd.DataFrame(transformData, columns=Transformer.get_feature_names_out())
            predict           = model.predict(frameWithFeatures)
            label             = "Fraud" if predict[0] == 1 else "Not Fraud"

            st.session_state.show_result  = True
            st.session_state.result_label = label
            st.session_state.result_fraud = (predict[0] == 1)
            st.rerun()

# ════════════════════════════════════════════════
with tab2:

    st.markdown("""
        <div style="text-align:center;font-size:20px;font-weight:bold;color:#333;margin-bottom:8px;">
            Upload Transaction File
        </div>
        <div style="text-align:center;font-size:14px;color:#888;margin-bottom:16px;">
            Upload a CSV file to detect fraud across multiple transactions
        </div>
    """, unsafe_allow_html=True)

    # ── Dynamic upload card ───────────────────
    if "fraud_uploader" in st.session_state and st.session_state.fraud_uploader is not None:
        u_file    = st.session_state.fraud_uploader
        kb        = round(u_file.size / 1024, 1)
        size_info = f"{kb} KB" if kb < 1024 else f"{round(kb/1024,2)} MB"
        card_html = f"""
            <div class="upload-card" id="ucard">
                <div class="upload-icon">📄</div>
                <div class="upload-title">{u_file.name}</div>
                <div class="upload-sub">{size_info} • Ready to Predict</div>
            </div>
        """
    else:
        card_html = """
            <div class="upload-card" id="ucard">
                <div class="upload-icon">☁️</div>
                <div class="upload-title">Choose a file or drag &amp; drop it here</div>
                <div class="upload-sub">CSV format · up to 50 MB</div>
            </div>
        """

    st.markdown(card_html, unsafe_allow_html=True)

    fileUpload = st.file_uploader(
        "Upload CSV",
        type="csv",
        key="fraud_uploader",
        label_visibility="collapsed"
    )

    st.markdown("""
        <script>
        (function() {
            function init() {
                const card = document.getElementById('ucard');
                const zone = document.querySelector('[data-testid="stFileUploadDropzone"]');
                if (!card || !zone) { setTimeout(init, 150); return; }
                zone.addEventListener('dragenter', () => card.classList.add('active'));
                zone.addEventListener('dragover',  (e) => { e.preventDefault(); card.classList.add('active'); });
                zone.addEventListener('dragleave', () => card.classList.remove('active'));
                zone.addEventListener('drop',      () => setTimeout(() => card.classList.remove('active'), 250));
            }
            init();
        })();
        </script>
        <div style="margin-bottom: 16px;"></div>
    """, unsafe_allow_html=True)

    fileButton = st.button("Click to Predict", use_container_width=True, key="predict_file_btn")

    if fileButton:
        if fileUpload is None:
            st.markdown('<div class="error-box">⚠️ Please upload a CSV file before predicting</div>', unsafe_allow_html=True)
        else:
            try:
                file                = pd.read_csv(fileUpload)
                TransformFile       = Transformer.transform(file)
                withEncodedFeatures = pd.DataFrame(TransformFile, columns=Transformer.get_feature_names_out())
                predictFile         = model.predict(withEncodedFeatures)
                file['Prediction']  = ["Fraud" if x == 1 else "Not Fraud" for x in predictFile]

                st.session_state.predicted_file = file
                st.session_state.show_success   = True
                st.rerun()
            except Exception as e:
                st.error(f"Error processing file: {e}")

    # ── Results ───────────────────────────────
    if st.session_state.predicted_file is not None:
        if st.session_state.show_success:
            fraud    = (st.session_state.predicted_file["Prediction"] == "Fraud").sum()
            notFraud = (st.session_state.predicted_file["Prediction"] == "Not Fraud").sum()

            c1, c2 = st.columns([0.85, 0.15])
            with c1:
                st.markdown('<div class="success-badge">✅ Analysis Complete — Predictions ready!</div>', unsafe_allow_html=True)
            with c2:
                if st.button("✖", key="clear_success"):
                    st.session_state.show_success   = False
                    st.session_state.predicted_file = None
                    st.rerun()

            # ── Summary cards ─────────────────
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                    <div style="
                        background-color: #fff0f0;
                        border-left: 4px solid #ff4b4b;
                        border-radius: 8px;
                        padding: 16px;
                        text-align: center;
                        font-size: 18px;
                        font-weight: bold;
                        color: #cc0000;
                        margin-bottom: 16px;
                    ">🚨 Fraud<br>{fraud:,}</div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div style="
                        background-color: #f0fff4;
                        border-left: 4px solid #28a745;
                        border-radius: 8px;
                        padding: 16px;
                        text-align: center;
                        font-size: 18px;
                        font-weight: bold;
                        color: #155724;
                        margin-bottom: 16px;
                    ">✅ Not Fraud<br>{notFraud:,}</div>
                """, unsafe_allow_html=True)

            # ── Table ─────────────────────────
            MAX_STYLED_ROWS = 10000

            def highlightResult(row):
                if row['Prediction'] == 'Fraud':
                    return ['background-color: #f8d7da'] * len(row)
                else:
                    return ['background-color: #d4edda'] * len(row)

            if len(st.session_state.predicted_file) <= MAX_STYLED_ROWS:
                st.dataframe(
                    st.session_state.predicted_file.style.apply(highlightResult, axis=1),
                    use_container_width=True
                )
            else:
                st.info(f"File has {len(st.session_state.predicted_file):,} rows — row highlighting disabled for performance.")
                st.dataframe(
                    st.session_state.predicted_file,
                    use_container_width=True
                )