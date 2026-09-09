import re
import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predatory Conference Detector",
    page_icon="🔍",
    layout="wide",
)

# ── Load model artefacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load("best_model.pkl")
    tfidf    = joblib.load("tfidf_vectorizer.pkl")
    num_cols = joblib.load("numeric_features.pkl")
    return model, tfidf, num_cols

@st.cache_resource
def load_shap():
    try:
        return joblib.load("shap_explainer.pkl")
    except:
        return None

model, tfidf, NUM_COLS = load_model()
shap_data = load_shap()

# ── Feature extraction helpers ────────────────────────────────────────────────
FREE_EMAIL_DOMAINS = {
    "gmail.com","yahoo.com","hotmail.com","outlook.com",
    "live.com","aol.com","mail.com","ymail.com",
}
FEE_PATTERNS = [
    r"\$\s*\d+", r"USD\s*\d+", r"publication\s+fee",
    r"submission\s+fee", r"registration\s+fee", r"author\s+fee",
    r"article\s+processing\s+charge",
]
REVIEW_PATTERNS = [
    r"\d+\s*(hours?|days?)\s*(peer\s*)?review",
    r"rapid\s+review", r"fast[- ]track", r"quick\s+review",
    r"immediate\s+(publication|acceptance)",
]
INDEXING_PATTERNS = [
    r"impact\s+factor", r"indexed\s+in", r"thomson\s+reuters",
    r"web\s+of\s+science", r"scopus", r"pubmed",
    r"isi\s+(indexed|listed)", r"cite\s*score",
]
SCOPE_VAGUE = [
    "all areas","all fields","multidisciplinary","broad scope",
    "wide range of topics","all disciplines","any topic",
]
URGENCY_PHRASES = [
    "submit now","limited slots","deadline extended",
    "call for papers","invitation to submit",
    "fast publication","guaranteed publication",
]

def count_hits(text, patterns):
    t = text.lower()
    return sum(1 for p in patterns if re.search(p, t))

def extract_features_from_text(text, url=""):
    emails     = re.findall(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    email_doms = {e.split("@")[-1].lower() for e in emails}
    t_lower    = text.lower()
    # FIX 1: correctly detect HTTP vs HTTPS from the actual URL
    is_http = int(url.startswith("http://") and not url.startswith("https://"))
    return {
        "has_free_email":          int(bool(email_doms & FREE_EMAIL_DOMAINS)),
        "fee_mention_count":       count_hits(text, FEE_PATTERNS),
        "review_claim_count":      count_hits(text, REVIEW_PATTERNS),
        "indexing_claim_count":    count_hits(text, INDEXING_PATTERNS),
        "vague_scope_count":       sum(1 for p in SCOPE_VAGUE if p in t_lower),
        "urgency_phrase_count":    sum(1 for p in URGENCY_PHRASES if p in t_lower),
        "has_impact_factor_claim": int(bool(re.search(r"impact\s+factor", text, re.I))),
        "has_rapid_review_claim":  int(bool(re.search(r"rapid\s+review|fast[- ]track", text, re.I))),
        "domain_uses_http":        is_http,
        "fetch_success":           1,
        "has_other_fees":          int(bool(re.search(r"other\s+fee|handling\s+fee", text, re.I))),
        "plagiarism_check":        int(bool(re.search(r"plagiarism", text, re.I))),
    }

def features_from_manual(answers):
    """Build feature dict from user's manual answers when URL scraping fails."""
    return {
        "has_free_email":          int(answers.get("free_email", False)),
        "fee_mention_count":       int(answers.get("fee_count", 0)),
        "review_claim_count":      int(answers.get("rapid_review", False)),
        "indexing_claim_count":    int(answers.get("indexing_claims", 0)),
        "vague_scope_count":       int(answers.get("vague_scope", False)),
        "urgency_phrase_count":    int(answers.get("urgency", False)),
        "has_impact_factor_claim": int(answers.get("impact_factor", False)),
        "has_rapid_review_claim":  int(answers.get("rapid_review", False)),
        "domain_uses_http":        int(answers.get("uses_http", False)),
        "fetch_success":           0,   # scraping failed
        "has_other_fees":          int(answers.get("other_fees", False)),
        "plagiarism_check":        int(answers.get("plagiarism", False)),
    }

def scrape_url(url):
    """
    Returns (text, final_url, error_message)
    final_url is the URL after any redirects — used for accurate HTTP/HTTPS detection.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=12, allow_redirects=True)
        resp.raise_for_status()
        # FIX 1: use final URL after redirect for accurate HTTP detection
        final_url = resp.url
        soup      = BeautifulSoup(resp.text, "html.parser")
        text      = soup.get_text(separator=" ", strip=True)[:8000]
        return text, final_url, None
    except Exception as e:
        return "", url, str(e)

def predict(text, url=""):
    feats     = extract_features_from_text(text, url)
    num_vec   = np.array([[feats.get(c, 0) for c in NUM_COLS]])
    tfidf_vec = tfidf.transform([text]).toarray()
    X         = np.hstack([num_vec, tfidf_vec])
    prob      = model.predict_proba(X)[0][1]
    pred      = int(prob >= 0.5)
    return pred, prob, feats

def predict_from_features(feats):
    """Predict using manually provided features (no text available)."""
    num_vec   = np.array([[feats.get(c, 0) for c in NUM_COLS]])
    tfidf_vec = tfidf.transform([""]).toarray()   # empty text
    X         = np.hstack([num_vec, tfidf_vec])
    prob      = model.predict_proba(X)[0][1]
    pred      = int(prob >= 0.5)
    return pred, prob

def show_result(pred, prob, feats):
    # ── Verdict banner ────────────────────────────────────────────────────────
    pct = prob * 100
    if prob >= 0.75:
        st.error(f"🚨 HIGH RISK — This appears to be a **Predatory** conference")
        bar_color = "#FF4B4B"
    elif prob >= 0.50:
        st.warning(f"⚠️ MEDIUM RISK — This conference shows **suspicious** signals")
        bar_color = "#FFA500"
    elif prob >= 0.25:
        st.info(f"🔵 LOW RISK — This conference appears mostly **legitimate**, with minor concerns")
        bar_color = "#4B9EFF"
    else:
        st.success(f"✅ VERY LOW RISK — This conference appears **Legitimate**")
        bar_color = "#00C853"

    # ── Probability meter ─────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="margin: 12px 0 6px;">
        <div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:4px;">
            <span style="color:#aaa">Legitimate</span>
            <span style="font-weight:600; font-size:15px;">Predatory Probability: {pct:.1f}%</span>
            <span style="color:#aaa">Predatory</span>
        </div>
        <div style="background:#2a2a2a; border-radius:8px; height:18px; overflow:hidden;">
            <div style="width:{pct:.1f}%; background:{bar_color}; height:100%; border-radius:8px;
                        transition: width 0.5s ease;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:11px; color:#888; margin-top:3px;">
            <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Flag definitions: (label, value, what_it_means, bad_when) ────────────
    flag_defs = [
        ("Contact Email",        feats["has_free_email"],
         "Uses free email (Gmail/Yahoo/Hotmail) as official contact",
         "Uses professional institutional email",
         True),
        ("Publication Fees",     feats["fee_mention_count"],
         f"Fee-related language found {feats['fee_mention_count']} time(s) on the page",
         "No excessive fee emphasis found",
         True),
        ("Peer Review Claims",   feats["has_rapid_review_claim"],
         "Claims unusually rapid review (24–48 hrs / fast-track)",
         "No suspicious rapid review claims",
         True),
        ("Impact Factor",        feats["has_impact_factor_claim"],
         "Claims to have an Impact Factor (common fake claim)",
         "No unverified impact factor claims",
         True),
        ("Scope of Topics",      feats["vague_scope_count"],
         f"Uses vague scope language ({feats['vague_scope_count']} instance(s)) — 'all fields', 'all disciplines'",
         "Clearly defined specific scope",
         True),
        ("Urgency Language",     feats["urgency_phrase_count"],
         f"Uses urgency phrases ({feats['urgency_phrase_count']} instance(s)) — 'submit now', 'guaranteed publication'",
         "No pressure or urgency language detected",
         True),
        ("Indexing Claims",      feats["indexing_claim_count"],
         f"Makes {feats['indexing_claim_count']} indexing claim(s) — verify these independently",
         "No excessive unverified indexing claims",
         True),
        ("Website Security",     feats["domain_uses_http"],
         "Website uses plain HTTP (no SSL/HTTPS encryption)",
         "Website uses HTTPS (secure)",
         True),
        ("Plagiarism Policy",    int(feats["plagiarism_check"] == 0),
         "No mention of plagiarism screening policy found",
         "Mentions plagiarism screening policy",
         True),
    ]

    # ── Separate into red flags vs safe signals ───────────────────────────────
    red_flags = [(lbl, msg, detail) for lbl, val, msg, detail, _ in flag_defs if val]
    safe_ok   = [(lbl, detail) for lbl, val, msg, detail, _ in flag_defs if not val]

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"#### 🚩 Red Flags Found ({len(red_flags)})")
        if red_flags:
            for lbl, msg, _ in red_flags:
                st.markdown(f"""
                <div style="background:#3d1f1f; border-left:4px solid #FF4B4B;
                            border-radius:6px; padding:10px 14px; margin-bottom:8px;">
                    <div style="font-weight:600; color:#FF4B4B; font-size:13px;">⛔ {lbl}</div>
                    <div style="color:#e0b0b0; font-size:12px; margin-top:3px;">{msg}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No red flags detected!")

    with col_b:
        st.markdown(f"#### ✅ Safe Signals ({len(safe_ok)})")
        if safe_ok:
            for lbl, detail in safe_ok:
                st.markdown(f"""
                <div style="background:#1a3d2b; border-left:4px solid #00C853;
                            border-radius:6px; padding:10px 14px; margin-bottom:8px;">
                    <div style="font-weight:600; color:#00C853; font-size:13px;">✔ {lbl}</div>
                    <div style="color:#a0d0b0; font-size:12px; margin-top:3px;">{detail}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("No safe signals detected.")

    # ── Summary score card — NOW DRIVEN BY THE MODEL'S PROBABILITY ────────────
    # (previously this was a flat count of 9 flags, unrelated to the model's
    #  actual prediction — that caused the 96% predatory + "7/9 safe" contradiction)
    st.markdown("---")
    total = len(flag_defs)
    score = len(safe_ok)

    # Risk score now matches the model verdict, not an independent flag count
    risk_pct = pct  # same probability used for the verdict banner above

    if risk_pct >= 75:
        risk_color_ = "#FF4B4B"
        risk_label  = "High Risk"
    elif risk_pct >= 50:
        risk_color_ = "#FFA500"
        risk_label  = "Medium Risk"
    elif risk_pct >= 25:
        risk_color_ = "#4B9EFF"
        risk_label  = "Low Risk"
    else:
        risk_color_ = "#00C853"
        risk_label  = "Very Low Risk"

    st.markdown(f"""
    <div style="background:#1e1e2e; border-radius:10px; padding:14px 18px;
                display:flex; align-items:center; justify-content:space-between;">
        <div>
            <div style="font-size:13px; color:#aaa;">Model Risk Score</div>
            <div style="font-size:22px; font-weight:700; color:{risk_color_}">
                {risk_pct:.0f}% — {risk_label}
            </div>
            <div style="font-size:12px; color:#888;">based on full ML model prediction</div>
        </div>
        <div style="text-align:right;">
            <div style="font-size:13px; color:#aaa;">Flag Count</div>
            <div style="font-size:22px; font-weight:700; color:#FF4B4B;">
                {len(red_flags)} / {total}
            </div>
            <div style="font-size:12px; color:#888;">individual signals flagged</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Explicitly explain the difference so it's never confusing again
    if len(red_flags) <= total // 2 and risk_pct >= 50:
        st.caption(
            f"ℹ️ Note: only {len(red_flags)} of {total} individual signals were flagged, "
            f"but the ML model weighs text patterns and feature combinations together — "
            f"not a simple flag count — which is why the overall risk is {risk_label.lower()} "
            f"({risk_pct:.0f}%). A small number of strong indicators (e.g. fake indexing claims "
            f"or urgency language) can outweigh several 'clean' individual signals."
        )

# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center; padding: 20px 0 10px;">
    <h1 style="font-size:2.2rem; font-weight:700; margin-bottom:6px;">
        🔍 Predatory Conference Detector
    </h1>
    <p style="color:#aaa; font-size:15px; margin:0;">
        AI-powered detection of Predatory Conferences using Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔎 Single Check", "📂 Batch Upload", "📊 Model Performance", "ℹ️ How It Works"
])

# ── TAB 1: Single checker ─────────────────────────────────────────────────────
with tab1:
    st.markdown("<h3 style='text-align:center;'>Check a Conference</h3>", unsafe_allow_html=True)

    # Centre the input using padding columns
    _, centre_col, _ = st.columns([1, 3, 1])
    with centre_col:
        input_mode = st.radio("Input method", ["🌐 Enter URL", "📝 Paste CFP / About text"],
                              horizontal=True, label_visibility="collapsed")
        st.markdown("<div style='text-align:center; font-size:13px; color:#aaa; margin-bottom:8px;'>Choose input method</div>", unsafe_allow_html=True)

        if input_mode == "🌐 Enter URL":
            url_input = st.text_input(
                "Conference URL",
                placeholder="https://example-conference.org",
                label_visibility="collapsed"
            )
            st.markdown("<div style='text-align:center; font-size:12px; color:#aaa; margin-top:-10px; margin-bottom:8px;'>Enter the conference website URL</div>", unsafe_allow_html=True)

            btn_col1, btn_col2, btn_col3 = st.columns([2,2,2])
            with btn_col2:
                analyse_clicked = st.button("🔍 Analyse URL", type="primary", use_container_width=True)

            if analyse_clicked:
                if url_input.strip():
                    with st.spinner("Fetching website ..."):
                        text, final_url, err = scrape_url(url_input.strip())

                    if err or not text.strip() or len(text.strip()) < 150:
                        st.session_state["scrape_failed"] = True
                        st.session_state["failed_url"]    = url_input.strip()
                        st.session_state.pop("result", None)
                    else:
                        st.session_state["scrape_failed"] = False
                        pred, prob, feats = predict(text, final_url)
                        st.session_state["result"] = (pred, prob, feats)
                else:
                    st.warning("Please enter a URL.")

            # ── Manual questionnaire shown when scraping fails ─────────────────
            if st.session_state.get("scrape_failed"):
                failed_url = st.session_state.get("failed_url", "")
                st.warning(
                    f"⚠️ The URL **{failed_url}** could not be scraped "
                    f"(site may be offline, blocked, or returning a blank page).\n\n"
                    f"Please answer the questions below so we can still analyse it:"
                )
                st.markdown("---")
                st.markdown("#### 📋 Manual Feature Entry")

                with st.form("manual_form"):
                    free_email    = st.radio("Does the conference use a free email (Gmail, Yahoo, Hotmail)?",
                                             ["No", "Yes"], horizontal=True)
                    uses_http     = st.radio("Does the website use plain HTTP (not HTTPS)?",
                                             ["No", "Yes"], horizontal=True)
                    rapid_review  = st.radio("Does it claim rapid/fast-track peer review (e.g. within 24–48 hours)?",
                                             ["No", "Yes"], horizontal=True)
                    impact_factor = st.radio("Does it claim to have an Impact Factor?",
                                             ["No", "Yes"], horizontal=True)
                    vague_scope   = st.radio("Is the scope vague or covers 'all fields / all disciplines'?",
                                             ["No", "Yes"], horizontal=True)
                    urgency       = st.radio("Does it use urgency language like 'Submit Now' or 'Guaranteed Publication'?",
                                             ["No", "Yes"], horizontal=True)
                    plagiarism    = st.radio("Does it mention plagiarism screening?",
                                             ["No", "Yes"], horizontal=True)
                    fee_count     = st.slider("How many times are publication/registration fees mentioned?",
                                              0, 10, 0)
                    indexing_cnt  = st.slider("How many indexing databases does it claim to be listed in?",
                                              0, 10, 0)
                    other_fees    = st.radio("Are there other hidden fees mentioned?",
                                             ["No", "Yes"], horizontal=True)

                    submitted = st.form_submit_button("Analyse with Manual Answers", type="primary")

                if submitted:
                    answers = {
                        "free_email":     free_email == "Yes",
                        "uses_http":      uses_http == "Yes",
                        "rapid_review":   rapid_review == "Yes",
                        "impact_factor":  impact_factor == "Yes",
                        "vague_scope":    vague_scope == "Yes",
                        "urgency":        urgency == "Yes",
                        "plagiarism":     plagiarism == "Yes",
                        "fee_count":      fee_count,
                        "indexing_claims":indexing_cnt,
                        "other_fees":     other_fees == "Yes",
                    }
                    feats = features_from_manual(answers)
                    pred, prob = predict_from_features(feats)
                    st.session_state["result"] = (pred, prob, feats)
                    st.session_state["scrape_failed"] = False

        else:
            text_input = st.text_area(
                "Paste CFP / About page text here",
                height=220,
                placeholder="Paste the full call-for-papers or about page text ...",
                label_visibility="collapsed"
            )
            st.markdown("<div style='text-align:center; font-size:12px; color:#aaa; margin-top:-10px; margin-bottom:8px;'>Paste the full CFP / About page text from the conference website</div>", unsafe_allow_html=True)

            btn_col1, btn_col2, btn_col3 = st.columns([2,2,2])
            with btn_col2:
                analyse_text_clicked = st.button("🔍 Analyse Text", type="primary", use_container_width=True)

            if analyse_text_clicked:
                if text_input.strip():
                    pred, prob, feats = predict(text_input.strip())
                    st.session_state["result"] = (pred, prob, feats)
                    st.session_state["scrape_failed"] = False
                else:
                    st.warning("Please paste some text.")

    # Results shown full-width below input — centred
    if "result" in st.session_state:
        st.markdown("---")
        pred, prob, feats = st.session_state["result"]
        show_result(pred, prob, feats)

# ── TAB 2: Batch Upload ───────────────────────────────────────────────────────
with tab2:
    st.markdown("<h3 style='text-align:center;'>Batch Classification</h3>", unsafe_allow_html=True)
    st.info("Upload a CSV with a column named **url** or **text**. The tool will classify each conference.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.write(f"Loaded {len(df_up)} rows. Columns: {list(df_up.columns)}")

        mode_col = None
        if "url" in df_up.columns:
            mode_col = "url"
        elif "text" in df_up.columns:
            mode_col = "text"
        else:
            st.error("CSV must have a column named 'url' or 'text'.")

        if mode_col and st.button("Run Batch Classification", type="primary"):
            preds, probs, statuses = [], [], []
            progress = st.progress(0)
            status_box = st.empty()

            for i, row in df_up.iterrows():
                val = str(row[mode_col])
                status_box.info(f"Processing {i+1}/{len(df_up)}: {val[:60]}")

                if mode_col == "url":
                    text, final_url, err = scrape_url(val)
                    if err or not text.strip() or len(text.strip()) < 150:
                        preds.append("Unable to scrape")
                        probs.append("-")
                        statuses.append("Scrape failed")
                        progress.progress((i + 1) / len(df_up))
                        continue
                    pred, prob, _ = predict(text, final_url)
                else:
                    text = val
                    pred, prob, _ = predict(text)

                preds.append("Predatory" if pred else "Legitimate")
                probs.append(round(prob * 100, 1))
                statuses.append("OK")
                progress.progress((i + 1) / len(df_up))

            status_box.empty()
            df_up["prediction"]       = preds
            df_up["predatory_prob_%"] = probs
            df_up["status"]           = statuses
            st.dataframe(df_up)

            csv_out = df_up.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results CSV", csv_out,
                               "batch_results.csv", "text/csv")

# ── TAB 3: Model Performance ──────────────────────────────────────────────────
with tab3:
    st.markdown("<h3 style='text-align:center;'>Model Comparison & Performance</h3>", unsafe_allow_html=True)
    try:
        results_df = pd.read_csv("model_results.csv")
        st.dataframe(results_df.style.highlight_max(
            subset=["accuracy","f1_macro","auc_roc","cv_f1"], color="#d4f4dd"
        ), use_container_width=True)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, metric in zip(axes, ["accuracy", "f1_macro", "auc_roc"]):
            ax.bar(results_df["model"], results_df[metric],
                   color=["#4C72B0","#DD8452","#55A868","#C44E52"])
            ax.set_title(metric.replace("_", " ").title())
            ax.set_ylim(0.5, 1.0)
            ax.axhline(0.95, color="red", linestyle="--", linewidth=1, label="95% target")
            ax.legend(fontsize=8)
            ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        st.pyplot(fig)
    except FileNotFoundError:
        st.warning("model_results.csv not found. Run step3_train_model.py first.")

    st.markdown("#### SHAP Feature Importance")
    try:
        st.image("shap_summary.png", caption="Top features driving predatory classification")
    except:
        st.info("shap_summary.png not found. Will appear after model training.")

# ── TAB 4: How it works ───────────────────────────────────────────────────────
with tab4:
    st.markdown("<h3 style='text-align:center;'>How the Detector Works</h3>", unsafe_allow_html=True)
    st.markdown("""
**Data sources:**
- 🔴 **Predatory** class: Beall's List (beallslist.net) — ~1,646 clean scraped conference/publisher entries
- 🟢 **Legitimate** class: DOAJ — ~1,646 sampled entries (used as legitimate baseline)

**Features used:**
| Feature | Description |
|---|---|
| TF-IDF text (3,000 n-grams) | Language patterns in CFP/About text |
| Free email contact | Gmail/Yahoo contact = red flag |
| Fee mention count | Unusual emphasis on fees |
| Rapid review claims | "24-hour review", "fast-track" etc. |
| Indexing claim count | Fake Scopus/WoS claims |
| Vague scope language | "All fields", "multidisciplinary" |
| Urgency phrases | "Submit now", "guaranteed publication" |
| Domain uses HTTP | No HTTPS = red flag |
| Plagiarism screening | Absence of plagiarism policy = red flag |

**Model pipeline:**
1. TF-IDF vectorization of website text
2. Numeric feature extraction (12 features)
3. SMOTE oversampling for class balance
4. Stacking ensemble: Random Forest + XGBoost → Logistic meta-learner
5. 5-fold stratified cross-validation

**When URL scraping fails:**
The app asks 10 manual questions about the conference to still produce a prediction using numeric features only.

**Accuracy target:** > 95% F1-macro on hold-out test set
    """)