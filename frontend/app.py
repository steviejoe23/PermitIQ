"""
PermitIQ Frontend v3 — Demo-Ready
Streamlit app for Boston Zoning Intelligence
"""

import streamlit as st
import requests
import pandas as pd
import pydeck as pdk
import os
from html import escape as _html_escape


def esc(val):
    """Escape a value for safe HTML rendering."""
    return _html_escape(str(val))

# =========================
# CONFIG
# =========================

API_URL = os.environ.get("PERMITIQ_API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="PermitIQ — Boston Zoning Intelligence",
    page_icon="🏛️",
    layout="wide"
)

# =========================
# CUSTOM STYLING
# =========================

st.markdown("""
<style>
    /* Hide Streamlit branding for clean demo */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none !important;}
    header[data-testid="stHeader"] {display: none !important;}

    /* Global spacing */
    .main .block-container { padding-top: 1rem; }

    /* Typography */
    .hero-title {
        font-size: 52px; font-weight: 900; margin-bottom: 0;
        background: linear-gradient(135deg, #4a9eff, #8b5cf6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    .hero-subtitle { font-size: 16px; color: #777; margin-top: -8px; line-height: 1.4; }

    /* Probability display */
    .big-number { font-size: 72px; font-weight: 800; text-align: center; margin: 5px 0; line-height: 1; }
    .label-text { font-size: 20px; font-weight: 700; text-align: center; margin-top: 0; letter-spacing: 2px; }
    .confidence-text { text-align: center; color: #888; font-size: 14px; margin-top: 5px; }

    /* Cards */
    .factor-card {
        background: #1a1a2e; padding: 12px 16px; border-radius: 8px;
        margin: 6px 0; border-left: 4px solid #4a9eff; font-size: 14px;
    }
    .factor-positive { border-left-color: #00cc66; }
    .factor-negative { border-left-color: #ff4444; }
    .factor-neutral { border-left-color: #ffaa00; }
    .case-card {
        background: #1a1a2e; padding: 10px 14px; border-radius: 6px;
        margin: 5px 0; font-size: 13px; line-height: 1.5;
    }

    /* Search results */
    .search-result {
        background: linear-gradient(145deg, #1a1a2e, #16162a);
        padding: 16px 20px; border-radius: 10px;
        margin: 8px 0; border: 1px solid #2a2a4e; cursor: pointer;
        transition: border-color 0.2s, transform 0.2s;
    }
    .search-result:hover { border-color: #4a9eff; transform: translateY(-1px); }
    .search-addr { font-size: 17px; font-weight: 700; color: #fff; }
    .search-meta { color: #999; font-size: 13px; margin-top: 6px; line-height: 1.5; }

    /* Stats row */
    .stat-box {
        background: linear-gradient(145deg, #1a1a2e, #16162a);
        padding: 20px 16px; border-radius: 12px;
        text-align: center; border: 1px solid #2a2a4e;
        transition: transform 0.2s, border-color 0.2s;
    }
    .stat-box:hover { transform: translateY(-2px); border-color: #4a9eff; }
    .stat-number { font-size: 36px; font-weight: 800; color: #fff; }
    .stat-label { color: #888; font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 4px; }

    /* Disclaimer */
    .disclaimer { font-size: 11px; color: #555; font-style: italic; margin-top: 20px; }

    /* Info boxes */
    .info-row { display: flex; gap: 8px; margin: 4px 0; }
    .info-label { color: #888; font-size: 13px; min-width: 100px; }
    .info-value { font-size: 13px; font-weight: 600; }

    /* Scenario cards for What-If */
    .scenario-card {
        background: #1a1a2e; padding: 14px 16px; border-radius: 8px;
        margin: 6px 0; border: 1px solid #2a2a4e;
    }
    .scenario-positive { border-left: 4px solid #00cc66; }
    .scenario-negative { border-left: 4px solid #ff4444; }
    .scenario-neutral { border-left: 4px solid #888; }

    /* Confidence badge */
    .confidence-badge {
        display: inline-block; padding: 4px 12px; border-radius: 12px;
        font-size: 12px; font-weight: 600; text-transform: uppercase;
        letter-spacing: 1px;
    }
    .confidence-high { background: #00cc6620; color: #00cc66; }
    .confidence-medium { background: #ffaa0020; color: #ffaa00; }
    .confidence-low { background: #ff444420; color: #ff4444; }

    /* Prediction button */
    .stButton > button[kind="primary"] {
        transition: all 0.3s ease;
        font-weight: 700;
        letter-spacing: 0.5px;
        border-radius: 10px;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.4);
    }

    /* Form submit buttons */
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #4a9eff, #6366f1);
        color: white;
        border: none;
        font-weight: 600;
        border-radius: 8px;
        padding: 8px 24px;
        transition: all 0.2s;
    }
    .stFormSubmitButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(74, 158, 255, 0.3);
    }

    /* Sidebar demo buttons */
    section[data-testid="stSidebar"] .stButton > button {
        font-size: 13px;
        padding: 8px 12px;
        border: 1px solid #333;
        background: linear-gradient(145deg, #1a1a2e, #222244);
        border-radius: 8px;
        width: 100%;
        transition: all 0.2s;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        border-color: #4a9eff;
        background: linear-gradient(145deg, #222244, #2a2a5e);
        transform: translateX(2px);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1a, #12122a);
    }

    /* Progress bar for OCR */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #4a9eff, #8b5cf6);
    }

    /* Download buttons */
    .stDownloadButton > button {
        font-size: 13px;
        border: 1px solid #333;
    }

    /* Expander headers */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 15px;
    }
    details[data-testid="stExpander"] {
        border: 1px solid #2a2a4e;
        border-radius: 10px;
        background: linear-gradient(145deg, #12122a, #16162e);
    }
    details[data-testid="stExpander"] summary {
        padding: 12px 16px;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 800;
    }
    [data-testid="stMetricLabel"] {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #888;
    }

    /* Section dividers */
    hr { border-color: #2a2a4e; margin: 24px 0; }

    /* Loading skeleton animation */
    @keyframes skeleton-pulse {
        0% { opacity: 0.4; }
        50% { opacity: 0.8; }
        100% { opacity: 0.4; }
    }
    .skeleton-box {
        background: #2a2a4e;
        border-radius: 6px;
        animation: skeleton-pulse 1.5s ease-in-out infinite;
        height: 24px;
        margin: 4px 0;
    }
    .skeleton-wide { width: 80%; }
    .skeleton-med { width: 50%; }
    .skeleton-short { width: 30%; }

    /* Print-friendly styles */
    @media print {
        #MainMenu, footer, .stDeployButton, .stSidebar,
        .stButton, .stDownloadButton, .stProgress { display: none !important; }
        .main .block-container { padding: 0 !important; max-width: 100% !important; }
        .big-number { font-size: 48px !important; }
        .stat-box { border: 1px solid #ccc !important; background: #fff !important; }
        .factor-card { border: 1px solid #ccc !important; background: #fff !important; color: #000 !important; }
        .search-result { border: 1px solid #ccc !important; background: #fff !important; }
    }

    /* Bookmark star */
    .bookmark-btn { cursor: pointer; font-size: 18px; opacity: 0.5; transition: opacity 0.2s; }
    .bookmark-btn:hover { opacity: 1; }
    .bookmark-btn.active { opacity: 1; color: #ffaa00; }
</style>
""", unsafe_allow_html=True)


# =========================
# HEADER
# =========================

st.markdown('<div class="hero-title">PermitIQ</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Boston Zoning Intelligence &amp; ZBA Prediction Engine — Powered by 7,500+ real ZBA decisions</div>', unsafe_allow_html=True)

# API connection status — cached to avoid re-fetching on every Streamlit rerun
@st.cache_data(ttl=30)
def _fetch_startup_data():
    """Fetch health + stats in one pass, cached for 30 seconds."""
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    stats = requests.get(f"{API_URL}/stats", timeout=5).json()
    return health, stats

try:
    _health, _stats_res = _fetch_startup_data()
    _model_name = _health.get('model_name', 'none')
    _auc = _health.get('model_auc', 0)
    _cases = _health.get('total_cases', 0)
    _calibrated = " (calibrated)" if _health.get('model_brier') else ""
    st.caption(f"API connected | Model: {_model_name}{_calibrated} (AUC: {_auc:.3f}) | {_cases:,} cases loaded")
except Exception:
    _stats_res = None
    st.caption("API offline — start with: `cd api && uvicorn main:app --reload --port 8000`")

# --- Platform Stats Row ---
try:
    if _stats_res is None:
        _stats_res = requests.get(f"{API_URL}/stats", timeout=5).json()
    st1, st2, st3, st4, st5 = st.columns(5)
    with st1:
        _n_decisions = _stats_res.get('cases_with_decisions', 0) or _stats_res.get('total_cases', 0)
        st.markdown(f"""<div class="stat-box">
            <div class="stat-number">{_n_decisions:,}</div>
            <div class="stat-label">ZBA Decisions</div>
        </div>""", unsafe_allow_html=True)
    with st2:
        st.markdown(f"""<div class="stat-box">
            <div class="stat-number">{_stats_res.get('total_parcels', 0):,}</div>
            <div class="stat-label">Parcels Mapped</div>
        </div>""", unsafe_allow_html=True)
    with st3:
        _oar = _stats_res.get('overall_approval_rate', 0)
        st.markdown(f"""<div class="stat-box">
            <div class="stat-number">{_oar:.0%}</div>
            <div class="stat-label">Avg Approval</div>
        </div>""", unsafe_allow_html=True)
    with st4:
        st.markdown(f"""<div class="stat-box">
            <div class="stat-number">{_stats_res.get('total_wards', 0)}</div>
            <div class="stat-label">Wards Covered</div>
        </div>""", unsafe_allow_html=True)
    with st5:
        try:
            _feats = _health.get('features', 0)
        except Exception:
            _feats = _stats_res.get('features', 57)
        st.markdown(f"""<div class="stat-box">
            <div class="stat-number">{_feats}</div>
            <div class="stat-label">ML Features</div>
        </div>""", unsafe_allow_html=True)
except Exception as e:
    st.caption(f"Stats dashboard unavailable: {e}")

# --- OCR Pipeline Status ---
try:
    _data_status = requests.get(f"{API_URL}/data_status", timeout=3).json()
    _ocr = _data_status.get("ocr_pipeline", {})
    if _ocr.get("running"):
        completed = _ocr.get("completed_pdfs", 0)
        total_pdfs = 262
        pct = completed / total_pdfs
        st.progress(pct, text=f"OCR Pipeline: {completed}/{total_pdfs} PDFs processed ({pct:.0%}) — model will retrain when complete")
except Exception:
    pass

st.markdown("---")


# =========================
# SIDEBAR — QUICK DEMO ACCESS
# =========================

with st.sidebar:
    st.markdown("### Quick Lookup")
    st.markdown("**Try a sample address:**")
    demo_addresses = {
        "1001 Boylston St": "3 cases, 0%",
        "437 Frankfort St": "3 cases, 33%",
        "753 East Broadway": "3 cases, 67%",
        "60 Oakridge St": "3 cases, 67%",
        "124 Glendower Rd": "3 cases, 33%",
        "36 Gaston St": "3 cases, 100%",
    }
    for addr, desc in demo_addresses.items():
        if st.button(f"{addr} ({desc})", key=f"demo_{addr}", use_container_width=True):
            try:
                res = requests.get(f"{API_URL}/search", params={"q": addr}, timeout=15)
                if res.status_code == 200:
                    st.session_state.search_results = res.json().get("results", [])
                    st.session_state.parcel_data = None  # Clear old parcel when searching new address
                    st.rerun()
            except Exception:
                pass

    st.markdown("")
    st.markdown("**Try a sample parcel:**")
    demo_parcels = {
        "0100001000": "East Boston",
        "0302951010": "South Boston",
        "1000358010": "Jamaica Plain",
        "2100394000": "Allston/Brighton",
    }
    for pid, desc in demo_parcels.items():
        if st.button(f"{pid} — {desc}", key=f"demo_p_{pid}", use_container_width=True):
            try:
                res = requests.get(f"{API_URL}/parcels/{pid}", timeout=15)
                if res.status_code == 200:
                    st.session_state.parcel_data = res.json()
                    st.rerun()
            except Exception:
                pass

    st.markdown("---")

    # Bookmarks
    if 'bookmarks' not in st.session_state:
        st.session_state.bookmarks = []
    if st.session_state.bookmarks:
        st.markdown("**Saved Analyses:**")
        for i, bm in enumerate(st.session_state.bookmarks):
            label = bm.get('label', f"Bookmark {i+1}")
            prob_str = f" ({bm.get('probability', 0):.0%})" if bm.get('probability') else ""
            st.markdown(f"- {label}{prob_str}")
        if st.button("Clear bookmarks", key="clear_bm"):
            st.session_state.bookmarks = []
            st.rerun()

    st.markdown("---")
    st.markdown("**Demo flow:**")
    st.markdown("1. Click an address above")
    st.markdown("2. Expand case history")
    st.markdown("3. Enter parcel + variances below")
    st.markdown("4. Hit Predict, explore What-If")
    st.markdown("5. Download the HTML report")


# =========================
# SESSION STATE
# =========================

if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'selected_address' not in st.session_state:
    st.session_state.selected_address = None
if 'parcel_data' not in st.session_state:
    st.session_state.parcel_data = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'bookmarks' not in st.session_state:
    st.session_state.bookmarks = []


# =========================
# MAIN LAYOUT: SEARCH + PREDICT
# =========================

# --- TOP SEARCH BAR ---
st.markdown("### 🔍 Look Up Any Boston Address")

search_col1, search_col2 = st.columns([3, 1])

with search_col1:
    address_query = st.text_input(
        "Street Address",
        placeholder="e.g. 123 Main Street, 45 Beacon St, 70 Burbank...",
        label_visibility="collapsed"
    )

with search_col2:
    search_clicked = st.button("Search Address", type="primary", use_container_width=True)

# Address autocomplete — show parcel suggestions as user types
if address_query and len(address_query) >= 3 and not search_clicked:
    try:
        _ac_res = requests.get(f"{API_URL}/autocomplete", params={"q": address_query, "limit": 5}, timeout=3)
        if _ac_res.status_code == 200:
            _suggestions = _ac_res.json().get("suggestions", [])
            if _suggestions:
                st.caption("Matching parcels:")
                ac_cols = st.columns(len(_suggestions))
                for j, sg in enumerate(_suggestions):
                    with ac_cols[j]:
                        district = f" · {sg['district']}" if sg.get('district') else ""
                        if st.button(f"{sg['address']}{district}", key=f"ac_{j}", use_container_width=True):
                            try:
                                p_res = requests.get(f"{API_URL}/parcels/{sg['parcel_id']}", timeout=10)
                                if p_res.status_code == 200:
                                    st.session_state.parcel_data = p_res.json()
                                    st.rerun()
                            except Exception:
                                pass
    except Exception:
        pass

# Parcel lookup — by ID or find from address
with st.expander("Parcel Lookup (by ID or address)"):
    pid_col1, pid_col2, pid_col3 = st.columns([2, 1, 1])
    with pid_col1:
        parcel_id_input = st.text_input(
            "Parcel ID",
            placeholder="e.g. 0102500000 or type address to find it",
            label_visibility="collapsed"
        )
    with pid_col2:
        parcel_clicked = st.button("Look Up Parcel", use_container_width=True)
    with pid_col3:
        find_parcel_clicked = st.button("Find Parcel from Address", use_container_width=True)

    # Geocode: find parcel ID from address
    if find_parcel_clicked and address_query:
        try:
            geo_res = requests.get(f"{API_URL}/geocode", params={"q": address_query}, timeout=10)
            if geo_res.status_code == 200:
                geo_data = geo_res.json().get("results", [])
                if geo_data:
                    st.markdown(f"**Found {len(geo_data)} parcel(s) matching \"{address_query}\":**")
                    for g in geo_data:
                        zoning_info = f" · {g.get('zoning_code', '')}" if g.get('zoning_code') else ""
                        district_info = f" · {g.get('district', '')}" if g.get('district') else ""
                        if st.button(
                            f"{g['parcel_id']} — {g['address']}{zoning_info}{district_info}",
                            key=f"geo_{g['parcel_id']}",
                            use_container_width=True
                        ):
                            # Look up this parcel
                            try:
                                p_res = requests.get(f"{API_URL}/parcels/{g['parcel_id']}", timeout=15)
                                if p_res.status_code == 200:
                                    st.session_state.parcel_data = p_res.json()
                                    st.rerun()
                            except Exception:
                                pass
                else:
                    st.warning(f"No parcels found for \"{address_query}\". Try a street number + name.")
        except Exception as e:
            st.caption(f"Geocoding unavailable: {e}")


# =========================
# ADDRESS SEARCH RESULTS
# =========================

if search_clicked and address_query:
    # Clear stale prediction when doing a new search
    st.session_state.prediction_result = None
    try:
        res = requests.get(f"{API_URL}/search", params={"q": address_query}, timeout=15)
        if res.status_code == 200:
            data = res.json()
            st.session_state.search_results = data.get("results", [])
            if not st.session_state.search_results:
                st.warning(f"No ZBA cases found for \"{address_query}\". Try a different address or use the parcel ID lookup.")
        else:
            st.error(f"Search error: {res.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure the server is running on port 8000.")

if st.session_state.search_results:
    results = st.session_state.search_results
    st.markdown(f"**Found {len(results)} matching address(es):**")

    for i, result in enumerate(results):
        approved = result.get("approved", 0)
        denied = result.get("denied", 0)
        total = result.get("total_cases", 0)
        rate = result.get("approval_rate")
        rate_str = f"{rate:.0%}" if rate is not None else "N/A"

        # Color code the rate
        if rate and rate >= 0.7:
            rate_color = "#00cc66"
        elif rate and rate >= 0.4:
            rate_color = "#ffaa00"
        else:
            rate_color = "#ff4444"

        ward_str = f" · Ward {esc(result['ward'])}" if result.get('ward') and result['ward'] not in ['', 'nan'] else ""
        zoning_str = f" · {esc(result['zoning'])}" if result.get('zoning') and result['zoning'] not in ['', 'nan'] else ""

        st.markdown(f"""
        <div class="search-result">
            <div class="search-addr">{esc(result['address'])}</div>
            <div class="search-meta">
                {total} ZBA case(s){ward_str}{zoning_str} ·
                <span style="color:{rate_color}; font-weight:600;">{approved} approved, {denied} denied ({rate_str})</span>
                · Latest: {esc(result.get('latest_date', 'N/A'))}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick action buttons
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            if st.button(f"Find Parcel for {result['address']}", key=f"find_{i}", use_container_width=True):
                try:
                    geo_res = requests.get(f"{API_URL}/geocode", params={"q": result['address']}, timeout=10)
                    if geo_res.status_code == 200:
                        geo_results = geo_res.json().get("results", [])
                        if geo_results:
                            pid = geo_results[0]["parcel_id"]
                            p_res = requests.get(f"{API_URL}/parcels/{pid}", timeout=15)
                            if p_res.status_code == 200:
                                st.session_state.parcel_data = p_res.json()
                                st.rerun()
                except Exception:
                    pass
        with btn_col2:
            pass  # Reserve for future actions

        # Expandable case history
        with st.expander(f"View case history for {result['address']}", expanded=False):
            try:
                cases_res = requests.get(
                    f"{API_URL}/address/{result['address']}/cases",
                    timeout=10
                )
                if cases_res.status_code == 200:
                    case_data = cases_res.json()
                    cases_list = case_data.get("cases", [])
                    if cases_list:
                        for case in cases_list:
                            emoji = "✅" if case.get('decision') == 'APPROVED' else "❌" if case.get('decision') == 'DENIED' else "⏳"
                            variances_raw = case.get('variances', '')
                            variances_str = str(variances_raw) if variances_raw and str(variances_raw).lower() not in ('nan', 'none', '') else 'none listed'
                            date_raw = case.get('date', '')
                            date_str = str(date_raw) if date_raw and str(date_raw).lower() not in ('nan', 'none', '') else ''
                            date_part = f"{esc(date_str)} — " if date_str else ""
                            st.markdown(
                                f"{emoji} **{esc(case.get('case_number', 'N/A'))}** — "
                                f"{esc(case.get('decision', 'N/A'))} — "
                                f"{date_part}"
                                f"Variances: {esc(variances_str)}"
                            )
                    else:
                        st.caption("No detailed case records available.")
                else:
                    st.caption("Could not load case history.")
            except Exception:
                st.caption("Could not load case history.")

    st.markdown("---")


# =========================
# PARCEL LOOKUP
# =========================

if parcel_clicked and parcel_id_input:
    try:
        res = requests.get(f"{API_URL}/parcels/{parcel_id_input}", timeout=15)
        if res.status_code == 404:
            st.error("Parcel not found. Check the Parcel ID.")
        elif res.status_code != 200:
            st.error(f"API Error: {res.status_code}")
        else:
            st.session_state.parcel_data = res.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure the server is running on port 8000.")

if st.session_state.parcel_data:
    data = st.session_state.parcel_data
    parcel_title = f"### 📍 Parcel {data.get('parcel_id', 'N/A')}"
    if data.get("address"):
        parcel_title += f" — {data['address']}"
    st.markdown(parcel_title)

    # Show ward and ZBA history if available
    extra_info = []
    if data.get("ward"):
        extra_info.append(f"Ward {data['ward']}")
    if data.get("zba_cases"):
        extra_info.append(f"{data['zba_cases']} ZBA case(s) on file")
    if extra_info:
        st.caption(" · ".join(extra_info))

    z1, z2, z3 = st.columns(3)
    with z1:
        st.metric("Zoning Code", data.get("zoning_code", "N/A"))
    with z2:
        st.metric("District", data.get("district", "N/A"))
    with z3:
        st.metric("Article", data.get("article", "N/A"))

    summary = data.get('zoning_summary', '')
    if summary and summary != 'N/A' and summary.strip():
        st.markdown(f"**Summary:** {esc(summary)}")

    if data.get("multi_zoning"):
        st.warning(f"Multi-zoning parcel — {data.get('zoning_count', 0)} zones overlap")

    # Zoning Requirements — from zoning analysis endpoint
    try:
        _zoning_res = requests.get(f"{API_URL}/zoning/{data.get('parcel_id', '')}", timeout=5)
        if _zoning_res.status_code == 200:
            zdata = _zoning_res.json()
            dreqs = zdata.get("dimensional_requirements", {})
            if dreqs.get("max_far"):
                with st.expander("📐 Zoning Dimensional Requirements", expanded=False):
                    st.markdown(f"**District:** {esc(zdata.get('district_name', ''))} — {esc(zdata.get('description', ''))}")
                    r1, r2, r3, r4 = st.columns(4)
                    with r1:
                        st.metric("Max FAR", dreqs.get("max_far", "N/A"))
                    with r2:
                        st.metric("Max Height", f"{dreqs.get('max_height_ft', 'N/A')} ft")
                    with r3:
                        st.metric("Max Stories", dreqs.get("max_stories", "N/A"))
                    with r4:
                        st.metric("Parking/Unit", dreqs.get("parking_per_unit", "N/A"))

                    r5, r6, r7, r8 = st.columns(4)
                    with r5:
                        st.metric("Min Lot Size", f"{dreqs.get('min_lot_sf', 'N/A'):,} sf" if isinstance(dreqs.get('min_lot_sf'), (int, float)) else "N/A")
                    with r6:
                        st.metric("Min Frontage", f"{dreqs.get('min_frontage_ft', 'N/A')} ft")
                    with r7:
                        st.metric("Front Setback", f"{dreqs.get('min_front_yard_ft', 'N/A')} ft")
                    with r8:
                        st.metric("Max Coverage", f"{dreqs.get('max_lot_coverage_pct', 'N/A')}%")

                    uses = zdata.get("allowed_uses", [])
                    if uses:
                        st.markdown(f"**Allowed Uses:** {', '.join(esc(u) for u in uses)}")

                    area_rate = zdata.get("area_approval_rate", 0)
                    area_cases = zdata.get("area_zba_cases", 0)
                    if area_cases > 0:
                        st.markdown(f"**Area ZBA History:** {area_rate:.0%} approval rate across {area_cases:,} cases")
    except Exception:
        pass

    # --- Zoning Compliance Checker ---
    with st.expander("🔍 Check: Does My Project Need a Variance?", expanded=False):
        st.markdown("Enter your proposed project details to see if you need zoning relief.")
        with st.form("compliance_check_form"):
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                cc_far = st.number_input("Proposed FAR", min_value=0.0, max_value=20.0, value=0.0, step=0.1, key="cc_far")
            with cc2:
                cc_height = st.number_input("Proposed Height (ft)", min_value=0, max_value=500, value=0, step=5, key="cc_height")
            with cc3:
                cc_stories = st.number_input("Proposed Stories", min_value=0, max_value=50, value=0, step=1, key="cc_stories")
            cc4, cc5, cc6 = st.columns(3)
            with cc4:
                cc_units = st.number_input("Proposed Units", min_value=0, max_value=500, value=0, step=1, key="cc_units")
            with cc5:
                cc_parking = st.number_input("Parking Spaces", min_value=0, max_value=500, value=0, step=1, key="cc_parking")
            with cc6:
                cc_use = st.selectbox("Proposed Use", ["Residential", "Commercial", "Mixed-Use"], key="cc_use")

            cc_submitted = st.form_submit_button("Check Compliance")
        if cc_submitted:
            _cc_payload = {
                "parcel_id": data.get("parcel_id", ""),
                "proposed_far": cc_far if cc_far > 0 else None,
                "proposed_height_ft": cc_height if cc_height > 0 else None,
                "proposed_stories": cc_stories if cc_stories > 0 else None,
                "proposed_units": cc_units if cc_units > 0 else None,
                "parking_spaces": cc_parking if cc_parking > 0 or cc_units > 0 else None,
                "proposed_use": cc_use.lower(),
            }
            # Remove None values
            _cc_payload = {k: v for k, v in _cc_payload.items() if v is not None}

            try:
                _cc_res = requests.post(f"{API_URL}/zoning/check_compliance", json=_cc_payload, timeout=10)
                if _cc_res.status_code == 200:
                    cc_data = _cc_res.json()

                    if cc_data.get("compliant"):
                        st.success("**No variances needed.** Your project appears to comply with zoning requirements for this district.")
                    else:
                        variances = cc_data.get("variances_needed", [])
                        complexity = cc_data.get("complexity", "unknown")
                        violations = cc_data.get("violations", [])

                        if complexity == "high":
                            st.error(f"**{len(violations)} violation(s) found — {len(variances)} variance(s) needed**")
                        elif complexity == "moderate":
                            st.warning(f"**{len(violations)} violation(s) found — {len(variances)} variance(s) needed**")
                        else:
                            st.info(f"**{len(violations)} violation(s) found — {len(variances)} variance(s) needed**")

                        st.markdown(f"*{esc(cc_data.get('complexity_note', ''))}*")

                        for v in violations:
                            v_color = "#ff4444" if v.get("type") in ("far", "height") else "#ffaa00"
                            excess = v.get("excess", v.get("deficit", ""))
                            st.markdown(
                                f'<div style="background:#1a1a2e;border-left:4px solid {v_color};padding:12px 16px;margin:6px 0;border-radius:6px;">'
                                f'<span style="color:{v_color};font-weight:700;font-size:14px;">{esc(v.get("type", "").upper())}</span><br>'
                                f'<span style="color:#ccc;font-size:13px;">{esc(v.get("requirement", ""))}</span>'
                                f' <span style="color:#888;">→</span> '
                                f'<span style="color:#fff;font-weight:600;font-size:13px;">{esc(v.get("proposed", ""))}</span>'
                                f'{"<br><span style=color:#888;font-size:12px;>" + esc(excess) + "</span>" if excess else ""}'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                        # Show historical rates for these variances
                        hist_rates = cc_data.get("variance_historical_rates", {})
                        if hist_rates:
                            st.markdown("**Historical approval rates for these variances:**")
                            for vtype, vinfo in hist_rates.items():
                                rate = vinfo.get("approval_rate", 0)
                                color = "#00cc66" if rate >= 0.7 else "#ffaa00" if rate >= 0.5 else "#ff4444"
                                st.markdown(
                                    f'<span style="color:{color};font-weight:bold;">{rate:.0%}</span> for {esc(vtype)} variances ({vinfo.get("total_cases", 0)} cases)',
                                    unsafe_allow_html=True
                                )
                else:
                    st.error(f"Error: {_cc_res.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Could not check compliance: {e}")

    # MAP
    if "geometry" in data:
        try:
            coords = data["geometry"]["coordinates"][0]
            if isinstance(coords[0], list) and isinstance(coords[0][0], list):
                coords = coords[0]

            df_coords = pd.DataFrame(coords, columns=["lon", "lat"])

            st.pydeck_chart(pdk.Deck(
                map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                initial_view_state=pdk.ViewState(
                    latitude=df_coords["lat"].mean(),
                    longitude=df_coords["lon"].mean(),
                    zoom=16, pitch=0,
                ),
                layers=[
                    pdk.Layer(
                        "PolygonLayer",
                        data=[{"polygon": coords}],
                        get_polygon="polygon",
                        get_fill_color=[74, 158, 255, 140],
                        get_line_color=[255, 255, 255],
                        line_width_min_pixels=2,
                    )
                ],
            ))
        except Exception as e:
            st.warning(f"Could not render map: {e}")

    # Nearby ZBA cases
    with st.expander(f"Nearby ZBA Cases in {data.get('district', 'this area')}", expanded=False):
        try:
            nearby_res = requests.get(f"{API_URL}/parcels/{data['parcel_id']}/nearby_cases?limit=8", timeout=10)
            if nearby_res.status_code == 200:
                nearby_data = nearby_res.json()
                nearby_cases = nearby_data.get("cases", [])
                if nearby_cases:
                    approved_count = sum(1 for c in nearby_cases if c.get("decision") == "APPROVED")
                    denied_count = sum(1 for c in nearby_cases if c.get("decision") == "DENIED")
                    st.markdown(f"**{len(nearby_cases)} recent cases** in {esc(nearby_data.get('district', 'area'))}: "
                                f"{approved_count} approved, {denied_count} denied")
                    for case in nearby_cases:
                        emoji = "✅" if case.get('decision') == 'APPROVED' else "❌"
                        st.markdown(
                            f"{emoji} **{esc(case.get('case_number', ''))}** — "
                            f"{esc(case.get('address', ''))} — {esc(case.get('decision', ''))} ({esc(case.get('date', ''))})"
                        )
                else:
                    st.caption("No nearby ZBA cases found.")
        except Exception:
            st.caption("Could not load nearby cases.")

    st.markdown("---")


# =========================
# PREDICTION PANEL
# =========================

st.markdown("### 🧠 ZBA Approval Risk Assessment")
st.markdown("*Configure your proposal and get an instant prediction based on 7,500+ historical decisions.*")

# Proposal input row
p1, p2, p3 = st.columns(3)

with p1:
    parcel_id = st.text_input("Parcel ID (for zoning lookup)", value=parcel_id_input if parcel_id_input else "", placeholder="e.g. 0102500000")
    proposed_use = st.selectbox(
        "Proposed Use",
        ["Residential", "Commercial", "Mixed Use", "Industrial", "Institutional", "Other"]
    )

with p2:
    project_type = st.selectbox(
        "Project Type",
        [
            "Addition / Extension",
            "New Construction",
            "Renovation / Rehab",
            "Conversion (Change of Use)",
            "Roof Deck",
            "Multi-Family Development",
            "Single-Family Development",
            "Mixed Use Development",
            "Demolition & Rebuild",
            "ADU (Accessory Dwelling Unit)",
            "Subdivision",
            "Parking Structure",
            "Other"
        ]
    )
    # Auto-fill ward from parcel lookup if available
    default_ward = ""
    if st.session_state.parcel_data and st.session_state.parcel_data.get("ward"):
        default_ward = st.session_state.parcel_data["ward"]
    ward = st.text_input("Ward (optional)", value=default_ward, placeholder="e.g. 17")

with p3:
    variances = st.multiselect(
        "Variances / Relief Needed",
        [
            "Height", "FAR (Floor Area Ratio)", "Lot Area", "Lot Frontage",
            "Front Setback", "Rear Setback", "Side Setback",
            "Parking", "Conditional Use", "Open Space", "Density", "Nonconforming"
        ]
    )
    has_attorney = st.checkbox("Will have legal representation", value=False)

# Additional project details
with st.expander("Project Details (optional — improves prediction accuracy)"):
    d1, d2 = st.columns(2)
    with d1:
        proposed_units = st.number_input("Proposed Units", min_value=0, max_value=500, value=0, help="Number of residential units in the proposal")
    with d2:
        proposed_stories = st.number_input("Proposed Stories", min_value=0, max_value=50, value=0, help="Number of stories/floors in the proposal")

# Map display names to API values
project_type_map = {
    "Addition / Extension": "addition",
    "New Construction": "new_construction",
    "Renovation / Rehab": "renovation",
    "Conversion (Change of Use)": "conversion",
    "Roof Deck": "roof_deck",
    "Multi-Family Development": "multi_family",
    "Single-Family Development": "single_family",
    "Mixed Use Development": "mixed_use",
    "Demolition & Rebuild": "demolition",
    "ADU (Accessory Dwelling Unit)": "adu",
    "Subdivision": "subdivision",
    "Parking Structure": "parking",
    "Other": "other",
}

variance_map = {
    "Height": "height", "FAR (Floor Area Ratio)": "far",
    "Lot Area": "lot_area", "Lot Frontage": "lot_frontage",
    "Front Setback": "front_setback", "Rear Setback": "rear_setback",
    "Side Setback": "side_setback", "Parking": "parking",
    "Conditional Use": "conditional_use", "Open Space": "open_space",
    "Density": "density", "Nonconforming": "nonconforming",
}
clean_variances = [variance_map.get(v, v.lower()) for v in variances]

# Show approval rate hints for selected variances
if variances:
    try:
        _var_stats_res = requests.get(f"{API_URL}/variance_stats", timeout=5)
        if _var_stats_res.status_code == 200:
            _var_stats = {v["variance_type"]: v["approval_rate"] for v in _var_stats_res.json().get("variance_stats", [])}
            hints = []
            for v in clean_variances:
                rate = _var_stats.get(v)
                if rate is not None:
                    hints.append(f"{v.replace('_', ' ').title()}: {rate:.0%}")
            if hints:
                st.caption(f"Historical approval rates — {' · '.join(hints)}")
    except Exception:
        pass


# --- PREDICT BUTTON ---
predict_clicked = st.button("⚡ Predict Approval Likelihood", type="primary", use_container_width=True)

if predict_clicked:
    if not variances:
        st.warning("Select at least one variance to predict.")
    else:
        try:
            payload = {
                "parcel_id": parcel_id,
                "proposed_use": proposed_use.lower(),
                "variances": clean_variances,
                "project_type": project_type_map.get(project_type, "other"),
                "ward": ward if ward else None,
                "has_attorney": has_attorney,
                "proposed_units": proposed_units,
                "proposed_stories": proposed_stories,
            }

            with st.spinner("Analyzing proposal against 7,500+ ZBA decisions..."):
                res = requests.post(f"{API_URL}/analyze_proposal", json=payload, timeout=30)

            if res.status_code == 404:
                st.error("Parcel not found. Check the Parcel ID.")
            elif res.status_code != 200:
                st.error(f"API Error: {res.status_code} — {res.text}")
            else:
                st.session_state.prediction_result = res.json()

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Start it with: `cd api && uvicorn main:app --reload --port 8000`")
        except requests.exceptions.Timeout:
            st.error("API request timed out. The model may still be loading — try again in a few seconds.")
        except Exception as e:
            st.error(f"Error: {e}")


# =========================
# PREDICTION RESULTS
# =========================

if st.session_state.prediction_result:
    result = st.session_state.prediction_result
    st.markdown("---")

    prob = result.get("approval_probability", 0)
    confidence = result.get("confidence", "unknown")
    based_on = result.get("based_on_cases", 0)
    model_name = result.get("model", "unknown")
    model_auc = result.get("model_auc", 0)

    # Color code
    if prob >= 0.7:
        color = "#00cc66"
        label = "LOW RISK"
    elif prob >= 0.4:
        color = "#ffaa00"
        label = "MODERATE RISK"
    else:
        color = "#ff4444"
        label = "HIGH RISK"

    # Big probability display
    r1, r2, r3 = st.columns([1, 2, 1])
    with r2:
        st.markdown(
            f'<div class="big-number" style="color: {color};">{prob:.0%}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="label-text" style="color: {color};">{label}</div>',
            unsafe_allow_html=True
        )
        conf_class = f"confidence-{confidence}" if confidence in ('high', 'medium', 'low') else "confidence-low"
        prob_range = result.get("probability_range")
        range_str = f" · Range: {prob_range[0]:.0%}–{prob_range[1]:.0%}" if prob_range else ""
        st.markdown(
            f'<div class="confidence-text">'
            f'<span class="confidence-badge {conf_class}">{confidence.upper()} confidence</span>'
            f' · Based on {based_on:,} similar cases{range_str}'
            f'</div>',
            unsafe_allow_html=True
        )

    # --- Risk Assessment ---
    st.markdown("")
    war = result.get('ward_approval_rate')
    num_variances = len(result.get('variances', []))
    has_atty = result.get('has_attorney', False)

    if prob >= 0.7:
        risk_level = "LOW RISK"
        risk_color = "#00cc66"
        risk_msg = "This proposal aligns well with historical ZBA patterns. Strong likelihood of approval with proper preparation."
    elif prob >= 0.5:
        risk_level = "MODERATE RISK"
        risk_color = "#ffaa00"
        risk_msg = "This proposal has reasonable odds but faces some headwinds. Consider the recommendations below to strengthen your position."
    elif prob >= 0.3:
        risk_level = "ELEVATED RISK"
        risk_color = "#ff8800"
        risk_msg = "This proposal faces significant challenges. Review the key factors carefully and consider scope adjustments before filing."
    else:
        risk_level = "HIGH RISK"
        risk_color = "#ff4444"
        risk_msg = "This proposal is unlikely to be approved as configured. Strongly recommend redesigning the project to reduce variances or scope."

    # Cost-at-risk estimate
    filing_cost_low, filing_cost_high = 30000, 100000
    expected_loss_low = int(filing_cost_low * (1 - prob))
    expected_loss_high = int(filing_cost_high * (1 - prob))

    st.markdown(
        f'<div style="background:#1a1a2e;padding:16px 20px;border-radius:10px;border-left:5px solid {risk_color};margin:8px 0;">'
        f'<div style="font-size:14px;font-weight:700;color:{risk_color};letter-spacing:1px;">{risk_level}</div>'
        f'<div style="color:#ccc;font-size:14px;margin-top:6px;">{risk_msg}</div>'
        f'<div style="color:#888;font-size:12px;margin-top:8px;">'
        f'Expected cost at risk: ${expected_loss_low:,}–${expected_loss_high:,} '
        f'(based on typical $30K–$100K permitting spend × {1-prob:.0%} denial probability)'
        f'</div></div>',
        unsafe_allow_html=True
    )
    st.markdown("")

    # --- Proposal Summary ---
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f"""<div class="stat-box">
            <div class="stat-label">Zoning</div>
            <div class="stat-number" style="font-size:18px;">{result.get('zoning', 'N/A')}</div>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""<div class="stat-box">
            <div class="stat-label">Project Type</div>
            <div class="stat-number" style="font-size:18px;">{result.get('project_type', 'N/A')}</div>
        </div>""", unsafe_allow_html=True)
    with s3:
        st.markdown(f"""<div class="stat-box">
            <div class="stat-label">Variances</div>
            <div class="stat-number" style="font-size:18px;">{len(result.get('variances', []))}</div>
        </div>""", unsafe_allow_html=True)
    with s4:
        war = result.get('ward_approval_rate')
        war_str = f"{war:.0%}" if war else "N/A"
        st.markdown(f"""<div class="stat-box">
            <div class="stat-label">Ward Approval Rate</div>
            <div class="stat-number" style="font-size:18px;">{war_str}</div>
        </div>""", unsafe_allow_html=True)

    # --- Key Factors ---
    key_factors = result.get("key_factors", [])
    if key_factors:
        st.markdown("")
        st.markdown("**⚡ Key Factors**")
        for factor in key_factors:
            # Color code based on content
            f_lower = factor.lower()
            if any(w in f_lower for w in ['increases', 'higher', 'support', 'positive']):
                css_class = "factor-card factor-positive"
            elif any(w in f_lower for w in ['reduces', 'lower', 'scrutiny', 'opposition', 'unlikely']):
                css_class = "factor-card factor-negative"
            else:
                css_class = "factor-card factor-neutral"
            st.markdown(f'<div class="{css_class}">{esc(factor)}</div>', unsafe_allow_html=True)

    # --- Model Explainability ---
    top_drivers = result.get("top_drivers", [])
    if top_drivers:
        st.markdown("")
        with st.expander("Model Explainability — What drove this prediction?", expanded=False):
            is_shap = top_drivers and "shap_value" in top_drivers[0]
            if is_shap:
                st.markdown("*SHAP values show how each feature pushed the prediction up or down:*")
            else:
                st.markdown("*Feature importance scores from the ML model:*")
            for d in top_drivers:
                shap_val = d.get("shap_value", d.get("importance", 0))
                direction = d.get("direction", "unknown")
                abs_val = abs(shap_val)
                bar_width = min(int(abs_val * 500), 100)
                if direction == "increases":
                    bar_color = "#00cc66"
                    arrow = "+"
                elif direction == "decreases":
                    bar_color = "#ff4444"
                    arrow = ""
                else:
                    bar_color = "#4a9eff"
                    arrow = ""
                st.markdown(
                    f'<div style="display:flex;align-items:center;padding:3px 0;">'
                    f'<span style="width:220px;font-size:14px;">{esc(d["feature"])}</span>'
                    f'<div style="background:{bar_color};height:14px;width:{bar_width}%;border-radius:3px;margin:0 10px;"></div>'
                    f'<span style="color:{bar_color};font-size:13px;font-weight:600;">{arrow}{shap_val:.3f}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            if is_shap:
                st.caption("SHAP values: positive (green) = increases approval odds, negative (red) = decreases odds.")
            else:
                st.caption("Global feature importance. SHAP per-prediction analysis available after model retrain.")

    # --- Similar Cases ---
    similar = result.get("similar_cases", [])
    if similar:
        st.markdown("")
        st.markdown("**📁 Similar Historical Cases**")
        for case in similar:
            emoji = "✅" if case.get('decision') == 'APPROVED' else "❌"
            date = case.get('date', '')
            # Clean up OCR artifacts in display
            if not date or date == 'nan' or str(date).strip().lower() in ('nan', 'none', 'nat', ''):
                date = ''
            address = case.get('address', 'Unknown')
            # Truncate garbled OCR addresses
            if len(str(address)) > 60:
                address = str(address)[:60] + '...'
            date_str = f' <span style="color:#666;">({esc(str(date))})</span>' if date else ''
            st.markdown(
                f'<div class="case-card">{emoji} <strong>{esc(case.get("case_number", "N/A"))}</strong>'
                f' — {esc(address)} — {esc(case.get("decision", "N/A"))}'
                f'{date_str}</div>',
                unsafe_allow_html=True
            )

    # --- What-If Comparison (uses /compare endpoint for real model numbers) ---
    st.markdown("")
    st.markdown("**What If...?**")
    st.markdown("*Model-computed scenarios showing how changes affect your approval odds:*")

    compare_res = None  # Initialize for report section
    try:
        compare_payload = {
            "parcel_id": parcel_id,
            "proposed_use": proposed_use.lower(),
            "variances": clean_variances,
            "project_type": project_type_map.get(project_type, "other"),
            "ward": ward if ward else None,
            "has_attorney": has_attorney,
            "proposed_units": proposed_units,
            "proposed_stories": proposed_stories,
        }
        compare_res = requests.post(f"{API_URL}/compare", json=compare_payload, timeout=15)  # noqa
        if compare_res.status_code == 200:
            compare_data = compare_res.json()
            scenarios = compare_data.get("scenarios", [])

            # Display scenarios in a grid
            cols = st.columns(min(len(scenarios), 3)) if scenarios else []
            for i, scenario in enumerate(scenarios[:6]):
                col_idx = i % min(len(scenarios), 3)
                with cols[col_idx]:
                    diff = scenario.get("difference", 0)
                    new_prob = scenario.get("probability", 0)
                    if diff > 0.02:
                        css_class = "factor-card factor-positive"
                        arrow = "↑"
                    elif diff < -0.02:
                        css_class = "factor-card factor-negative"
                        arrow = "↓"
                    else:
                        css_class = "factor-card factor-neutral"
                        arrow = "→"

                    st.markdown(
                        f'<div class="{css_class}">'
                        f'<strong>{esc(scenario["scenario"])}</strong><br>'
                        f'{arrow} {new_prob:.0%} ({diff:+.1%})'
                        f'</div>', unsafe_allow_html=True)
        else:
            # Fallback to static estimates if /compare fails
            wi1, wi2 = st.columns(2)
            with wi1:
                if not has_attorney:
                    st.markdown(
                        '<div class="factor-card factor-positive">'
                        'With an attorney: probability would likely increase ~15-20%'
                        '</div>', unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div class="factor-card factor-negative">'
                        'Without an attorney: probability would likely drop ~15-20%'
                        '</div>', unsafe_allow_html=True)
            with wi2:
                if len(variances) > 2:
                    fewer = max(1, len(variances) - 2)
                    st.markdown(
                        f'<div class="factor-card factor-positive">'
                        f'With only {fewer} variance(s): approval odds would increase'
                        f'</div>', unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div class="factor-card factor-neutral">'
                        'Adding more variances: each additional variance reduces approval odds'
                        '</div>', unsafe_allow_html=True)
    except Exception as e:
        st.caption(f"Could not load key factors: {e}")

    # --- Actionable Recommendations ---
    st.markdown("")
    recommendations = []
    if not result.get("has_attorney"):
        recommendations.append("Hire an attorney — legal representation significantly boosts approval odds")
    if len(result.get("variances", [])) > 3:
        recommendations.append(f"Reduce scope — you're requesting {len(result['variances'])} variances; fewer = better odds")
    if prob < 0.4 and result.get("ward_approval_rate") and result["ward_approval_rate"] > 0.6:
        recommendations.append("Your project specifics are dragging you below the ward average — review variance selections")
    if prob >= 0.6:
        recommendations.append("Strong position — ensure community engagement and clean application to maintain odds")
    if prob < 0.3:
        recommendations.append("Consider redesigning the project to require fewer variances before filing")

    if recommendations:
        with st.expander("Recommended Next Steps", expanded=prob < 0.5):
            for r in recommendations:
                st.markdown(f"→ {r}")

    # --- Model Info ---
    # --- Save / Bookmark ---
    bm_col1, bm_col2 = st.columns([1, 3])
    with bm_col1:
        if st.button("Save Analysis", key="save_analysis"):
            bm = {
                "label": f"{result.get('parcel_id', '?')} — {result.get('district', '?')}",
                "probability": prob,
                "parcel_id": result.get("parcel_id", ""),
                "variances": result.get("variances", []),
                "project_type": result.get("project_type", ""),
            }
            st.session_state.bookmarks.append(bm)
            st.success("Saved to sidebar bookmarks")
    with bm_col2:
        st.markdown(
            f'<div style="color:#555; font-size:12px; padding-top:10px;">'
            f'Model: {model_name} · AUC: {model_auc:.3f} · '
            f'Trained on {result.get("total_training_cases", 0):,} ZBA decisions'
            f'</div>',
            unsafe_allow_html=True
        )

    # --- Timeline Estimate ---
    timeline = result.get("estimated_timeline_days")
    if timeline and timeline.get("median_days"):
        days = timeline["median_days"]
        months = days / 30
        cases_used = timeline.get('cases_used', 0)
        note = timeline.get('note', '')
        if note:
            ward_note = f" ({esc(note)})"
        elif timeline.get("ward_specific"):
            ward_note = f" (Ward-specific, {cases_used} cases)"
        elif cases_used > 0:
            ward_note = f" (city-wide avg, {cases_used} cases)"
        else:
            ward_note = ""
        st.markdown(
            f'<div style="color:#888; font-size:13px; margin-top:5px;">'
            f'Estimated timeline: ~{months:.0f} months ({days} days){ward_note}'
            f'</div>',
            unsafe_allow_html=True
        )

    # --- Export Report ---
    st.markdown("")
    # Build What-If section for report
    whatif_section = ""
    try:
        if compare_res and compare_res.status_code == 200:
            compare_data = compare_res.json()
            scenarios = compare_data.get("scenarios", [])
            if scenarios:
                whatif_lines = []
                for s in scenarios:
                    diff = s.get('difference', 0)
                    whatif_lines.append(f"- {s['scenario']}: {s['probability']:.0%} ({diff:+.1%})")
                whatif_section = f"\nWHAT-IF SCENARIOS\n{chr(10).join(whatif_lines)}\n"
    except Exception:
        whatif_section = ""

    report_text = f"""PERMITIQ ZONING ANALYSIS REPORT
{'='*50}
Generated: {pd.Timestamp.now().strftime('%B %d, %Y %I:%M %p')}

PROPOSAL SUMMARY
Parcel ID: {result.get('parcel_id', 'N/A')}
Zoning: {result.get('zoning', 'N/A')}
District: {result.get('district', 'N/A')}
Proposed Use: {result.get('proposed_use', 'N/A')}
Project Type: {result.get('project_type', 'N/A')}
Variances: {', '.join(result.get('variances', []))}
Attorney: {'Yes' if result.get('has_attorney') else 'No'}

PREDICTION
Approval Probability: {prob:.0%}
Confidence: {confidence.upper()}
Based on: {based_on:,} similar historical cases

KEY FACTORS
{chr(10).join(f'- {f}' for f in key_factors)}
{whatif_section}
SIMILAR CASES
{chr(10).join(f"- {c.get('case_number','N/A')} | {c.get('address','N/A')} | {c.get('decision','N/A')} ({c.get('date','')})" for c in similar[:5])}

MODEL INFO
Model: {model_name} | AUC: {model_auc:.3f}
Training Cases: {result.get('total_training_cases', 0):,}

DISCLAIMER
PermitIQ predictions are based on historical ZBA decision data and
do not constitute legal advice. Individual outcomes may vary.
{'='*50}
Generated by PermitIQ v2.0 — Boston Zoning Intelligence
"""
    # Build HTML report
    prob_color = "#00cc66" if prob >= 0.7 else "#ffaa00" if prob >= 0.4 else "#ff4444"
    prob_range = result.get("probability_range")
    range_html = f"<p style='color:#888;'>Range: {prob_range[0]:.0%} – {prob_range[1]:.0%}</p>" if prob_range else ""
    factors_html = "".join(f"<li>{esc(f)}</li>" for f in key_factors)
    similar_html = "".join(
        f"<tr><td>{esc(c.get('case_number',''))}</td><td>{esc(c.get('address',''))}</td>"
        f"<td>{esc(c.get('decision',''))}</td><td>{esc(c.get('date',''))}</td></tr>"
        for c in similar[:5]
    )
    whatif_html = ""
    if whatif_section:
        whatif_html = f"<h3>What-If Scenarios</h3><pre>{esc(whatif_section)}</pre>"

    html_report = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>PermitIQ Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; color: #222; }}
h1 {{ color: #1a1a2e; border-bottom: 3px solid #1a1a2e; padding-bottom: 10px; }}
h2 {{ color: #333; margin-top: 30px; }}
.prob {{ font-size: 72px; font-weight: bold; color: {prob_color}; text-align: center; margin: 20px 0 5px; }}
.label {{ text-align: center; font-size: 18px; color: {prob_color}; font-weight: 600; }}
.confidence {{ text-align: center; color: #888; margin-bottom: 20px; }}
.summary-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
.summary-item {{ background: #f5f5f5; padding: 12px; border-radius: 6px; }}
.summary-item strong {{ display: block; color: #888; font-size: 12px; text-transform: uppercase; }}
table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #eee; }}
th {{ background: #f5f5f5; font-weight: 600; }}
ul {{ line-height: 1.8; }}
.disclaimer {{ background: #fff3cd; padding: 12px; border-radius: 6px; font-size: 13px; margin-top: 30px; }}
.footer {{ text-align: center; color: #888; font-size: 12px; margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; }}
</style></head><body>
<h1>PermitIQ Zoning Analysis Report</h1>
<p style="color:#888;">Generated {pd.Timestamp.now().strftime('%B %d, %Y at %I:%M %p')}</p>

<div class="prob">{prob:.0%}</div>
<div class="label">{'LIKELY TO PASS' if prob >= 0.7 else 'UNCERTAIN' if prob >= 0.4 else 'UNLIKELY TO PASS'}</div>
<div class="confidence">{confidence.upper()} confidence · Based on {based_on:,} similar cases</div>
{range_html}

<div style="background:#f8f8f8;padding:16px;border-radius:8px;border-left:5px solid {prob_color};margin:15px 0;">
<strong style="color:{prob_color};">{risk_level}</strong><br>
{risk_msg}<br>
<span style="color:#888;font-size:13px;">Expected cost at risk: ${expected_loss_low:,}–${expected_loss_high:,}</span>
</div>

<h2>Proposal Summary</h2>
<div class="summary-grid">
<div class="summary-item"><strong>Parcel ID</strong>{esc(str(result.get('parcel_id', 'N/A')))}</div>
<div class="summary-item"><strong>Zoning</strong>{esc(str(result.get('zoning', 'N/A')))}</div>
<div class="summary-item"><strong>District</strong>{esc(str(result.get('district', 'N/A')))}</div>
<div class="summary-item"><strong>Proposed Use</strong>{esc(str(result.get('proposed_use', 'N/A')))}</div>
<div class="summary-item"><strong>Project Type</strong>{esc(str(result.get('project_type', 'N/A')))}</div>
<div class="summary-item"><strong>Variances</strong>{esc(', '.join(result.get('variances', [])))}</div>
<div class="summary-item"><strong>Attorney</strong>{'Yes' if result.get('has_attorney') else 'No'}</div>
<div class="summary-item"><strong>Ward Approval Rate</strong>{result.get('ward_approval_rate', 'N/A')}</div>
</div>

<h2>Key Factors</h2>
<ul>{factors_html}</ul>

{whatif_html}

<h2>Similar Historical Cases</h2>
<table><tr><th>Case #</th><th>Address</th><th>Decision</th><th>Date</th></tr>
{similar_html}</table>

<h2>Model Info</h2>
<p>Model: {esc(model_name)} · AUC: {model_auc:.3f} · Trained on {result.get('total_training_cases', 0):,} ZBA decisions</p>

<div class="disclaimer">
<strong>⚠️ Risk Assessment Disclaimer</strong><br>
PermitIQ provides risk assessments based on statistical analysis of {result.get('total_training_cases', 7500):,}+ historical ZBA decisions.
This is NOT a prediction of your specific outcome and does NOT constitute legal advice.
Actual ZBA decisions depend on many factors not captured in the model including: board member composition,
quality of presentation, neighborhood politics, project design details, and community engagement.
Always consult a qualified zoning attorney before making financial decisions based on this analysis.
Probabilities reflect historical patterns, not guarantees.
</div>
<div class="footer">Generated by PermitIQ v2.0 — Boston Zoning Intelligence</div>
</body></html>"""

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            label="Download Report (HTML)",
            data=html_report,
            file_name=f"PermitIQ_Report_{result.get('parcel_id', 'analysis')}.html",
            mime="text/html",
        )
    with dl2:
        st.download_button(
            label="Download Report (Text)",
            data=report_text,
            file_name=f"PermitIQ_Report_{result.get('parcel_id', 'analysis')}.txt",
            mime="text/plain",
        )

    # Risk Assessment Disclaimer
    st.markdown(
        '<div style="background:#2a1f00;border:1px solid #856404;border-radius:8px;'
        'padding:14px;margin-top:20px;font-size:12px;color:#d4a017;">'
        '<strong>⚠️ Risk Assessment Disclaimer</strong><br>'
        'This is a statistical risk assessment, not a prediction or guarantee. '
        'Actual ZBA outcomes depend on factors not captured in the model: board composition, '
        'presentation quality, neighborhood dynamics, and project design. '
        'Always consult a qualified zoning attorney before making financial decisions. '
        'PermitIQ does not provide legal advice.</div>',
        unsafe_allow_html=True
    )

    # --- Quick Comparison Tool ---
    with st.expander("Compare Alternative Scenario", expanded=False):
        st.markdown("*Quickly test a different configuration against your current prediction:*")
        alt_c1, alt_c2 = st.columns(2)
        with alt_c1:
            alt_variances = st.multiselect(
                "Alternative Variances",
                ["Height", "FAR (Floor Area Ratio)", "Lot Area", "Lot Frontage",
                 "Front Setback", "Rear Setback", "Side Setback",
                 "Parking", "Conditional Use", "Open Space", "Density", "Nonconforming"],
                key="alt_variances"
            )
            alt_attorney = st.checkbox("With attorney?", value=True, key="alt_attorney")
        with alt_c2:
            alt_project = st.selectbox(
                "Alternative Project Type",
                ["Addition / Extension", "New Construction", "Renovation / Rehab",
                 "Conversion (Change of Use)", "Roof Deck", "Multi-Family Development",
                 "Single-Family Development", "Mixed Use Development", "Other"],
                key="alt_project"
            )
            alt_units = st.number_input("Units", min_value=0, max_value=500, value=0, key="alt_units")

        if st.button("Compare", key="compare_alt"):
            alt_variance_map = {
                "Height": "height", "FAR (Floor Area Ratio)": "far",
                "Lot Area": "lot_area", "Lot Frontage": "lot_frontage",
                "Front Setback": "front_setback", "Rear Setback": "rear_setback",
                "Side Setback": "side_setback", "Parking": "parking",
                "Conditional Use": "conditional_use", "Open Space": "open_space",
                "Density": "density", "Nonconforming": "nonconforming",
            }
            alt_project_map = {
                "Addition / Extension": "addition", "New Construction": "new_construction",
                "Renovation / Rehab": "renovation", "Conversion (Change of Use)": "conversion",
                "Roof Deck": "roof_deck", "Multi-Family Development": "multi_family",
                "Single-Family Development": "single_family", "Mixed Use Development": "mixed_use",
                "Other": "other",
            }
            alt_payload = {
                "parcel_id": result.get("parcel_id", ""),
                "proposed_use": result.get("proposed_use", "residential"),
                "variances": [alt_variance_map.get(v, v.lower()) for v in alt_variances],
                "project_type": alt_project_map.get(alt_project, "other"),
                "has_attorney": alt_attorney,
                "proposed_units": alt_units,
            }
            try:
                alt_res = requests.post(f"{API_URL}/analyze_proposal", json=alt_payload, timeout=30)
                if alt_res.status_code == 200:
                    alt_result = alt_res.json()
                    alt_prob = alt_result.get("approval_probability", 0)
                    diff = alt_prob - prob
                    diff_color = "#00cc66" if diff > 0 else "#ff4444" if diff < 0 else "#888"

                    cmp1, cmp2, cmp3 = st.columns(3)
                    with cmp1:
                        st.metric("Current Proposal", f"{prob:.0%}")
                    with cmp2:
                        st.metric("Alternative", f"{alt_prob:.0%}")
                    with cmp3:
                        st.markdown(
                            f'<div style="text-align:center;padding-top:15px;">'
                            f'<span style="font-size:28px;font-weight:700;color:{diff_color};">{diff:+.0%}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
            except Exception as e:
                st.caption(f"Comparison failed: {e}")


# =========================
# WARD INSIGHTS
# =========================

with st.expander("Ward Insights — Compare approval rates across Boston"):
    # Ward heatmap — all wards at a glance
    try:
        _all_wards_res = requests.get(f"{API_URL}/wards/all", timeout=10)
        all_ward_data = _all_wards_res.json().get("wards", []) if _all_wards_res.status_code == 200 else []

        if all_ward_data:
            st.markdown("**All Boston Wards — Approval Rates:**")
            # Render as a colored grid
            cols = st.columns(4)
            for i, wd in enumerate(sorted(all_ward_data, key=lambda x: -x.get('approval_rate', 0))):
                rate = wd.get('approval_rate', 0)
                total = wd.get('total_cases', 0)
                if rate >= 0.7:
                    bg = "#1a3d1a"
                    border = "#00cc66"
                elif rate >= 0.5:
                    bg = "#3d3d1a"
                    border = "#ffaa00"
                else:
                    bg = "#3d1a1a"
                    border = "#ff4444"
                with cols[i % 4]:
                    st.markdown(
                        f'<div style="background:{bg};border:1px solid {border};border-radius:8px;'
                        f'padding:10px;text-align:center;margin:4px 0;">'
                        f'<div style="font-size:20px;font-weight:700;color:{border};">Ward {esc(wd["ward"])}</div>'
                        f'<div style="font-size:24px;font-weight:800;">{rate:.0%}</div>'
                        f'<div style="color:#888;font-size:11px;">{total} cases</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            st.markdown("")
    except Exception:
        pass

    # Individual ward lookup
    ward_input = st.text_input("Look up specific ward", placeholder="e.g. 17", key="ward_insights")
    if ward_input:
        try:
            ward_res = requests.get(f"{API_URL}/wards/{ward_input}/stats", timeout=10)
            if ward_res.status_code == 200:
                ws = ward_res.json()
                w1, w2, w3, w4 = st.columns(4)
                with w1:
                    st.metric("Total Cases", ws['total_cases'])
                with w2:
                    st.metric("Approved", ws['approved'])
                with w3:
                    st.metric("Denied", ws['denied'])
                with w4:
                    rate = ws['approval_rate']
                    st.metric("Approval Rate", f"{rate:.0%}")

                # Timeline for this ward
                try:
                    tl_res = requests.get(f"{API_URL}/timeline_stats", timeout=10)
                    if tl_res.status_code == 200:
                        tl_data = tl_res.json()
                        for tw in tl_data.get("by_ward", []):
                            if tw["ward"] == ward_input:
                                st.caption(f"Median decision timeline for Ward {ward_input}: {tw['median_days']} days ({tw['cases']} cases)")
                                break
                except Exception:
                    pass

                if rate > 0.7:
                    st.success(f"Ward {ward_input} has a strong approval rate — above Boston average.")
                elif rate > 0.5:
                    st.info(f"Ward {ward_input} has a moderate approval rate — close to Boston average.")
                else:
                    st.warning(f"Ward {ward_input} has a below-average approval rate — projects face more scrutiny here.")
            elif ward_res.status_code == 404:
                st.warning(f"No ZBA cases found for Ward {ward_input}.")
        except Exception as e:
            st.caption(f"Ward insights unavailable: {e}")


# =========================
# MARKET INTELLIGENCE
# =========================

with st.expander("Market Intelligence — Trends, Variance Stats & Top Attorneys", expanded=False):
    intel_tab1, intel_tab2, intel_tab3, intel_tab4, intel_tab5, intel_tab6, intel_tab7 = st.tabs([
        "Approval Trends", "Variance Rates", "Project Types", "Attorneys",
        "Neighborhoods", "Denial Patterns", "Voting & Provisos"
    ])

    # --- Approval Trends Chart ---
    with intel_tab1:
        try:
            trends_res = requests.get(f"{API_URL}/trends", timeout=10)
            if trends_res.status_code == 200:
                trends_data = trends_res.json().get("years", [])
                if trends_data:
                    trends_df = pd.DataFrame(trends_data)
                    col_chart, col_table = st.columns([2, 1])
                    with col_chart:
                        st.bar_chart(trends_df.set_index("year")["approval_rate"], use_container_width=True)
                        st.caption("ZBA approval rate by year")
                    with col_table:
                        for row in trends_data:
                            yr = row["year"]
                            rate = row["approval_rate"]
                            total = row["total_cases"]
                            st.markdown(f"**{yr}**: {rate:.0%} ({total} cases)")
                    st.download_button(
                        "Export Trends CSV", trends_df.to_csv(index=False),
                        "permitiq_trends.csv", "text/csv", key="dl_trends"
                    )
        except Exception as e:
            st.caption(f"Trends unavailable: {e}")

    # --- Variance Success Rates ---
    with intel_tab2:
        try:
            var_res = requests.get(f"{API_URL}/variance_stats", timeout=10)
            if var_res.status_code == 200:
                var_data = var_res.json().get("variance_stats", [])
                if var_data:
                    for v in var_data:
                        rate = v["approval_rate"]
                        name = v["variance_type"].replace("_", " ").title()
                        total = v["total_cases"]
                        if rate >= 0.7:
                            color = "#00cc66"
                        elif rate >= 0.5:
                            color = "#ffaa00"
                        else:
                            color = "#ff4444"
                        st.markdown(
                            f'<div style="display:flex;justify-content:space-between;padding:4px 0;">'
                            f'<span>{esc(name)} ({total} cases)</span>'
                            f'<span style="color:{color};font-weight:600;">{rate:.0%}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    st.caption("Approval rate by variance type (higher = easier to get approved)")
        except Exception as e:
            st.caption(f"Variance stats unavailable: {e}")

    # --- Project Type Rates ---
    with intel_tab3:
        try:
            pt_res = requests.get(f"{API_URL}/project_type_stats", timeout=10)
            if pt_res.status_code == 200:
                pt_data = pt_res.json().get("project_type_stats", [])
                if pt_data:
                    for p in pt_data:
                        rate = p["approval_rate"]
                        name = p["project_type"]
                        total = p["total_cases"]
                        if rate >= 0.7:
                            color = "#00cc66"
                        elif rate >= 0.5:
                            color = "#ffaa00"
                        else:
                            color = "#ff4444"
                        st.markdown(
                            f'<div style="display:flex;justify-content:space-between;padding:4px 0;">'
                            f'<span>{esc(name)} ({total} cases)</span>'
                            f'<span style="color:{color};font-weight:600;">{rate:.0%}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    st.caption("Approval rate by project type (higher = easier to get approved)")
        except Exception as e:
            st.caption(f"Project type stats unavailable: {e}")

    # --- Attorney Leaderboard ---
    with intel_tab4:
        try:
            atty_res = requests.get(f"{API_URL}/attorneys/leaderboard?min_cases=10&limit=15", timeout=10)
            if atty_res.status_code == 200:
                atty_data = atty_res.json()
                attorneys = atty_data.get("attorneys", [])

                # Context stats
                atty_rate = atty_data.get("attorney_approval_rate")
                no_atty_rate = atty_data.get("no_attorney_approval_rate")
                if atty_rate is not None and no_atty_rate is not None:
                    st.markdown(
                        f"With attorney: **{atty_rate:.0%}** approval · "
                        f"Without: **{no_atty_rate:.0%}** approval · "
                        f"Difference: **+{(atty_rate - no_atty_rate):.0%}**"
                    )
                    st.markdown("")

                if attorneys:
                    for rank, a in enumerate(attorneys, 1):
                        rate = a["approval_rate"]
                        st.markdown(
                            f'<div style="display:flex;justify-content:space-between;padding:4px 0;">'
                            f'<span>{rank}. {esc(a["name"])} ({a["total_cases"]} cases)</span>'
                            f'<span style="color:#00cc66;font-weight:600;">{rate:.0%} '
                            f'({a["approved"]}W-{a["denied"]}L)</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    st.caption("Top applicants/attorneys by ZBA approval rate (min. 10 cases)")
                    st.download_button(
                        "Export Attorney CSV", pd.DataFrame(attorneys).to_csv(index=False),
                        "permitiq_attorneys.csv", "text/csv", key="dl_attorneys"
                    )
        except Exception as e:
            st.caption(f"Attorney data unavailable: {e}")

    # --- Neighborhoods ---
    with intel_tab5:
        try:
            nb_res = requests.get(f"{API_URL}/neighborhoods", timeout=10)
            if nb_res.status_code == 200:
                nb_data = nb_res.json().get("neighborhoods", [])
                if nb_data:
                    for n in nb_data:
                        rate = n["approval_rate"]
                        name = n["neighborhood"]
                        total = n["total_cases"]
                        if rate >= 0.7:
                            color = "#00cc66"
                        elif rate >= 0.5:
                            color = "#ffaa00"
                        else:
                            color = "#ff4444"
                        st.markdown(
                            f'<div style="display:flex;justify-content:space-between;padding:4px 0;">'
                            f'<span>{esc(name)} ({total} cases)</span>'
                            f'<span style="color:{color};font-weight:600;">{rate:.0%}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    st.caption("Approval rate by zoning district/neighborhood (min. 10 cases)")
                    st.download_button(
                        "Export Neighborhoods CSV", pd.DataFrame(nb_data).to_csv(index=False),
                        "permitiq_neighborhoods.csv", "text/csv", key="dl_neighborhoods"
                    )
                else:
                    st.caption("No neighborhood data available.")
        except Exception as e:
            st.caption(f"Neighborhood data unavailable: {e}")

    # --- Denial Patterns ---
    with intel_tab6:
        try:
            dp_res = requests.get(f"{API_URL}/denial_patterns", timeout=10)
            if dp_res.status_code == 200:
                dp_data = dp_res.json()
                st.markdown(
                    f"**{dp_data.get('total_approved', 0):,} approved** vs "
                    f"**{dp_data.get('total_denied', 0):,} denied** — what separates them?"
                )
                st.markdown("")
                patterns = dp_data.get("patterns", [])
                for p in patterns:
                    factor = p["factor"]
                    app_r = p["approved_rate"]
                    den_r = p["denied_rate"]
                    diff = p["difference"]
                    if diff > 0:
                        arrow_color = "#00cc66"
                        arrow = "+"
                    else:
                        arrow_color = "#ff4444"
                        arrow = ""
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #222;">'
                        f'<span>{esc(factor)}</span>'
                        f'<span>Approved: {app_r:.0%} · Denied: {den_r:.0%} · '
                        f'<span style="color:{arrow_color};font-weight:600;">{arrow}{diff:.0%}</span></span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                st.caption("Difference = approved rate minus denied rate. Positive = more common in approved cases.")
        except Exception as e:
            st.caption(f"Denial patterns unavailable: {e}")

    # --- Voting & Provisos ---
    with intel_tab7:
        vc1, vc2 = st.columns(2)
        with vc1:
            st.markdown("**Voting Patterns**")
            try:
                vp_res = requests.get(f"{API_URL}/voting_patterns", timeout=10)
                if vp_res.status_code == 200:
                    vp = vp_res.json()
                    if vp.get("unanimous_total"):
                        st.metric("Unanimous Decisions", f"{vp['unanimous_total']:,}")
                        st.metric("Unanimous Approval Rate", f"{vp['unanimous_approval_rate']:.0%}")
                    if vp.get("split_total"):
                        st.metric("Split Decisions", f"{vp['split_total']:,}")
                        st.metric("Split Approval Rate", f"{vp['split_approval_rate']:.0%}")
            except Exception as e:
                st.caption(f"Voting data unavailable: {e}")

        with vc2:
            st.markdown("**Common Approval Conditions**")
            try:
                pv_res = requests.get(f"{API_URL}/proviso_stats", timeout=10)
                if pv_res.status_code == 200:
                    pv = pv_res.json()
                    conditions = pv.get("conditions", [])
                    if conditions:
                        for c in conditions:
                            st.markdown(f"- **{esc(c['condition'])}**: {c['count']:,} cases ({c['rate']:.0%})")
                    else:
                        st.caption("No proviso data in current dataset.")
            except Exception as e:
                st.caption(f"Proviso data unavailable: {e}")


# =========================
# SITE SELECTION
# =========================

with st.expander("Site Selection — Where Should I Build?", expanded=False):
    st.markdown(
        "Instead of *'What are my odds on this parcel?'*, this answers "
        "*'Which areas of Boston are best for my project type?'*"
    )

    ss_col1, ss_col2, ss_col3 = st.columns(3)
    with ss_col1:
        ss_project = st.selectbox(
            "Project Type", [
                "residential", "commercial", "new_construction", "renovation",
                "addition", "conversion", "adu", "mixed_use", "multi_family"
            ], key="ss_project"
        )
    with ss_col2:
        ss_min_rate = st.slider("Min Approval Rate", 0.3, 0.9, 0.5, 0.05, key="ss_rate")
    with ss_col3:
        ss_limit = st.number_input("Results", 5, 20, 10, key="ss_limit")

    if st.button("Find Best Locations", key="ss_go", type="primary"):
        try:
            ss_res = requests.get(f"{API_URL}/recommend", params={
                "project_type": ss_project,
                "min_approval_rate": ss_min_rate,
                "limit": ss_limit,
            }, timeout=30)

            if ss_res.status_code == 200:
                ss_data = ss_res.json()
                parcels = ss_data.get("parcels", [])

                if parcels:
                    st.success(f"Found {len(parcels)} recommended parcels from "
                               f"{ss_data.get('total_candidates', 0):,} candidates")

                    for i, p in enumerate(parcels[:ss_limit]):
                        prob = p.get("predicted_probability", 0)
                        pid = esc(str(p.get("parcel_id", "")))
                        zoning = esc(str(p.get("zoning", "")))

                        if prob >= 0.7:
                            color = "#00cc66"
                        elif prob >= 0.5:
                            color = "#ffaa00"
                        else:
                            color = "#ff4444"

                        st.markdown(
                            f'<div style="background:#1a1a2e;border:1px solid {color};'
                            f'border-radius:8px;padding:12px;margin:6px 0;">'
                            f'<span style="font-size:18px;font-weight:700;color:{color};">'
                            f'{prob:.0%}</span> &nbsp; '
                            f'Parcel <code>{pid}</code> &nbsp; Zoning: {zoning}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No parcels match your criteria. Try lowering the minimum approval rate.")
            elif ss_res.status_code == 503:
                st.warning("Model not loaded. Start the API with a trained model.")
            else:
                st.error(f"API error: {ss_res.status_code}")
        except Exception as e:
            st.error(f"Site selection unavailable: {e}")


# =========================
# FOOTER
# =========================

st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #555; font-size: 12px;">'
    'PermitIQ v3.0 — Boston Zoning Risk Assessment Platform<br>'
    'Built with 7,500+ real ZBA decisions &middot; 57 leakage-free features &middot; PostGIS spatial data<br>'
    '<em>Statistical risk assessment only. Not legal advice. Consult a zoning attorney.</em>'
    '</div>',
    unsafe_allow_html=True
)
