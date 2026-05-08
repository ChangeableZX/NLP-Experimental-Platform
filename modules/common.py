"""共用样式与辅助函数"""

import streamlit as st

# ── 模块颜色主题 ───────────────────────────────────────────────────────────────
MODULE_GRADIENTS = {
    "m1": "linear-gradient(135deg, #1a3a6e 0%, #2667cc 100%)",
    "m2": "linear-gradient(135deg, #7928ca 0%, #ff0080 100%)",
    "m3": "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)",
    "m4": "linear-gradient(135deg, #8e44ad 0%, #c0392b 100%)",
    "m5": "linear-gradient(135deg, #e67e22 0%, #e74c3c 100%)",
    "m6": "linear-gradient(135deg, #2c3e50 0%, #3498db 100%)",
    "m7": "linear-gradient(135deg, #134e5e 0%, #71b280 100%)",
    "m8": "linear-gradient(135deg, #2980b9 0%, #6dd5fa 100%)",
    "m9": "linear-gradient(135deg, #ee0979 0%, #ff6a00 100%)",
}

SEC_GRADIENTS = {
    "blue":   "linear-gradient(135deg, #667eea, #764ba2)",
    "pink":   "linear-gradient(135deg, #f093fb, #f5576c)",
    "cyan":   "linear-gradient(135deg, #4facfe, #00f2fe)",
    "green":  "linear-gradient(135deg, #43e97b, #38f9d7)",
    "orange": "linear-gradient(135deg, #fa709a, #fee140)",
}


def render_module_header(module_id: str, title: str, subtitle: str) -> None:
    gradient = MODULE_GRADIENTS.get(module_id, MODULE_GRADIENTS["m1"])
    st.markdown(
        f"""
        <div style="background:{gradient};padding:1.4rem 2rem;border-radius:14px;
                    margin-bottom:1.5rem;color:white;box-shadow:0 4px 18px rgba(0,0,0,.15);">
            <h2 style="font-size:1.65rem;font-weight:800;letter-spacing:1px;
                       margin:0 0 .35rem;text-shadow:0 2px 4px rgba(0,0,0,.2);">{title}</h2>
            <p style="opacity:.85;font-size:.88rem;margin:0;letter-spacing:.5px;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sec_header(text: str, color: str = "blue") -> None:
    g = SEC_GRADIENTS.get(color, SEC_GRADIENTS["blue"])
    st.markdown(
        f'<div style="background:{g};color:white;padding:.5rem 1rem;border-radius:8px;'
        f'font-weight:700;font-size:.93rem;margin-bottom:.75rem;">{text}</div>',
        unsafe_allow_html=True,
    )


COMMON_CSS = """
<style>
/* ── App background ────────────────────────────────────────────────────────── */
.stApp, [data-testid="stAppViewContainer"] { background: #f0f4fa !important; }

/* ── Global font size boost ────────────────────────────────────────────────── */
/* Raise the base so Streamlit's default widgets and markdown feel more readable */
html { font-size: 17px !important; }
.stMarkdown p, .stMarkdown li          { font-size: 1rem   !important; line-height: 1.78 !important; }
.stMarkdown h1                         { font-size: 1.65rem !important; }
.stMarkdown h2                         { font-size: 1.35rem !important; }
.stMarkdown h3                         { font-size: 1.15rem !important; }
[data-testid="stWidgetLabel"] p        { font-size: .97rem  !important; font-weight: 500 !important; }
.stTextInput input, .stTextArea textarea { font-size: .97rem !important; }
.stSelectbox div[data-baseweb="select"] { font-size: .97rem !important; }

/* ── Sidebar shell ──────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    min-width: 350px !important;
    max-width: 350px !important;
}
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #0b1828 0%, #122040 55%, #0b1a34 100%) !important;
    min-width: 350px !important;
    max-width: 350px !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #a8c8e8 !important; }
[data-testid="stSidebar"] hr   { border-color: #1e3a5a !important; }

/* ── Sidebar card navigation ───────────────────────────────────────────────── */
[data-testid="stSidebar"] div[role="radiogroup"] {
    display: flex !important;
    flex-direction: column !important;
    gap: 5px !important;
    padding: 2px 0 4px !important;
}

/* Base card */
[data-testid="stSidebar"] div[role="radiogroup"] label {
    display: flex !important;
    align-items: center !important;
    background: rgba(255,255,255,.045) !important;
    border: 1px solid rgba(255,255,255,.08) !important;
    border-left: 3px solid transparent !important;
    border-radius: 10px !important;
    padding: 10px 13px !important;
    cursor: pointer !important;
    transition: background .18s, border-color .18s, box-shadow .18s !important;
    color: #8ab4d4 !important;
    font-size: .875rem !important;
    font-weight: 500 !important;
    margin: 0 !important;
    line-height: 1.35 !important;
    letter-spacing: .01em !important;
}

/* Hover */
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(80,140,255,.11) !important;
    border-color: rgba(80,150,255,.28) !important;
    border-left-color: rgba(100,180,255,.55) !important;
    color: #c4dcf4 !important;
}

/* Hide the round radio-dot indicator */
[data-testid="stSidebar"] div[role="radiogroup"] [data-testid="stRadioButton"],
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child > div:first-child {
    display: none !important;
}

/* Selected card — aria-checked on label (Streamlit ≥ 1.30) */
[data-testid="stSidebar"] div[role="radiogroup"] label[aria-checked="true"] {
    background: linear-gradient(135deg,
        rgba(38,103,204,.30) 0%,
        rgba(72,40,190,.20) 100%) !important;
    border-color: rgba(100,165,255,.42) !important;
    border-left-color: #6aabff !important;
    color: #d8eeff !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 14px rgba(38,103,204,.22) !important;
}

/* Fallback: :has() for environments where aria-checked lands on the input */
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg,
        rgba(38,103,204,.30) 0%,
        rgba(72,40,190,.20) 100%) !important;
    border-color: rgba(100,165,255,.42) !important;
    border-left-color: #6aabff !important;
    color: #d8eeff !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 14px rgba(38,103,204,.22) !important;
}

/* ── Tabs ───────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #e4eaf4;
    border-radius: 10px;
    padding: 4px 6px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #4a6080;
    font-weight: 600;
    padding: 7px 20px;
    border: none !important;
    font-size: .92rem !important;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #1a3a6e !important;
    box-shadow: 0 2px 8px rgba(0,0,0,.10);
}

/* ── Metrics ─────────────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: white;
    border: 1px solid #dde6f0;
    border-radius: 10px;
    padding: .75rem 1rem;
    box-shadow: 0 2px 6px rgba(0,0,0,.05);
}

/* ── Misc ────────────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
hr { border-color: #d0dcea !important; }
[data-testid="stInfo"]    { border-radius: 8px; }
[data-testid="stSuccess"] { border-radius: 8px; }
[data-testid="stWarning"] { border-radius: 8px; }
[data-testid="stError"]   { border-radius: 8px; }

/* ── Buttons (main content) ──────────────────────────────────────────────────── */
.main .stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: .95rem !important;
}
</style>
"""
