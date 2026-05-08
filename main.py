import streamlit as st

st.set_page_config(
    page_title="NLP 综合实验平台",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from modules.common import COMMON_CSS  # noqa: E402  (after set_page_config)

st.markdown(COMMON_CSS, unsafe_allow_html=True)

MODULES = [
    ("m1", "🀄 中文词法分析"),
    ("m2", "🔭 句法双引擎透视仪与歧义侦探"),
    ("m3", "📐 语义表示与对比分析"),
    ("m4", "🧠 词义消歧与语义角色标注"),
    ("m5", "💬 篇章分析与指代消解系统"),
    ("m6", "📈 语言模型训练与对比分析"),
    ("m7", "🕸️ 信息抽取与知识图谱构建"),
    ("m8", "🌐 机器翻译机制与质量评测"),
    ("m9", "⚡ 电商评论情感分析与意见挖掘"),
]

# ── Sidebar navigation ─────────────────────────────────────────────────────
with st.sidebar:
    # Logo / title block
    st.markdown(
        """
        <div style="padding:22px 4px 16px;text-align:center">
            <div style="font-size:2.2rem;line-height:1;margin-bottom:8px">🧠</div>
            <div style="color:#ddeeff;font-size:1.02rem;font-weight:800;
                        letter-spacing:.8px;line-height:1.2">
                NLP 综合实验平台
            </div>
            <div style="color:#3d6090;font-size:.68rem;margin-top:5px;
                        letter-spacing:2.5px;text-transform:uppercase">
                Natural Language Processing
            </div>
        </div>
        <div style="height:1px;background:linear-gradient(90deg,
             transparent,rgba(100,160,255,.35),transparent);margin:0 4px 14px"></div>
        <div style="color:#3d6090;font-size:.72rem;font-weight:700;
                    letter-spacing:1.8px;text-transform:uppercase;
                    padding:0 4px 8px">
            实验模块
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = st.radio(
        "模块选择",
        options=[m[0] for m in MODULES],
        format_func=lambda x: next(m[1] for m in MODULES if m[0] == x),
        label_visibility="collapsed",
    )

    # Footer
    st.markdown(
        """
        <div style="height:1px;background:linear-gradient(90deg,
             transparent,rgba(100,160,255,.25),transparent);margin:14px 4px 10px"></div>
        <div style="color:#2a4a6a;font-size:.7rem;text-align:center;
                    letter-spacing:.5px;padding-bottom:8px">
            自然语言处理综合实验 © 2025
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Module routing ─────────────────────────────────────────────────────────
# Wrap in st.empty().container() so the entire page area is replaced as a
# single unit on module switch, preventing residual content from the previous
# module bleeding through during Streamlit's incremental re-render.
_page = st.empty()
with _page.container():
    if selected == "m1":
        from modules.m1_lexical import render
        render()
    elif selected == "m2":
        from modules.m2_syntax import render
        render()
    elif selected == "m3":
        from modules.m3_semantic import render
        render()
    elif selected == "m4":
        from modules.m4_wsd_srl import render
        render()
    elif selected == "m5":
        from modules.m5_discourse import render
        render()
    elif selected == "m6":
        from modules.m6_lm import render
        render()
    elif selected == "m7":
        from modules.m7_ie_kg import render
        render()
    elif selected == "m8":
        from modules.m8_mt import render
        render()
    elif selected == "m9":
        from modules.m9_sentiment import render
        render()
