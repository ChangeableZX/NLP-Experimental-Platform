"""模块9：电商评论情感分析与意见挖掘"""

import random
import streamlit as st
import plotly.graph_objects as go
from transformers import pipeline

from .common import render_module_header

_LABEL_META = {
    "positive": ("积极 Positive", "#16a34a", "😊"),
    "negative": ("消极 Negative", "#dc2626", "😔"),
    "neutral":  ("中性 Neutral",  "#d97706", "😐"),
}

_SENTIMENT_CSS = """
<style>
.s9-badge-pos { display:inline-block; padding:7px 22px; border-radius:20px;
    background:rgba(22,163,74,.12); border:1px solid #16a34a; color:#15803d; font-weight:700; font-size:1.05rem; }
.s9-badge-neg { display:inline-block; padding:7px 22px; border-radius:20px;
    background:rgba(220,38,38,.10); border:1px solid #dc2626; color:#b91c1c; font-weight:700; font-size:1.05rem; }
.s9-badge-neu { display:inline-block; padding:7px 22px; border-radius:20px;
    background:rgba(217,119,6,.10); border:1px solid #d97706; color:#b45309; font-weight:700; font-size:1.05rem; }
.s9-sec-hdr {
    color:#1a3a6e; font-size:1rem; font-weight:700;
    padding-bottom:6px; border-bottom:1px solid rgba(26,58,110,.2);
    margin-bottom:12px; letter-spacing:.5px;
}
.s9-review-row {
    background:rgba(26,58,110,.03); border:1px solid rgba(26,58,110,.10);
    border-radius:8px; padding:10px 14px; margin:5px 0;
    display:flex; justify-content:space-between; align-items:center; gap:10px;
}
.s9-stat-row {
    border-radius:8px; padding:12px 16px; margin:6px 0;
    display:flex; justify-content:space-between; align-items:center;
}
</style>
"""


def _norm(label: str) -> str:
    return label.strip().lower()


def _meta(label: str):
    return _LABEL_META.get(_norm(label), (label, "#2667cc", "❓"))


@st.cache_resource(show_spinner="⏳ 正在加载情感分析模型（首次需要下载，请稍候）…")
def load_sentiment_model():
    return pipeline(
        "text-classification",
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        top_k=None,
    )


def _analyze(pipe, text: str):
    scores = pipe(text.strip())[0]
    top = max(scores, key=lambda x: x["score"])
    return _norm(top["label"]), top["score"], scores


def _gauge_chart(score: float, label: str) -> go.Figure:
    _, color, _ = _meta(label)
    pct = round(score * 100, 1)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 40, "color": color, "family": "Arial Black"}},
        title={"text": "置信度 Confidence", "font": {"size": 14, "color": "#4a6a8a"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#c0cce0",
                     "tickfont": {"color": "#6a8aaa", "size": 10}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "rgba(220,38,38,0.08)"},
                {"range": [40, 70], "color": "rgba(217,119,6,0.08)"},
                {"range": [70,100], "color": "rgba(22,163,74,0.08)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.80, "value": pct},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=260, margin=dict(t=50, b=0, l=20, r=20),
    )
    return fig


def _score_bars(all_scores: list):
    COLOR_MAP = {"positive": "#16a34a", "negative": "#dc2626", "neutral": "#d97706"}
    for item in sorted(all_scores, key=lambda x: x["score"], reverse=True):
        lbl_cn, _, icon = _meta(_norm(item["label"]))
        color = COLOR_MAP.get(_norm(item["label"]), "#2667cc")
        pct = item["score"] * 100
        st.html(f"""
        <div style="margin:7px 0">
            <div style="display:flex;justify-content:space-between;margin-bottom:3px">
                <span style="color:{color};font-size:.85rem">{icon} {lbl_cn}</span>
                <span style="color:{color};font-size:.85rem;font-weight:700">{pct:.1f}%</span>
            </div>
            <div style="background:rgba(0,0,0,.08);border-radius:4px;height:7px;overflow:hidden">
                <div style="width:{pct:.1f}%;height:100%;background:{color};border-radius:4px"></div>
            </div>
        </div>
        """)


def _render_sentiment_panel(pipe, text: str, container=None):
    tgt = container or st
    label, score, all_scores = _analyze(pipe, text)
    lbl_cn = _LABEL_META[label][0] if label in _LABEL_META else label
    _, color, icon = _meta(label)
    badge = "pos" if label == "positive" else "neg" if label == "negative" else "neu"

    tgt.html(f"""
    <div style="text-align:center;margin:6px 0 0">
        <div style="font-size:.8rem;color:#4a6a8a;margin-bottom:8px">分析结果</div>
        <div class="s9-badge-{badge}">{icon} {lbl_cn}</div>
    </div>
    """)
    tgt.plotly_chart(_gauge_chart(score, label), width="stretch")
    tgt.html('<div class="s9-sec-hdr" style="margin-top:4px">各情感得分明细</div>')
    with tgt:
        _score_bars(all_scores)


def render():
    render_module_header(
        "m9",
        "⚡ 电商评论情感分析与意见挖掘",
        "DistilBERT 多语言情感分类 · 显式 vs 隐式情感对比 · 批量舆情仪表盘",
    )

    with st.expander("📖 功能说明与技术栈", expanded=False):
        st.markdown("""
| 功能 | 说明 | 技术 / 库 |
|------|------|-----------|
| 单文本情感分析 | 输入任意中文评论，输出积极 / 消极 / 中性分类结果及各类别置信度仪表盘 | `transformers` (distilbert-base-multilingual-cased-sentiments-student) |
| 显式 vs 隐式情感对比 | 对比直接含褒贬词的显式情感与通过客观事实描述传达的隐式情感的模型识别效果差异 | `transformers` |
| 批量舆情仪表盘 | 从预设评论池中随机采样并批量分析，生成好评率、饼图及逐条情感明细列表 | `transformers`, `plotly` |
""")

    st.markdown(_SENTIMENT_CSS, unsafe_allow_html=True)

    pipe = load_sentiment_model()

    tab1, tab2, tab3 = st.tabs([
        "🎯  单文本情感分析",
        "🔬  显式 vs 隐式情感",
        "📊  批量舆情仪表盘",
    ])

    # ── Tab 1: 单文本情感分析 ──────────────────────────────────────────────
    with tab1:
        left, right = st.columns([1, 1], gap="large")

        with left:
            st.html('<div class="s9-sec-hdr">📝 输入商品评论</div>')
            if "t1_comment" not in st.session_state:
                st.session_state["t1_comment"] = ""
            if "t1_autorun" not in st.session_state:
                st.session_state["t1_autorun"] = False
            _T1_EX = [
                ("好评示例", "这款手机拍照效果非常好，续航也很棒，性价比超高，强烈推荐！"),
                ("差评示例", "收到货发现屏幕有划痕，客服态度恶劣，完全不处理，非常失望。"),
                ("中性示例", "商品按时到达，包装完好，和描述基本一致，中规中矩。"),
            ]
            st.caption("🏷️ 示例评论：")
            _t1_cols = st.columns(len(_T1_EX))
            for _i, (_lbl, _txt) in enumerate(_T1_EX):
                if _t1_cols[_i].button(_lbl, key=f"t1_ex_{_i}", use_container_width=True):
                    st.session_state["t1_comment"] = _txt
                    st.session_state["t1_autorun"] = True
            user_text = st.text_area(
                "comment_input",
                placeholder="在此输入中文商品评论…\n\n例如：这款手机拍照效果非常好，续航也很棒，性价比超高，强烈推荐！",
                height=130,
                label_visibility="collapsed",
                key="t1_comment",
            )
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("🔍  开始情感分析", key="t1_run")

        with right:
            st.html('<div class="s9-sec-hdr">📈 分析结果</div>')
            result_slot = st.container()
            if run_btn or st.session_state.get("t1_autorun", False):
                st.session_state["t1_autorun"] = False
                if not (user_text or "").strip():
                    result_slot.warning("请先输入一段评论文本！")
                else:
                    with st.spinner("正在推理…"):
                        _render_sentiment_panel(pipe, user_text.strip(), result_slot)
            else:
                result_slot.html("""
                <div style="height:300px;display:flex;align-items:center;justify-content:center;
                     border:1px dashed rgba(26,58,110,.2);border-radius:12px">
                    <div style="text-align:center;color:#4a6a8a">
                        <div style="font-size:2.5rem;margin-bottom:10px">📈</div>
                        <div style="font-size:.9rem">输入评论后点击
                            <b style="color:#2667cc">「开始情感分析」</b>
                        </div>
                        <div style="font-size:.8rem;margin-top:6px;color:#6a8aaa">结果将在此处实时展示</div>
                    </div>
                </div>
                """)

    # ── Tab 2: 显式 vs 隐式情感 ───────────────────────────────────────────
    with tab2:
        st.html("""
        <div style="background:linear-gradient(135deg,rgba(38,103,204,.07),rgba(80,0,180,.05));
             border:1px solid rgba(38,103,204,.2);border-radius:14px;padding:20px 24px;margin-bottom:16px">
            <div style="color:#1a3a6e;font-weight:800;font-size:1.05rem;margin-bottom:14px;letter-spacing:.5px">
                🧠 情感语言学科普 — 什么是显式情感 vs 隐式情感？
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
                <div style="background:rgba(22,163,74,.06);border:1px solid rgba(22,163,74,.2);
                     border-left:3px solid #16a34a;border-radius:10px;padding:14px 16px">
                    <div style="color:#15803d;font-weight:700;margin-bottom:8px;font-size:.95rem">
                        ✨ 显式情感（Explicit Sentiment）
                    </div>
                    <div style="color:#374151;font-size:.87rem;line-height:1.75">
                        直接使用<b style="color:#15803d">褒贬义词汇</b>或<b style="color:#15803d">情感形容词</b>
                        表达观点，情感信号强烈，模型易于捕捉。<br><br>
                        <span style="color:#6a8aaa">典型特征：</span>
                        情感副词（"非常好"、"极差"）、感叹词（"太棒了！"）、主观评价动词（"喜欢"、"讨厌"）
                        <br><br>
                        <div style="background:rgba(22,163,74,.08);border-radius:6px;padding:8px 12px;
                             font-size:.85rem;color:#374151;border-left:2px solid #16a34a">
                            💬 "这款产品<b style="color:#15803d">质量超级棒</b>，我
                            <b style="color:#15803d">非常喜欢</b>，<b style="color:#15803d">强烈推荐</b>！"
                        </div>
                    </div>
                </div>
                <div style="background:rgba(217,119,6,.06);border:1px solid rgba(217,119,6,.2);
                     border-left:3px solid #d97706;border-radius:10px;padding:14px 16px">
                    <div style="color:#b45309;font-weight:700;margin-bottom:8px;font-size:.95rem">
                        🔍 隐式情感（Implicit Sentiment）
                    </div>
                    <div style="color:#374151;font-size:.87rem;line-height:1.75">
                        通过<b style="color:#b45309">客观事实陈述</b>间接传递情感，无明显情感词，
                        需结合常识与语境推断倾向。<br><br>
                        <span style="color:#6a8aaa">典型特征：</span>
                        客观数据（"只用了4小时"）、期望落差对比、使用结果描述
                        <br><br>
                        <div style="background:rgba(217,119,6,.08);border-radius:6px;padding:8px 12px;
                             font-size:.85rem;color:#374151;border-left:2px solid #d97706">
                            💬 "手机<b style="color:#b45309">玩游戏半小时就没电了</b>，
                            充一次电<b style="color:#b45309">只能用 4 小时</b>。"
                        </div>
                    </div>
                </div>
            </div>
            <div style="margin-top:14px;padding:10px 14px;background:rgba(38,103,204,.05);
                 border-radius:8px;font-size:.83rem;color:#4a6a8a;text-align:center">
                ⚠️ 提示：对于隐式情感，轻量模型的识别准确率通常低于显式情感，这正是自然语言理解的挑战所在。
            </div>
        </div>
        """)

        col_exp, col_imp = st.columns(2, gap="large")

        with col_exp:
            st.html("""<div class="s9-sec-hdr" style="color:#15803d;border-color:rgba(22,163,74,.25)">
                ✨ 显式情感评价 — Explicit</div>""")
            if "t2_exp" not in st.session_state:
                st.session_state["t2_exp"] = ""
            _EXP_EX = [
                ("正向显式", "这款手机太棒了！屏幕超级清晰，手感非常好，强烈推荐！"),
                ("负向显式", "质量极差，做工粗糙，完全不值这个价格，后悔购买，一星差评！"),
            ]
            _exp_cols = st.columns(len(_EXP_EX))
            for _i, (_lbl, _txt) in enumerate(_EXP_EX):
                if _exp_cols[_i].button(_lbl, key=f"t2_exp_ex_{_i}", use_container_width=True):
                    st.session_state["t2_exp"] = _txt
            st.text_area(
                "exp_input",
                placeholder="请输入带有明显褒贬词的评论…\n\n例如：这款手机太棒了！屏幕超级清晰，手感非常好，强烈推荐！",
                height=110,
                key="t2_exp",
                label_visibility="collapsed",
            )
            run_exp = st.button("分析显式情感 →", key="t2_run_exp")
            exp_result = st.container()
            if run_exp:
                txt = (st.session_state.get("t2_exp") or "").strip()
                if not txt:
                    exp_result.warning("请输入显式情感评论！")
                else:
                    with st.spinner("分析中…"):
                        _render_sentiment_panel(pipe, txt, exp_result)

        with col_imp:
            st.html("""<div class="s9-sec-hdr" style="color:#b45309;border-color:rgba(217,119,6,.25)">
                🔍 隐式客观描述 — Implicit</div>""")
            if "t2_imp" not in st.session_state:
                st.session_state["t2_imp"] = ""
            _IMP_EX = [
                ("负向隐式", "手机玩游戏半小时就没电了，充一次电只能用 4 小时。"),
                ("正向隐式", "这双鞋穿了三个月，鞋底完好，外形依旧如新，每天都在穿。"),
            ]
            _imp_cols = st.columns(len(_IMP_EX))
            for _i, (_lbl, _txt) in enumerate(_IMP_EX):
                if _imp_cols[_i].button(_lbl, key=f"t2_imp_ex_{_i}", use_container_width=True):
                    st.session_state["t2_imp"] = _txt
            st.text_area(
                "imp_input",
                placeholder="请输入无明显情感词的客观事实描述…\n\n例如：手机玩游戏半小时就没电了，充一次电只能用 4 小时。",
                height=110,
                key="t2_imp",
                label_visibility="collapsed",
            )
            run_imp = st.button("分析隐式情感 →", key="t2_run_imp")
            imp_result = st.container()
            if run_imp:
                txt = (st.session_state.get("t2_imp") or "").strip()
                if not txt:
                    imp_result.warning("请输入隐式情感描述！")
                else:
                    with st.spinner("分析中…"):
                        _render_sentiment_panel(pipe, txt, imp_result)

    # ── Tab 3: 批量舆情仪表盘 ─────────────────────────────────────────────
    with tab3:
        SAMPLE_POOL = [
            "这款耳机音质超好，低音浑厚，佩戴舒适，续航强劲，性价比极高，五星好评！",
            "手机到手发现屏幕有划痕，简直太失望了，卖家还拒绝退货！",
            "商品已收到，和图片描述一致，包装完好，快递速度正常。",
            "口红颜色太美了，上唇持久不脱色，滋润不干燥，完美产品！",
            "这双鞋穿了三天鞋底就开胶了，做工粗糙，质量令人失望，不推荐！",
            "面膜用了一周，感觉皮肤水润了一些，效果还可以，会继续回购。",
            "笔记本电脑开机极速，散热优秀，做视频剪辑完全够用，超值！",
            "充电宝充了五个小时才充满，输出功率非常慢，远不如描述所言。",
            "衣服面料普通，做工一般，但价格便宜，凑合着穿吧。",
            "这款咖啡机操作简单，萃取咖啡香气浓郁，每天早上必备！",
            "买回来的键盘有两个按键不灵敏，联系客服多次也没人处理，极差！",
            "洗发水洗完头发清爽，没有特别香味，也没有不舒服，中规中矩。",
            "这个价位能买到这种品质真的难得，外观精美，用料扎实！",
            "空调安装完师傅直接走了，留下一地垃圾，制冷效果也一般。",
            "蓝牙音箱连接稳定，音量足够日常使用，适合放在书桌上。",
            "连衣裙收到后颜色和图片差很多，而且做工很粗糙，很不满意！",
            "平板电脑用来看视频刷剧非常流畅，画质清晰，性价比不错。",
            "这款护肤霜保湿效果很好，用了两周皮肤明显改善，强力推荐！",
        ]

        if "batch_results" not in st.session_state:
            st.session_state["batch_results"] = None

        ctrl_col, dash_col = st.columns([1, 2], gap="large")

        with ctrl_col:
            st.html('<div class="s9-sec-hdr">⚡ 数据生成与分析控制</div>')
            n = st.slider("生成评论数量", min_value=5, max_value=15, value=12)
            gen_btn = st.button("🚀  生成测试舆情数据并分析", key="t3_gen")

            if gen_btn:
                selected = random.sample(SAMPLE_POOL, min(n, len(SAMPLE_POOL)))
                results = []
                prog = st.progress(0, text="正在批量分析…")
                for i, text in enumerate(selected):
                    label, score, all_sc = _analyze(pipe, text)
                    results.append({"text": text, "label": label, "score": score, "all": all_sc})
                    prog.progress((i + 1) / len(selected), text=f"正在分析第 {i+1}/{len(selected)} 条…")
                prog.empty()
                st.session_state["batch_results"] = results

            if st.session_state["batch_results"]:
                res   = st.session_state["batch_results"]
                pos   = sum(1 for r in res if r["label"] == "positive")
                neg   = sum(1 for r in res if r["label"] == "negative")
                neu   = sum(1 for r in res if r["label"] == "neutral")
                total = len(res)

                st.markdown("<br>", unsafe_allow_html=True)
                st.html('<div class="s9-sec-hdr">📊 统计概览</div>')

                COLOR_MAP = {"positive": "#16a34a", "negative": "#dc2626", "neutral": "#d97706"}
                BG_MAP    = {
                    "positive": "rgba(22,163,74,.08)",
                    "negative": "rgba(220,38,38,.08)",
                    "neutral":  "rgba(217,119,6,.08)",
                }
                for count, lbl, icon, key in [
                    (pos,   "积极评价", "😊", "positive"),
                    (neg,   "消极评价", "😔", "negative"),
                    (neu,   "中性评价", "😐", "neutral"),
                    (total, "总计分析", "📋", None),
                ]:
                    color = COLOR_MAP.get(key, "#2667cc") if key else "#2667cc"
                    bg    = BG_MAP.get(key, "rgba(38,103,204,.05)") if key else "rgba(38,103,204,.05)"
                    st.html(f"""
                    <div class="s9-stat-row"
                         style="background:{bg};border:1px solid {color}33;border-left:3px solid {color}">
                        <span style="color:#4a6a8a;font-size:.88rem">{icon} {lbl}</span>
                        <span style="color:{color};font-size:1.6rem;font-weight:800">{count}</span>
                    </div>
                    """)

                rate = pos / total * 100 if total else 0
                st.html(f"""
                <div style="margin-top:12px;background:rgba(38,103,204,.05);
                     border:1px solid rgba(38,103,204,.15);border-radius:10px;
                     padding:14px 18px;text-align:center">
                    <div style="color:#4a6a8a;font-size:.8rem;margin-bottom:4px">好评率</div>
                    <div style="font-size:2rem;font-weight:900;color:#15803d">{rate:.1f}%</div>
                </div>
                """)

        with dash_col:
            if not st.session_state["batch_results"]:
                st.html("""
                <div style="height:420px;display:flex;align-items:center;justify-content:center;
                     border:1px dashed rgba(26,58,110,.2);border-radius:14px">
                    <div style="text-align:center;color:#4a6a8a">
                        <div style="font-size:3rem;margin-bottom:14px">📊</div>
                        <div style="font-size:.95rem">点击左侧按钮生成舆情数据</div>
                        <div style="font-size:.82rem;margin-top:6px;color:#6a8aaa">
                            系统将自动分析并渲染可视化仪表盘
                        </div>
                    </div>
                </div>
                """)
            else:
                res   = st.session_state["batch_results"]
                pos   = sum(1 for r in res if r["label"] == "positive")
                neg   = sum(1 for r in res if r["label"] == "negative")
                neu   = sum(1 for r in res if r["label"] == "neutral")
                total = len(res)

                st.html('<div class="s9-sec-hdr">🥧 口碑比例分布 — Sentiment Distribution</div>')

                fig_pie = go.Figure(data=[go.Pie(
                    labels=["积极 Positive", "消极 Negative", "中性 Neutral"],
                    values=[pos, neg, neu],
                    hole=0.52,
                    marker=dict(
                        colors=["#16a34a", "#dc2626", "#d97706"],
                        line=dict(color="white", width=2),
                    ),
                    textfont=dict(size=12, color="white"),
                    hovertemplate="<b>%{label}</b><br>数量: %{value}<br>占比: %{percent}<extra></extra>",
                    pull=[0.04, 0.04, 0.04],
                )])
                fig_pie.add_annotation(
                    text=f"{total}<br>条评论",
                    x=0.5, y=0.5,
                    font=dict(size=18, color="#1a3a6e"),
                    showarrow=False,
                )
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=300, margin=dict(t=10, b=10, l=20, r=20),
                    legend=dict(
                        orientation="h", yanchor="bottom", y=-0.12,
                        xanchor="center", x=0.5,
                        font=dict(color="#4a6a8a", size=12),
                    ),
                )
                st.plotly_chart(fig_pie, width="stretch")

                st.html('<div class="s9-sec-hdr">📋 逐条情感明细列表</div>')

                COLOR_MAP = {"positive": "#16a34a", "negative": "#dc2626", "neutral": "#d97706"}
                ICON_MAP  = {"positive": "😊", "negative": "😔", "neutral": "😐"}
                LABEL_CN  = {"positive": "积极", "negative": "消极", "neutral": "中性"}

                for r in res:
                    lbl    = r["label"]
                    color  = COLOR_MAP.get(lbl, "#2667cc")
                    icon   = ICON_MAP.get(lbl, "❓")
                    lbl_cn = LABEL_CN.get(lbl, lbl)
                    pct    = r["score"] * 100
                    short  = r["text"][:52] + ("…" if len(r["text"]) > 52 else "")
                    st.html(f"""
                    <div class="s9-review-row" style="border-left:3px solid {color}">
                        <div style="color:#374151;font-size:.86rem;flex:1">{short}</div>
                        <div style="text-align:right;white-space:nowrap">
                            <div style="color:{color};font-size:.88rem;font-weight:700">{icon} {lbl_cn}</div>
                            <div style="color:#6a8aaa;font-size:.78rem;margin-top:2px">{pct:.1f}%</div>
                        </div>
                    </div>
                    """)
