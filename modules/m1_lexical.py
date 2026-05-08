"""模块1：中文词法分析（spaCy 版，兼容 Python 3.14）"""

import re
import html as htmllib
from collections import Counter

import zhconv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import spacy
import streamlit as st

from .common import render_module_header, sec_header


# ── spaCy 中文模型 ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="正在加载中文语言模型…")
def _load_zh_nlp():
    return spacy.load("zh_core_web_sm")


# ── 保留字符白名单（文本规范化用） ────────────────────────────────────────────
_KEEP_RE = re.compile(
    r"[^一-鿿㐀-䶿"
    r" -~\s"
    r"，。！？、；："
    r"‘’“”"
    r"（）【】《》"
    r"…—～·]+"
)

STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也",
    "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那",
    "他", "她", "们", "我们", "这个", "那个", "什么", "怎么", "但", "但是", "还", "就是", "可以",
}

# spaCy Universal POS → 中文标签 + 颜色
POS_MAP = {
    "NOUN":  ("名词",   "#e74c3c"),
    "VERB":  ("动词",   "#2980b9"),
    "ADJ":   ("形容词", "#27ae60"),
    "ADV":   ("副词",   "#7d3c98"),
    "PROPN": ("专有名词","#8e44ad"),
    "PRON":  ("代词",   "#17a589"),
    "NUM":   ("数词",   "#b7950b"),
    "PUNCT": ("标点",   "#ccd1d1"),
    "PART":  ("助词",   "#909497"),
    "ADP":   ("介词",   "#e67e22"),
    "CCONJ": ("连词",   "#626567"),
    "CONJ":  ("连词",   "#626567"),
    "DET":   ("限定词", "#f39c12"),
    "INTJ":  ("感叹词", "#e67e22"),
    "SYM":   ("符号",   "#95a5a6"),
    "X":     ("其他",   "#aab7b8"),
}

PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
           "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]


# ── 文本规范化 ────────────────────────────────────────────────────────────────
def _fullwidth_to_halfwidth(text):
    out = []
    for ch in text:
        cp = ord(ch)
        if cp == 0x3000:
            out.append(" ")
        elif 0xFF01 <= cp <= 0xFF5E:
            out.append(chr(cp - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def normalize_text(text):
    steps, cur = [], text
    for fn, name in [
        (lambda t: zhconv.convert(t, "zh-hans"), "繁体 → 简体"),
        (_fullwidth_to_halfwidth, "全角 → 半角"),
    ]:
        nxt = fn(cur)
        steps.append({"name": name, "result": nxt, "changed": nxt != cur})
        cur = nxt
    nxt = _KEEP_RE.sub("", cur)
    steps.append({"name": "去除特殊符号", "result": nxt, "changed": nxt != cur})
    cur = nxt
    nxt = re.sub(r"[ \t\r\n]+", " ", cur).strip()
    steps.append({"name": "规范空白字符", "result": nxt, "changed": nxt != cur})
    return {"original": text, "normalized": nxt, "steps": steps}


# ── 分词（三种模式） ──────────────────────────────────────────────────────────
def do_segment(text, nlp):
    """
    精确模式：spaCy 统计模型（最优分词）
    全模式：字符级拆分（召回率最高，颗粒度最细）
    搜索引擎模式：在精确分词基础上，将 3 字以上词条再切为 2-gram 叠加输出
    """
    doc = nlp(text)
    precise = [t.text for t in doc if t.text.strip()]

    # 全模式：每个汉字单独输出，ASCII 序列保留
    full = []
    for ch in text:
        if ch.strip():
            full.append(ch)

    # 搜索引擎模式：精确词 + 长词拆分
    search = list(precise)
    for tok in precise:
        hanzi = [c for c in tok if "一" <= c <= "鿿"]
        if len(hanzi) >= 3:
            for i in range(len(tok) - 1):
                search.append(tok[i:i + 2])

    return {"precise": precise, "full": full, "search": search}


def top_freq(words, n=5):
    filt = [w for w in words if len(w) >= 2 and w not in STOPWORDS and re.search(r"\w", w)]
    return Counter(filt).most_common(n)


# ── 词性标注 ──────────────────────────────────────────────────────────────────
def do_pos_tag(text, nlp):
    doc = nlp(text)
    result = []
    for t in doc:
        if not t.text.strip():
            continue
        label, color = POS_MAP.get(t.pos_, ("其他", "#aab7b8"))
        result.append({"word": t.text, "flag": t.pos_, "label": label, "color": color})
    return result


# ── 图表 ──────────────────────────────────────────────────────────────────────
def chart_freq(freq):
    if not freq:
        return None
    words  = [w for w, _ in freq]
    counts = [c for _, c in freq]
    fig = go.Figure(go.Bar(
        x=counts[::-1], y=words[::-1],
        orientation="h",
        marker_color=PALETTE[:len(words)],
        text=[str(c) for c in counts[::-1]],
        textposition="outside",
    ))
    fig.update_layout(
        title="高频词 Top 5",
        xaxis_title="频次",
        height=220,
        margin=dict(l=10, r=50, t=40, b=10),
        plot_bgcolor="rgba(247,250,252,1)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
    )
    return fig


def chart_comparison(results, sentence):
    keys   = ["precise", "full", "search"]
    labels = ["精确模式", "全模式", "搜索引擎模式"]
    colors = ["#4e79a7", "#f28e2b", "#e15759"]
    token_cnt  = [len(results[k]) for k in keys]
    unique_cnt = [len(set(results[k])) for k in keys]
    avg_len    = [sum(len(w) for w in results[k]) / max(len(results[k]), 1) for k in keys]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["分词总数", "唯一词数", "平均词长（字）"],
    )
    for col, vals in enumerate([token_cnt, unique_cnt, avg_len], 1):
        fig.add_trace(
            go.Bar(
                x=labels, y=vals,
                marker_color=colors,
                text=[f"{v:.2f}" if isinstance(v, float) else str(v) for v in vals],
                textposition="outside",
                showlegend=False,
            ),
            row=1, col=col,
        )
    short = sentence[:18] + "…" if len(sentence) > 18 else sentence
    fig.update_layout(
        title=dict(text=f'"{short}" — 三种算法统计对比', x=0.5),
        height=300,
        margin=dict(l=10, r=10, t=70, b=10),
        plot_bgcolor="rgba(247,250,252,1)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )
    return fig


# ── HTML 辅助 ─────────────────────────────────────────────────────────────────
def e(s):
    return htmllib.escape(str(s))


def chips_html(words, diff=None):
    parts = []
    for w in words:
        cls = "chip chip-diff" if (diff and w in diff) else "chip"
        parts.append(f'<span class="{cls}">{e(w)}</span>')
    return '<div class="chips-box">' + "".join(parts) + "</div>"


# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = """
<style>
.step-box{padding:.48rem .75rem;border-radius:6px;background:#f7fafc;
          border-left:3px solid #cbd5e0;margin-bottom:.4rem;font-size:.85rem;}
.step-box.chg{border-left-color:#48bb78;}
.step-name{font-size:.68rem;font-weight:700;text-transform:uppercase;
           letter-spacing:.4px;color:#718096;margin-bottom:.2rem;}
.badge-chg{display:inline-block;font-size:.6rem;padding:.05rem .35rem;
           background:#c6f6d5;color:#276749;border-radius:8px;margin-left:.35rem;}
.step-text{color:#2d3748;line-height:1.5;word-break:break-all;}
.chips-box{display:flex;flex-wrap:wrap;gap:4px;padding:8px;background:#f7fafc;
           border-radius:6px;max-height:150px;overflow-y:auto;margin-bottom:.2rem;}
.chip{display:inline-block;padding:2px 7px;border-radius:4px;
      font-size:.83rem;background:white;border:1px solid #e2e8f0;color:#2d3748;}
.chip-diff{background:#fef3c7!important;border-color:#fcd34d!important;}
.pos-box{display:flex;flex-wrap:wrap;gap:4px;padding:8px;background:#f7fafc;
         border-radius:8px;max-height:260px;overflow-y:auto;}
.pw{display:inline-flex;flex-direction:column;align-items:center;
    padding:3px 7px;border-radius:5px;cursor:default;}
.pw .w{font-size:.88rem;color:white;font-weight:600;white-space:nowrap;}
.pw .ft{font-size:.58rem;color:rgba(255,255,255,.75);margin-top:1px;}
.pos-legend{display:flex;flex-wrap:wrap;gap:7px;margin-top:8px;}
.leg{display:flex;align-items:center;gap:4px;font-size:.72rem;color:#718096;}
.leg-dot{width:9px;height:9px;border-radius:50%;display:inline-block;flex-shrink:0;}
.algo-box{background:#f7fafc;border-radius:8px;padding:.75rem;}
.algo-title{font-size:.76rem;font-weight:700;text-transform:uppercase;
            letter-spacing:.4px;margin-bottom:.45rem;padding-bottom:.3rem;border-bottom:2px solid #e2e8f0;}
.algo-stats{font-size:.71rem;color:#718096;margin-top:.4rem;}
.cmp-note{padding:.65rem 1rem;background:#ebf8ff;border-left:3px solid #4a6fa5;
          border-radius:6px;font-size:.82rem;color:#2c5282;line-height:1.55;margin-top:.7rem;}
</style>
"""


# ── 主渲染函数 ────────────────────────────────────────────────────────────────
def render():
    render_module_header(
        "m1",
        "📝 中文词法分析",
        "文本规范化 · 中文分词 · 词频统计 · 词性标注 · 分词算法对比",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    with st.expander("📖 功能说明与技术栈", expanded=False):
        st.markdown("""
| 功能 | 说明 | 技术 / 库 |
|------|------|-----------|
| 文本规范化 | 繁简体互转、全角→半角、去除特殊符号、统一空白字符 | `zhconv`, `re` |
| 中文分词 | 精确模式（spaCy 统计模型）、全模式（字符级拆分）、搜索引擎模式（精确词 + 长词拆分）| `spaCy` zh_core_web_sm |
| 词性标注 | Universal POS 标注（名词/动词/形容词等），彩色标签展示 | `spaCy` zh_core_web_sm |
| 高频词统计 | 过滤停用词后统计词频，绘制 Top-5 柱状图 | `collections.Counter`, `matplotlib` |
| 算法对比 | 三种分词模式在词数、唯一词数、平均词长三维度上的可视化对比 | `matplotlib` |
""")

    if "m1_text" not in st.session_state:
        st.session_state["m1_text"] = ""
    if "m1_autorun" not in st.session_state:
        st.session_state["m1_autorun"] = False
    _M1_EX = [
        ("📰 新闻语料", "人工智能技术正在深刻改变各行各业的生产方式，推动经济高质量发展。"),
        ("📖 古诗词", "月落乌啼霜满天，江枫渔火对愁眠。姑苏城外寒山寺，夜半钟声到客船。"),
        ("💬 口语对话", "今天天气不错，我们去公园散步，顺便买点水果回来吧。"),
    ]
    _ex_cols = st.columns(len(_M1_EX))
    for _i, (_lbl, _txt) in enumerate(_M1_EX):
        if _ex_cols[_i].button(_lbl, key=f"m1_ex_{_i}", use_container_width=True):
            st.session_state["m1_text"] = _txt
            st.session_state["m1_autorun"] = True
    main_text = st.text_area("输入文本", height=130,
                              placeholder="请输入中文文本……",
                              label_visibility="collapsed",
                              key="m1_text")
    run = st.button("▶ 开始分析", type="primary")

    if run or st.session_state.get("m1_autorun", False):
        st.session_state["m1_autorun"] = False
        if not main_text.strip():
            st.error("请先输入文本！")
            return

        nlp = _load_zh_nlp()
        if nlp is None:
            st.error("中文语言模型加载失败，请检查 zh_core_web_sm 是否已安装。")
            return

        with st.spinner("正在分析，请稍候…"):
            norm     = normalize_text(main_text)
            seg      = do_segment(norm["normalized"], nlp)
            freq     = top_freq(seg["precise"])
            pos_tags = do_pos_tag(norm["normalized"], nlp)
            amb      = do_segment(main_text.strip(), nlp)
            fig_freq = chart_freq(freq)
            fig_cmp  = chart_comparison(amb, main_text.strip())

        st.divider()
        c1, c2, c3 = st.columns(3, gap="medium")

        # 区块1：文本规范化
        with c1:
            sec_header("⚙️ 文本规范化", "blue")
            st.caption("**原始文本**")
            st.info(norm["original"])
            st.caption("**处理步骤**")
            steps_h = ""
            for s in norm["steps"]:
                badge = '<span class="badge-chg">已修改</span>' if s["changed"] else ""
                cls   = "step-box chg" if s["changed"] else "step-box"
                steps_h += (
                    f'<div class="{cls}"><div class="step-name">{e(s["name"])}{badge}</div>'
                    f'<div class="step-text">{e(s["result"])}</div></div>'
                )
            st.markdown(steps_h, unsafe_allow_html=True)
            st.caption("**规范化结果**")
            st.success(norm["normalized"])

        # 区块2：分词 & 词频
        with c2:
            sec_header("✂️ 分词 & 词频统计", "pink")
            MODES = [
                ("精确模式", "precise", "spaCy 统计模型，最优分词"),
                ("全模式",   "full",    "字符级拆分，最大召回率"),
                ("搜索引擎模式", "search", "精确词 + 长词拆分，提升检索覆盖"),
            ]
            for name, key, desc in MODES:
                words = seg[key]
                st.caption(f"**{name}** — {desc}")
                st.markdown(chips_html(words), unsafe_allow_html=True)
                st.caption(f"共 {len(words)} 词 · 唯一 {len(set(words))} 词")
            if fig_freq:
                st.caption("**词频统计（精确模式 Top 5）**")
                st.plotly_chart(fig_freq, use_container_width=True)

        # 区块3：词性标注
        with c3:
            sec_header("🏷️ 词性标注", "cyan")
            seen = {}
            pw_h = '<div class="pos-box">'
            for t in pos_tags:
                w = t["word"]
                if t["flag"] == "PUNCT" or t["flag"] == "SPACE":
                    pw_h += f'<span style="font-size:.9rem;color:#999;align-self:center">{e(w)}</span>'
                else:
                    pw_h += (
                        f'<span class="pw" style="background:{t["color"]}" '
                        f'title="{e(t["label"])}({e(t["flag"])})">'
                        f'<span class="w">{e(w)}</span>'
                        f'<span class="ft">{e(t["flag"])}</span></span>'
                    )
                seen.setdefault(t["label"], t["color"])
            pw_h += "</div>"
            st.markdown(pw_h, unsafe_allow_html=True)
            leg_h = '<div class="pos-legend">' + "".join(
                f'<div class="leg"><span class="leg-dot" style="background:{c}"></span>'
                f'<span>{e(lbl)}</span></div>'
                for lbl, c in seen.items()
            ) + "</div>"
            st.markdown(leg_h, unsafe_allow_html=True)

        st.divider()

        # 分词算法对比
        sec_header("📊 分词算法对比分析", "green")
        short = main_text.strip()[:30] + ("…" if len(main_text.strip()) > 30 else "")
        st.caption(f'分析句子：**"{short}"**')

        all_cnt = Counter()
        for k in ["precise", "full", "search"]:
            for w in set(amb[k]):
                all_cnt[w] += 1
        diff_words = {w for w, cnt in all_cnt.items() if cnt < 3}

        ac1, ac2, ac3 = st.columns(3, gap="medium")
        ALGO_DEFS = [
            ("精确模式", "precise", "#4e79a7"),
            ("全模式",   "full",    "#f28e2b"),
            ("搜索引擎模式", "search", "#e15759"),
        ]
        for acol, (name, key, color) in zip([ac1, ac2, ac3], ALGO_DEFS):
            with acol:
                words = amb[key]
                avg   = sum(len(w) for w in words) / max(len(words), 1)
                st.markdown(
                    f'<div class="algo-box">'
                    f'<div class="algo-title" style="color:{color};border-color:{color}40">{name}</div>'
                    + chips_html(words, diff_words) +
                    f'<div class="algo-stats">共 {len(words)} 词 &nbsp;·&nbsp; '
                    f'唯一 {len(set(words))} 词 &nbsp;·&nbsp; 平均词长 {avg:.2f} 字</div></div>',
                    unsafe_allow_html=True,
                )

        st.plotly_chart(fig_cmp, use_container_width=True)

        st.markdown(
            '<div class="cmp-note"><b>💡 算法说明</b><br>'
            '<b>精确模式</b>：spaCy 统计模型，基于训练语料的最优切分，适合文本分析。<br>'
            '<b>全模式</b>：字符级拆分，每个汉字独立成词，召回率最高但含大量单字。<br>'
            '<b>搜索引擎模式</b>：在精确模式基础上对长词（3字+）进行 2-gram 拆分叠加，提升检索覆盖率。<br>'
            '<span style="background:#fef3c7;padding:0 3px;border-radius:3px;color:#92400e">'
            '🟡 黄色高亮</span>表示各算法切分结果不一致的词，体现分词粒度差异。</div>',
            unsafe_allow_html=True,
        )
