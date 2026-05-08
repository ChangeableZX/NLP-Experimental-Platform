"""模块1：中文词法分析"""

import re
import html as htmllib
from collections import Counter

import jieba
import jieba.posseg as pseg
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import streamlit as st

from .common import render_module_header, sec_header

logging.getLogger("jieba").setLevel(logging.WARNING)


# ── matplotlib 中文字体 ──────────────────────────────────────────────────────
@st.cache_resource
def _cn_font():
    try:
        fm._rebuild()
    except Exception:
        pass
    cands = [
        "Noto Sans CJK SC", "Noto Serif CJK SC", "Noto Sans SC",
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
        "Microsoft YaHei", "SimHei", "SimSun", "FangSong", "KaiTi",
        "STHeiti", "STSong", "Arial Unicode MS",
    ]
    avail = {f.name for f in fm.fontManager.ttflist}
    return next((f for f in cands if f in avail), None)


_FONT = _cn_font()
if _FONT:
    plt.rcParams["font.sans-serif"] = [_FONT] + plt.rcParams.get("font.sans-serif", [])
plt.rcParams["axes.unicode_minus"] = False

# ── 保留字符白名单 ────────────────────────────────────────────────────────────
_KEEP_RE = re.compile(
    r"[^一-鿿㐀-䶿"
    r" -~\s"
    r"，。！？、；："
    "“”‘’"
    r"（）【】《》"
    r"…—～·]+"
)

STOPWORDS = {
    "的","了","在","是","我","有","和","就","不","人","都","一","一个","上","也",
    "很","到","说","要","去","你","会","着","没有","看","好","自己","这","那",
    "他","她","们","我们","这个","那个","什么","怎么","但","但是","还","就是","可以",
}

POS_MAP = {
    "nr": ("人名",   "#c0392b"), "ns": ("地名",   "#16a085"), "nt": ("机构名", "#2980b9"),
    "nz": ("专有名词","#8e44ad"), "n":  ("名词",   "#e74c3c"), "vn": ("动名词", "#1a6ea8"),
    "v":  ("动词",   "#2980b9"), "a":  ("形容词", "#27ae60"), "ad": ("副形词", "#1e8449"),
    "d":  ("副词",   "#7d3c98"), "p":  ("介词",   "#e67e22"), "r":  ("代词",   "#17a589"),
    "m":  ("数词",   "#b7950b"), "q":  ("量词",   "#1d8348"), "c":  ("连词",   "#626567"),
    "u":  ("助词",   "#909497"), "x":  ("标点",   "#ccd1d1"), "eng":("English","#5dade2"),
}

PALETTE = ["#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
           "#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ac"]


def fullwidth_to_halfwidth(text):
    out = []
    for ch in text:
        cp = ord(ch)
        if cp == 0x3000:   out.append(" ")
        elif 0xFF01 <= cp <= 0xFF5E: out.append(chr(cp - 0xFEE0))
        else: out.append(ch)
    return "".join(out)


def trad_to_simp(text):
    try:
        import zhconv
        return zhconv.convert(text, "zh-hans")
    except ImportError:
        return text


def normalize_text(text):
    steps, cur = [], text
    for fn, name in [(trad_to_simp, "繁体 → 简体"), (fullwidth_to_halfwidth, "全角 → 半角")]:
        nxt = fn(cur)
        steps.append({"name": name, "result": nxt, "changed": nxt != cur})
        cur = nxt
    nxt = _KEEP_RE.sub("", cur)
    steps.append({"name": "去除特殊符号", "result": nxt, "changed": nxt != cur})
    cur = nxt
    nxt = re.sub(r"[ \t\r\n]+", " ", cur).strip()
    steps.append({"name": "规范空白字符", "result": nxt, "changed": nxt != cur})
    return {"original": text, "normalized": nxt, "steps": steps}


def do_segment(text):
    return {
        "precise": [w for w in jieba.cut(text, cut_all=False) if w.strip()],
        "full":    [w for w in jieba.cut(text, cut_all=True)  if w.strip()],
        "search":  [w for w in jieba.cut_for_search(text)     if w.strip()],
    }


def top_freq(words, n=5):
    filt = [w for w in words if len(w) >= 2 and w not in STOPWORDS and re.search(r"\w", w)]
    return Counter(filt).most_common(n)


def resolve_pos(flag):
    if flag in POS_MAP: return POS_MAP[flag]
    for pfx, info in POS_MAP.items():
        if flag.startswith(pfx): return info
    return ("其他", "#aab7b8")


def do_pos_tag(text):
    return [{"word": w, "flag": f, **dict(zip(("label", "color"), resolve_pos(f)))}
            for w, f in pseg.cut(text) if w.strip()]


def chart_freq(freq):
    if not freq: return None
    words  = [w for w, _ in freq]
    counts = [c for _, c in freq]
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    bars = ax.barh(words[::-1], counts[::-1], color=PALETTE[:len(words)], height=0.52)
    ax.set_title("高频词 Top 5", fontsize=12, fontweight="bold", pad=7)
    ax.set_xlabel("频次", fontsize=9)
    for bar, c in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + .06, bar.get_y() + bar.get_height()/2, str(c), va="center", fontsize=9)
    ax.set_xlim(0, max(counts)*1.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def chart_comparison(results, sentence):
    keys   = ["precise", "full", "search"]
    labels = ["精确模式", "全模式", "搜索引擎"]
    colors = ["#4e79a7", "#f28e2b", "#e15759"]
    token_cnt  = [len(results[k]) for k in keys]
    unique_cnt = [len(set(results[k])) for k in keys]
    avg_len    = [sum(len(w) for w in results[k])/max(len(results[k]),1) for k in keys]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.0))
    for ax, title, vals in zip(axes, ["分词总数","唯一词数","平均词长（字）"], [token_cnt, unique_cnt, avg_len]):
        bars = ax.bar(labels, vals, color=colors, width=0.5)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.02*max(vals, default=1),
                    f"{v:.2f}" if isinstance(v, float) else str(v), ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, max(vals)*1.3 if max(vals)>0 else 1)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(f'"{sentence}" — 三种算法统计对比', fontsize=10, fontweight="bold", y=1.03)
    plt.tight_layout()
    return fig


def e(s): return htmllib.escape(str(s))


def chips_html(words, diff=None):
    parts = []
    for w in words:
        cls = "chip chip-diff" if (diff and w in diff) else "chip"
        parts.append(f'<span class="{cls}">{e(w)}</span>')
    return '<div class="chips-box">' + "".join(parts) + "</div>"


# ── CSS 局部样式 ──────────────────────────────────────────────────────────────
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
| 文本规范化 | 繁简体互转、全角→半角、去除特殊符号、统一空白字符，输出清洁文本 | `zhconv`, `re` |
| 中文分词 | 提供精确模式（最小粒度）、全模式（所有可能词语）、搜索引擎模式（适合检索）三种切分策略 | `jieba` |
| 词性标注 | 对每个分词结果标注词性（名词 / 动词 / 形容词等），并按词性分类着色展示 | `jieba.posseg` |
| 高频词统计 | 过滤停用词后统计词频，绘制 Top-5 词条水平柱状图 | `collections.Counter`, `matplotlib` |
| 算法对比 | 三种分词模式在分词总数、唯一词数、平均词长三个维度上的可视化对比 | `matplotlib` |
""")

    main_text = st.text_area("输入文本", height=130, placeholder="请输入中文文本……",
                              label_visibility="collapsed")
    run = st.button("▶ 开始分析", type="primary")

    if run:
        if not main_text.strip():
            st.error("请先输入文本！")
            return

        ambiguous = main_text.strip()
        with st.spinner("正在分析，请稍候…"):
            norm     = normalize_text(main_text)
            seg      = do_segment(norm["normalized"])
            freq     = top_freq(seg["precise"])
            pos_tags = do_pos_tag(norm["normalized"])
            amb      = do_segment(ambiguous)
            fig_freq = chart_freq(freq)
            fig_cmp  = chart_comparison(amb, ambiguous)

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
                steps_h += (f'<div class="{cls}"><div class="step-name">{e(s["name"])}{badge}</div>'
                            f'<div class="step-text">{e(s["result"])}</div></div>')
            st.markdown(steps_h, unsafe_allow_html=True)
            st.caption("**规范化结果**")
            st.success(norm["normalized"])

        # 区块2：分词 & 词频
        with c2:
            sec_header("✂️ 分词 & 词频统计", "pink")
            MODES = [
                ("精确模式", "precise", "最精确切分，适合文本分析"),
                ("全模式",   "full",    "扫描所有可能词语，含冗余"),
                ("搜索引擎模式","search","在精确模式基础上对长词再切分"),
            ]
            for name, key, desc in MODES:
                words = seg[key]
                st.caption(f"**{name}** — {desc}")
                st.markdown(chips_html(words), unsafe_allow_html=True)
                st.caption(f"共 {len(words)} 词 · 唯一 {len(set(words))} 词")
            if fig_freq:
                st.caption("**词频统计（精确模式 Top 5）**")
                st.pyplot(fig_freq, use_container_width=True)
                plt.close(fig_freq)

        # 区块3：词性标注
        with c3:
            sec_header("🏷️ 词性标注", "cyan")
            PUNCT_CP = {0xff0c,0x3002,0xff01,0xff1f,0x3001,0xff1b,0xff1a,
                        0x201c,0x201d,0x2018,0x2019,0xff08,0xff09,0x3010,0x3011,
                        0x300a,0x300b,0x2026,0x2014,0xff5e,0x00b7,0x0020}
            seen = {}
            pw_h = '<div class="pos-box">'
            for t in pos_tags:
                w = t["word"]
                if t["flag"] == "x" or (len(w)==1 and ord(w) in PUNCT_CP):
                    pw_h += f'<span style="font-size:.9rem;color:#999;align-self:center">{e(w)}</span>'
                else:
                    pw_h += (f'<span class="pw" style="background:{t["color"]}" title="{e(t["label"])}({e(t["flag"])})">'
                             f'<span class="w">{e(w)}</span><span class="ft">{e(t["flag"])}</span></span>')
                seen.setdefault(t["label"], t["color"])
            pw_h += "</div>"
            st.markdown(pw_h, unsafe_allow_html=True)
            leg_h = '<div class="pos-legend">' + "".join(
                f'<div class="leg"><span class="leg-dot" style="background:{c}"></span><span>{e(lbl)}</span></div>'
                for lbl, c in seen.items()
            ) + "</div>"
            st.markdown(leg_h, unsafe_allow_html=True)

        st.divider()

        # 分词算法对比
        sec_header("📊 分词算法对比分析", "green")
        st.caption(f'分析句子：**"{ambiguous}"**')

        all_cnt = Counter()
        for k in ["precise","full","search"]:
            for w in set(amb[k]): all_cnt[w] += 1
        diff_words = {w for w, cnt in all_cnt.items() if cnt < 3}

        ac1, ac2, ac3 = st.columns(3, gap="medium")
        ALGO_DEFS = [
            ("精确模式", "precise", "#4e79a7"),
            ("全模式",   "full",    "#f28e2b"),
            ("搜索引擎模式","search","#e15759"),
        ]
        for acol, (name, key, color) in zip([ac1, ac2, ac3], ALGO_DEFS):
            with acol:
                words = amb[key]
                avg   = sum(len(w) for w in words)/max(len(words),1)
                st.markdown(
                    f'<div class="algo-box">'
                    f'<div class="algo-title" style="color:{color};border-color:{color}40">{name}</div>'
                    + chips_html(words, diff_words) +
                    f'<div class="algo-stats">共 {len(words)} 词 &nbsp;·&nbsp; 唯一 {len(set(words))} 词 &nbsp;·&nbsp; 平均词长 {avg:.2f} 字</div></div>',
                    unsafe_allow_html=True,
                )

        st.write("")
        st.pyplot(fig_cmp, use_container_width=True)
        plt.close(fig_cmp)

        st.markdown(
            '<div class="cmp-note"><b>💡 算法说明</b><br>'
            '<b>精确模式</b>：最精确切分，适合文本分析任务。<br>'
            '<b>全模式</b>：扫描所有可能词语，召回率高但含冗余（重叠词）。<br>'
            '<b>搜索引擎模式</b>：在精确模式基础上对长词再次切分，提升召回。<br>'
            '<span style="background:#fef3c7;padding:0 3px;border-radius:3px;color:#92400e">'
            '🟡 黄色高亮</span>表示各算法切分结果不一致的词，体现分词歧义现象。</div>',
            unsafe_allow_html=True,
        )
