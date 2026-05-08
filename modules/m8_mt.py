"""模块8：机器翻译机制与质量评测"""

import re

import nltk
import streamlit as st
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from transformers import pipeline

from .common import render_module_header, sec_header


def _zh_tokenize(text: str) -> list:
    """字符级中文分词（用于 BLEU 计算，无需 jieba）"""
    return list(text.replace(" ", ""))


# ── 规则词典（RBMT）────────────────────────────────────────────────────────────
EN_ZH_DICT: dict[str, str] = {
    "i":"我","me":"我","my":"我的","myself":"我自己","you":"你","your":"你的","yourself":"你自己",
    "he":"他","him":"他","his":"他的","she":"她","her":"她的","it":"它","its":"它的",
    "we":"我们","us":"我们","our":"我们的","they":"他们","them":"他们","their":"他们的",
    "this":"这","that":"那","these":"这些","those":"那些",
    "a":"一个","an":"一个","the":"这","some":"一些","any":"任何","every":"每个","all":"所有",
    "no":"没有","not":"不","never":"从不",
    "in":"在","on":"在…上","at":"在","of":"的","to":"到","for":"为了","with":"与","by":"通过",
    "from":"从","about":"关于","into":"进入","through":"通过","between":"在…之间",
    "among":"在…之中","over":"在…上方","under":"在…下面","after":"在…之后","before":"在…之前",
    "during":"在…期间","without":"没有","within":"在…之内",
    "and":"和","or":"或者","but":"但是","because":"因为","if":"如果","when":"当",
    "while":"当…时","although":"虽然","so":"所以","than":"比","as":"作为","until":"直到",
    "is":"是","are":"是","was":"是","were":"是","be":"是","been":"是","being":"是",
    "have":"有","has":"有","had":"有","do":"做","does":"做","did":"做","done":"完成",
    "go":"去","get":"获得","make":"制造","say":"说","know":"知道","think":"认为",
    "take":"拿","see":"看","come":"来","want":"想要","need":"需要","use":"使用",
    "find":"发现","give":"给","tell":"告诉","ask":"问","seem":"似乎","feel":"感觉",
    "try":"尝试","leave":"离开","put":"放","mean":"意味着","keep":"保持","let":"让",
    "begin":"开始","start":"开始","show":"显示","hear":"听","run":"跑","move":"移动",
    "live":"居住","believe":"相信","bring":"带来","happen":"发生","write":"写",
    "provide":"提供","learn":"学习","change":"改变","lead":"领导","understand":"理解",
    "watch":"观看","follow":"跟随","create":"创建","help":"帮助","develop":"发展",
    "build":"构建","speak":"说话","read":"读","spend":"花费","grow":"增长","open":"打开",
    "work":"工作","include":"包括","continue":"继续","translate":"翻译",
    "transforms":"改变","transform":"改变","improve":"改善","achieve":"实现",
    "time":"时间","year":"年","people":"人们","way":"方式","day":"天","man":"男人",
    "world":"世界","life":"生活","part":"部分","place":"地方","case":"情况",
    "week":"周","company":"公司","system":"系统","program":"程序","question":"问题",
    "government":"政府","number":"数字","money":"金钱","story":"故事","fact":"事实",
    "month":"月","right":"权利","book":"书","job":"工作","word":"词语","business":"商业",
    "issue":"问题","city":"城市","team":"团队","idea":"想法","information":"信息",
    "level":"级别","office":"办公室","health":"健康","person":"人","history":"历史",
    "reason":"原因","research":"研究","country":"国家","nation":"国家","society":"社会",
    "policy":"政策","law":"法律","plan":"计划","goal":"目标","project":"项目",
    "report":"报告","news":"新闻","event":"事件","problem":"问题","solution":"解决方案",
    "effect":"效果","impact":"影响","process":"过程","method":"方法","language":"语言",
    "machine":"机器","translation":"翻译","model":"模型","data":"数据","network":"网络",
    "intelligence":"智能","artificial":"人工的","technology":"技术","science":"科学",
    "computer":"计算机","algorithm":"算法","sentence":"句子","text":"文本",
    "good":"好","new":"新","first":"第一","last":"最后","long":"长","great":"伟大",
    "little":"小","other":"其他","old":"老","big":"大","high":"高","different":"不同",
    "small":"小","large":"大","next":"下一个","important":"重要","real":"真实",
    "best":"最好","free":"自由","strong":"强","special":"特别","easy":"容易",
    "recent":"最近","possible":"可能","social":"社会的","economic":"经济的",
    "deep":"深","neural":"神经的","natural":"自然的","quick":"快速","fast":"快",
    "also":"也","very":"非常","just":"只是","more":"更多","now":"现在","then":"然后",
    "here":"这里","there":"那里","only":"只","still":"仍然","even":"甚至",
    "already":"已经","yet":"还","again":"再次","most":"最多","always":"总是",
    "sometimes":"有时","often":"经常","really":"真的","much":"很多","many":"许多",
    "too":"也","however":"然而","therefore":"因此","thus":"因此","especially":"特别",
    "quickly":"迅速地","easily":"容易地","significantly":"显著地","recently":"最近",
    "yesterday":"昨天","today":"今天","tomorrow":"明天",
}


def rbmt_translate(text: str):
    tokens = text.strip().split()
    details: list = []
    output_parts: list = []
    for token in tokens:
        prefix_m = re.match(r"^([^a-zA-Z0-9]*)(.*?)([^a-zA-Z0-9]*)$", token)
        prefix, core, suffix = prefix_m.groups() if prefix_m else ("", token, "")
        key = core.lower()
        if key in EN_ZH_DICT:
            zh = EN_ZH_DICT[key]
            details.append((token, zh, True)); output_parts.append(prefix+zh+suffix)
        else:
            details.append((token, token, False)); output_parts.append(token)
    return " ".join(output_parts), details


@st.cache_resource(show_spinner=False)
def load_translation_pipeline():
    return pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")


def render():
    render_module_header(
        "m8",
        "🌐 机器翻译机制与质量评测",
        "神经机器翻译 (NMT) · 基于规则翻译 (RBMT) · BLEU 自动评测",
    )

    with st.expander("📖 功能说明与技术栈", expanded=False):
        st.markdown("""
| 功能 | 说明 | 技术 / 库 |
|------|------|-----------|
| 神经机器翻译 (NMT) | 基于 Transformer Seq2Seq 架构进行英→中翻译，模型由 Helsinki-NLP 在平行语料上训练 | `transformers` (Helsinki-NLP/opus-mt-en-zh) |
| 基于规则翻译 (RBMT) | 使用内置约 250 词的英中词典逐词替换，模拟传统规则翻译方法，展示其局限性 | 内置词典，`re` |
| NMT vs RBMT 对比 | 并排展示两种方法的译文，对差异词条进行颜色标注，直观呈现两类方法的优劣 | 自定义对比逻辑 |
| BLEU 自动评测 | 输入参考译文与候选译文，计算 BLEU-1 至 BLEU-4 精确率及综合得分，量化翻译质量 | `NLTK` (sentence_bleu, SmoothingFunction)，字符级分词 |
""")

    tab1, tab2, tab3 = st.tabs([
        "🤖 模块一：神经机器翻译",
        "⚔️ 模块二：RBMT vs NMT 对比",
        "📊 模块三：翻译质量评测（BLEU）",
    ])

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 1: NMT
    # ════════════════════════════════════════════════════════════════════════════
    with tab1:
        st.header("🤖 神经机器翻译")
        st.markdown("基于 **Helsinki-NLP/opus-mt-en-zh** 模型，将英文句子翻译为中文。")
        st.markdown("---")

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("📝 输入英文")
            if "nmt_input" not in st.session_state:
                st.session_state["nmt_input"] = ""
            _NMT_EX = [
                ("人工智能", "Artificial intelligence is transforming the world and changing the way we live and work."),
                ("气候变化", "Climate change is one of the most urgent challenges facing humanity today."),
                ("科学探索", "Scientists have discovered a new method to generate clean energy from solar power."),
            ]
            st.caption("🏷️ 示例英文句子：")
            _nmt_cols = st.columns(len(_NMT_EX))
            for _i, (_lbl, _txt) in enumerate(_NMT_EX):
                if _nmt_cols[_i].button(_lbl, key=f"nmt_ex_{_i}", use_container_width=True):
                    st.session_state["nmt_input"] = _txt
            user_input = st.text_area(
                label="请在下方输入英文句子：",
                placeholder="e.g. Artificial intelligence is transforming the world.",
                height=160, key="nmt_input",
            )
            translate_btn = st.button("🚀 开始翻译", use_container_width=True)

        with col_right:
            st.subheader("🈶 中文译文")
            result_placeholder = st.empty()
            if translate_btn:
                if not user_input.strip():
                    result_placeholder.warning("⚠️ 请先输入英文文本。")
                else:
                    with st.spinner("模型加载中，正在翻译，请稍候…"):
                        translator = load_translation_pipeline()
                        output = translator(user_input.strip(), max_length=512)
                    translation = output[0]["translation_text"]
                    result_placeholder.success("翻译完成！")
                    st.markdown(f"""
                    <div style="background:#f0f7ff;border-left:4px solid #1a73e8;
                                padding:16px 20px;border-radius:6px;font-size:1.1rem;line-height:1.8;">
                        {translation}
                    </div>""", unsafe_allow_html=True)

        st.markdown("---")
        with st.expander("ℹ️ 模型说明"):
            st.markdown("""
- **模型**：`Helsinki-NLP/opus-mt-en-zh`
- **架构**：基于 MarianMT（Transformer 编码器-解码器）
- **训练数据**：OPUS 多语言平行语料库
- **特点**：轻量、快速，适合英→中翻译演示
            """)

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 2: RBMT vs NMT
    # ════════════════════════════════════════════════════════════════════════════
    with tab2:
        st.header("⚔️ 基于规则翻译 vs 神经机器翻译")
        st.markdown("在同一句话上，对比 **逐词直译（RBMT）** 与 **神经机器翻译（NMT）** 的效果，直观感受两种范式的差距。")
        st.markdown("---")

        example_sentences = [
            "自定义输入…",
            "Artificial intelligence is transforming the world.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine translation has made great progress in recent years.",
            "I need to go to the hospital because I feel sick.",
            "The government announced a new economic policy yesterday.",
            "She believes that education is the key to a better life.",
            "Natural language processing is a branch of computer science.",
        ]
        choice = st.selectbox("💡 选择示例句子（或自定义输入）", example_sentences)
        if choice == "自定义输入…":
            src_sentence = st.text_area("请输入英文句子：", placeholder="Type an English sentence here…", height=100, key="rbmt_custom")
        else:
            src_sentence = choice; st.info(f"当前输入：**{src_sentence}**")

        run_btn = st.button("🚀 运行对比翻译", use_container_width=True)

        if run_btn:
            if not src_sentence.strip():
                st.warning("⚠️ 请输入英文句子。")
            else:
                rbmt_result, token_details = rbmt_translate(src_sentence.strip())
                with st.spinner("NMT 模型推理中，请稍候…"):
                    translator = load_translation_pipeline()
                    nmt_output = translator(src_sentence.strip(), max_length=512)
                nmt_result = nmt_output[0]["translation_text"]

                st.success("翻译完成！"); st.markdown("---")
                col_rbmt, col_nmt = st.columns(2)
                _card_style = "padding:18px 22px;border-radius:8px;font-size:1.15rem;line-height:1.9;min-height:90px;"
                with col_rbmt:
                    st.subheader("📖 RBMT — 逐词直译")
                    st.markdown(f'<div style="background:#fff8e1;border-left:4px solid #f9a825;{_card_style}">{rbmt_result}</div>', unsafe_allow_html=True)
                with col_nmt:
                    st.subheader("🤖 NMT — 神经机器翻译")
                    st.markdown(f'<div style="background:#f0f7ff;border-left:4px solid #1a73e8;{_card_style}">{nmt_result}</div>', unsafe_allow_html=True)

                st.markdown("---"); st.subheader("🔍 逐词直译 — 词典命中明细")
                hit_count  = sum(1 for _,_,hit in token_details if hit)
                miss_count = len(token_details) - hit_count
                hit_rate   = hit_count/len(token_details)*100 if token_details else 0
                m1c, m2c, m3c = st.columns(3)
                m1c.metric("总词数", len(token_details))
                m2c.metric("词典命中", hit_count)
                m3c.metric("未命中（保留英文）", miss_count)

                rows_html = ""
                for orig, zh, hit in token_details:
                    row_bg = "#e8f5e9" if hit else "#fce4ec"
                    badge  = '<span style="color:#2e7d32;font-weight:bold;">✔ 命中</span>' if hit else '<span style="color:#c62828;font-weight:bold;">✘ 未命中</span>'
                    rows_html += f'<tr style="background:{row_bg};"><td style="padding:6px 12px;border:1px solid #ddd;">{orig}</td><td style="padding:6px 12px;border:1px solid #ddd;">{zh}</td><td style="padding:6px 12px;border:1px solid #ddd;">{badge}</td></tr>'

                st.markdown(f"""
                <table style="width:100%;border-collapse:collapse;font-size:.95rem;">
                  <thead><tr style="background:#eeeeee;">
                    <th style="padding:8px 12px;border:1px solid #ddd;text-align:left;">原词</th>
                    <th style="padding:8px 12px;border:1px solid #ddd;text-align:left;">译词</th>
                    <th style="padding:8px 12px;border:1px solid #ddd;text-align:left;">状态</th>
                  </tr></thead><tbody>{rows_html}</tbody></table>
                  <p style="color:#555;font-size:.85rem;margin-top:6px;">词典命中率：{hit_rate:.1f}%</p>
                """, unsafe_allow_html=True)

                st.markdown("---"); st.subheader("💬 观察与分析")
                st.markdown(f"""
| 维度 | RBMT 逐词直译 | NMT 神经翻译 |
|------|--------------|-------------|
| **词序** | 保持英文原始词序，中文读起来生硬 | 自动调整为中文习惯词序 |
| **上下文** | 每词独立查表，无上下文感知 | 编解码器全局建模，语义连贯 |
| **未登录词** | 直接保留英文（命中率 {hit_rate:.1f}%） | 模型可泛化推断 |
| **流畅度** | 通常较差，逐字堆砌 | 通常自然流畅 |
| **可解释性** | 完全透明，每步可追溯 | 黑盒，难以解释 |
                """)

        with st.expander("ℹ️ 什么是 RBMT？"):
            st.markdown("""
**基于规则的机器翻译（Rule-Based MT，RBMT）** 是最早的机器翻译范式（1950s–1980s）。

核心思路：构建双语词典 → 分词 → 逐词查表 → 未登录词保留原词

**局限性**：无法处理词序差异、多义词消歧、上下文感知等问题，这正是 NMT 崛起的动因。
            """)

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 3: BLEU
    # ════════════════════════════════════════════════════════════════════════════
    with tab3:
        st.header("📊 机器翻译自动评测（BLEU）")
        st.markdown("提供英文原文、人工参考译文与候选译文，系统将计算 **BLEU 分数**并解释其含义。")
        st.markdown("---")

        def interpret_bleu(score):
            if score >= 0.6: return "优秀","#2e7d32","译文与参考译文高度吻合，词汇和短语匹配率极高，质量接近人工翻译水平。"
            elif score >= 0.4: return "良好","#1565c0","译文传达了原文主要意思，部分表达与参考译文一致，但仍有改进空间。"
            elif score >= 0.2: return "一般","#e65100","译文能体现原文大意，但措辞差距明显，流畅性或准确性有所欠缺。"
            else: return "较差","#b71c1c","译文与参考译文 n-gram 重叠率很低，可能存在严重漏译、误译或语序混乱问题。"

        st.subheader("① 待翻译英文原文")
        src_en = st.text_area("输入英文句子（用于 NMT 自动生成候选译文）：",
                              placeholder="e.g. Machine translation has made great progress in recent years.",
                              height=100, key="bleu_src")

        st.subheader("② 标准参考译文（Reference）")
        ref_zh = st.text_area("输入人工参考中文译文：",
                              placeholder="例：近年来，机器翻译取得了很大进展。",
                              height=100, key="bleu_ref")

        st.subheader("③ 候选译文（Candidate）")
        cand_mode = st.radio("候选译文来源", ["手动输入","调用 NMT 自动生成"], horizontal=True, key="cand_mode")
        cand_zh = ""

        if cand_mode == "手动输入":
            cand_zh = st.text_area("输入机器生成的中文候选译文：",
                                   placeholder="粘贴任意机器翻译结果…",
                                   height=100, key="bleu_cand_manual")
        else:
            auto_btn = st.button("⚡ 调用 NMT 生成候选译文", use_container_width=False)
            nmt_result_holder = st.empty()
            if "bleu_nmt_cache" not in st.session_state:
                st.session_state["bleu_nmt_cache"] = {"src":"","result":""}
            if auto_btn:
                if not src_en.strip():
                    nmt_result_holder.warning("⚠️ 请先填写英文原文。")
                else:
                    with st.spinner("NMT 模型推理中，请稍候…"):
                        translator = load_translation_pipeline()
                        nmt_out = translator(src_en.strip(), max_length=512)
                    st.session_state["bleu_nmt_cache"] = {"src":src_en.strip(),"result":nmt_out[0]["translation_text"]}
            cached = st.session_state["bleu_nmt_cache"]
            if cached["result"]:
                nmt_result_holder.success(f"NMT 生成完成：**{cached['result']}**")
                cand_zh = cached["result"]
            else:
                nmt_result_holder.info("点击上方按钮，系统将调用 NMT 模型自动生成候选译文。")

        st.markdown("---")
        calc_btn = st.button("📈 计算 BLEU 分数", use_container_width=True)

        if calc_btn:
            errors = []
            if not ref_zh.strip():  errors.append("参考译文（②）不能为空")
            if not cand_zh.strip(): errors.append("候选译文（③）不能为空")
            if errors:
                st.warning("⚠️ " + "；".join(errors) + "。")
            else:
                with st.spinner("计算中…"):
                    ref_tokens  = _zh_tokenize(ref_zh.strip())
                    cand_tokens = _zh_tokenize(cand_zh.strip())
                    smoother = SmoothingFunction().method1
                    bleu1 = sentence_bleu([ref_tokens], cand_tokens, weights=(1,0,0,0), smoothing_function=smoother)
                    bleu2 = sentence_bleu([ref_tokens], cand_tokens, weights=(.5,.5,0,0), smoothing_function=smoother)
                    bleu4 = sentence_bleu([ref_tokens], cand_tokens, weights=(.25,.25,.25,.25), smoothing_function=smoother)

                level, color, explanation = interpret_bleu(bleu4)
                st.success("计算完成！")

                st.subheader("📊 BLEU 分数结果")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("BLEU-1（Unigram）", f"{bleu1:.4f}")
                mc2.metric("BLEU-2（Bigram）",  f"{bleu2:.4f}")
                mc3.metric("BLEU-4（标准）",    f"{bleu4:.4f}")

                st.markdown("#### 🔍 分数解读")
                st.markdown(f"""
                <div style="border:2px solid {color};border-radius:10px;padding:18px 24px;background:#fafafa;">
                    <p style="font-size:1.4rem;font-weight:bold;color:{color};margin:0 0 8px;">
                        质量等级：{level} &nbsp;｜&nbsp; BLEU-4 = {bleu4:.4f}
                    </p>
                    <p style="font-size:1rem;color:#333;margin:0;line-height:1.7;">{explanation}</p>
                </div>""", unsafe_allow_html=True)

                st.markdown("---"); st.subheader("📋 输入内容回顾")
                r1, r2, r3 = st.columns(3)
                with r1: st.caption("英文原文"); st.write(src_en if src_en.strip() else "（未填写）")
                with r2: st.caption("参考译文 tokens（前 20）"); st.write(" / ".join(ref_tokens[:20])+("…" if len(ref_tokens)>20 else ""))
                with r3: st.caption("候选译文 tokens（前 20）"); st.write(" / ".join(cand_tokens[:20])+("…" if len(cand_tokens)>20 else ""))

        with st.expander("ℹ️ BLEU 评测原理详解"):
            st.markdown("""
### 什么是 BLEU？
**BLEU（Bilingual Evaluation Understudy）** 是 2002 年由 IBM 提出的机器翻译自动评测指标。

**计算思路**：n-gram 精确率 → 修正精确率 → 简短惩罚（BP）→ 几何平均

| BLEU-4 分数 | 通常含义 |
|------------|---------|
| > 0.60 | 接近人工翻译质量 |
| 0.40 – 0.60 | 良好，可理解，有少量错误 |
| 0.20 – 0.40 | 一般，主要意思正确，表达欠自然 |
| < 0.20 | 较差，存在明显误译或漏译 |

> **注意**：BLEU 仅衡量词汇重叠，不能评价语义正确性和语言流畅度，实际评测通常结合人工评分综合判断。
            """)
