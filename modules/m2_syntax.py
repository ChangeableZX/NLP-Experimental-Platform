"""模块2：句法双引擎透视仪与"歧义侦探" """

import streamlit as st
from .common import render_module_header, sec_header

# ── T5Tokenizer 兼容补丁（必须在 benepar 之前） ───────────────────────────────
import transformers
if not hasattr(transformers.models.t5.tokenization_t5.T5Tokenizer,
               "build_inputs_with_special_tokens"):
    transformers.models.t5.tokenization_t5.T5Tokenizer.build_inputs_with_special_tokens = (
        lambda self, token_ids_0, token_ids_1=None: token_ids_0 + (token_ids_1 if token_ids_1 else [])
    )

import spacy
from spacy import displacy
import nltk
import benepar
from nltk.tree import Tree
import svgling


@st.cache_resource(show_spinner="正在预热句法引擎并拉取模型…")
def load_nlp_models():
    model_name = "en_core_web_sm"
    try:
        nlp_dep = spacy.load(model_name)
    except OSError:
        from spacy.cli import download
        download(model_name)
        nlp_dep = spacy.load(model_name)

    try:
        nltk.data.find("models/benepar_en3")
    except LookupError:
        benepar.download("benepar_en3")

    nlp_const = spacy.load(model_name)
    nlp_const.add_pipe("benepar", config={"model": "benepar_en3"})
    return nlp_dep, nlp_const


def render():
    render_module_header(
        "m2",
        "🕵️ 句法双引擎透视仪 & 歧义侦探",
        "依存句法分析 · 成分句法分析 · 核心论元提取 · 歧义结构透视",
    )

    with st.expander("📖 功能说明与技术栈", expanded=False):
        st.markdown("""
| 功能 | 说明 | 技术 / 库 |
|------|------|-----------|
| 依存句法分析 | 分析词语间的修饰 / 支配依存关系，以弧线图可视化词语间的语法联系 | `spaCy` (en_core_web_sm), `displacy` |
| 成分句法分析 | 将句子解析为 NP / VP 等短语嵌套层级结构，生成短语结构树 | `benepar` (benepar_en3), `svgling`, `NLTK` |
| 核心论元提取 | 基于依存标签规则提取句子的主语、谓语、宾语、状语，以表格形式呈现 | `spaCy` 依存标签 |
| 歧义结构透视 | 对同一句话同时展示依存图与成分树，直观感受结构歧义来源（如 PP-attachment） | `spaCy`, `benepar` |
""")

    st.markdown(
        "将一句话同时送入 **依存句法（Dependency）** 与 **成分句法（Constituency）** "
        "两个引擎，透视其底层结构，直观感受歧义的来源。"
    )

    sec_header("📝 输入检测文本", "pink")
    default_text = "The boy saw the man with the telescope."
    text = st.text_input("请输入需要透视的英文句子：", value=default_text)

    st.divider()

    if not text.strip():
        st.info("请在上方输入英文句子以开始分析。")
        return

    with st.spinner("正在加载句法引擎…"):
        nlp_dep, nlp_const = load_nlp_models()

    tab_dep, tab_const = st.tabs(["🔗 依存关系", "🌳 成分结构"])

    with tab_dep:
        st.subheader("依存句法 (Dependency Parsing)")
        st.markdown("*关注词与词之间的修饰与支配关系 (Who modifies whom?)*")
        doc_dep = nlp_dep(text)
        html_dep = displacy.render(doc_dep, style="dep", page=False)
        st.markdown(
            f"<div style='overflow-x:auto;background:white;padding:20px;"
            f"border-radius:8px;border:1px solid #dde6f0;'>{html_dep}</div>",
            unsafe_allow_html=True,
        )

    with tab_const:
        st.subheader("成分句法 (Constituency Parsing)")
        st.markdown("*关注短语如何嵌套组合 (How are words grouped?)*")
        doc_const = nlp_const(text)
        sents = list(doc_const.sents)
        if sents:
            sent = sents[0]
            parse_string = sent._.parse_string
            tree = Tree.fromstring(parse_string)
            svg_obj = svgling.draw_tree(tree)
            svg_str = svg_obj._repr_svg_()
            st.markdown(
                f"<div style='overflow-x:auto;background:white;padding:20px;"
                f"border-radius:8px;border:1px solid #dde6f0;'>{svg_str}</div>",
                unsafe_allow_html=True,
            )
            with st.expander("查看原始多级文本缩进 (Text format)"):
                st.code(tree.pformat(), language="lisp")

    st.divider()

    sec_header("🧲 核心论元提取器 (Core Argument Extractor)", "blue")
    st.markdown("*直接遍历 SpaCy 依存分析对象（Token），精准剥离句子的核心骨架。*")

    target_deps = ["ROOT", "nsubj", "dobj", "pobj"]
    extracted_args = []
    for token in doc_dep:
        if token.dep_ in target_deps:
            label_desc = {
                "ROOT":  "👑 根节点 (核心动词)",
                "nsubj": "👤 主语 (Nominal Subject)",
                "dobj":  "🎯 直接宾语 (Direct Object)",
                "pobj":  "📦 介词宾语 (Object of Preposition)",
            }.get(token.dep_, token.dep_)
            extracted_args.append({
                "提取词汇 (Token)":    token.text,
                "依存角色 (Dep Label)": token.dep_,
                "角色说明":            label_desc,
                "被谁支配 (Head Token)": token.head.text,
                "词性 (POS)":          token.pos_,
            })

    if extracted_args:
        st.dataframe(extracted_args, use_container_width=True)
    else:
        st.info("在这句话中没有检测到标准的主、谓、宾等核心论元。")
