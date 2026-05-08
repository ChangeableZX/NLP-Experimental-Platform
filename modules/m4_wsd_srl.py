"""模块4：词义消歧与语义角色标注"""

import streamlit as st
import torch
import numpy as np
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy import displacy
import nltk

from .common import render_module_header, sec_header


@st.cache_data
def download_nltk_data():
    download_list = ["wordnet", "omw-1.4", "punkt", "averaged_perceptron_tagger"]
    for item in download_list:
        try:
            nltk.data.find(f"corpora/{item}")
        except LookupError:
            try:
                nltk.download(item, quiet=True, raise_on_error=True)
            except Exception as e:
                st.error(f"下载 {item} 失败: {e}")
                return False
    return True


@st.cache_resource
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    return tokenizer, model


@st.cache_resource
def load_spacy_model():
    import subprocess, sys
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                       check=True, capture_output=True)
        nlp = spacy.load("en_core_web_sm")
    return nlp


def get_span_text(token):
    parts = [token.text]
    for child in token.children:
        if child.dep_ in ("compound", "amod", "det", "nummod"):
            parts.append(child.text)
    return " ".join(sorted(parts, key=lambda x: token.doc.text.find(x)))


def extract_prep_phrase(token, doc):
    if token.dep_ == "prep":
        parts = [token.text]
        for child in token.children:
            if child.dep_ == "pobj":
                parts.append(get_span_text(child))
        return " ".join(parts)
    elif token.dep_ == "pobj":
        prep = token.head
        if prep.dep_ == "prep":
            return f"{prep.text} {get_span_text(token)}"
        return get_span_text(token)
    return None


def extract_srl(sentence, nlp):
    doc = nlp(sentence)
    srl_result = {"A0": None, "Predicate": None, "A1": None, "AM-LOC": None, "AM-TMP": None}
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            srl_result["Predicate"] = token.text
            aux_tokens = [c.text for c in token.children if c.dep_ in ("aux","auxpass")]
            if aux_tokens:
                srl_result["Predicate"] = " ".join(sorted(aux_tokens+[token.text], key=lambda x: doc.text.find(x)))
        if token.dep_ in ("nsubj","nsubjpass") and token.head.pos_ == "VERB":
            srl_result["A0"] = get_span_text(token)
        if token.dep_ == "dobj" and token.head.pos_ == "VERB":
            srl_result["A1"] = get_span_text(token)
        if token.dep_ in ("prep","pobj"):
            loc_preps = ["in","at","on","near","by","under","over"]
            if token.text.lower() in loc_preps or token.head.text.lower() in loc_preps:
                loc_phrase = extract_prep_phrase(token, doc)
                if loc_phrase: srl_result["AM-LOC"] = loc_phrase
        if token.ent_type_ in ("DATE","TIME"):
            srl_result["AM-TMP"] = get_span_text(token)
        if token.dep_ in ("prep","pobj"):
            tmp_preps = ["in","at","on","during","before","after","this","last","next"]
            if token.text.lower() in tmp_preps or token.head.text.lower() in tmp_preps:
                tmp_phrase = extract_prep_phrase(token, doc)
                if tmp_phrase and not srl_result["AM-TMP"]:
                    srl_result["AM-TMP"] = tmp_phrase
    return srl_result, doc


def get_contextual_embedding(sentence, target_word, tokenizer, model):
    words = sentence.split()
    target_indices = [i for i, word in enumerate(words) if target_word.lower() in word.lower()]
    if not target_indices:
        return None, None, "目标词未在句子中找到"
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    tokens = tokenizer.tokenize(sentence)
    word_tokens = [len(tokenizer.tokenize(word)) for word in words]
    target_token_indices = []
    current_pos = 1
    for i, token_count in enumerate(word_tokens):
        if i in target_indices:
            for j in range(token_count):
                target_token_indices.append(current_pos + j)
        current_pos += token_count
    if not target_token_indices:
        return None, None, "无法对齐目标词"
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    target_embeddings = last_hidden_state[0, target_token_indices, :]
    embedding = torch.mean(target_embeddings, dim=0).numpy()
    return embedding, tokens, None


def perform_lesk_wsd(sentence, target_word):
    synset = lesk(sentence.split(), target_word)
    if synset is None:
        return None, "无法找到匹配的词义"
    return synset, None


def render():
    render_module_header(
        "m4",
        "🧠 词义消歧与语义角色标注",
        "Lesk 算法 · BERT 上下文向量 · 依存句法 SRL · 语义角色可视化",
    )

    with st.expander("📖 功能说明与技术栈", expanded=False):
        st.markdown("""
| 功能 | 说明 | 技术 / 库 |
|------|------|-----------|
| Lesk 算法 WSD | 传统词典方法：计算句子上下文词汇与 WordNet 各词义定义的重叠度来消歧 | `NLTK` (lesk, WordNet) |
| BERT 上下文向量 WSD | 提取目标词在具体语境中的动态语境向量，用余弦相似度量化同词在不同句义中的差异 | `transformers` (bert-base-uncased), `scikit-learn` |
| 语义角色标注 SRL | 基于依存句法启发式规则提取谓词及 A0（施事）、A1（受事）、AM-LOC（地点）、AM-TMP（时间）等论元 | `spaCy` (en_core_web_sm), `displacy` |
""")

    download_nltk_data()

    with st.spinner("正在加载 BERT 模型（首次加载可能需要一些时间）…"):
        tokenizer, model = load_bert_model()

    tab1, tab2 = st.tabs(["🔍 词义消歧 (WSD)", "📋 语义角色标注 (SRL)"])

    # ── WSD Tab ───────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("词义消歧 (Word Sense Disambiguation)")
        st.markdown("通过对比传统 Lesk 算法与 BERT 上下文向量表示来分析多义词。")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📄 句子 1")
            sentence1 = st.text_input("输入包含多义词的句子",
                value="I went to the bank to deposit my money.", key="sentence1")
        with col2:
            st.subheader("📄 句子 2")
            sentence2 = st.text_input("输入第二个句子进行对比",
                value="I sat by the river bank.", key="sentence2")

        target_word = st.text_input("🎯 目标多义词（如：bank）", value="bank")

        if st.button("🚀 开始分析", type="primary"):
            if not sentence1 or not target_word:
                st.error("请输入句子1和目标词！")
                return

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.markdown("---")
                st.markdown("### 📊 句子1分析结果")
                st.info(f"**句子：** {sentence1}")
                st.info(f"**目标词：** `{target_word}`")

                st.markdown("#### 1️⃣ Lesk算法（传统方法）")
                synset1, error1 = perform_lesk_wsd(sentence1, target_word)
                if error1:
                    st.error(error1)
                else:
                    st.success(f"**预测Synset：** `{synset1.name()}`")
                    st.success(f"**定义：** {synset1.definition()}")
                    examples = synset1.examples()
                    if examples:
                        st.markdown("**例句：**")
                        for ex in examples[:2]: st.markdown(f"- *{ex}*")

                st.markdown("#### 2️⃣ BERT上下文向量")
                embedding1, tokens1, error_emb1 = get_contextual_embedding(sentence1, target_word, tokenizer, model)
                if error_emb1:
                    st.error(error_emb1)
                else:
                    st.success(f"**Token序列：** `{' '.join(tokens1[:15])}{'...' if len(tokens1)>15 else ''}`")
                    st.success(f"**向量维度：** {embedding1.shape}")
                    st.success(f"**向量前5维：** {embedding1[:5].round(4)}")
                    st.markdown(f"**向量统计：** 均值={embedding1.mean():.4f}, 标准差={embedding1.std():.4f}")

            with result_col2:
                st.markdown("---")
                if sentence2:
                    st.markdown("### 📊 句子2分析结果")
                    st.info(f"**句子：** {sentence2}")
                    st.info(f"**目标词：** `{target_word}`")

                    st.markdown("#### 1️⃣ Lesk算法（传统方法）")
                    synset2, error2 = perform_lesk_wsd(sentence2, target_word)
                    if error2:
                        st.error(error2)
                    else:
                        st.success(f"**预测Synset：** `{synset2.name()}`")
                        st.success(f"**定义：** {synset2.definition()}")
                        examples = synset2.examples()
                        if examples:
                            st.markdown("**例句：**")
                            for ex in examples[:2]: st.markdown(f"- *{ex}*")

                    st.markdown("#### 2️⃣ BERT上下文向量")
                    embedding2, tokens2, error_emb2 = get_contextual_embedding(sentence2, target_word, tokenizer, model)
                    if error_emb2:
                        st.error(error_emb2)
                    else:
                        st.success(f"**Token序列：** `{' '.join(tokens2[:15])}{'...' if len(tokens2)>15 else ''}`")
                        st.success(f"**向量维度：** {embedding2.shape}")
                        st.success(f"**向量前5维：** {embedding2[:5].round(4)}")
                        st.markdown(f"**向量统计：** 均值={embedding2.mean():.4f}, 标准差={embedding2.std():.4f}")
                else:
                    st.info("请输入句子2以进行对比分析")

            if sentence2:
                try:
                    if embedding1 is not None and embedding2 is not None:
                        st.divider()
                        st.markdown("### 🔬 对比验证")
                        similarity = cosine_similarity(embedding1.reshape(1,-1), embedding2.reshape(1,-1))[0][0]
                        st.metric(label="两个BERT词向量的余弦相似度", value=f"{similarity:.4f}")
                except Exception:
                    pass

    # ── SRL Tab ───────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("语义角色标注 (Semantic Role Labeling)")
        st.markdown("使用 spaCy 依存句法分析进行轻量级语义角色标注，通过启发式规则提取谓词和论元。")

        with st.spinner("正在加载 spaCy 模型…"):
            nlp = load_spacy_model()

        srl_sentence = st.text_input(
            "输入句子进行语义角色标注",
            value="Apple is manufacturing new smartphones in China this year.",
            key="srl_sentence",
        )

        if st.button("🔍 开始SRL分析", type="primary", key="srl_button"):
            if not srl_sentence:
                st.error("请输入句子！")
            else:
                srl_result, doc = extract_srl(srl_sentence, nlp)

                st.divider()
                st.markdown("### 📊 语义角色标注结果")
                table_data = {
                    "语义角色": ["A0 (施事者)","谓词 (Predicate)","A1 (受事者)","AM-LOC (地点)","AM-TMP (时间)"],
                    "内容": [
                        srl_result["A0"]     or "未检测到",
                        srl_result["Predicate"] or "未检测到",
                        srl_result["A1"]     or "未检测到",
                        srl_result["AM-LOC"] or "未检测到",
                        srl_result["AM-TMP"] or "未检测到",
                    ],
                }
                st.table(table_data)

                st.divider()
                st.markdown("### 🔗 依存句法分析图")
                html = displacy.render(doc, style="dep", options={"compact": True, "distance": 100})
                st.write(html, unsafe_allow_html=True)
