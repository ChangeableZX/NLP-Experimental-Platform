"""模块3：语义表示与对比分析（无 gensim，纯 sklearn/numpy 实现）"""

import re
import textwrap
from collections import Counter

import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from nltk.tokenize import sent_tokenize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .common import render_module_header, sec_header

DEFAULT_CORPUS = textwrap.dedent("""
    In the middle of the nineteenth century, a small public library stood near a river port where merchants, teachers, sailors, and carpenters crossed paths every day. The building was plain, with limestone walls and wide windows, but people said the reading room felt larger than its size because every desk held a different world. Children copied maps from travel journals, mechanics compared diagrams of steam engines, and clerks practiced letter writing by reading newspaper editorials aloud. No one used the phrase "knowledge network," yet the town had already built one in practice.

    The librarian, Mrs. Alton, believed that books were only one part of literacy. She arranged weekly evenings where residents explained what they had learned from reading and how they applied it at work. A baker who read chemistry pamphlets adjusted oven temperatures and reduced wasted flour. A shipwright who studied geometry redesigned hull ribs and made riverboats lighter. A nurse who borrowed essays on sanitation persuaded two factories to improve drainage. The library did not produce isolated scholars; it cultivated people who translated abstract ideas into practical decisions.

    During winters, when trade slowed and days became short, attendance grew. Workers who had never attended formal school arrived after sunset and asked for introductions to history, arithmetic, and law. Mrs. Alton created hand-written guides that linked difficult books to easier ones, so readers could build confidence step by step. She also paired newcomers with experienced volunteers, not as instructors in a strict classroom sense, but as companions in conversation. This social structure mattered because many learners were less afraid of difficult content when questions could be asked without embarrassment.

    At the same time, the town experienced disagreement about what should be taught. Some business owners wanted only technical manuals, arguing that public funds should support immediate economic gain. Others argued for poetry, philosophy, and political speeches, claiming that a community without imagination becomes obedient rather than thoughtful. The library board debated these priorities for months. Eventually, they adopted a mixed policy: practical collections would expand, but no category would be removed entirely. Their compromise acknowledged that citizens need both employable skills and reflective judgment.

    Years later, a railway connected the town to larger cities. New books arrived faster, and periodicals from distant regions changed local debates. People discovered farming techniques suited to dry climates, courtroom reforms from neighboring states, and serialized novels that described life in industrial districts. Exposure to unfamiliar voices sometimes caused tension, yet it also widened the vocabulary people used to describe their own experience. Problems once treated as private failure began to be discussed as policy questions. The library became a place where language itself evolved through shared reading.

    Today, the old library still faces the river, though the port now handles containers tracked by satellite data. Teenagers attend coding clubs in the same hall where mechanics once discussed steam engines. Retired workers mentor students on oral history projects, recording stories about strikes, storms, and neighborhood festivals. The collection now includes ebooks, podcasts, and community datasets alongside fragile paper volumes. Despite changing formats, the institution keeps one constant principle: knowledge becomes most powerful when people connect ideas across generations, occupations, and forms of media.
""").strip()


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def split_into_sentences(text):
    normalized = " ".join(text.split())
    if not normalized:
        return []
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
        except Exception:
            pass
    try:
        sentences = sent_tokenize(normalized)
    except LookupError:
        sentences = re.split(r"(?<=[.!?])\s+", normalized)
    return [s.strip() for s in sentences if s.strip()]


def vectorize_documents(sentences, mode):
    if mode == "TF-IDF":
        vectorizer = TfidfVectorizer(stop_words="english")
    else:
        vectorizer = CountVectorizer(stop_words="english", binary=True)
    matrix = vectorizer.fit_transform(sentences)
    return matrix, vectorizer.get_feature_names_out()


def tokenize_corpus(sentences):
    tokenized = []
    for sent in sentences:
        tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", sent.lower())
        if tokens:
            tokenized.append(tokens)
    return tokenized


def _cosine_sim(v1, v2):
    d = np.linalg.norm(v1) * np.linalg.norm(v2)
    return 0.0 if d == 0 else float(np.dot(v1, v2) / d)


# ── 词-共现矩阵 + SVD 词向量（Word2Vec 替代） ─────────────────────────────────

@st.cache_resource(show_spinner=False)
def build_cooc_word_vectors(tokenized_tuple, window, vector_size, min_count):
    """共现矩阵 + TruncatedSVD，产生与 Word2Vec 原理相近的分布式词向量。"""
    tokenized = [list(s) for s in tokenized_tuple]
    freq = Counter(w for sent in tokenized for w in sent)
    vocab_list = sorted(w for w, c in freq.items() if c >= min_count)
    if len(vocab_list) < 3:
        return None, [], {}
    word_to_idx = {w: i for i, w in enumerate(vocab_list)}
    n = len(vocab_list)

    matrix = np.zeros((n, n), dtype=np.float32)
    for sent in tokenized:
        for i, word in enumerate(sent):
            if word not in word_to_idx:
                continue
            wi = word_to_idx[word]
            lo, hi = max(0, i - window), min(len(sent), i + window + 1)
            for j in range(lo, hi):
                if j != i and sent[j] in word_to_idx:
                    matrix[wi][word_to_idx[sent[j]]] += 1.0

    n_comp = min(vector_size, n - 1, 150)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    vecs = svd.fit_transform(matrix)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vecs / norms, vocab_list, word_to_idx


def most_similar(word, vecs, vocab_list, word_to_idx, topn=5):
    if word not in word_to_idx or vecs is None:
        return []
    idx = word_to_idx[word]
    sims = vecs @ vecs[idx]
    sims[idx] = -2.0
    top_idx = np.argsort(sims)[::-1][:topn]
    return [(vocab_list[i], float(sims[i])) for i in top_idx]


def analogy(pos_a, neg_b, pos_c, vecs, vocab_list, word_to_idx, topn=5):
    """向量运算：pos_a − neg_b + pos_c → 最近邻。"""
    exclude = {pos_a, neg_b, pos_c}
    missing = [w for w in exclude if w not in word_to_idx]
    if missing:
        return None, missing
    result_vec = vecs[word_to_idx[pos_a]] - vecs[word_to_idx[neg_b]] + vecs[word_to_idx[pos_c]]
    norm = np.linalg.norm(result_vec)
    if norm > 0:
        result_vec = result_vec / norm
    sims = vecs @ result_vec
    candidates = [(vocab_list[i], float(sims[i]))
                  for i in np.argsort(sims)[::-1]
                  if vocab_list[i] not in exclude]
    return candidates[:topn], []


# ── 字符 n-gram + SVD 词向量（FastText 替代） ─────────────────────────────────

@st.cache_resource(show_spinner=False)
def build_char_word_vectors(tokenized_tuple, vector_size, min_count):
    """字符 3-6-gram TF-IDF + SVD，复现 FastText 处理 OOV 的能力。"""
    tokenized = [list(s) for s in tokenized_tuple]
    freq = Counter(w for sent in tokenized for w in sent)
    vocab_list = sorted(w for w, c in freq.items() if c >= min_count)
    if len(vocab_list) < 3:
        return None, [], {}, None, None

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), min_df=1)
    char_matrix = vectorizer.fit_transform(vocab_list)

    n_comp = min(vector_size, char_matrix.shape[0] - 1, char_matrix.shape[1] - 1, 100)
    if n_comp < 2:
        return None, vocab_list, {w: i for i, w in enumerate(vocab_list)}, vectorizer, None

    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    vecs = svd.fit_transform(char_matrix)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    word_to_idx = {w: i for i, w in enumerate(vocab_list)}
    return vecs / norms, vocab_list, word_to_idx, vectorizer, svd


def get_oov_vector(oov_word, vecs, vocab_list, word_to_idx, vectorizer, svd):
    """利用字符 n-gram 为词汇表外的词生成向量（FastText 核心能力）。"""
    if vectorizer is None or svd is None:
        return None
    try:
        char_vec = vectorizer.transform([oov_word])
        raw = svd.transform(char_vec)[0]
        n = np.linalg.norm(raw)
        return raw / n if n > 0 else raw
    except Exception:
        return None


def avg_sentence_vec(sentence, vecs, word_to_idx):
    tokens = re.findall(r"[A-Za-z]+", sentence.lower())
    v_list, t_list = [], []
    for t in tokens:
        if t in word_to_idx:
            v_list.append(vecs[word_to_idx[t]])
            t_list.append(t)
    if not v_list:
        return None, []
    return np.mean(v_list, axis=0), t_list


# ── 渲染主函数 ────────────────────────────────────────────────────────────────

def render():
    render_module_header(
        "m3",
        "🔬 语义表示与对比分析",
        "统计语义表示 · 共现矩阵词向量 · 向量类比推理 · 字符 n-gram 嵌入",
    )

    with st.expander("📖 功能说明与技术栈", expanded=False):
        st.markdown("""
| 功能 | 说明 | 技术 / 库 |
|------|------|-----------|
| TF-IDF / LSA | 构建词-文档矩阵，经截断 SVD 降维得到低维语义空间 | `scikit-learn` (TfidfVectorizer, TruncatedSVD) |
| 共现矩阵词向量 | 统计词语共现频次，SVD 分解得到分布式词向量（Word2Vec 同源思路） | `numpy`, `scikit-learn` TruncatedSVD |
| 向量类比推理 | 在 LSA 词空间执行 A − B + C 类比运算，验证语义结构 | `numpy` 向量运算 |
| 字符 n-gram 嵌入 | 字符级 TF-IDF + SVD，可为词汇表外词（OOV）生成向量（FastText 同源思路） | `scikit-learn` char_wb TF-IDF |
""")

    tab1, tab2, tab3, tab4 = st.tabs([
        "1️⃣ 统计表示 (TF-IDF/LSA)",
        "2️⃣ 共现矩阵词向量",
        "3️⃣ 向量类比推理",
        "4️⃣ 字符 n-gram 嵌入 (FastText 思路)",
    ])

    # ── Tab 1：TF-IDF + LSA ────────────────────────────────────────────────────
    with tab1:
        st.subheader("传统统计文本表示（TF-IDF + LSA）")
        st.write("输入英文语料（推荐 500-1000 词），系统切分为句子、计算 TF-IDF，并以 LSA 2D 投影可视化。")

        text_input = st.text_area("English Corpus", value=DEFAULT_CORPUS, height=280,
                                  help="可替换为自己的语料。")
        if not text_input.strip():
            st.warning("Please provide an English corpus.")
            return

        word_count = len(text_input.split())
        sentences = split_into_sentences(text_input)
        st.caption(f"Word count: {word_count} | Sentence count: {len(sentences)}")

        if len(sentences) < 2:
            st.error("需要至少两个句子才能构建文档级向量。")
            return

        st.markdown("**Sentence-Level Documents**")
        sent_df = pd.DataFrame({"Sentence ID": [f"S{i+1}" for i in range(len(sentences))],
                                "Text": sentences})
        st.dataframe(sent_df, use_container_width=True, height=220)

        tfidf_matrix, tfidf_terms = vectorize_documents(sentences, mode="TF-IDF")
        tfidf_dense = tfidf_matrix.toarray()
        st.markdown("**TF-IDF Matrix**")
        tfidf_df = pd.DataFrame(tfidf_dense,
                                index=[f"S{i+1}" for i in range(tfidf_dense.shape[0])],
                                columns=tfidf_terms)
        st.dataframe(tfidf_df, use_container_width=True, height=260)

        term_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
        top_indices = np.argsort(term_scores)[::-1][:5]
        st.markdown("**Top 5 Keywords (by Mean TF-IDF Weight)**")
        st.table(pd.DataFrame({"Keyword": tfidf_terms[top_indices],
                               "Mean TF-IDF Weight": term_scores[top_indices]
                               }).style.format({"Mean TF-IDF Weight": "{:.4f}"}))

        st.markdown("**LSA (TruncatedSVD) 2D Vocabulary Projection**")
        basis = st.radio("Choose matrix for LSA:", ["TF-IDF", "One-hot (CountVectorizer)"],
                         horizontal=True)
        lsa_matrix, lsa_terms = vectorize_documents(
            sentences, mode="TF-IDF" if basis == "TF-IDF" else "One-hot")
        if lsa_matrix.shape[1] < 2:
            st.error("需要至少两个不同词汇。")
            return
        svd = TruncatedSVD(n_components=2, random_state=42)
        svd.fit(lsa_matrix)
        term_coords = svd.components_.T
        term_strength = np.asarray(lsa_matrix.sum(axis=0)).ravel()
        plot_df = pd.DataFrame({"Term": lsa_terms, "LSA-1": term_coords[:, 0],
                                "LSA-2": term_coords[:, 1], "Strength": term_strength})
        fig = px.scatter(plot_df, x="LSA-1", y="LSA-2", size="Strength",
                         hover_name="Term",
                         title=f"2D LSA Term Space ({basis})", opacity=0.85)
        fig.update_traces(marker=dict(line=dict(width=0.5, color="white")))
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2：共现矩阵词向量 ──────────────────────────────────────────────────
    with tab2:
        st.subheader("共现矩阵词向量（Word2Vec 同源思路）")
        st.info("**原理**：统计词语在滑动窗口内的共现频次，对共现矩阵做截断 SVD，"
                "得到低维分布式词向量。与 Word2Vec 同源于分布假设：语义相近的词出现在相同上下文中。")

        w2v_text = st.text_area("Training Corpus", value=DEFAULT_CORPUS, height=200,
                                key="w2v_corpus")
        w2v_sentences = split_into_sentences(w2v_text)
        tokenized = tokenize_corpus(w2v_sentences)

        if len(tokenized) < 2:
            st.error("需要至少两个有效句子。")
            return

        col1, col2 = st.columns(2)
        with col1:
            window    = st.slider("Context Window", 2, 10, 5, key="w2v_window")
            min_count = st.slider("Min Word Count", 1, 3, 1, key="w2v_min")
        with col2:
            vector_size = st.slider("Embedding Dimension", 20, 150, 80, 10, key="w2v_dim")

        tokenized_tuple = tuple(tuple(s) for s in tokenized)
        with st.spinner("构建共现矩阵并 SVD 分解…"):
            vecs, vocab_list, word_to_idx = build_cooc_word_vectors(
                tokenized_tuple, window, vector_size, min_count)

        if vecs is None or len(vocab_list) == 0:
            st.error("词汇量过小，请增加语料或减小 Min Word Count。")
            return

        st.success(f"词向量就绪 — 词汇量: {len(vocab_list)} | 维度: {vecs.shape[1]}")
        vocab_preview = vocab_list[:20]

        if "w2v_query" not in st.session_state:
            st.session_state["w2v_query"] = "library"
        _WQ_EX = ["library", "knowledge", "history", "reading"]
        st.caption("🏷️ 示例查询词：")
        _wq_cols = st.columns(len(_WQ_EX))
        for _i, _w in enumerate(_WQ_EX):
            if _wq_cols[_i].button(_w, key=f"w2v_qex_{_i}", use_container_width=True):
                st.session_state["w2v_query"] = _w
        query_word = st.text_input(
            "查询最相似词（Top-5）", key="w2v_query").strip().lower()

        if not query_word:
            st.info("请输入一个词。")
        elif query_word not in word_to_idx:
            st.warning("OOV。词汇表示例：" + ", ".join(vocab_preview))
        else:
            similar = most_similar(query_word, vecs, vocab_list, word_to_idx, topn=5)
            st.markdown("**Top 5 最相似词**")
            st.table(pd.DataFrame(similar, columns=["Word", "Cosine Similarity"]
                                  ).style.format({"Cosine Similarity": "{:.4f}"}))

        st.markdown("**2D 词空间可视化（PCA on SVD vectors）**")
        if len(vocab_list) >= 20:
            show_n = st.slider("显示词数", 10, min(60, len(vocab_list)), 30, key="w2v_show")
            freq = Counter(w for sent in tokenized for w in sent)
            top_words = [w for w, _ in freq.most_common(show_n) if w in word_to_idx]
            indices = [word_to_idx[w] for w in top_words]
            coords_2d = TruncatedSVD(n_components=2, random_state=0).fit_transform(
                vecs[indices])
            scatter_df = pd.DataFrame({"Word": top_words, "x": coords_2d[:, 0],
                                       "y": coords_2d[:, 1]})
            fig2 = px.scatter(scatter_df, x="x", y="y", text="Word",
                              title="词向量 2D 投影（高频词）")
            fig2.update_traces(textposition="top center", marker_size=6)
            fig2.update_layout(height=440)
            st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 3：向量类比推理 ────────────────────────────────────────────────────
    with tab3:
        st.subheader("向量类比推理（A − B + C ≈ ?）")
        st.info("在共现矩阵词向量空间中执行类比运算，验证「语义关系可通过向量差表达」这一核心假设。"
                "GloVe、Word2Vec 的类比能力均来源于此。")

        if "vecs" not in dir() or vecs is None:
            st.warning("请先在 Tab 2 中构建词向量。")
        else:
            if "ana_a" not in st.session_state: st.session_state["ana_a"] = "library"
            if "ana_b" not in st.session_state: st.session_state["ana_b"] = "books"
            if "ana_c" not in st.session_state: st.session_state["ana_c"] = "knowledge"
            _ANA_EX = [
                ("library−books+knowledge", "library", "books", "knowledge"),
                ("history−years+people",    "history", "years", "people"),
                ("reading−books+library",   "reading", "books", "library"),
            ]
            st.caption("🏷️ 预设类比三元组（点击填入）：")
            _ana_cols = st.columns(len(_ANA_EX))
            for _i, (_lbl, _a, _b, _c) in enumerate(_ANA_EX):
                if _ana_cols[_i].button(_lbl, key=f"ana_ex_{_i}", use_container_width=True):
                    st.session_state["ana_a"] = _a
                    st.session_state["ana_b"] = _b
                    st.session_state["ana_c"] = _c
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                word_a = st.text_input("A（正向）", key="ana_a").strip().lower()
            with col_b:
                word_b = st.text_input("B（负向）", key="ana_b").strip().lower()
            with col_c:
                word_c = st.text_input("C（正向）", key="ana_c").strip().lower()

            st.caption("运算：**A − B + C** = ?  "
                       "（类似经典 `king − man + woman ≈ queen`）")

            if st.button("执行类比推理", key="run_analogy"):
                candidates, missing = analogy(word_a, word_b, word_c,
                                              vecs, vocab_list, word_to_idx)
                if missing:
                    st.warning("以下词不在词汇表中：" + ", ".join(missing))
                elif candidates:
                    best, best_score = candidates[0]
                    st.success(f"最接近结果：`{best}`（余弦相似度: {best_score:.4f}）")
                    st.table(pd.DataFrame(candidates, columns=["Word", "Cosine Similarity"]
                                         ).style.format({"Cosine Similarity": "{:.4f}"}))
                else:
                    st.warning("未找到有效候选词。")

            st.markdown("**词语相似度计算**")
            sc1, sc2 = st.columns(2)
            with sc1:
                sim_w1 = st.text_input("词 1", value="library", key="sim_w1").strip().lower()
            with sc2:
                sim_w2 = st.text_input("词 2", value="reading",  key="sim_w2").strip().lower()

            if st.button("计算余弦相似度", key="compute_sim"):
                miss = [w for w in [sim_w1, sim_w2] if w not in word_to_idx]
                if miss:
                    st.warning("OOV：" + ", ".join(miss))
                else:
                    score = _cosine_sim(vecs[word_to_idx[sim_w1]],
                                        vecs[word_to_idx[sim_w2]])
                    st.metric("Cosine Similarity", f"{score:.4f}")

    # ── Tab 4：字符 n-gram 嵌入 ───────────────────────────────────────────────
    with tab4:
        st.subheader("字符 n-gram 嵌入（FastText 同源思路）")
        st.info("**FastText 核心思想**：每个词由其字符 n-gram 集合表达，即使词汇表外的词（OOV）"
                "也能通过共享 n-gram 获得向量。本 Tab 用 `scikit-learn` 字符 TF-IDF + SVD 复现该能力。")

        ft_tokenized = tokenize_corpus(split_into_sentences(DEFAULT_CORPUS))
        ft_tuple = tuple(tuple(s) for s in ft_tokenized)

        ft_dim     = st.slider("Embedding Dimension", 20, 100, 50, 10, key="ft_dim")
        ft_min     = st.slider("Min Word Count", 1, 3, 1, key="ft_min")

        with st.spinner("训练字符 n-gram 嵌入…"):
            ft_result = build_char_word_vectors(ft_tuple, ft_dim, ft_min)
        ft_vecs, ft_vocab, ft_w2i, ft_vec, ft_svd = ft_result

        if ft_vecs is None:
            st.error("词汇量过小，无法构建字符嵌入。")
            return

        st.success(f"字符嵌入就绪 — 词汇量: {len(ft_vocab)} | 维度: {ft_vecs.shape[1]}")

        st.markdown("**OOV 测试：Word2Vec（共现矩阵）vs FastText（字符 n-gram）**")
        if "oov_test" not in st.session_state:
            st.session_state["oov_test"] = "computeer"
        _OOV_EX = ["computeer", "librari", "histori", "languaje"]
        st.caption("🏷️ OOV 示例词：")
        _oov_cols = st.columns(len(_OOV_EX))
        for _i, _w in enumerate(_OOV_EX):
            if _oov_cols[_i].button(_w, key=f"oov_ex_{_i}", use_container_width=True):
                st.session_state["oov_test"] = _w
        oov_word = st.text_input(
            "输入一个可能拼写错误的词（如 computeer）",
            key="oov_test").strip().lower()

        if st.button("运行 OOV 对比", key="run_oov"):
            # 共现词向量：仅表内词有效
            if oov_word in word_to_idx:
                st.success(f"共现矩阵词向量：词汇表内 ✓（词汇表大小 {len(vocab_list)}）")
            else:
                st.warning("共现矩阵词向量：**OOV** — 无法生成向量")

            # 字符 n-gram：可处理 OOV
            oov_vec = get_oov_vector(oov_word, ft_vecs, ft_vocab, ft_w2i, ft_vec, ft_svd)
            if oov_vec is not None:
                sims = ft_vecs @ oov_vec
                top_idx = np.argsort(sims)[::-1][:5]
                similar_ft = [(ft_vocab[i], float(sims[i])) for i in top_idx]
                st.success("字符 n-gram 嵌入：**OOV 向量已生成** ✓（基于字符 n-gram 组合）")
                st.table(pd.DataFrame(similar_ft,
                                      columns=["FastText-style Similar Word", "Cosine Similarity"]
                                      ).style.format({"Cosine Similarity": "{:.4f}"}))
            else:
                st.warning("字符 n-gram 嵌入：无法为该词生成向量（字符序列过短）。")

        st.markdown("**句子相似度（平均词向量）**")
        sent_col1, sent_col2 = st.columns(2)
        with sent_col1:
            sentence_1 = st.text_area(
                "Sentence 1",
                value="The library provides digital archives and coding workshops.",
                key="sent1", height=100)
        with sent_col2:
            sentence_2 = st.text_area(
                "Sentence 2",
                value="Students learn practical skills through community programming projects.",
                key="sent2", height=100)

        if st.button("计算句子相似度", key="sent_sim_btn"):
            v1, t1 = avg_sentence_vec(sentence_1, ft_vecs, ft_w2i)
            v2, t2 = avg_sentence_vec(sentence_2, ft_vecs, ft_w2i)
            if v1 is None or v2 is None:
                st.warning("请输入两个非空英文句子。")
            else:
                score = _cosine_sim(v1, v2)
                st.metric("Sentence Cosine Similarity", f"{score:.4f}")
                st.caption(f"S1 有效词: {len(t1)} | S2 有效词: {len(t2)}")
