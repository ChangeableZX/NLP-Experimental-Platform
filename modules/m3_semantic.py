"""模块3：语义表示与对比分析"""

import os
import re
import ssl
import textwrap
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from gensim import downloader as api
from gensim.models import FastText, KeyedVectors, Word2Vec
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


def split_into_sentences(text):
    normalized = " ".join(text.split())
    if not normalized: return []
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        try: nltk.download("punkt", quiet=True)
        except Exception: pass
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


def tokenize_for_word2vec(sentences):
    tokenized = []
    for sent in sentences:
        tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", sent.lower())
        if tokens: tokenized.append(tokens)
    return tokenized


@st.cache_resource(show_spinner=False)
def train_word2vec_model(tokenized_sentences, sg, window, vector_size, min_count, epochs):
    corpus = [list(x) for x in tokenized_sentences]
    return Word2Vec(sentences=corpus, sg=sg, window=window, vector_size=vector_size,
                    min_count=min_count, workers=1, seed=42, epochs=epochs)


@st.cache_resource(show_spinner=False)
def train_fasttext_model(tokenized_sentences, sg, window, vector_size, min_count, epochs):
    corpus = [list(x) for x in tokenized_sentences]
    return FastText(sentences=corpus, sg=sg, window=window, vector_size=vector_size,
                    min_count=min_count, workers=1, seed=42, epochs=epochs)


def average_sentence_vector(sentence, keyed_vectors):
    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", sentence.lower())
    if not tokens: return None, []
    vectors, valid_tokens = [], []
    for token in tokens:
        try:
            vectors.append(keyed_vectors[token]); valid_tokens.append(token)
        except KeyError:
            continue
    if not vectors: return None, []
    return np.mean(vectors, axis=0), valid_tokens


def cosine_similarity(vec1, vec2):
    denom = float(np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return 0.0 if denom == 0 else float(np.dot(vec1, vec2) / denom)


@st.cache_resource(show_spinner=False)
def load_pretrained_glove(model_name="glove-twitter-25"):
    base_dir = Path(os.environ.get("GENSIM_DATA_DIR", str(Path.home()/"gensim-data")))
    local_candidates = [
        base_dir/model_name/f"{model_name}.gz",
        base_dir/model_name/f"{model_name}.txt",
    ]
    for local_path in local_candidates:
        if local_path.exists():
            for no_header in (False, True):
                try:
                    model = KeyedVectors.load_word2vec_format(str(local_path), binary=False, no_header=no_header)
                    return model, f"Loaded from local cache: {local_path}"
                except Exception:
                    continue
    attempts = []
    for ctx_fn, label in [
        (None, "default"),
        (lambda: ssl.create_default_context(cafile=__import__("certifi").where()), "certifi"),
        (ssl._create_unverified_context, "unverified_ssl"),
    ]:
        try:
            if ctx_fn:
                orig = ssl._create_default_https_context
                ssl._create_default_https_context = ctx_fn if callable(ctx_fn) and label == "unverified_ssl" else lambda: ctx_fn()
            model = api.load(model_name)
            if ctx_fn and label != "default":
                ssl._create_default_https_context = orig
            return model, f"Downloaded via {label} context."
        except Exception as exc:
            attempts.append(f"{label}: {exc}")
            try:
                ssl._create_default_https_context = orig
            except Exception:
                pass
    raise RuntimeError("Unable to load glove-twitter-25.\n" + "\n".join(attempts))


def render():
    render_module_header(
        "m3",
        "🔬 语义表示与对比分析",
        "统计语义表示 · Word2Vec · GloVe · FastText · 句子嵌入",
    )

    with st.expander("📖 功能说明与技术栈", expanded=False):
        st.markdown("""
| 功能 | 说明 | 技术 / 库 |
|------|------|-----------|
| TF-IDF / LSA | 用词频-逆文档频率构建词-文档矩阵，再经截断 SVD 降维，得到低维语义空间表示 | `scikit-learn` (TfidfVectorizer, TruncatedSVD) |
| Word2Vec 训练 | 在自定义语料上以 CBOW / Skip-gram 训练词向量，展示近邻词与向量加减运算 | `gensim` Word2Vec |
| GloVe 类比推理 | 加载预训练 GloVe 向量，完成"king − man + woman ≈ queen"类型的类比推理任务 | `gensim` (glove-wiki-gigaword-100) |
| FastText & 句子嵌入 | 子词级词向量（可处理未登录词），以及句子级平均向量的相似度计算与可视化 | `gensim` FastText |
""")

    tab1, tab2, tab3, tab4 = st.tabs([
        "1️⃣ 统计表示 (TF-IDF/LSA)",
        "2️⃣ Word2Vec 训练",
        "3️⃣ 预训练 GloVe",
        "4️⃣ FastText & Sent2Vec",
    ])

    # ── Tab 1 ──────────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("传统统计文本表示（TF-IDF + LSA）")
        st.write("输入英文语料（推荐 500-1000 词），系统将切分为句子文档、计算 TF-IDF 并以 LSA 2D 投影可视化。")

        text_input = st.text_area("English Corpus", value=DEFAULT_CORPUS, height=280,
                                  help="可替换为自己的语料。")
        if not text_input.strip():
            st.warning("Please provide an English corpus.")
            return

        word_count = len(text_input.split())
        sentences = split_into_sentences(text_input)
        st.caption(f"Word count: {word_count} | Sentence count: {len(sentences)}")

        if len(sentences) < 2:
            st.error("Need at least two sentences to build sentence-level document vectors.")
            return

        st.markdown("**Sentence-Level Documents**")
        sent_df = pd.DataFrame({"Sentence ID": [f"S{i+1}" for i in range(len(sentences))], "Text": sentences})
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
        basis = st.radio("Choose matrix for LSA:", ["TF-IDF", "One-hot (CountVectorizer)"], horizontal=True)
        lsa_matrix, lsa_terms = vectorize_documents(sentences, mode="TF-IDF" if basis == "TF-IDF" else "One-hot")
        if lsa_matrix.shape[1] < 2:
            st.error("Need at least two unique vocabulary terms.")
            return
        svd = TruncatedSVD(n_components=2, random_state=42)
        svd.fit(lsa_matrix)
        term_coords = svd.components_.T
        term_strength = np.asarray(lsa_matrix.sum(axis=0)).ravel()
        plot_df = pd.DataFrame({"Term": lsa_terms, "LSA-1": term_coords[:,0],
                                "LSA-2": term_coords[:,1], "Strength": term_strength})
        fig = px.scatter(plot_df, x="LSA-1", y="LSA-2", size="Strength", hover_name="Term",
                         title=f"2D LSA Term Space ({basis})", opacity=0.85)
        fig.update_traces(marker=dict(line=dict(width=0.5, color="white")))
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2 ──────────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Word2Vec 实时训练与测试")
        w2v_text = st.text_area("Training Corpus (Word2Vec)", value=DEFAULT_CORPUS, height=240,
                                key="w2v_corpus")
        w2v_sentences = split_into_sentences(w2v_text)
        tokenized_sentences = tokenize_for_word2vec(w2v_sentences)
        vocab_preview = sorted({w for sent in tokenized_sentences for w in sent})

        if len(tokenized_sentences) < 2:
            st.error("Need at least two valid sentences with words to train Word2Vec.")
            return

        col1, col2 = st.columns(2)
        with col1:
            arch = st.radio("Training Architecture", ["CBOW (sg=0)", "Skip-Gram (sg=1)"],
                            horizontal=True, key="w2v_arch")
            window    = st.slider("Context Window", 2, 10, 5)
            min_count = st.slider("Min Word Count", 1, 3, 1)
        with col2:
            vector_size = st.slider("Embedding Dimension", 20, 200, 100, 10)
            epochs      = st.slider("Training Epochs", 10, 200, 80, 10)
            st.caption(f"Sentences: {len(tokenized_sentences)} | Vocab: {len(vocab_preview)}")

        sg = 0 if arch.startswith("CBOW") else 1
        tokenized_tuple = tuple(tuple(s) for s in tokenized_sentences)
        word2vec_model = train_word2vec_model(tokenized_tuple, sg, window, vector_size, min_count, epochs)

        st.markdown(f"**Model Ready** — Current mode: `{arch}`")
        query_word = st.text_input("Find top-5 most similar words", value="library", key="w2v_query").strip().lower()

        if not query_word:
            st.info("Please input one word.")
        elif query_word not in word2vec_model.wv.key_to_index:
            st.warning("OOV. Try: " + ", ".join(vocab_preview[:15]))
        else:
            similar_words = word2vec_model.wv.most_similar(query_word, topn=5)
            st.markdown("**Top 5 Most Similar Words**")
            st.table(pd.DataFrame(similar_words, columns=["Word", "Cosine Similarity"]
                                  ).style.format({"Cosine Similarity": "{:.4f}"}))

    # ── Tab 3 ──────────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("预训练 GloVe：词语类比与相似度")
        try:
            with st.spinner("Loading pretrained model `glove-twitter-25`…"):
                glove_model, load_status = load_pretrained_glove("glove-twitter-25")
        except Exception as exc:
            st.error("Failed to load `glove-twitter-25`. Check network/SSL or pre-download to gensim cache.")
            st.exception(exc)
            return

        st.caption(f"Vocabulary: {len(glove_model.key_to_index)} | Dim: {glove_model.vector_size} | {load_status}")

        st.markdown("**Word Analogy Calculator (A − B + C)**")
        col_a, col_b, col_c = st.columns(3)
        with col_a: word_a = st.text_input("A", value="king", key="glove_analogy_a").strip().lower()
        with col_b: word_b = st.text_input("B", value="man",  key="glove_analogy_b").strip().lower()
        with col_c: word_c = st.text_input("C", value="woman",key="glove_analogy_c").strip().lower()

        if st.button("Run Word Analogy", key="glove_run_analogy"):
            missing = [w for w in [word_a, word_b, word_c] if w not in glove_model.key_to_index]
            if missing: st.warning("OOV: " + ", ".join(missing))
            else:
                result_vector = glove_model[word_a] - glove_model[word_b] + glove_model[word_c]
                candidates = [(w, s) for w, s in glove_model.similar_by_vector(result_vector, topn=10)
                              if w not in {word_a, word_b, word_c}]
                if not candidates: st.warning("No valid candidate found.")
                else:
                    best_word, best_score = candidates[0]
                    st.success(f"Closest word: `{best_word}` (cosine similarity: {best_score:.4f})")
                    st.table(pd.DataFrame(candidates[:5], columns=["Word","Cosine Similarity"]
                                         ).style.format({"Cosine Similarity": "{:.4f}"}))

        st.markdown("**Word Similarity Score**")
        sc1, sc2 = st.columns(2)
        with sc1: sim_word1 = st.text_input("Word 1", value="car", key="glove_sim_word1").strip().lower()
        with sc2: sim_word2 = st.text_input("Word 2", value="bus", key="glove_sim_word2").strip().lower()

        if st.button("Compute Similarity", key="glove_compute_similarity"):
            missing = [w for w in [sim_word1, sim_word2] if w not in glove_model.key_to_index]
            if missing: st.warning("OOV: " + ", ".join(missing))
            else:
                st.metric("Cosine Similarity", f"{glove_model.similarity(sim_word1, sim_word2):.4f}")

    # ── Tab 4 ──────────────────────────────────────────────────────────────────
    with tab4:
        st.subheader("FastText & Sentence-Level Representation (Sent2Vec)")
        ft_tokenized_sentences = tokenized_sentences
        if len(ft_tokenized_sentences) < 2:
            st.error("Need at least two tokenized sentences.")
            return

        st.caption(f"Corpus: {len(ft_tokenized_sentences)} sentences | Vocab: {len(vocab_preview)}")
        with st.spinner("Training FastText model…"):
            fasttext_model = train_fasttext_model(tokenized_tuple, sg, window, vector_size, min_count, epochs)

        st.markdown("**OOV Test: Word2Vec vs FastText**")
        oov_word = st.text_input("Potentially misspelled word (e.g., computeer)",
                                 value="computeer", key="oov_word_test").strip().lower()

        if st.button("Run OOV Comparison", key="run_oov_compare"):
            try:
                _ = word2vec_model.wv[oov_word]
                st.success("Word2Vec: This word exists in vocabulary.")
            except KeyError:
                st.warning("Word2Vec: OOV (not in vocabulary).")
            try:
                ft_similar = fasttext_model.wv.most_similar(oov_word, topn=5)
                st.success("FastText: Vector computed successfully.")
                query_vec = fasttext_model.wv[oov_word]
                ft_rows = [(w, cs, float(np.linalg.norm(query_vec - fasttext_model.wv[w])))
                           for w, cs in ft_similar]
                st.table(pd.DataFrame(ft_rows, columns=["FastText Similar Word","Cosine Similarity","Euclidean Distance"]
                                      ).style.format({"Cosine Similarity":"{:.8f}","Euclidean Distance":"{:.8f}"}))
            except KeyError:
                st.warning("FastText could not compose a vector. Try a longer alphabetic word.")

        st.markdown("**Sent2Vec (Average Pooling with FastText Vectors)**")
        sent_col1, sent_col2 = st.columns(2)
        with sent_col1:
            sentence_1 = st.text_area("Sentence 1",
                value="The library provides digital archives and coding workshops that help students learn practical skills.",
                key="sent2vec_sentence1", height=110)
        with sent_col2:
            sentence_2 = st.text_area("Sentence 2",
                value="Students improve their abilities through community projects, archived data, and collaborative programming activities.",
                key="sent2vec_sentence2", height=110)

        if st.button("Compute Sentence Similarity", key="compute_sent2vec_similarity"):
            vec1, tokens1 = average_sentence_vector(sentence_1, fasttext_model.wv)
            vec2, tokens2 = average_sentence_vector(sentence_2, fasttext_model.wv)
            if vec1 is None or vec2 is None:
                st.warning("Please provide two non-empty English sentences.")
            else:
                sent_sim = cosine_similarity(vec1, vec2)
                st.metric("Sentence Cosine Similarity", f"{sent_sim:.4f}")
                st.caption(f"S1 tokens: {len(tokens1)} | S2 tokens: {len(tokens2)}")
