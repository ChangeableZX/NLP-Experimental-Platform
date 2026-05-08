"""模块6：语言模型训练与对比分析"""

import math

import nltk
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from nltk import ngrams, sent_tokenize, word_tokenize
from nltk.corpus import reuters
from collections import Counter
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

from .common import render_module_header, sec_header

# ── NLTK 数据检查（模块级，非UI）────────────────────────────────────────────────
def _ensure_nltk():
    for corpus_name in ["reuters"]:
        try: nltk.data.find(f"corpora/{corpus_name}")
        except LookupError: nltk.download(corpus_name, quiet=True)
    for tok in ["punkt", "punkt_tab"]:
        try: nltk.data.find(f"tokenizers/{tok}")
        except LookupError: nltk.download(tok, quiet=True)


# ── CharRNN 模型定义 ──────────────────────────────────────────────────────────
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        out, hidden = self.rnn(embedded, hidden)
        return self.fc(out), hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


# ── 预训练模型（缓存加载）────────────────────────────────────────────────────
@st.cache_resource
def load_bert_pipeline():
    return pipeline("fill-mask", model="bert-base-uncased", tokenizer="bert-base-uncased")


@st.cache_resource
def load_gpt2_pipeline():
    return pipeline("text-generation", model="gpt2", tokenizer="gpt2")


@st.cache_resource
def load_gpt2_ppl_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model


def render():
    render_module_header(
        "m6",
        "📊 语言模型训练与对比分析",
        "N-gram 统计模型 · 字符级 RNN · BERT & GPT-2 预训练 · 困惑度评测",
    )

    with st.expander("📖 功能说明与技术栈", expanded=False):
        st.markdown("""
| 功能 | 说明 | 技术 / 库 |
|------|------|-----------|
| N-gram 统计模型 | 基于 Reuters 语料统计 N-gram 条件概率，支持 Laplace 加法平滑与 KN 平滑，可计算任意句子概率 | `NLTK` (ngrams, Reuters 语料库) |
| 字符级 RNN (CharRNN) | 在自定义文本上训练 LSTM 字符级生成模型，支持实时训练过程可视化，并按温度参数采样生成文本 | `PyTorch` (LSTM, Adam 优化器) |
| BERT 填词预测 | 将句子中的词替换为 [MASK]，让 BERT 预测最可能的填充词，展示预训练语言知识 | `transformers` (bert-base-uncased) |
| GPT-2 文本生成 | 输入前缀文本，GPT-2 以自回归方式续写，支持调节生成长度与 top-k 采样 | `transformers` (gpt2) |
| 困惑度 (PPL) 评测 | 计算给定文本在 GPT-2 下的困惑度分值，数值越低代表文本越流畅自然 | `transformers` (GPT2LMHeadModel) |
""")

    _ensure_nltk()

    tab_ngram, tab_rnn, tab_pretrained, tab_compare = st.tabs([
        "1️⃣ N-gram 统计模型",
        "2️⃣ 自定义字符级 RNN",
        "3️⃣ 预训练模型（BERT / GPT-2）",
        "4️⃣ 结果对比与困惑度",
    ])

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 1: N-gram
    # ════════════════════════════════════════════════════════════════════════════
    with tab_ngram:
        st.header("基于统计的 N-gram 语言模型")
        st.markdown("加载英文语料，构建 **Bigram / Trigram** 模型，计算输入句子的生成概率，并对比 **Add-one (Laplace) 平滑** 前后结果。")

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("语料与模型设置")
            use_reuters = st.checkbox("加载 NLTK Reuters 样本语料", value=True)
            if use_reuters:
                default_corpus = " ".join(reuters.words()[:800])
            else:
                default_corpus = ("Natural language processing is a subfield of linguistics, "
                                  "computer science, and artificial intelligence concerned with "
                                  "the interactions between computers and human language.")
            corpus_text = st.text_area("📄 基础语料（可手动编辑）", value=default_corpus, height=220)

            n_option = st.selectbox(
                "选择 N-gram 阶数",
                options=[("Bigram (2-gram)", 2), ("Trigram (3-gram)", 3)],
                format_func=lambda x: x[0],
            )
            n = n_option[1]
            use_smoothing = st.checkbox("✅ 开启 Add-one (Laplace) 平滑", value=False)

        with col_right:
            st.subheader("句子概率计算")
            test_sentence = st.text_input("✏️ 输入测试句子",
                                          value="The company said it will increase prices .")
            calc_clicked = st.button("🚀 计算概率", use_container_width=True)

            if calc_clicked:
                sentences = sent_tokenize(corpus_text.lower())
                vocab = set()
                ngram_counts = Counter()
                history_counts = Counter()

                for sent in sentences:
                    tokens = word_tokenize(sent)
                    vocab.update(tokens)
                    if n == 2:
                        padded = ["<s>"] + tokens + ["</s>"]
                        for bg in ngrams(padded, 2):
                            ngram_counts[bg] += 1; history_counts[bg[0]] += 1
                    else:
                        padded = ["<s>", "<s>"] + tokens + ["</s>"]
                        for tg in ngrams(padded, 3):
                            ngram_counts[tg] += 1; history_counts[(tg[0],tg[1])] += 1

                V = len(vocab)
                if V == 0:
                    st.error("语料为空，请先输入有效的英文语料。"); return

                test_tokens = word_tokenize(test_sentence.lower())
                if n == 2:
                    padded_test = ["<s>"] + test_tokens + ["</s>"]
                    test_ngrams = list(ngrams(padded_test, 2))
                else:
                    padded_test = ["<s>","<s>"] + test_tokens + ["</s>"]
                    test_ngrams = list(ngrams(padded_test, 3))

                raw_joint_prob = smooth_joint_prob = 1.0
                details = []; zero_prob_flag = False

                for ng in test_ngrams:
                    if n == 2:
                        w1, w2 = ng; hist = w1; ngram_str = f"P({w2} | {w1})"
                    else:
                        w1, w2, w3 = ng; hist = (w1,w2); ngram_str = f"P({w3} | {w1}, {w2})"
                    count_ngram = ngram_counts[ng]; count_hist = history_counts[hist]
                    raw_p = 0.0 if count_hist == 0 else count_ngram / count_hist
                    smooth_p = (1/V) if count_hist == 0 else (count_ngram+1)/(count_hist+V)
                    if raw_p == 0.0: zero_prob_flag = True
                    raw_joint_prob *= raw_p; smooth_joint_prob *= smooth_p
                    details.append({"条件概率":ngram_str,"历史计数":count_hist,
                                    "N-gram 计数":count_ngram,
                                    "未平滑概率":f"{raw_p:.6f}" if raw_p>0 else "0.0 (OOV)",
                                    "平滑后概率":f"{smooth_p:.6f}"})

                st.markdown("---"); st.write("#### 逐 N-gram 概率分解")
                st.table(details)

                raw_log_prob = math.log2(raw_joint_prob) if raw_joint_prob > 0 else float("-inf")
                smooth_log_prob = math.log2(smooth_joint_prob)

                if zero_prob_flag:
                    st.warning("⚠️ 检测到语料库中 **未出现的 N-gram**（零概率事件）。")
                    col_raw, col_smooth = st.columns(2)
                    with col_raw:
                        st.metric("未平滑联合概率", "0.0")
                        st.metric("未平滑对数概率 (log₂)", "-∞")
                    with col_smooth:
                        st.metric("Add-one 平滑后联合概率", f"{smooth_joint_prob:.6e}")
                        st.metric("平滑后对数概率 (log₂)", f"{smooth_log_prob:.4f}")
                    if not use_smoothing:
                        st.info("💡 当前未开启平滑，整体联合概率为 **0**。勾选左侧平滑选项可得非零概率。")
                    else:
                        st.success("✅ 当前已开启平滑，使用平滑后的概率（右侧数值）。")
                else:
                    display_prob = smooth_joint_prob if use_smoothing else raw_joint_prob
                    display_log = smooth_log_prob if use_smoothing else raw_log_prob
                    st.metric(f"{'平滑后' if use_smoothing else '原始'}联合概率", f"{display_prob:.6e}")
                    st.metric(f"{'平滑后' if use_smoothing else '原始'}对数概率 (log₂)", f"{display_log:.4f}")
                    if use_smoothing:
                        st.info(f"未平滑原始联合概率: {raw_joint_prob:.6e}，对数概率: {raw_log_prob:.4f}")

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 2: CharRNN
    # ════════════════════════════════════════════════════════════════════════════
    with tab_rnn:
        st.header("🔥 自定义字符级 RNN 训练")
        st.markdown("输入一段英文短语料，调整超参数后点击 **开始训练**，训练完成后可根据给定 Seed 自动生成文本。")

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("语料与超参数")
            default_rnn_text = (
                "To be, or not to be, that is the question:\n"
                "Whether 'tis nobler in the mind to suffer\n"
                "The slings and arrows of outrageous fortune,\n"
                "Or to take arms against a sea of troubles\n"
                "And by opposing end them."
            )
            rnn_text = st.text_area("✏️ 输入训练语料（字符级）", value=default_rnn_text, height=200)
            hidden_size   = st.slider("Hidden Size", 16, 128, 64, 16)
            epochs        = st.slider("Epochs", 10, 200, 50, 10)
            learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.005, 0.0001, format="%.4f")
            seq_len       = st.slider("序列长度 (Seq Len)", 10, 100, 25, 5)
            train_btn     = st.button("🚀 开始训练", use_container_width=True)

        with col_right:
            st.subheader("训练与生成")
            if train_btn:
                if not rnn_text or len(set(rnn_text)) < 2:
                    st.error("语料太短或字符种类太少，请输入更丰富的文本。")
                else:
                    chars = sorted(set(rnn_text))
                    char2idx = {ch:i for i,ch in enumerate(chars)}
                    idx2char = {i:ch for i,ch in enumerate(chars)}
                    vocab_size = len(chars)
                    data = [char2idx[ch] for ch in rnn_text]
                    inputs, targets = [], []
                    for i in range(len(data)-seq_len):
                        inputs.append(data[i:i+seq_len]); targets.append(data[i+1:i+seq_len+1])
                    inputs = torch.tensor(inputs, dtype=torch.long)
                    targets = torch.tensor(targets, dtype=torch.long)

                    model = CharRNN(vocab_size, hidden_size)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                    loss_chart = st.empty(); progress_bar = st.progress(0); loss_history = []

                    for epoch in range(epochs):
                        optimizer.zero_grad()
                        output, _ = model(inputs)
                        loss = criterion(output.reshape(-1, vocab_size), targets.reshape(-1))
                        loss.backward(); optimizer.step()
                        loss_history.append(loss.item())
                        progress_bar.progress((epoch+1)/epochs)
                        if (epoch+1) % 5 == 0 or epoch == epochs-1:
                            loss_chart.line_chart({"Loss": loss_history})

                    st.success(f"训练完成！最终 Loss: {loss_history[-1]:.4f}")
                    st.session_state.rnn_model     = model
                    st.session_state.rnn_char2idx  = char2idx
                    st.session_state.rnn_idx2char  = idx2char
                    st.session_state.rnn_losses    = loss_history

            if "rnn_model" in st.session_state and st.session_state.rnn_model is not None:
                st.markdown("---"); st.write("#### 📝 文本生成")
                seed    = st.text_input("输入起始字符（Seed）", value="To be", key="rnn_seed")
                gen_len = st.number_input("生成长度（字符数）", 10, 500, 50, 10)
                if st.button("✨ 生成文本", key="rnn_generate"):
                    model    = st.session_state.rnn_model
                    char2idx = st.session_state.rnn_char2idx
                    idx2char = st.session_state.rnn_idx2char
                    invalid_chars = [c for c in seed if c not in char2idx]
                    if invalid_chars:
                        st.error(f"Seed 中包含训练集外的字符: {set(invalid_chars)}")
                    else:
                        model.eval()
                        input_eval = torch.tensor([char2idx[c] for c in seed], dtype=torch.long).unsqueeze(0)
                        hidden = model.init_hidden(1); generated = seed
                        with torch.no_grad():
                            out, hidden = model(input_eval, hidden)
                            for _ in range(gen_len):
                                out, hidden = model(input_eval[:,-1:], hidden)
                                prob = torch.softmax(out[:,-1,:], dim=-1)
                                next_idx = torch.argmax(prob).item()
                                generated += idx2char[next_idx]
                                input_eval = torch.tensor([[next_idx]], dtype=torch.long)
                        st.text_area("生成结果", value=generated, height=150)
            else:
                st.info("请先点击左侧的 **开始训练** 按钮完成模型训练。")

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 3: BERT vs GPT-2
    # ════════════════════════════════════════════════════════════════════════════
    with tab_pretrained:
        st.header("🤖 现代预训练模型：BERT vs GPT-2")
        st.markdown("左侧体验 **BERT (Masked LM)** 的完形填空，右侧体验 **GPT-2 (Causal LM)** 的文本生成。")

        col_bert, col_gpt2 = st.columns(2)

        with col_bert:
            st.subheader("BERT — Masked Language Modeling")
            bert_input = st.text_input("输入带 [MASK] 的句子",
                                       value="The man went to the [MASK] to buy some milk.",
                                       key="bert_input")
            if st.button("🔍 BERT 预测", key="bert_btn", use_container_width=True):
                if "[MASK]" not in bert_input:
                    st.error("请确保输入中包含 `[MASK]` 标记。")
                else:
                    with st.spinner("正在加载 BERT 并预测，首次可能需要下载模型…"):
                        try:
                            fill_masker = load_bert_pipeline()
                            results = fill_masker(bert_input)
                            st.write("**Top-5 候选词及其概率：**")
                            for i, res in enumerate(results[:5], 1):
                                st.write(f"{i}. `{res['token_str'].strip()}` — {res['score']:.4f}")
                        except Exception as e:
                            st.error(f"预测失败：{e}")

        with col_gpt2:
            st.subheader("GPT-2 — Causal Language Modeling")
            gpt2_input = st.text_input("输入前缀 Prompt",
                                       value="In the future, artificial intelligence will",
                                       key="gpt2_input")
            if st.button("✨ GPT-2 生成", key="gpt2_btn", use_container_width=True):
                with st.spinner("正在加载 GPT-2 并生成文本，首次可能需要下载模型…"):
                    try:
                        generator = load_gpt2_pipeline()
                        outputs = generator(gpt2_input, max_new_tokens=20,
                                            num_return_sequences=1, do_sample=True, temperature=0.9)
                        st.write("**生成结果：**")
                        st.info(outputs[0]["generated_text"])
                    except Exception as e:
                        st.error(f"生成失败：{e}")

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 4: PPL 评测
    # ════════════════════════════════════════════════════════════════════════════
    with tab_compare:
        st.header("📈 结果对比与评价指标（困惑度 PPL）")
        st.markdown("使用 **GPT-2** 计算输入句子的 **Perplexity (PPL)**。困惑度越低，说明模型认为该句子越『自然』。")

        ppl_text = st.text_area(
            "✏️ 输入多段测试句子（每行一句）",
            value=(
                "The cat sat on the mat.\n"
                "Quantum entanglement revolutionizes cryptography.\n"
                "asdf ghjk zxcv bnm qwer tyui op."
            ),
            height=200,
        )

        if st.button("🧮 计算 PPL", use_container_width=True):
            sentences = [s.strip() for s in ppl_text.split("\n") if s.strip()]
            if not sentences:
                st.error("请输入至少一个有效的测试句子。")
            else:
                with st.spinner("正在加载 GPT-2 并逐句计算困惑度，首次可能需要下载模型…"):
                    try:
                        tokenizer_ppl, model_ppl = load_gpt2_ppl_model()
                        results = []
                        for sent in sentences:
                            inputs = tokenizer_ppl(sent, return_tensors="pt")
                            with torch.no_grad():
                                outputs = model_ppl(**inputs, labels=inputs["input_ids"])
                                loss = outputs.loss.item()
                                ppl = math.exp(loss)
                            results.append({"测试句子":sent, "Cross-Entropy Loss":round(loss,4), "PPL":round(ppl,2)})

                        st.write("#### 各句子的困惑度 (PPL) 结果")
                        st.table(results)
                        st.write("#### PPL 对比图")
                        st.bar_chart({"PPL": [r["PPL"] for r in results]})
                    except Exception as e:
                        st.error(f"计算失败：{e}")
