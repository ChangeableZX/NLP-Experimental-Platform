# NLP 综合实验平台

> 基于 Streamlit 构建的自然语言处理综合实验平台，涵盖词法分析、句法解析、语义表示、语言模型、机器翻译、情感分析等九大核心模块。

---

## 目录

- [项目概览](#项目概览)
- [功能模块](#功能模块)
- [技术栈](#技术栈)
- [环境要求](#环境要求)
- [安装与启动](#安装与启动)
- [模块详情](#模块详情)
- [项目结构](#项目结构)

---

## 项目概览

本平台是一套面向自然语言处理课程实验的一体化 Web 应用，所有模块以侧边栏导航的形式组织在单一 Streamlit 应用中。用户可在浏览器内完成从文本输入、算法分析到结果可视化的全流程操作，无需额外配置。

**平台特性**

- 九个独立实验模块，覆盖 NLP 主要研究方向
- 中英文双语语料支持
- 实时可视化（依存弧线图、短语结构树、词向量散点图、知识图谱等）
- 深色主题侧边栏 + 浅色内容区，界面简洁专业
- 模型首次使用时自动下载，无需手动管理权重文件

---

## 功能模块

| 编号 | 模块名称 | 核心功能 |
|------|----------|----------|
| M1 | 🀄 中文词法分析 | 文本规范化、中文分词（三种模式）、词频统计、词性标注、分词算法对比 |
| M2 | 🔭 句法双引擎透视仪与歧义侦探 | 依存句法分析、成分句法分析、核心论元提取、歧义结构透视 |
| M3 | 📐 语义表示与对比分析 | TF-IDF / 词袋向量化、词向量（Word2Vec / FastText）、SVD 降维可视化、语义相似度对比 |
| M4 | 🧠 词义消歧与语义角色标注 | Lesk 算法、BERT 上下文嵌入消歧、WordNet 义项查询、基于依存的 SRL |
| M5 | 💬 篇章分析与指代消解系统 | EDU 基本话语单元切分、RST 修辞关系标注、指代链识别（规则 + 神经网络） |
| M6 | 📈 语言模型训练与对比分析 | N-gram 语言模型、CharRNN 训练（Reuters 语料）、GPT-2 文本生成、困惑度对比 |
| M7 | 🕸️ 信息抽取与知识图谱构建 | 命名实体识别（中英双语）、关系抽取、知识图谱交互式可视化 |
| M8 | 🌐 机器翻译机制与质量评测 | 规则翻译（RBMT）、神经机器翻译（Helsinki-NLP MarianMT）、BLEU 分数评测 |
| M9 | ⚡ 电商评论情感分析与意见挖掘 | 多语言情感分类（积极 / 消极 / 中性）、批量评论分析、意见统计可视化 |

---

## 技术栈

**核心框架**

| 类别 | 库 / 框架 | 版本要求 |
|------|-----------|----------|
| Web UI | Streamlit | ≥ 1.31.0 |
| 深度学习 | PyTorch | ≥ 2.0.0 |
| 预训练模型 | Transformers (HuggingFace) | ≥ 4.38.0 |

**NLP 工具链**

| 库 | 用途 |
|----|------|
| spaCy (`en_core_web_sm`, `zh_core_web_sm`) | 依存分析、NER、SRL |
| benepar (`benepar_en3`) | 成分句法分析 |
| NLTK | 分词、WordNet WSD、N-gram LM、BLEU |
| jieba | 中文分词与词性标注 |
| zhconv | 繁简体转换 |
| gensim | Word2Vec / FastText 词向量 |
| fastcoref | 神经网络指代消解 |

**可视化与数据处理**

| 库 | 用途 |
|----|------|
| matplotlib | 词频柱状图、分词对比图 |
| plotly | 词向量散点图、情感统计图 |
| svgling | 短语结构树 SVG 渲染 |
| pandas / numpy | 数据处理 |
| scikit-learn | TF-IDF、SVD 降维 |

**预训练模型（HuggingFace 自动下载）**

| 模型 | 用途 |
|------|------|
| `bert-base-uncased` | M4 词义消歧、M6 完形填空 |
| `gpt2` | M6 文本生成 |
| `Helsinki-NLP/opus-mt-en-zh` | M8 神经机器翻译 |
| `lxyuan/distilbert-base-multilingual-cased-sentiments-student` | M9 情感分类 |

---

## 环境要求

- Python **3.9 – 3.11**（推荐 3.10）
- pip ≥ 23.0
- （可选）CUDA 兼容 GPU，可加速 PyTorch 推理

---

## 安装与启动

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载语言模型与语料（一次性）

```bash
# spaCy 语言模型
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm

# NLTK 语料
python -c "import nltk; nltk.download(['wordnet','omw-1.4','punkt','punkt_tab','averaged_perceptron_tagger','reuters','brown'])"

# benepar 成分句法模型
python -c "import benepar; benepar.download('benepar_en3')"
```

> HuggingFace 模型（BERT、GPT-2、MarianMT、DistilBERT）将在首次访问对应模块时**自动下载**，无需手动操作。

### 3. 启动应用

```bash
streamlit run main.py
```

浏览器访问 `http://localhost:8501` 即可使用平台。

---

## 模块详情

### M1 — 中文词法分析

对输入中文文本依次执行**文本规范化**（繁简转换、全角→半角、去除特殊字符）、**三模式分词**（精确 / 全模式 / 搜索引擎）、**词性标注**（18 类词性彩色标注）及**词频统计**，并以柱状图对比三种分词算法在词数、唯一词数、平均词长上的差异。黄色高亮标出各算法切分结果存在歧义的词。

### M2 — 句法双引擎透视仪

接受英文句子，同时运行 **spaCy 依存分析引擎**（生成弧线依存图）和 **benepar 成分分析引擎**（生成 SVG 短语结构树），并列展示两种句法视图。另设**歧义侦探**功能，对经典 PP-attachment 歧义句显式对比两种句法解读。

### M3 — 语义表示与对比分析

在内置英文语料（可替换）上演示：
- TF-IDF / 词袋矩阵构建与词汇权重排名
- Word2Vec / FastText 词向量训练与近义词查询
- SVD 降维后的二维语义空间 Plotly 散点图
- 跨方法语义相似度对比

### M4 — 词义消歧与语义角色标注

**WSD 部分**：对歧义词分别使用 Lesk 算法（基于词义定义重叠）和 BERT 上下文嵌入（余弦相似度选择最优义项），并列出 WordNet 候选义项。  
**SRL 部分**：基于 spaCy 依存标签规则提取主语、谓语、宾语、介词短语，以结构化表格呈现语义角色。

### M5 — 篇章分析与指代消解

- **EDU 切分**：基于连接词规则将段落切分为基本话语单元，标注边界词
- **RST 修辞关系**：对相邻 EDU 对标注 Elaboration / Contrast / Cause 等修辞关系
- **指代消解**：优先使用 fastcoref 神经模型，不可用时回退到规则（代词-先行词距离匹配），以颜色高亮指代链

### M6 — 语言模型训练与对比分析

在 Reuters 语料上训练 Bigram / Trigram N-gram 模型，同时训练字符级 RNN（CharRNN），对比三者在测试集上的**困惑度（Perplexity）**。另集成 GPT-2 进行开放域文本续写，并使用 BERT MLM 展示完形填空。

### M7 — 信息抽取与知识图谱构建

自动检测输入语言（中 / 英），调用对应 spaCy 模型进行**命名实体识别**（PER / ORG / LOC / MISC），再通过依存规则抽取**实体间关系三元组**，最终以交互式力导向图渲染知识图谱（节点可拖拽）。

### M8 — 机器翻译机制与质量评测

对同一英文原句分别使用：
1. **RBMT 规则翻译**（内置 500+ 条词典 + 简单词序调整）
2. **NMT 神经翻译**（Helsinki-NLP/opus-mt-en-zh MarianMT）

输入参考译文后，两路输出分别计算 **BLEU 分数**并可视化对比，直观展示规则翻译与神经翻译的质量差异。

### M9 — 电商评论情感分析

基于多语言 DistilBERT 模型对电商评论进行三分类情感分析（积极 / 消极 / 中性），支持单条分析与批量分析两种模式，输出置信度进度条、分类分布饼图及按情感类别分组的评论列表。

---

## 项目结构

```
汇总/
├── main.py              # 应用入口，侧边栏导航与模块路由
├── requirements.txt     # Python 依赖（含安装后操作说明）
└── modules/
    ├── __init__.py
    ├── common.py        # 共用 CSS 样式、模块头渲染函数
    ├── m1_lexical.py    # 中文词法分析
    ├── m2_syntax.py     # 句法双引擎透视仪
    ├── m3_semantic.py   # 语义表示与对比分析
    ├── m4_wsd_srl.py    # 词义消歧与语义角色标注
    ├── m5_discourse.py  # 篇章分析与指代消解
    ├── m6_lm.py         # 语言模型训练与对比
    ├── m7_ie_kg.py      # 信息抽取与知识图谱
    ├── m8_mt.py         # 机器翻译与质量评测
    └── m9_sentiment.py  # 电商评论情感分析
```

---

> 自然语言处理综合实验 © 2025
