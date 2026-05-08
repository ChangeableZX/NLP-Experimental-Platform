# NLP 综合实验平台

> 基于 Streamlit 构建的自然语言处理综合实验平台，涵盖词法分析、句法解析、语义表示、语言模型、机器翻译、情感分析等九大核心模块，支持一键部署至 Streamlit Cloud。本项目已经部署于[https://nlp-experimental-platform-rxkj4jgw45qpmpbvmtwvte.streamlit.app/](https://nlp-experimental-platform-rxkj4jgw45qpmpbvmtwvte.streamlit.app/)

---

## 目录

- [项目概览](#项目概览)
- [功能模块](#功能模块)
- [技术栈](#技术栈)
- [本地安装与启动](#本地安装与启动)
- [Streamlit Cloud 部署](#streamlit-cloud-部署)
- [模块详情](#模块详情)
- [项目结构](#项目结构)

---

## 项目概览

本平台是一套面向自然语言处理课程实验的一体化 Web 应用，所有模块以侧边栏导航的形式组织在单一 Streamlit 应用中。用户可在浏览器内完成从文本输入、算法分析到结果可视化的全流程操作。

**平台特性**

- 九个独立实验模块，覆盖 NLP 主要研究方向
- **一键演示**：每个输入框均配有预设例句按钮，点击即可自动填入并运行分析
- 中英文双语语料支持，中英双模型 NER 与分词
- 实时交互式可视化（依存弧线图、短语结构树、词向量散点图、知识图谱等）
- 深色主题侧边栏 + 浅色内容区，界面简洁专业
- 预训练模型首次使用时自动下载，无需手动管理权重文件
- 兼容 Python 3.14，可直接部署至 Streamlit Cloud

---

## 功能模块

| 编号 | 模块名称 | 核心功能 | 例句按钮 |
|------|----------|----------|----------|
| M1 | 🀄 中文词法分析 | 文本规范化、三模式分词、词频统计、词性标注、算法对比 | 3 个中文示例 |
| M2 | 🔭 句法双引擎透视仪 | 依存句法图、成分结构树、核心论元提取、歧义对比 | 3 个英文句型 |
| M3 | 📐 语义表示与对比分析 | TF-IDF/LSA、共现矩阵词向量、向量类比推理、字符 n-gram | 词汇 / 类比三元组 / OOV |
| M4 | 🧠 词义消歧与语义角色标注 | Lesk 算法、BERT 上下文向量 WSD、依存启发式 SRL | 3 个多义词场景 + 3 个 SRL 句 |
| M5 | 💬 篇章分析与指代消解 | EDU 话语切分、PDTB 篇章关系、神经网络指代消解 | 3 段 EDU / PDTB / 指代文本 |
| M6 | 📈 语言模型训练与对比 | N-gram 统计模型、CharRNN、BERT 完形填空、GPT-2 生成、PPL | 测试句 / Prompt / PPL 组 |
| M7 | 🕸️ 信息抽取与知识图谱 | 命名实体识别（中英）、关系抽取、知识图谱交互可视化 | 3 段中英实体文本 |
| M8 | 🌐 机器翻译与质量评测 | RBMT 规则翻译、NMT 神经翻译、BLEU 分数评测 | 3 个英文句 |
| M9 | ⚡ 情感分析与意见挖掘 | 三分类情感（积极/消极/中性）、显式 vs 隐式对比、批量舆情仪表盘 | 好评 / 差评 / 中性各 1 条 |

---

## 技术栈

### 核心框架

| 类别 | 库 / 框架 | 版本约束 |
|------|-----------|----------|
| Web UI | Streamlit | ≥ 1.31.0 |
| 深度学习 | PyTorch | ≥ 2.0.0 |
| 预训练模型 | Transformers (HuggingFace) | ≥ 4.38.0, < 5.0.0 |

### NLP 工具链

| 库 | 用途 | 说明 |
|----|------|------|
| spaCy `en_core_web_sm` | 依存分析、英文 NER、句法 SRL | 通过 wheel URL 预安装 |
| spaCy `zh_core_web_sm` | 中文分词、中文 NER | 替代 jieba，兼容 Python 3.14 |
| benepar `benepar_en3` | 成分句法分析 | 运行时自动下载 |
| NLTK | WordNet WSD、N-gram LM、句子分割 | 运行时按需下载语料 |
| zhconv | 繁简体转换 | 纯 Python，零依赖 |
| fastcoref | 神经网络指代消解 | 不可用时自动回退规则方法 |
| svgling | 短语结构树 SVG 渲染 | — |

### 向量与可视化

| 库 | 用途 | 说明 |
|----|------|------|
| numpy + scikit-learn | 共现矩阵词向量、TruncatedSVD、TF-IDF | 替代 gensim，兼容 Python 3.14 |
| plotly | 所有统计图表（柱图、散点图、饼图、Gauge） | 浏览器端渲染，原生 Unicode |
| vis-network.js | 知识图谱力导向图 | CDN 引入，无需安装 |
| pandas | 数据框展示 | — |

### 预训练模型（HuggingFace 自动下载）

| 模型 | 大小（约） | 用途 |
|------|-----------|------|
| `bert-base-uncased` | 440 MB | M4 词义消歧、M6 完形填空 |
| `gpt2` | 548 MB | M6 文本生成与困惑度评测 |
| `Helsinki-NLP/opus-mt-en-zh` | 310 MB | M8 英→中神经机器翻译 |
| `lxyuan/distilbert-base-multilingual-cased-sentiments-student` | 280 MB | M9 多语言情感分类 |
| `benepar_en3` | ～75 MB | M2 成分句法分析 |
| `FCoref` (fastcoref) | 362 MB | M5 神经网络指代消解 |

---

## 本地安装与启动

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

spaCy 语言模型（`en_core_web_sm`、`zh_core_web_sm`）已通过 `requirements.txt` 中的 wheel URL 一并安装，**无需单独执行** `python -m spacy download`。

### 2. 下载 NLTK 语料（一次性）

```bash
python -c "import nltk; nltk.download(['wordnet','omw-1.4','punkt','punkt_tab','averaged_perceptron_tagger','reuters'])"
```

### 3. 下载 benepar 成分句法模型（M2 模块使用）

```bash
python -c "import benepar; benepar.download('benepar_en3')"
```

> HuggingFace 预训练模型（BERT、GPT-2、MarianMT、DistilBERT、FCoref）将在**首次访问对应模块**时自动下载，缓存至本地 `~/.cache/huggingface`，后续启动无需重复下载。

### 4. 启动应用

```bash
streamlit run main.py
```

浏览器访问 `http://localhost:8501` 即可使用。

---

## Streamlit Cloud 部署

本项目已针对 Streamlit Cloud 的 Python 3.14 环境做全面兼容处理，可直接推送 GitHub 仓库后一键部署：

1. 登录 [share.streamlit.io](https://share.streamlit.io)，选择该仓库与 `main.py` 作为入口文件
2. Streamlit Cloud 将自动读取 `requirements.txt` 与 `packages.txt` 安装依赖
3. spaCy 语言模型通过 requirements.txt 中的 wheel URL 预安装，避免运行时权限报错

**关键兼容处理说明**

| 问题 | 根因 | 解决方案 |
|------|------|----------|
| gensim 构建失败 | Python 3.14 移除 C API 内部字段（`PyDictObject.ma_version_tag` 等） | 用 numpy + scikit-learn 共现矩阵 + TruncatedSVD 完整替代 |
| jieba SyntaxError | jieba 源码含非法转义序列（`\s`、`\.`），Python 3.14 视为语法错误 | 用 spaCy `zh_core_web_sm` 替代，功能等效 |
| matplotlib 中文乱码 | 云端 Linux 无 CJK 字体，`fm._rebuild()` 在容器内不可靠 | 全面替换为 plotly（浏览器渲染，原生 Unicode） |
| fastcoref AttributeError | PyArrow 18.x 移除 `PyExtensionType`，旧 datasets 仍引用 | 锁定 `datasets>=3.0.0`，异常捕获扩展为 `except Exception` |
| transformers 噪音 import | transformers 5.x 引入 Aria 视觉模型，依赖未安装的 torchvision | 锁定 `transformers>=4.38.0,<5.0.0` |
| spaCy 模型权限错误 | 运行时 `pip install` 写入只读 venv | 通过 wheel URL 在 requirements.txt 中预安装 |

---

## 模块详情

### M1 — 中文词法分析

对输入中文文本依次执行**文本规范化**（繁简转换、全角→半角、去除特殊字符）、**三模式分词**：

| 模式 | 算法 | 特点 |
|------|------|------|
| 精确模式 | spaCy 统计模型 | 最优切分，适合文本分析 |
| 全模式 | 字符级拆分 | 最高召回率，每字独立输出 |
| 搜索引擎模式 | 精确词 + 长词 2-gram 扩展 | 提升检索覆盖率 |

此外提供 18 类词性彩色标注，以及词频 Top-5 柱状图和三模式对比统计图（均使用 plotly 渲染）。黄色高亮各算法切分结果存在差异的词。

**一键演示**：新闻语料 / 古诗词 / 口语对话 3 个预设例句。

---

### M2 — 句法双引擎透视仪

接受英文句子，同时运行两套引擎：

- **spaCy 依存分析**：生成弧线依存图（via displacy）
- **benepar 成分分析**：生成 SVG 嵌套短语结构树（via svgling）

另设**核心论元提取器**，基于依存标签规则提取 ROOT / nsubj / dobj / pobj，以表格结构化展示。

**一键演示**：PP-attachment 歧义句 / 被动结构 / 嵌套从句 3 类经典例句。

---

### M3 — 语义表示与对比分析

四个 Tab 逐步演示从统计向量到分布式词向量的进化路径：

| Tab | 方法 | 核心技术 |
|-----|------|----------|
| 1 | TF-IDF + LSA | sklearn TfidfVectorizer + TruncatedSVD，2D 词汇空间 Plotly 散点图 |
| 2 | 共现矩阵词向量 | 滑动窗口共现统计 + SVD，Word2Vec 同源思路，支持近义词查询 |
| 3 | 向量类比推理 | A − B + C 向量运算，验证"语义关系可通过向量差表达"假设 |
| 4 | 字符 n-gram 嵌入 | char_wb TF-IDF + SVD，FastText 同源思路，可处理 OOV 词 |

**一键演示**：Tab 2 提供 4 个查询词示例；Tab 3 提供 3 组预设类比三元组；Tab 4 提供 4 个 OOV 拼写错误示例。

---

### M4 — 词义消歧与语义角色标注

**词义消歧（WSD）**：针对同一多义词在两个语境下的用法对比：

- **Lesk 算法**：计算上下文词汇与 WordNet 各词义定义的重叠度
- **BERT 上下文向量**：提取目标词的动态语境嵌入，以余弦相似度量化语义差异

**语义角色标注（SRL）**：基于 spaCy 依存标签规则提取谓词框架（A0 施事者 / 谓词 / A1 受事者 / AM-LOC 地点 / AM-TMP 时间），结合 displacy 依存图可视化。

**一键演示**：WSD 提供 bank（金融/河岸）/ crane（起重机/鹤）/ well（水井/良好）3 个多义词场景；SRL 提供科技句 / 被动句 / 时间地点句 3 个示例。

---

### M5 — 篇章分析与指代消解

三个 Tab 覆盖篇章分析核心任务：

| Tab | 功能 | 方法 |
|-----|------|------|
| 话语分割（EDU） | 将篇章切分为基本话语单元，对比规则基线与 NeuralEDUSeg 真实标注 | spaCy 依存 + 启发式规则 |
| 浅层篇章关系（PDTB） | 检测显式连接词，标注 Temporal / Contingency / Comparison / Expansion 关系 | 内置 PDTB 连接词词典 |
| 指代消解 | 识别代词/名词短语与所指实体，以颜色聚类高亮标注 | fastcoref 神经模型（规则方法兜底） |

**一键演示**：各 Tab 均提供 3 段预设文本，点击即填入并运行。

---

### M6 — 语言模型训练与对比分析

| Tab | 内容 |
|-----|------|
| N-gram 统计模型 | 加载 Reuters 语料构建 Bigram / Trigram 模型，支持 Add-one Laplace 平滑，逐 N-gram 展示条件概率 |
| 字符级 RNN（CharRNN） | 在自定义短语料上训练字符级 LSTM，实时绘制 Loss 曲线，训练后按给定 Seed 采样生成文本 |
| 预训练模型 | BERT Masked LM（完形填空 Top-5 预测）和 GPT-2（开放域文本续写） |
| 困惑度评测（PPL） | 用 GPT-2 计算多条测试句子的 Perplexity，以柱状图对比流畅度差异 |

**一键演示**：测试句 / BERT [MASK] 句 / GPT-2 Prompt / PPL 句组均配有预设示例按钮。

---

### M7 — 信息抽取与知识图谱构建

自动检测输入语言（中 / 英），调用对应 spaCy 模型执行：

1. **命名实体识别**：PER（人物）/ ORG（组织）/ LOC（地点）/ MISC（其他），支持中英双语
2. **BIO 序列标注**：可切换为 BIO 格式的字符级标注视图
3. **关系抽取**：基于依存句法规则提取（主体, 关系谓词, 客体）三元组，支持被动句处理
4. **知识图谱可视化**：vis-network.js 力导向图，节点可拖拽，悬停查看详情

**一键演示**：科技英文 / 政治英文 / 中文人物组织 3 个预设示例，点击即分析。

---

### M8 — 机器翻译机制与质量评测

三个 Tab 演示机器翻译发展脉络：

| Tab | 内容 |
|-----|------|
| 神经机器翻译（NMT） | Helsinki-NLP/opus-mt-en-zh（MarianMT），英→中翻译 |
| RBMT vs NMT 对比 | 逐词规则翻译与神经翻译并排对比，词典命中率统计，差距直观可见 |
| BLEU 自动评测 | 输入参考译文与候选译文，计算 BLEU-1 / BLEU-2 / BLEU-4，提供等级解读 |

**说明**：中文 BLEU 分词采用字符级切分（无需 jieba）。

**一键演示**：NMT Tab 提供人工智能 / 气候变化 / 科学探索 3 个预设英文句。

---

### M9 — 电商评论情感分析与意见挖掘

基于多语言 DistilBERT 模型对中文电商评论进行三分类（积极 / 消极 / 中性）：

| Tab | 内容 |
|-----|------|
| 单文本分析 | 输入任意中文评论，输出分类标签、置信度仪表盘（Gauge 图）、三类得分明细 |
| 显式 vs 隐式情感 | 并排对比直接含褒贬词的显式情感与通过客观事实传达的隐式情感，展示模型识别差异 |
| 批量舆情仪表盘 | 随机采样预设评论池，批量分析后生成好评率、情感分布饼图及逐条明细列表 |

**一键演示**：Tab 1 提供好评 / 差评 / 中性 3 条预设评论，点击即分析；Tab 2 提供正向/负向显式与正向/负向隐式共 4 个预设示例。

---

## 项目结构

```
汇总/
├── main.py                 # 应用入口，侧边栏导航与模块路由
├── requirements.txt        # Python 依赖（含 spaCy wheel URL、版本锁定）
├── packages.txt            # Streamlit Cloud 系统包（build-essential, fonts-noto-cjk 等）
└── modules/
    ├── __init__.py
    ├── common.py           # 共用 CSS 样式、模块头渲染函数
    ├── m1_lexical.py       # 中文词法分析（spaCy zh_core_web_sm + plotly）
    ├── m2_syntax.py        # 句法双引擎透视仪（spaCy dep + benepar const）
    ├── m3_semantic.py      # 语义表示与对比（numpy/sklearn 替代 gensim）
    ├── m4_wsd_srl.py       # 词义消歧与语义角色标注（BERT + NLTK WordNet）
    ├── m5_discourse.py     # 篇章分析与指代消解（fastcoref + 规则兜底）
    ├── m6_lm.py            # 语言模型训练与对比（N-gram + RNN + GPT-2）
    ├── m7_ie_kg.py         # 信息抽取与知识图谱（spaCy 双语 + vis-network）
    ├── m8_mt.py            # 机器翻译与质量评测（MarianMT + BLEU，字符级分词）
    └── m9_sentiment.py     # 电商评论情感分析（distilbert 多语言）
```

---

> 自然语言处理综合实验 © 2025
