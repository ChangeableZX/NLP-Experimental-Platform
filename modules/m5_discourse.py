"""模块5：篇章分析与指代消解系统"""

import re
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional

import requests
import spacy
import streamlit as st

from .common import render_module_header, sec_header


# ── CSS 局部样式 ──────────────────────────────────────────────────────────────
_CSS = """
<style>
.edu-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 12px 16px; margin: 8px 0; border-radius: 8px;
    border-left: 4px solid #4a5568; color: white;
    font-size: 14px; line-height: 1.5;
}
.edu-card-gold {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 12px 16px; margin: 8px 0; border-radius: 8px;
    border-left: 4px solid #e53e3e; color: white;
    font-size: 14px; line-height: 1.5;
}
.boundary-highlight {
    background-color: #ffd700; color: #000; padding: 2px 6px;
    border-radius: 4px; font-weight: bold;
    text-decoration: underline; text-decoration-color: #e53e3e;
    text-decoration-thickness: 2px;
}
.m5-metric-card {
    background: #f7fafc; padding: 16px; border-radius: 8px;
    border: 1px solid #e2e8f0;
}
</style>
"""


@dataclass
class EDUSegment:
    text: str
    boundary_token: str
    token_index: int
    edu_index: int


class EDUSegmentationModule:
    def __init__(self):
        self.base_url = "https://cdn.jsdelivr.net/gh/PKU-TANGENT/NeuralEDUSeg@master/data/rst/"
        self.nlp = None

    def load_spacy_model(self):
        if self.nlp is None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                st.error("请先运行: python -m spacy download en_core_web_sm")
                return None
        return self.nlp

    def fetch_neuraleduseg_data(self, dataset="TRAINING", filename=None):
        if filename is None:
            filename = "wsj_0600.out.rst"
        edu_file_url = f"{self.base_url}{dataset}/{filename}"
        try:
            response = requests.get(edu_file_url, timeout=10)
            response.raise_for_status()
            content = response.text
            edus = []
            for line in content.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    edu_text = re.sub(r"<[^>]+>", "", line).strip()
                    if edu_text: edus.append(edu_text)
            full_text = " ".join(edus)
            return full_text, content, edus
        except requests.RequestException as e:
            st.error(f"无法获取数据: {e}")
            sample_text = "Although the economy is improving, many people still struggle to find jobs. The government has implemented new policies to address this issue."
            sample_edus = [
                "Although the economy is improving,",
                "many people still struggle to find jobs.",
                "The government has implemented new policies to address this issue."
            ]
            return sample_text, "\n".join(sample_edus), sample_edus

    def parse_gold_edus(self, edu_content):
        edus = []
        for idx, line in enumerate(edu_content.strip().split("\n"), 1):
            line = line.strip()
            if not line or line.startswith("#"): continue
            clean_text = re.sub(r"<[^>]+>", "", line).strip()
            words = clean_text.split()
            if words:
                boundary_token = words[-1].rstrip(".,;:!?")
                edus.append(EDUSegment(text=clean_text, boundary_token=boundary_token,
                                       token_index=len(words)-1, edu_index=idx))
        return edus

    def rule_based_segmentation(self, text):
        nlp = self.load_spacy_model()
        if nlp is None: return []
        doc = nlp(text)
        subordinating_conjunctions = {
            "although","though","while","whereas","because","since","if","unless",
            "when","after","before","until","even","once","till","whenever","where",
            "wherever","whether","as"
        }
        split_indices = set()
        for i, token in enumerate(doc):
            if token.text in ".!?" or (token.text == '"' and i>0 and doc[i-1].text in ".!?"):
                split_indices.add(i); continue
            if token.pos_ == "SCONJ" and i>0:
                prev_token = doc[i-1]
                if prev_token.text in ".!?;:" or (prev_token.text=="," and i>1):
                    split_indices.add(i-1)
                elif token.text.lower() in subordinating_conjunctions:
                    if i>0 and doc[i-1].text==",": split_indices.add(i-1)
            if token.dep_ in {"advcl","ccomp","xcomp","parataxis"}:
                if token.head.i < token.i:
                    for j in range(token.head.i+1, token.i):
                        if doc[j].text==",": split_indices.add(j); break
            if token.pos_=="CCONJ" and token.text in {"but","yet","so","for"}:
                if i>0 and doc[i-1].text==",": split_indices.add(i-1)
        split_indices = sorted(split_indices)
        edus = []; start_idx = 0; edu_counter = 1
        for split_idx in split_indices:
            if split_idx > start_idx:
                edu_text = doc[start_idx:split_idx+1].text.strip()
                if edu_text:
                    edus.append(EDUSegment(text=edu_text, boundary_token=doc[split_idx].text,
                                           token_index=split_idx, edu_index=edu_counter))
                    edu_counter += 1
                start_idx = split_idx + 1
        if start_idx < len(doc):
            edu_text = doc[start_idx:].text.strip()
            if edu_text:
                edus.append(EDUSegment(text=edu_text, boundary_token=doc[-1].text if len(doc)>0 else "",
                                       token_index=len(doc)-1, edu_index=edu_counter))
        return edus

    def highlight_boundary_tokens(self, text, boundary_tokens):
        highlighted = text
        for token in boundary_tokens:
            if token and len(token)>0:
                pattern = r"\b" + re.escape(token) + r"\b"
                highlighted = re.sub(pattern, f'<span class="boundary-highlight">{token}</span>', highlighted)
        return highlighted


class ShallowDiscourseModule:
    CONNECTIVE_MARKERS = {
        "Temporal":    {"color":"#3182ce","bg_color":"#ebf8ff",
                        "markers":["when","while","after","before","until","till","once","since","whenever","meantime","meanwhile","as"]},
        "Contingency": {"color":"#38a169","bg_color":"#f0fff4",
                        "markers":["because","since","as","if","unless","provided","so","therefore","thus","consequently","hence","accordingly","as a result","because of","due to"]},
        "Comparison":  {"color":"#e53e3e","bg_color":"#fff5f5",
                        "markers":["although","though","but","however","nevertheless","yet","still","while","whereas","instead","rather","on the other hand","in contrast","despite","in spite of"]},
        "Expansion":   {"color":"#805ad5","bg_color":"#faf5ff",
                        "markers":["and","also","or","nor","furthermore","moreover","besides","additionally","in addition","for example","for instance","specifically","in particular","indeed","in fact","that is","i.e.","namely"]},
    }

    def __init__(self):
        self.nlp = None

    def load_spacy_model(self):
        if self.nlp is None:
            try: self.nlp = spacy.load("en_core_web_sm")
            except OSError: st.error("请先运行: python -m spacy download en_core_web_sm"); return None
        return self.nlp

    def detect_connectives(self, text):
        found_connectives = []
        text_lower = text.lower()
        for relation_type, info in self.CONNECTIVE_MARKERS.items():
            for marker in info["markers"]:
                start = 0
                while True:
                    idx = text_lower.find(marker, start)
                    if idx == -1: break
                    before = idx==0 or not text_lower[idx-1].isalpha()
                    after  = idx+len(marker)>=len(text_lower) or not text_lower[idx+len(marker)].isalpha()
                    if before and after:
                        found_connectives.append({
                            "marker": text[idx:idx+len(marker)], "marker_lower": marker,
                            "relation_type": relation_type, "start": idx, "end": idx+len(marker),
                            "color": info["color"], "bg_color": info["bg_color"],
                        })
                    start = idx + 1
        found_connectives.sort(key=lambda x: x["start"])
        return found_connectives

    def extract_arguments(self, text, connective):
        conn_start, conn_end = connective["start"], connective["end"]
        arg1_start = 0
        for i in range(conn_start-1, -1, -1):
            if text[i] in ".;": arg1_start = i+1; break
            if text[i]=="-" and i>0: arg1_start = i+1; break
        arg2_start = conn_end
        while arg2_start < len(text) and text[arg2_start].isspace(): arg2_start += 1
        if arg2_start < len(text) and text[arg2_start]==",":
            arg2_start += 1
            while arg2_start < len(text) and text[arg2_start].isspace(): arg2_start += 1
        arg2_end = len(text)
        for i in range(arg2_start, len(text)):
            if text[i] in ".;" and i>arg2_start: arg2_end = i+1; break
        return text[arg1_start:conn_start].strip(), connective["marker"], text[arg2_start:arg2_end].strip()

    def render_highlighted_text(self, text, connectives):
        if not connectives: return text
        result = []; last_end = 0
        for conn in connectives:
            result.append(text[last_end:conn["start"]])
            result.append(
                f'<span style="background-color:{conn["bg_color"]};color:{conn["color"]};'
                f'font-weight:bold;padding:2px 6px;border-radius:4px;border:2px solid {conn["color"]};">'
                f'{conn["marker"]} <small>[{conn["relation_type"].upper()}]</small></span>'
            )
            last_end = conn["end"]
        result.append(text[last_end:])
        return "".join(result)


CLUSTER_COLORS = [
    {"bg":"#FEE2E2","border":"#EF4444","text":"#991B1B"},
    {"bg":"#DBEAFE","border":"#3B82F6","text":"#1E40AF"},
    {"bg":"#D1FAE5","border":"#10B981","text":"#065F46"},
    {"bg":"#FEF3C7","border":"#F59E0B","text":"#92400E"},
    {"bg":"#E9D5FF","border":"#A855F7","text":"#6B21A8"},
    {"bg":"#FBCFE8","border":"#EC4899","text":"#9D174D"},
    {"bg":"#CFFAFE","border":"#06B6D4","text":"#155E75"},
    {"bg":"#FFEDD5","border":"#F97316","text":"#9A3412"},
]


class RuleBasedCorefModule:
    def __init__(self):
        self.nlp = None
        self.pronouns = {"he","she","it","they","him","her","them","his","its","their"}

    def load_spacy(self):
        if self.nlp is None:
            try: self.nlp = spacy.load("en_core_web_sm")
            except: return None
        return self.nlp

    def analyze(self, text):
        nlp = self.load_spacy()
        if nlp is None: return None
        doc = nlp(text)
        mentions = []
        for ent in doc.ents:
            if ent.label_ in ("PERSON","ORG","GPE","PRODUCT"):
                mentions.append({"text":ent.text,"start":ent.start_char,"end":ent.end_char,"type":ent.label_,"is_pronoun":False})
        for token in doc:
            if token.text.lower() in self.pronouns:
                mentions.append({"text":token.text,"start":token.idx,"end":token.idx+len(token.text),"type":"PRONOUN","is_pronoun":True})
        cluster_map = {}
        for m in mentions:
            key = m["text"].lower()
            if key not in cluster_map: cluster_map[key] = []
            cluster_map[key].append(m)
        clusters = []; all_mentions = []
        for idx,(key,mention_list) in enumerate(cluster_map.items()):
            if len(mention_list)>1 or not mention_list[0]["is_pronoun"]:
                color = CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]
                cluster_id = idx+1
                clusters.append({"cluster_id":cluster_id,"mentions":mention_list,"color":color,"main_entity":mention_list[0]["text"]})
                for m in mention_list:
                    m["cluster_id"] = cluster_id; m["color"] = color; all_mentions.append(m)
        if not clusters: return {"clusters":[],"highlighted_html":text}
        highlighted = self._build_highlighted_html(text, all_mentions)
        return {"clusters":clusters,"highlighted_html":highlighted}

    def _build_highlighted_html(self, text, mentions):
        if not mentions: return text
        mentions_sorted = sorted(mentions, key=lambda x: (x["start"], x["end"]))
        result_parts = []; last_end = 0
        for m in mentions_sorted:
            start, end = m["start"], m["end"]
            if start > last_end: result_parts.append(text[last_end:start])
            color = m["color"]; cluster_id = m.get("cluster_id",0); mention_text = text[start:end]
            result_parts.append(
                f'<span style="background-color:{color["bg"]};border:2px solid {color["border"]};'
                f'color:{color["text"]};padding:2px 4px;border-radius:4px;font-weight:bold;" '
                f'title="Cluster {cluster_id}">{mention_text}'
                f'<sup style="font-size:10px;margin-left:2px;">[{cluster_id}]</sup></span>'
            )
            last_end = end
        if last_end < len(text): result_parts.append(text[last_end:])
        return "".join(result_parts)


class CoreferenceResolutionModule:
    def __init__(self):
        self.model = None

    def load_model(self):
        if self.model is None:
            try:
                from fastcoref import FCoref
                self.model = FCoref(device="cpu")
                return self.model
            except ImportError:
                return None
            except Exception as e:
                st.error(f"模型加载失败: {e}")
                return None
        return self.model

    def analyze(self, text):
        model = self.load_model()
        if model is None: return None
        try:
            predictions = model.predict(texts=[text])
            if not predictions or not predictions[0].get_clusters():
                return {"clusters":[],"highlighted_html":text}
            pred = predictions[0]
            clusters_data = pred.get_clusters(as_strings=False)
            clusters = []; all_mentions = []
            for cluster_idx, cluster in enumerate(clusters_data):
                color = CLUSTER_COLORS[cluster_idx % len(CLUSTER_COLORS)]
                mentions = []
                for span in cluster:
                    if isinstance(span,(list,tuple)) and len(span)==2:
                        start, end = span; mention_text = text[start:end]
                        mentions.append({"text":mention_text,"start":start,"end":end})
                        all_mentions.append({"text":mention_text,"start":start,"end":end,"cluster_id":cluster_idx,"color":color})
                if mentions:
                    clusters.append({"cluster_id":cluster_idx+1,"mentions":mentions,"color":color,"main_entity":mentions[0]["text"]})
            if not clusters: return {"clusters":[],"highlighted_html":text}
            highlighted_html = self._build_highlighted_html(text, all_mentions)
            return {"clusters":clusters,"highlighted_html":highlighted_html}
        except Exception as e:
            st.error(f"指代消解分析失败: {e}")
            return None

    def _build_highlighted_html(self, text, mentions):
        if not mentions: return text
        mentions_sorted = sorted(mentions, key=lambda x: (x["start"], x["end"]))
        result_parts = []; last_end = 0
        for m in mentions_sorted:
            start, end, color = m["start"], m["end"], m["color"]
            cluster_id = m.get("cluster_id",0)+1
            if start > last_end: result_parts.append(text[last_end:start])
            mention_text = text[start:end]
            result_parts.append(
                f'<span style="background-color:{color["bg"]};border:2px solid {color["border"]};'
                f'color:{color["text"]};padding:2px 4px;border-radius:4px;font-weight:bold;" '
                f'title="Cluster {cluster_id}">{mention_text}'
                f'<sup style="font-size:10px;margin-left:2px;">[{cluster_id}]</sup></span>'
            )
            last_end = end
        if last_end < len(text): result_parts.append(text[last_end:])
        return "".join(result_parts)


def render():
    render_module_header(
        "m5",
        "📚 篇章分析与指代消解系统",
        "话语分割 (EDU) · 浅层篇章关系 (PDTB) · 指代消解 (Coreference)",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    with st.expander("📖 功能说明与技术栈", expanded=False):
        st.markdown("""
| 功能 | 说明 | 技术 / 库 |
|------|------|-----------|
| EDU 话语切分 | 将篇章切分为最小话语单元（Elementary Discourse Unit），支持规则基线与 NeuralEDUSeg 两种方式 | `spaCy` (en_core_web_sm)，规则启发式 |
| PDTB 篇章关系分析 | 检测句间显式连接词（"因为"、"然而"等），标注其关系类型（时序 / 条件 / 比较 / 扩展） | 内置 PDTB 连接词词典，`spaCy` |
| 指代消解 | 识别文中代词 / 名词短语与其所指实体的共指关系，以颜色聚类方式高亮标注 | `fastcoref`（规则方法回退） |
""")
        st.caption("PDTB 四大顶级关系：Temporal（时序）· Contingency（条件因果）· Comparison（比较对比）· Expansion（扩展补充）")

    tab1, tab2, tab3 = st.tabs([
        "📖 话语分割 (EDU Segmentation)",
        "🔗 浅层篇章关系 (PDTB-style)",
        "👥 指代消解 (Coreference Resolution)",
    ])

    # ── Tab 1: EDU 话语分割 ────────────────────────────────────────────────────
    with tab1:
        st.header("📖 话语分割 (Discourse Segmentation)")
        st.markdown("对比 **规则基线 (spaCy)** 与 **NeuralEDUSeg 真实标注** 的 EDU 切分结果。")
        module = EDUSegmentationModule()

        col1, col2, col3 = st.columns([2,2,2])
        with col1: dataset = st.selectbox("选择数据集", ["TRAINING","test","dev"])
        with col2: sample_id = st.number_input("样本编号 (0-99)", 0, 99, 0)
        with col3: use_custom = st.checkbox("使用自定义文本", value=False)

        filename = None
        if use_custom:
            custom_text = st.text_area("输入英文文本",
                value="Although the economy is improving, many people still struggle to find jobs. The government has implemented new policies to address this issue.",
                height=100)
            full_text = custom_text
            gold_edus = [EDUSegment(text=s.strip(), boundary_token=s.split()[-1] if s.split() else "",
                                    token_index=0, edu_index=i+1)
                         for i, s in enumerate(custom_text.split(". ")) if s.strip()]
        else:
            filename = f"wsj_{601+sample_id:04d}.out.edus"
            with st.spinner(f"正在从 GitHub 获取数据: {filename}…"):
                full_text, edu_content, gold_edu_list = module.fetch_neuraleduseg_data(dataset, filename)
                gold_edus = module.parse_gold_edus(edu_content)

        with st.expander("📄 原始文本", expanded=True):
            st.text(full_text[:2000] + ("…" if len(full_text)>2000 else ""))

        with st.spinner("正在进行依存句法分析和规则切分…"):
            baseline_edus = module.rule_based_segmentation(full_text)

        st.subheader("🔍 切分结果对比视图")
        left_col, right_col = st.columns(2)

        with left_col:
            st.markdown("""
            <div style="background:#edf2f7;padding:12px;border-radius:8px;margin-bottom:16px;">
                <h4 style="margin:0;color:#2d3748;">📐 规则基线切分</h4>
                <p style="margin:4px 0 0 0;font-size:12px;color:#718096;">基于 spaCy 依存句法分析 + 启发式规则</p>
            </div>""", unsafe_allow_html=True)
            with st.expander("📝 启发式规则说明"):
                st.markdown("**规则1**: 句尾标点 (`.`,`!`,`?`)  \n**规则2**: 从属连词 (`although`,`because`,`if`)  \n**规则3**: 依存关系从句 (advcl,ccomp,xcomp)  \n**规则4**: 并列连词 (`but`,`yet`,`so`)")
            if baseline_edus:
                for edu in baseline_edus:
                    boundary_word = edu.boundary_token if edu.boundary_token else (edu.text.split()[-1] if edu.text.split() else "")
                    highlighted_text = module.highlight_boundary_tokens(edu.text, [boundary_word])
                    st.markdown(f'<div class="edu-card"><div style="font-size:11px;opacity:.9;margin-bottom:4px;">EDU #{edu.edu_index} | 边界词: <b>{boundary_word}</b></div><div>{highlighted_text}</div></div>', unsafe_allow_html=True)
            else:
                st.info("未能生成规则基线切分")
            avg_len = sum(len(e.text.split()) for e in baseline_edus)/len(baseline_edus) if baseline_edus else 0
            st.markdown(f'<div class="m5-metric-card"><b>切分统计</b><br>EDU 数量: <b>{len(baseline_edus)}</b><br>平均长度: <b>{avg_len:.1f}</b> 词/EDU</div>', unsafe_allow_html=True)

        with right_col:
            st.markdown("""
            <div style="background:#fff5f5;padding:12px;border-radius:8px;margin-bottom:16px;">
                <h4 style="margin:0;color:#c53030;">⭐ NeuralEDUSeg 真实标注</h4>
                <p style="margin:4px 0 0 0;font-size:12px;color:#718096;">基于循环神经网络的标注数据 (Gold Standard)</p>
            </div>""", unsafe_allow_html=True)
            with st.expander("🔗 数据来源"):
                st.markdown(f"**GitHub**: [PKU-TANGENT/NeuralEDUSeg](https://github.com/PKU-TANGENT/NeuralEDUSeg)\n\n**文件**: `{filename if not use_custom else '自定义文本'}`")
            if gold_edus:
                for edu in gold_edus:
                    boundary_word = edu.boundary_token if hasattr(edu,"boundary_token") and edu.boundary_token else (edu.text.split()[-1].rstrip(".,;:!?") if edu.text.split() else "")
                    highlighted_text = module.highlight_boundary_tokens(edu.text, [boundary_word])
                    edu_idx = edu.edu_index if hasattr(edu,"edu_index") else 0
                    st.markdown(f'<div class="edu-card-gold"><div style="font-size:11px;opacity:.9;margin-bottom:4px;">EDU #{edu_idx} | 边界词: <b>{boundary_word}</b></div><div>{highlighted_text}</div></div>', unsafe_allow_html=True)
            else:
                st.info("未能获取真实标注数据")
            avg_len_gold = sum(len(e.text.split()) for e in gold_edus)/len(gold_edus) if gold_edus else 0
            st.markdown(f'<div class="m5-metric-card"><b>切分统计</b><br>EDU 数量: <b>{len(gold_edus)}</b><br>平均长度: <b>{avg_len_gold:.1f}</b> 词/EDU</div>', unsafe_allow_html=True)

        # 序列标注视角
        st.subheader("🏷️ 序列标注视角 (Token-level Boundary Detection)")
        if baseline_edus:
            col_data = []
            words = full_text.split()[:50]
            for i, word in enumerate(words):
                is_baseline_boundary = any(
                    word.rstrip(".,;:!?") == edu.boundary_token.rstrip(".,;:!?")
                    for edu in baseline_edus[:5] if hasattr(edu,"boundary_token")
                ) or word.endswith((".", "!", "?", ";"))
                is_gold_boundary = any(
                    word.rstrip(".,;:!?") in edu.text.split()[-1].rstrip(".,;:!?")
                    for edu in gold_edus[:5]
                ) if gold_edus else False
                col_data.append({"位置":i,"词":word,"基线标注":"B-EDU" if is_baseline_boundary else "I-EDU","真实标注":"B-EDU" if is_gold_boundary else "I-EDU"})
            st.dataframe(col_data, use_container_width=True, hide_index=True)

    # ── Tab 2: 浅层篇章关系 ────────────────────────────────────────────────────
    with tab2:
        st.header("🔗 浅层篇章关系提取 (PDTB-style)")
        st.markdown("基于 **显式连接词 (Explicit Connectives)** 的浅层篇章分析。")

        smodule = ShallowDiscourseModule()
        default_text = "Third-quarter sales in Europe were exceptionally strong, boosted by promotional programs and new products - although weaker foreign currencies reduced the company's earnings."
        input_text = st.text_area("输入句子进行篇章关系分析", value=default_text, height=100)

        connectives = smodule.detect_connectives(input_text)
        st.subheader("📌 显式连接词检测")

        if connectives:
            highlighted = smodule.render_highlighted_text(input_text, connectives)
            st.markdown(f'<div style="background:#f7fafc;padding:16px;border-radius:8px;border:1px solid #e2e8f0;font-size:16px;line-height:1.8;">{highlighted}</div>', unsafe_allow_html=True)
            st.markdown("**检测到的连接词:**")
            cols = st.columns(len(connectives))
            for i, conn in enumerate(connectives):
                with cols[i]:
                    st.markdown(f'<div style="background:{conn["bg_color"]};padding:12px;border-radius:8px;border:2px solid {conn["color"]};text-align:center;"><div style="font-size:18px;font-weight:bold;color:{conn["color"]};">{conn["marker"]}</div><div style="font-size:12px;color:#666;margin-top:4px;">{conn["relation_type"].upper()}</div></div>', unsafe_allow_html=True)
        else:
            st.info("⚠️ 未检测到显式连接词，尝试输入包含 although, because, when, but 等词的句子")

        st.subheader("📐 论据提取 (Argument Extraction)")
        if connectives:
            for i, conn in enumerate(connectives):
                arg1, conn_text, arg2 = smodule.extract_arguments(input_text, conn)
                st.markdown(f"**连接词 {i+1}: `{conn_text}` [{conn['relation_type'].upper()}]**")
                ac1, ac2 = st.columns(2)
                with ac1:
                    st.markdown(f'<div style="background:linear-gradient(135deg,#667eea,#764ba2);padding:16px;border-radius:8px;color:white;"><div style="font-size:12px;opacity:.9;margin-bottom:8px;">📝 Arg1 (前置论据)</div><div style="font-size:14px;line-height:1.5;">{arg1 if arg1 else "<i>(空)</i>"}</div></div>', unsafe_allow_html=True)
                with ac2:
                    st.markdown(f'<div style="background:linear-gradient(135deg,#f093fb,#f5576c);padding:16px;border-radius:8px;color:white;"><div style="font-size:12px;opacity:.9;margin-bottom:8px;">📝 Arg2 (后置论据)</div><div style="font-size:14px;line-height:1.5;">{arg2 if arg2 else "<i>(空)</i>"}</div></div>', unsafe_allow_html=True)
                st.markdown("---")

        with st.expander("📚 PDTB 连接词分类参考表"):
            for relation_type, info in ShallowDiscourseModule.CONNECTIVE_MARKERS.items():
                markers_str = ", ".join(info["markers"][:8])
                st.markdown(f'<div style="margin:8px 0;padding:12px;background:{info["bg_color"]};border-radius:6px;border-left:4px solid {info["color"]};"><span style="color:{info["color"]};font-weight:bold;">{relation_type.upper()}</span>: <code>{markers_str}</code>{"..." if len(info["markers"])>8 else ""}</div>', unsafe_allow_html=True)

    # ── Tab 3: 指代消解 ────────────────────────────────────────────────────────
    with tab3:
        st.header("👥 指代消解 (Coreference Resolution)")
        st.markdown("使用 **fastcoref** 神经网络模型（或基于规则的备选方案）进行端到端指代消解。")

        fastcoref_available = False
        try:
            import fastcoref
            fastcoref_available = True
        except Exception:
            pass

        if fastcoref_available:
            st.success("✅ fastcoref 已安装 — 将使用神经网络模型")
        else:
            st.warning("⚠️ fastcoref 未安装 — 将使用基于规则的备选方案")
            st.code("pip install fastcoref", language="bash")

        default_coref_text = """Barack Obama was born in Honolulu, Hawaii. He served as the 44th President of the United States from 2009 to 2017. Obama was the first African-American president in U.S. history. During his presidency, he signed many landmark bills into law. His signature achievement was the Affordable Care Act. After leaving office, he continued his work through the Obama Foundation."""

        input_text = st.text_area("输入包含指代关系的英文段落", value=default_coref_text, height=130)

        if st.button("🔍 开始分析", type="primary", key="coref_analyze"):
            result = None; use_rule_based = False
            if fastcoref_available:
                with st.spinner("正在加载 fastcoref 模型（首次使用需下载约362MB，请耐心等待）…"):
                    coref_module = CoreferenceResolutionModule()
                    try: result = coref_module.analyze(input_text)
                    except Exception as e:
                        st.error(f"fastcoref 运行失败: {e}")
                        use_rule_based = True
            else:
                use_rule_based = True

            if use_rule_based or result is None:
                with st.spinner("正在使用基于规则的指代分析…"):
                    rule_module = RuleBasedCorefModule()
                    result = rule_module.analyze(input_text)

            if result is None:
                st.error("分析失败"); return
            if not result["clusters"]:
                st.info("未检测到明显的指代关系"); return

            st.subheader("🎨 高亮渲染结果")
            st.markdown(f'<div style="background:#f8fafc;padding:20px;border-radius:12px;border:2px solid #e2e8f0;font-size:16px;line-height:2;">{result["highlighted_html"]}</div>', unsafe_allow_html=True)

            st.markdown("**图例:**")
            legend_cols = st.columns(min(len(result["clusters"]), 8))
            for i, cluster in enumerate(result["clusters"][:8]):
                with legend_cols[i]:
                    color = cluster["color"]
                    st.markdown(f'<div style="background:{color["bg"]};padding:8px;border-radius:6px;border:2px solid {color["border"]};text-align:center;"><span style="color:{color["text"]};font-weight:bold;font-size:12px;">簇 {cluster["cluster_id"]}</span></div>', unsafe_allow_html=True)

            st.subheader("📋 指代簇详情")
            for cluster in result["clusters"]:
                color = cluster["color"]
                mentions_str = ", ".join(f"'{m['text']}'" for m in cluster["mentions"])
                st.markdown(f'<div style="margin:12px 0;padding:16px;background:{color["bg"]};border-radius:8px;border-left:4px solid {color["border"]};"><div style="color:{color["text"]};font-weight:bold;margin-bottom:8px;">🏷️ Cluster {cluster["cluster_id"]} (主实体: {cluster["main_entity"]})</div><div style="color:#4a5568;font-family:monospace;font-size:14px;">[{mentions_str}]</div><div style="color:#718096;font-size:12px;margin-top:8px;">共 {len(cluster["mentions"])} 个表述</div></div>', unsafe_allow_html=True)

            total_mentions = sum(len(c["mentions"]) for c in result["clusters"])
            st.markdown(f"**统计信息**: 检测到 **{len(result['clusters'])}** 个指代簇，共 **{total_mentions}** 个表述（Mention）")

        with st.expander("📖 查看示例文本"):
            st.markdown("""
**示例 1 (人名-代词链)**:
```
Apple Inc. was founded by Steve Jobs. He was a visionary entrepreneur. Jobs revolutionized the tech industry with his innovative products.
```
**示例 2 (多实体交织)**:
```
Microsoft and Google are tech giants. They compete fiercely in many markets. While Microsoft focuses on cloud services, Google dominates search.
```
            """)
