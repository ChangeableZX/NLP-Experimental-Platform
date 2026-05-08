"""模块7：信息抽取与知识图谱构建"""

import json
import re
from dataclasses import dataclass
from typing import List, Optional

import spacy
import streamlit as st
import streamlit.components.v1 as components

from .common import render_module_header, sec_header


# ══════════════════════════════════════════════════════════════
# 模块一：NER Engine
# ══════════════════════════════════════════════════════════════

@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int


_nlp_models: dict = {"en": None, "zh": None}


def _load_model(lang: str) -> spacy.Language:
    global _nlp_models
    if _nlp_models[lang] is None:
        if lang == "en":
            _nlp_models[lang] = spacy.load("en_core_web_sm")
        elif lang == "zh":
            _nlp_models[lang] = spacy.load("zh_core_web_sm")
        else:
            raise ValueError(f"不支持的语言: {lang}")
    return _nlp_models[lang]


def _detect_language(text: str) -> str:
    if not text: return "en"
    chinese_chars = re.findall(r"[一-鿿]", text)
    return "zh" if len(chinese_chars)/len(text) > 0.15 else "en"


SPACY_LABEL_MAP = {
    "PERSON":"PER","PER":"PER","ORG":"ORG","GPE":"LOC","LOC":"LOC",
    "FAC":"MISC","PRODUCT":"MISC","EVENT":"MISC","WORK_OF_ART":"MISC",
    "LAW":"MISC","LANGUAGE":"MISC","NORP":"MISC",
}


def _map_label(spacy_label: str) -> str:
    return SPACY_LABEL_MAP.get(spacy_label, "MISC")


def _extract_with_model(text: str, lang: str) -> List[Entity]:
    if not text.strip(): return []
    try:
        nlp = _load_model(lang)
        doc = nlp(text)
        return [Entity(text=ent.text, label=_map_label(ent.label_),
                       start=ent.start_char, end=ent.end_char) for ent in doc.ents]
    except Exception:
        return []


def _merge_entities(entities: List[Entity]) -> List[Entity]:
    if not entities: return []
    sorted_ents = sorted(entities, key=lambda e: (e.start, -(e.end-e.start)))
    merged: List[Entity] = []
    for ent in sorted_ents:
        exact_dup = False
        for sel in merged:
            if ent.start == sel.start and ent.end == sel.end:
                exact_dup = True
                priority = {"PER":3,"ORG":3,"LOC":3,"MISC":1}
                if priority.get(ent.label,0) > priority.get(sel.label,0):
                    merged[merged.index(sel)] = ent
                break
        if exact_dup: continue
        overlap = any(not(ent.end<=sel.start or ent.start>=sel.end) for sel in merged)
        if not overlap: merged.append(ent)
    merged.sort(key=lambda x: x.start)
    return merged


def extract_entities(text: str) -> List[Entity]:
    if not text or not text.strip(): return []
    en_entities = _extract_with_model(text, "en")
    zh_entities = _extract_with_model(text, "zh")
    return _merge_entities(en_entities + zh_entities)


def generate_bio_sequence(text: str, entities: List[Entity]) -> List[str]:
    if not text: return []
    bio = ["O"] * len(text)
    for ent in entities:
        if ent.start >= len(text) or ent.end > len(text): continue
        for idx in range(ent.start, ent.end):
            bio[idx] = f"B-{ent.label}" if idx == ent.start else f"I-{ent.label}"
    return bio


def bio_to_display(text: str, bio: List[str]) -> str:
    if not text: return ""
    lines = []
    for i, char in enumerate(text):
        tag = bio[i] if i < len(bio) else "O"
        ch = "\\n" if char == chr(10) else char
        lines.append(f"{ch}/{tag}")
    return " ".join(lines)


ENTITY_COLORS = {"PER":"#FF6B6B","ORG":"#4ECDC4","LOC":"#45B7D1","MISC":"#96CEB4"}


def get_entity_color(label: str) -> str:
    return ENTITY_COLORS.get(label, "#FFD93D")


# ══════════════════════════════════════════════════════════════
# 模块二：Relation Engine
# ══════════════════════════════════════════════════════════════

@dataclass
class Relation:
    source: str; target: str; relation: str
    source_start: int; source_end: int
    target_start: int; target_end: int
    sentence: str


EN_RELATION_MAP = {
    "found":"FOUNDED","create":"FOUNDED","establish":"FOUNDED","founder":"FOUNDED",
    "work":"WORKS_AT","join":"WORKS_AT","live":"LIVES_IN","reside":"LIVES_IN",
    "acquire":"ACQUIRED","buy":"ACQUIRED","purchase":"ACQUIRED","own":"OWNS","possess":"OWNS",
    "lead":"LEADS","head":"LEADS","manage":"LEADS","invest":"INVESTED_IN",
    "cooperate":"COOPERATES_WITH","partner":"COOPERATES_WITH",
    "persuade":"PERSUADED","take":"TOOK_OVER","spend":"WORKED_AT",
}

ZH_RELATION_MAP = {
    "创立":"创始人","创建":"创始人","成立":"创始人","创办":"创始人",
    "工作":"工作于","任职":"工作于","加入":"工作于",
    "居住":"居住于","住在":"居住于","收购":"收购","购买":"收购",
    "拥有":"拥有","领导":"领导","管理":"领导","投资":"投资于","合作":"合作于",
}


def _normalize_relation(lemma: str, lang: str) -> str:
    lemma = lemma.strip().lower() if lang == "en" else lemma.strip()
    return EN_RELATION_MAP.get(lemma, lemma.upper().replace(" ","_")) if lang=="en" else ZH_RELATION_MAP.get(lemma, "关联")


def _token_to_entity(token, entities: List[Entity]) -> Optional[Entity]:
    for ent in entities:
        if ent.start <= token.idx < ent.end: return ent
    return None


def _extract_relations_with_model(text: str, entities: List[Entity], lang: str) -> List[Relation]:
    if not text.strip() or len(entities) < 2: return []
    try: nlp = _load_model(lang)
    except Exception: return []
    doc = nlp(text)
    relations: List[Relation] = []

    def _add_relation(subj, obj, rel_label, sentence):
        if subj.text == obj.text: return
        if not any(r.source==subj.text and r.target==obj.text and r.sentence==sentence for r in relations):
            relations.append(Relation(source=subj.text,target=obj.text,relation=rel_label,
                                      source_start=subj.start,source_end=subj.end,
                                      target_start=obj.start,target_end=obj.end,sentence=sentence))

    for sent in doc.sents:
        sent_ents = [e for e in entities if e.start>=sent.start_char and e.end<=sent.end_char]
        if len(sent_ents) < 2: continue
        verbs = [t for t in sent if t.pos_ in ("VERB","AUX")]
        for verb in verbs:
            logical_subjs, logical_objs = [], []
            is_passive = any(c.dep_ in ("nsubjpass","auxpass") for c in verb.children)

            def find_subjects(t):
                subjs = [c for c in t.children if c.dep_ in ("nsubj","nsubjpass")]
                if subjs: return subjs
                if t.dep_ == "conj": return find_subjects(t.head)
                if t.dep_ in ("xcomp","advcl"):
                    head_objs = [c for c in t.head.children if c.dep_ in ("dobj","obj")]
                    return head_objs if head_objs else find_subjects(t.head)
                return []

            raw_subjs = find_subjects(verb)
            raw_objs = [c for c in verb.children if c.dep_ in ("dobj","obj","attr","pobj")]
            for child in verb.children:
                if child.dep_ in ("prep","agent"):
                    raw_objs.extend([c for c in child.children if c.dep_ in ("pobj","dobj")])
                    if child.dep_ == "agent": is_passive = True

            if is_passive:
                logical_objs.extend(raw_subjs)
                agents = [c for child in verb.children if child.dep_=="agent" for c in child.children if c.dep_=="pobj"]
                logical_subjs.extend(agents if agents else raw_objs)
            else:
                logical_subjs.extend(raw_subjs); logical_objs.extend(raw_objs)

            rel_label = _normalize_relation(verb.lemma_, lang)
            for subj_token in logical_subjs:
                subj_ent = _token_to_entity(subj_token, sent_ents)
                if not subj_ent: continue
                for obj_token in logical_objs:
                    obj_ent = _token_to_entity(obj_token, sent_ents)
                    if not obj_ent: continue
                    _add_relation(subj_ent, obj_ent, rel_label, sent.text)
    return relations


def extract_relations(text: str, entities: List[Entity]) -> List[Relation]:
    en_rels = _extract_relations_with_model(text, entities, "en")
    zh_rels = _extract_relations_with_model(text, entities, "zh")
    seen = set(); unique_rels = []
    for rel in en_rels + zh_rels:
        key = (rel.source, rel.target, rel.relation, rel.sentence.strip())
        if key not in seen: seen.add(key); unique_rels.append(rel)
    return unique_rels


# ══════════════════════════════════════════════════════════════
# 模块三：KG Viz（知识图谱可视化）
# ══════════════════════════════════════════════════════════════

GROUP_STYLES = {
    "PER": {"shape":"dot","size":28,"font":{"size":16,"color":"#1f2937","face":"arial"},"borderWidth":3,"shadow":{"enabled":True,"color":"rgba(0,0,0,0.1)","size":10}},
    "ORG": {"shape":"box","size":22,"font":{"size":15,"color":"#1f2937","face":"arial"},"borderWidth":2,"shadow":{"enabled":True,"color":"rgba(0,0,0,0.1)","size":8},"margin":10,"borderRadius":6},
    "LOC": {"shape":"triangle","size":24,"font":{"size":15,"color":"#1f2937","face":"arial"},"borderWidth":2,"shadow":{"enabled":True,"color":"rgba(0,0,0,0.1)","size":8}},
    "MISC":{"shape":"diamond","size":22,"font":{"size":14,"color":"#1f2937","face":"arial"},"borderWidth":2,"shadow":{"enabled":True,"color":"rgba(0,0,0,0.1)","size":8}},
}


def _darken(hex_color: str, factor: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2],16), int(hex_color[2:4],16), int(hex_color[4:6],16)
    r, g, b = [min(255,max(0,int(c*factor))) for c in [r,g,b]]
    return f"#{r:02x}{g:02x}{b:02x}"


def _build_nodes(entities: List[Entity]):
    seen_texts, nodes, text_to_id = set(), [], {}
    for idx, ent in enumerate(entities):
        if ent.text in seen_texts: continue
        seen_texts.add(ent.text)
        node_id = f"ent_{idx}"; text_to_id[ent.text] = node_id
        color = get_entity_color(ent.label)
        group_style = GROUP_STYLES.get(ent.label, GROUP_STYLES["MISC"])
        node = {"id":node_id,"label":ent.text,"group":ent.label,
                "title":f"<b>{ent.text}</b><br>类别: {ent.label}",
                "color":{"background":color,"border":_darken(color,0.8),
                         "highlight":{"background":_darken(color,1.2),"border":_darken(color,0.6)},
                         "hover":{"background":_darken(color,1.1),"border":_darken(color,0.6)}}}
        node.update(group_style); nodes.append(node)
    return nodes, text_to_id


def _build_edges(relations: List[Relation], text_to_id: dict) -> list:
    edges, seen_pairs = [], set()
    for rel in relations:
        src_id = text_to_id.get(rel.source); tgt_id = text_to_id.get(rel.target)
        if not src_id or not tgt_id: continue
        pair_key = (src_id, tgt_id, rel.relation)
        if pair_key in seen_pairs: continue
        seen_pairs.add(pair_key)
        edges.append({"from":src_id,"to":tgt_id,"label":rel.relation,
                      "title":f"{rel.source} <b>{rel.relation}</b> {rel.target}",
                      "arrows":{"to":{"enabled":True,"scaleFactor":0.8,"type":"arrow"}},
                      "color":{"color":"#9ca3af","highlight":"#4b5563","hover":"#4b5563"},
                      "font":{"size":13,"color":"#374151","background":"rgba(255,255,255,0.85)","strokeWidth":0},
                      "smooth":{"type":"continuous","roundness":0.2},"width":2})
    return edges


def generate_visnetwork_html(entities: List[Entity], relations: List[Relation], height: int = 550) -> str:
    nodes, text_to_id = _build_nodes(entities)
    edges = _build_edges(relations, text_to_id)
    if not nodes:
        return f'<div style="display:flex;align-items:center;justify-content:center;height:{height}px;background:#f9fafb;border-radius:12px;color:#6b7280;font-family:arial;"><p>暂无实体数据，请先进行文本分析</p></div>'
    nodes_json = json.dumps(nodes, ensure_ascii=False)
    edges_json = json.dumps(edges, ensure_ascii=False)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>body,html{{margin:0;padding:0;width:100%;height:100%;overflow:hidden;}}
  #kg-network{{width:100%;height:{height}px;background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;}}
  .kg-legend{{position:absolute;top:12px;right:12px;background:rgba(255,255,255,.95);border:1px solid #e5e7eb;border-radius:8px;padding:8px 12px;font-family:Arial,sans-serif;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.08);z-index:10;}}
  .kg-legend-item{{display:flex;align-items:center;gap:6px;margin:4px 0;}}
  .kg-legend-dot{{width:10px;height:10px;border-radius:50%;display:inline-block;}}</style></head>
  <body><div id="kg-network"></div>
  <div class="kg-legend"><div style="font-weight:600;margin-bottom:4px;">图例</div>
  <div class="kg-legend-item"><span class="kg-legend-dot" style="background:#FF6B6B;"></span>人物(PER)</div>
  <div class="kg-legend-item"><span class="kg-legend-dot" style="background:#4ECDC4;"></span>组织(ORG)</div>
  <div class="kg-legend-item"><span class="kg-legend-dot" style="background:#45B7D1;"></span>地点(LOC)</div>
  <div class="kg-legend-item"><span class="kg-legend-dot" style="background:#96CEB4;"></span>其他(MISC)</div></div>
  <script>
  var nodes=new vis.DataSet({nodes_json});var edges=new vis.DataSet({edges_json});
  var container=document.getElementById('kg-network');
  var network=new vis.Network(container,{{nodes:nodes,edges:edges}},{{
    nodes:{{borderWidthSelected:4}},edges:{{selectionWidth:3,hoverWidth:1.5}},
    physics:{{enabled:true,forceAtlas2Based:{{gravitationalConstant:-60,centralGravity:.005,springLength:120,springConstant:.18,damping:.4,avoidOverlap:.5}},maxVelocity:50,minVelocity:.1,solver:'forceAtlas2Based',stabilization:{{enabled:true,iterations:1200,updateInterval:25}}}},
    interaction:{{hover:true,tooltipDelay:150,hideEdgesOnDrag:false,zoomView:true,dragView:true,dragNodes:true,navigationButtons:true,keyboard:true}},
    layout:{{randomSeed:42}}
  }});
  window.addEventListener('resize',function(){{network.fit();}});
  network.once("stabilizationIterationsDone",function(){{network.fit({{animation:{{duration:800,easingFunction:"easeInOutQuad"}}}});}});
  </script></body></html>"""


def build_highlighted_html(text: str, entities: List[Entity]) -> str:
    if not entities: return text.replace("\n", "<br>")
    html_parts = []; last_end = 0
    for ent in sorted(entities, key=lambda e: e.start):
        before = text[last_end:ent.start]
        if before: html_parts.append(before.replace("\n","<br>"))
        color = get_entity_color(ent.label)
        html_parts.append(f'<span class="entity-badge" style="background-color:{color};" title="{ent.label}">{ent.text}</span>')
        last_end = ent.end
    after = text[last_end:]
    if after: html_parts.append(after.replace("\n","<br>"))
    return "".join(html_parts)


_CSS = """<style>
.main-header{font-size:2.2rem;font-weight:700;color:#1f2937;margin-bottom:.5rem;}
.sub-header{font-size:1rem;color:#6b7280;margin-bottom:2rem;}
.highlight-box{background-color:#f9fafb;border:1px solid #e5e7eb;border-radius:12px;
               padding:1.5rem;font-size:1.15rem;line-height:2;min-height:120px;word-wrap:break-word;}
.bio-box{background-color:#1f2937;color:#e5e7eb;border-radius:12px;padding:1.5rem;
         font-family:'Courier New',monospace;font-size:1rem;line-height:2;
         min-height:120px;word-wrap:break-word;white-space:pre-wrap;}
.entity-badge{display:inline-block;padding:2px 10px;border-radius:6px;font-weight:600;
              color:#ffffff;text-shadow:0 1px 2px rgba(0,0,0,.2);box-shadow:0 2px 4px rgba(0,0,0,.1);}
.legend-item{display:inline-flex;align-items:center;gap:6px;margin-right:16px;margin-bottom:8px;font-size:.9rem;}
.legend-dot{width:14px;height:14px;border-radius:4px;display:inline-block;}
.stat-card{background:#ffffff;border:1px solid #e5e7eb;border-radius:10px;padding:1rem;text-align:center;}
.stat-number{font-size:1.8rem;font-weight:700;color:#111827;}
.stat-label{font-size:.85rem;color:#6b7280;margin-top:4px;}
</style>"""

LABELS_INFO = {"PER":"人物 (Person)","ORG":"组织 (Organization)","LOC":"地点 (Location)","MISC":"其他 (Miscellaneous)"}


def render():
    render_module_header(
        "m7",
        "🧠 信息抽取与知识图谱构建",
        "命名实体识别 (NER) · 关系抽取 (RE) · 知识图谱可视化 · BIO 序列标注",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    with st.expander("📖 功能说明与技术栈", expanded=False):
        st.markdown("""
| 功能 | 说明 | 技术 / 库 |
|------|------|-----------|
| 命名实体识别 (NER) | 识别文本中的人名 (PER)、地名 (LOC)、机构名 (ORG) 等实体类别，支持中英双语切换 | `spaCy` (en_core_web_sm / zh_core_web_sm) |
| 实体高亮展示 | 在原文中对不同类别实体分色标注，直观展示识别结果；可切换为 BIO 序列标注视图 | `spaCy`，自定义 HTML 渲染 |
| 关系抽取 | 基于依存句法分析从文本中提取（主体, 关系谓词, 客体）三元组，支持被动句处理 | `spaCy` 依存分析，规则模板 |
| 知识图谱可视化 | 将提取的实体-关系三元组渲染为可交互的网络拓扑图，支持拖拽、缩放 | `vis-network.js`（浏览器端渲染） |
""")

    # ── 侧边栏附加控件 ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("#### ⚙️ NER 显示设置")
        show_bio = st.checkbox("🔍 BIO 序列标注模式", value=False,
                               help="开启后展示 BIO 序列标注，而非彩色高亮文本")
        st.markdown("#### 📊 实体类别图例")
        for label, desc in LABELS_INFO.items():
            color = get_entity_color(label)
            st.markdown(f'<div class="legend-item"><span class="legend-dot" style="background-color:{color};"></span><span>{desc}</span></div>', unsafe_allow_html=True)
        st.markdown('<small style="color:#9ca3af">基于 spaCy 中英文双模型<br>支持 NER + RE + KG</small>', unsafe_allow_html=True)

    # ── 主内容 ────────────────────────────────────────────────────────────────
    col_input, col_btn = st.columns([6, 1])
    with col_input:
        default_text = "Steve Jobs founded Apple. Bill Gates established Microsoft in Albuquerque. Elon Musk leads Tesla and SpaceX."
        user_text = st.text_area("📝 请输入待分析的文本（支持英文或中文）：",
                                 value=default_text, height=140,
                                 placeholder="在此粘贴语料…", label_visibility="collapsed")
    with col_btn:
        st.write(""); st.write("")
        analyze_clicked = st.button("🔎 开始识别", use_container_width=True, type="primary")

    if analyze_clicked or (user_text and user_text != default_text):
        entities    = extract_entities(user_text)
        bio_sequence = generate_bio_sequence(user_text, entities)
        relations   = extract_relations(user_text, entities)

        total_entities  = len(entities); total_relations = len(relations)
        label_counts    = {}
        for ent in entities: label_counts[ent.label] = label_counts.get(ent.label,0)+1

        st.markdown("---")
        stat_cols = st.columns(6)
        for col, (label, value) in zip(stat_cols, [
            ("📄 总字符", len(user_text)), ("🏷️ 实体总数", total_entities),
            ("🔗 关系总数", total_relations), ("👤 人物", label_counts.get("PER",0)),
            ("🏢 组织", label_counts.get("ORG",0)), ("📍 地点", label_counts.get("LOC",0)),
        ]):
            col.markdown(f'<div class="stat-card"><div class="stat-number">{value}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        if show_bio:
            st.markdown("#### 🏷️ BIO 序列标注结果")
            bio_display = bio_to_display(user_text, bio_sequence)
            st.markdown(f'<div class="bio-box">{bio_display}</div>', unsafe_allow_html=True)
            with st.expander("📋 查看原始 BIO 标签数组"):
                def _ch_repr(c): return "\\n" if c == chr(10) else c
                bio_lines = [f"[{i:03d}] '{_ch_repr(ch)}' -> {tag}"
                             for i,(ch,tag) in enumerate(zip(user_text,bio_sequence))]
                st.code("\n".join(bio_lines), language="text")
        else:
            st.markdown("#### ✨ 实体高亮展示")
            highlighted = build_highlighted_html(user_text, entities)
            st.markdown(f'<div class="highlight-box">{highlighted}</div>', unsafe_allow_html=True)
            if entities:
                st.markdown("#### 📋 识别实体详情")
                st.dataframe([{"实体文本":e.text,"类别":e.label,"类别说明":LABELS_INFO.get(e.label,e.label),"起始":e.start,"结束":e.end} for e in entities],
                             use_container_width=True, hide_index=True)
            else:
                st.info("未检测到命名实体。请尝试输入包含人物、组织、地点等实体的文本。")

        st.markdown("---")
        st.markdown("#### 🔗 关系抽取结果（三元组）")
        if relations:
            st.dataframe([{"主体":r.source,"主体类别":LABELS_INFO.get(next((e.label for e in entities if e.text==r.source),"MISC"),"MISC"),"关系":r.relation,"客体":r.target,"客体类别":LABELS_INFO.get(next((e.label for e in entities if e.text==r.target),"MISC"),"MISC"),"所在句子":r.sentence} for r in relations],
                         use_container_width=True, hide_index=True)
        else:
            st.info("未检测到明确的实体间关系。建议输入包含主谓宾结构的完整句子。")

        st.markdown("---")
        st.markdown("#### 🕸️ 知识图谱可视化")
        if entities:
            kg_html = generate_visnetwork_html(entities, relations, height=550)
            components.html(kg_html, height=560, scrolling=False)
            st.caption("💡 提示：可拖拽节点调整布局，滚轮缩放，悬停查看详情，右上角有导航按钮。")
        else:
            st.info("暂无知识图谱数据，请先输入文本并识别实体。")

        with st.expander("🛠️ 开发者调试信息"):
            st.json({"input_length":len(user_text),"entity_count":total_entities,"relation_count":total_relations,
                     "entities":[{"text":e.text,"label":e.label,"start":e.start,"end":e.end} for e in entities],
                     "relations":[{"source":r.source,"target":r.target,"relation":r.relation} for r in relations],
                     "bio_length":len(bio_sequence)})
    else:
        st.info("👆 请在上方输入文本并点击「开始识别」按钮。")
