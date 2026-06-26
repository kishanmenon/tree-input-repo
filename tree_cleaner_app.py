import streamlit as st
import pandas as pd
from collections import defaultdict
import io
import re
import gspread
from google.oauth2 import service_account

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shopsy Node Remover",
    page_icon="🌿",
    layout="wide",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)

st.markdown("""
<style>
    .stApp { background-color: #f7f9fc; }
    .metric-card {
        background: white; border-radius: 12px; padding: 20px 24px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08); text-align: center;
    }
    .metric-card .value { font-size: 2.2rem; font-weight: 700; color: #1a73e8; }
    .metric-card .label { font-size: 0.85rem; color: #666; margin-top: 4px; }
    .tag {
        display: inline-block; background: #e8f0fe; color: #1a73e8;
        border-radius: 6px; padding: 2px 8px; font-size: 0.78rem; margin: 2px;
    }
    .output-box {
        background: white; border: 1px solid #d0e1fd; border-radius: 10px;
        padding: 16px; font-family: monospace; font-size: 0.9rem;
        word-break: break-all; line-height: 1.8; max-height: 200px; overflow-y: auto;
    }
    .tree-box {
        background: #1e1e2e; color: #cdd6f4; border-radius: 10px;
        padding: 16px; font-family: monospace; font-size: 0.78rem;
        line-height: 1.6; overflow: auto; max-height: 500px;
        white-space: pre; border: 1px solid #313244;
    }
    .tree-id   { color: #a6e3a1; font-weight: bold; }
    .tree-na   { color: #6c7086; }
    .tree-shopsy { color: #f38ba8; }
</style>
""", unsafe_allow_html=True)


# ── Google Sheets loader ───────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…", ttl=300)
def load_csv_from_sheet() -> bytes:
    creds = service_account.Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets.readonly",
            "https://www.googleapis.com/auth/drive.readonly",
        ],
    )
    gc    = gspread.authorize(creds)
    sheet = gc.open_by_key("1KoWMOArhrPP0Y-BxqafOh5uF7eIugB0McKsiymekgxE").worksheet("TreeData")
    data  = sheet.get_all_values()
    df    = pd.DataFrame(data[1:], columns=data[0])
    return df.to_csv(index=False).encode()


# ── Tree builder (vectorized) ──────────────────────────────────────────────────
@st.cache_data(show_spinner="Building tree index…")
def build_tree(csv_bytes: bytes):
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df["pathString"] = df["node_path"].str.replace(">", "/", regex=False)
    df["node_id"]    = pd.to_numeric(df["node_id"], errors="coerce")

    id_per_path = (
        df.dropna(subset=["node_id", "pathString"])
        .groupby("pathString")["node_id"].first().astype(int).to_dict()
    )
    name_per_path = (
        df.dropna(subset=["node_name", "pathString"])
        .groupby("pathString")["node_name"].first().astype(str).to_dict()
    )

    all_paths = df["pathString"].dropna().unique()
    path_map  = {
        p: {
            "node_id": id_per_path.get(p),
            "name":    name_per_path.get(p, p.rsplit("/", 1)[-1]),
        }
        for p in all_paths
    }

    id_to_path = {v["node_id"]: k for k, v in path_map.items() if v["node_id"] is not None}

    path_set         = set(path_map)
    children_by_path = defaultdict(list)
    for p in path_map:
        idx = p.rfind("/")
        if idx > 0:
            parent = p[:idx]
            if parent in path_set:
                children_by_path[parent].append(p)

    return path_map, id_to_path, children_by_path


# ── Distillation logic ─────────────────────────────────────────────────────────
def get_desc_names(path, path_map, children_by_path, cache):
    if path in cache:
        return cache[path]
    names = [path_map[path]["name"]]
    for cp in children_by_path[path]:
        names.extend(get_desc_names(cp, path_map, children_by_path, cache))
    cache[path] = names
    return names


def distill(path, path_map, children_by_path, desc_cache, is_root=False):
    name = path_map[path]["name"]
    if not is_root and "Shopsy" in name:
        return []
    has_shopsy = any("Shopsy" in n for n in get_desc_names(path, path_map, children_by_path, desc_cache))
    if not has_shopsy:
        nid = path_map[path]["node_id"]
        return [nid] if nid else []
    result = []
    for cp in children_by_path[path]:
        result.extend(distill(cp, path_map, children_by_path, desc_cache))
    return result


def run_distillation(input_ids, path_map, id_to_path, children_by_path):
    desc_cache = {}
    final_ids, not_found, per_node = [], [], {}
    for nid in input_ids:
        if nid in id_to_path:
            result = [x for x in distill(id_to_path[nid], path_map, children_by_path, desc_cache, is_root=True) if x]
            per_node[nid] = result
            final_ids.extend(result)
        else:
            not_found.append(nid)
    seen, deduped = set(), []
    for x in final_ids:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped, not_found, per_node


# ── Visual tree (ported from Visual_Validator.R) ───────────────────────────────
def get_all_descendant_ids(path, path_map, children_by_path):
    """All node IDs in subtree including self"""
    result = []
    nid = path_map[path]["node_id"]
    if nid:
        result.append(nid)
    for cp in children_by_path[path]:
        result.extend(get_all_descendant_ids(cp, path_map, children_by_path))
    return result


def build_relevant_paths(target_ids, id_to_path, path_map, children_by_path):
    """
    For each target ID: collect its path, all descendant paths,
    and all ancestor paths needed to connect the hierarchy.
    """
    relevant = set()
    for nid in target_ids:
        if nid not in id_to_path:
            continue
        path = id_to_path[nid]
        # Descendants
        for did in get_all_descendant_ids(path, path_map, children_by_path):
            if did in id_to_path:
                dp = id_to_path[did]
                relevant.add(dp)
                # Ancestors of each descendant
                parts = dp.split("/")
                for i in range(1, len(parts)):
                    ap = "/".join(parts[:i])
                    if ap in path_map:
                        relevant.add(ap)
        relevant.add(path)
        # Ancestors of target itself
        parts = path.split("/")
        for i in range(1, len(parts)):
            ap = "/".join(parts[:i])
            if ap in path_map:
                relevant.add(ap)
    return relevant


def render_tree(path, path_map, children_by_path, relevant, highlight_ids, prefix="", is_last=True):
    """Recursively render one node + children as lines of text."""
    name = path_map[path]["name"]
    nid  = path_map[path]["node_id"]

    connector  = "└── " if is_last else "├── "
    display_id = str(nid) if (nid and nid in highlight_ids) else "NA"
    shopsy_tag = " [Shopsy]" if "Shopsy" in name else ""

    line = "{}{}{:<55} {}{}".format(prefix, connector, name, display_id, shopsy_tag)
    lines = [line]

    rel_children = sorted([cp for cp in children_by_path[path] if cp in relevant])
    child_prefix = prefix + ("    " if is_last else "│   ")
    for i, cp in enumerate(rel_children):
        lines.extend(render_tree(
            cp, path_map, children_by_path, relevant, highlight_ids,
            child_prefix, i == len(rel_children) - 1
        ))
    return lines


def generate_visual_tree_text(target_ids, id_to_path, path_map, children_by_path, highlight_ids):
    """Full tree text for a set of target IDs (mirrors R Visual_Validator logic)."""
    relevant = build_relevant_paths(target_ids, id_to_path, path_map, children_by_path)
    if not relevant:
        return "No nodes to display."

    # Find roots: paths whose parent is not in relevant
    roots = []
    for p in relevant:
        idx = p.rfind("/")
        parent = p[:idx] if idx > 0 else None
        if parent is None or parent not in relevant:
            roots.append(p)
    roots = sorted(roots)

    lines = []
    for i, rp in enumerate(roots):
        name = path_map[rp]["name"]
        nid  = path_map[rp]["node_id"]
        display_id = str(nid) if (nid and nid in highlight_ids) else "NA"
        lines.append("{:<55} {}".format(name, display_id))
        rel_children = sorted([cp for cp in children_by_path[rp] if cp in relevant])
        for j, cp in enumerate(rel_children):
            lines.extend(render_tree(
                cp, path_map, children_by_path, relevant, highlight_ids,
                "", j == len(rel_children) - 1
            ))
        if i < len(roots) - 1:
            lines.append("")
    return "\n".join(lines)


def colorize_tree_html(tree_text):
    """Wrap tree text in HTML with syntax colouring."""
    html_lines = []
    for line in tree_text.split("\n"):
        # Escape HTML
        line_esc = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if "[Shopsy]" in line_esc:
            html_lines.append('<span class="tree-shopsy">' + line_esc + '</span>')
        elif line_esc.strip().endswith("NA"):
            html_lines.append('<span class="tree-na">' + line_esc + '</span>')
        else:
            # Highlight the ID at end of line
            parts = line_esc.rsplit(" ", 1)
            if len(parts) == 2 and parts[1].strip().isdigit():
                html_lines.append(parts[0] + ' <span class="tree-id">' + parts[1] + '</span>')
            else:
                html_lines.append(line_esc)
    return "\n".join(html_lines)


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🌿 Shopsy Node Remover")

tree_ready = False
try:
    csv_bytes = load_csv_from_sheet()
    path_map, id_to_path, children_by_path = build_tree(csv_bytes)
    col_info, col_btn = st.columns([5, 1])
    with col_info:
        st.success("✅ Ready")
    with col_btn:
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()
    tree_ready = True
except KeyError as e:
    st.error("Missing secret: " + str(e) + ". Add [gcp_service_account] in Streamlit Secrets.")
except Exception as e:
    st.error("Could not load data: " + str(e))

if tree_ready:
    st.markdown("---")
    st.markdown("### Enter input node IDs")

    input_mode = st.radio("Input method",
        ["Paste comma-separated IDs", "Upload a CSV file"], horizontal=True)

    raw_ids = []

    if input_mode == "Paste comma-separated IDs":
        text_input = st.text_area("Node IDs — comma or newline separated", height=110,
            placeholder="e.g.  22897, 22232, 23973, 21274")
        if text_input.strip():
            raw_ids = [int(x) for x in re.split(r"[\s,]+", text_input.strip()) if x.strip().isdigit()]
    else:
        id_file = st.file_uploader("CSV with a column named 'node_id'", type=["csv"])
        if id_file:
            try:
                id_df = pd.read_csv(id_file)
                if "node_id" not in id_df.columns:
                    st.error("CSV must have a 'node_id' column.")
                else:
                    raw_ids = id_df["node_id"].dropna().astype(int).tolist()
                    st.info("Loaded **" + str(len(raw_ids)) + "** IDs from file.")
            except Exception as e:
                st.error("Could not read file: " + str(e))

    if raw_ids:
        preview = " ".join(["<span class='tag'>" + str(i) + "</span>" for i in raw_ids[:40]])
        if len(raw_ids) > 40:
            preview += "<span class='tag'>+" + str(len(raw_ids) - 40) + " more</span>"
        st.markdown("<b>" + str(len(raw_ids)) + " IDs queued</b><br>" + preview, unsafe_allow_html=True)
        st.markdown("")

    if st.button("🚀 Run Distillation", disabled=not raw_ids, type="primary"):
        with st.spinner("Distilling…"):
            final_ids, not_found, per_node = run_distillation(raw_ids, path_map, id_to_path, children_by_path)

        st.markdown("---")
        st.markdown("### Results")

        c1, c2, c3, c4 = st.columns(4)
        reduction = round((1 - len(final_ids) / max(len(raw_ids), 1)) * 100, 1)
        for col, val, label in [
            (c1, len(raw_ids),         "Input IDs"),
            (c2, len(final_ids),       "Clean output IDs"),
            (c3, len(not_found),       "Not found in tree"),
            (c4, str(abs(reduction)) + "%", "↓ Reduction" if reduction >= 0 else "↑ Increase"),
        ]:
            with col:
                st.markdown(
                    "<div class='metric-card'><div class='value'>" + str(val) +
                    "</div><div class='label'>" + label + "</div></div>",
                    unsafe_allow_html=True
                )

        st.markdown("")

        # ── Output IDs (with not-found appended) ──────────────────────────────
        csv_string = ", ".join(str(x) for x in final_ids)
        if not_found:
            not_found_str = ", ".join(str(x) for x in not_found)
            display_string = csv_string + "\n\nTree not found: " + not_found_str
        else:
            display_string = csv_string

        st.markdown("#### Output node IDs")
        html_display = display_string.replace("\n", "<br>")
        st.markdown("<div class='output-box'>" + html_display + "</div>", unsafe_allow_html=True)
        st.code(display_string, language=None)

        st.markdown("")

        # ── Visual Tree Validator ──────────────────────────────────────────────
        st.markdown("### 🌳 Visual Tree Validator")
        st.caption("Green = node ID present · Grey = structural parent (NA) · Red = Shopsy node")

        with st.spinner("Generating trees…"):
            input_set  = set(raw_ids)
            output_set = set(final_ids)
            input_tree_text  = generate_visual_tree_text(raw_ids,   id_to_path, path_map, children_by_path, input_set)
            output_tree_text = generate_visual_tree_text(final_ids, id_to_path, path_map, children_by_path, output_set)

        vcol1, vcol2 = st.columns(2)
        with vcol1:
            st.markdown("**📥 Input Tree** — " + str(len(raw_ids)) + " nodes")
            st.markdown(
                "<div class='tree-box'>" + colorize_tree_html(input_tree_text) + "</div>",
                unsafe_allow_html=True
            )
            st.download_button("⬇️ Download input tree .txt",
                data=input_tree_text.encode(), file_name="visual_input_tree.txt",
                mime="text/plain")

        with vcol2:
            st.markdown("**📤 Output Tree** — " + str(len(final_ids)) + " nodes")
            st.markdown(
                "<div class='tree-box'>" + colorize_tree_html(output_tree_text) + "</div>",
                unsafe_allow_html=True
            )
            st.download_button("⬇️ Download output tree .txt",
                data=output_tree_text.encode(), file_name="visual_output_tree.txt",
                mime="text/plain")

        st.markdown("")

        # ── Detail tabs ───────────────────────────────────────────────────────
        tab1, tab2, tab3 = st.tabs(["📋 Full table", "🔍 Per-node breakdown", "⚠️ Not found"])

        with tab1:
            out_df = pd.DataFrame({"node_id": final_ids})
            out_df["node_name"] = out_df["node_id"].map(
                lambda x: path_map[id_to_path[x]]["name"] if x in id_to_path else "")
            out_df["node_path"] = out_df["node_id"].map(
                lambda x: id_to_path[x].replace("/", " > ") if x in id_to_path else "")
            st.dataframe(out_df, use_container_width=True, height=360)
            st.download_button("⬇️ Download distilled_clean_node_ids.csv",
                data=out_df.to_csv(index=False).encode(),
                file_name="distilled_clean_node_ids.csv", mime="text/csv", type="primary")

        with tab2:
            rows = []
            for inp_id, out_ids in per_node.items():
                inp_name = path_map[id_to_path[inp_id]]["name"] if inp_id in id_to_path else "?"
                rows.append({
                    "input_node_id":   inp_id,
                    "input_node_name": inp_name,
                    "resolved_ids":    ", ".join(str(x) for x in out_ids),
                    "count":           len(out_ids),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=360)

        with tab3:
            if not_found:
                st.warning(str(len(not_found)) + " ID(s) were not in the tree and were skipped.")
                st.dataframe(pd.DataFrame({"node_id": not_found}), use_container_width=True)
            else:
                st.success("All input IDs were found in the tree. ✅")
