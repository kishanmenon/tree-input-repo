import streamlit as st
import pandas as pd
from collections import defaultdict
import io
import re
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tree Purity Distiller",
    page_icon="🌿",
    layout="wide",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #f7f9fc; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        text-align: center;
    }
    .metric-card .value { font-size: 2.2rem; font-weight: 700; color: #1a73e8; }
    .metric-card .label { font-size: 0.85rem; color: #666; margin-top: 4px; }
    .tag {
        display: inline-block;
        background: #e8f0fe;
        color: #1a73e8;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.78rem;
        margin: 2px;
    }
    .section-header {
        font-size: 1rem; font-weight: 600; color: #333;
        margin-bottom: 8px; margin-top: 16px;
    }
    .output-box {
        background: white;
        border: 1px solid #d0e1fd;
        border-radius: 10px;
        padding: 16px;
        font-family: monospace;
        font-size: 0.9rem;
        word-break: break-all;
        line-height: 1.8;
        max-height: 200px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)


# ── Google Drive loader ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching latest tree.csv from Google Drive…", ttl=300)
def load_csv_from_drive():
    """
    Reads service account credentials from Streamlit secrets,
    searches the configured folder for tree.csv, and returns its bytes.
    """
    creds_dict = dict(st.secrets["gcp_service_account"])
    folder_id  = st.secrets["DRIVE_FOLDER_ID"]

    creds   = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    service = build("drive", "v3", credentials=creds, cache_discovery=False)

    # Find tree.csv in the folder
    query   = f"name='tree.csv' and '{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files   = results.get("files", [])

    if not files:
        raise FileNotFoundError(
            "tree.csv not found in the configured Google Drive folder. "
            "Make sure the service account has access to the shared folder."
        )

    file_id = files[0]["id"]
    request = service.files().get_media(fileId=file_id)
    buf     = io.BytesIO()
    dl      = MediaIoBaseDownload(buf, request)
    done    = False
    while not done:
        _, done = dl.next_chunk()

    buf.seek(0)
    return buf.read()


# ── Tree builder ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Building tree index…")
def build_tree(csv_bytes):
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df["pathString"] = df["node_path"].str.replace(">", "/", regex=False)

    path_map = {}
    for _, row in df.iterrows():
        p    = row["pathString"]
        nid  = int(row["node_id"]) if pd.notna(row["node_id"]) else None
        name = str(row["node_name"]) if pd.notna(row["node_name"]) else p.split("/")[-1]
        if p not in path_map:
            path_map[p] = {"node_id": nid, "name": name}
        elif nid is not None and path_map[p]["node_id"] is None:
            path_map[p]["node_id"] = nid

    id_to_path = {v["node_id"]: k for k, v in path_map.items() if v["node_id"] is not None}

    children_by_path = defaultdict(list)
    for p in sorted(path_map.keys(), key=lambda x: x.count("/")):
        parts = p.split("/")
        if len(parts) > 1:
            parent = "/".join(parts[:-1])
            if parent in path_map:
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


# ── App UI ─────────────────────────────────────────────────────────────────────
st.title("🌿 Tree Purity Distiller")
st.caption("Fetches tree.csv from Google Drive · strips Shopsy nodes · returns minimal clean IDs")

# ── Load tree from Drive ───────────────────────────────────────────────────────
try:
    csv_bytes = load_csv_from_drive()
    path_map, id_to_path, children_by_path = build_tree(csv_bytes)
    col_info, col_btn = st.columns([5, 1])
    with col_info:
        st.success(f"✅ Tree loaded from Drive — **{len(id_to_path):,}** nodes (cached 5 min)")
    with col_btn:
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()
    tree_ready = True
except FileNotFoundError as e:
    st.error(str(e))
    tree_ready = False
except KeyError:
    st.error(
        "Streamlit secrets not configured. "
        "Add `[gcp_service_account]` and `DRIVE_FOLDER_ID` in your app's Secrets settings."
    )
    tree_ready = False
except Exception as e:
    st.error(f"Could not load tree from Drive: {e}")
    tree_ready = False

if tree_ready:
    st.markdown("---")

    # ── Input ──────────────────────────────────────────────────────────────────
    st.markdown("### Enter input node IDs")

    input_mode = st.radio(
        "Input method",
        ["Paste comma-separated IDs", "Upload a CSV file"],
        horizontal=True,
    )

    raw_ids = []

    if input_mode == "Paste comma-separated IDs":
        text_input = st.text_area(
            "Node IDs — comma or newline separated",
            height=110,
            placeholder="e.g.  22897, 22232, 23973, 21274",
        )
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
                    st.info(f"Loaded **{len(raw_ids)}** IDs from file.")
            except Exception as e:
                st.error(f"Could not read file: {e}")

    if raw_ids:
        preview = " ".join([f"<span class='tag'>{i}</span>" for i in raw_ids[:40]])
        if len(raw_ids) > 40:
            preview += f"<span class='tag'>+{len(raw_ids)-40} more</span>"
        st.markdown(f"<div class='section-header'>{len(raw_ids)} IDs queued</div>{preview}", unsafe_allow_html=True)
        st.markdown("")

    # ── Run ────────────────────────────────────────────────────────────────────
    if st.button("🚀 Run Distillation", disabled=not raw_ids, type="primary"):
        with st.spinner("Distilling…"):
            final_ids, not_found, per_node = run_distillation(raw_ids, path_map, id_to_path, children_by_path)

        st.markdown("---")
        st.markdown("### Results")

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        reduction = round((1 - len(final_ids) / max(len(raw_ids), 1)) * 100, 1)
        for col, val, label in [
            (c1, len(raw_ids),   "Input IDs"),
            (c2, len(final_ids), "Clean output IDs"),
            (c3, len(not_found), "Not found in tree"),
            (c4, f"{abs(reduction)}%", "↓ Reduction" if reduction >= 0 else "↑ Increase"),
        ]:
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='value'>{val}</div>
                    <div class='label'>{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Output as comma-separated string (on-screen)
        csv_string = ", ".join(str(x) for x in final_ids)
        st.markdown("#### Output node IDs")
        st.markdown(f"<div class='output-box'>{csv_string}</div>", unsafe_allow_html=True)

        # Copy helper
        st.code(csv_string, language=None)  # easy select-all copy

        st.markdown("")

        # Tabs
        tab1, tab2, tab3 = st.tabs(["📋 Full table", "🔍 Per-node breakdown", "⚠️ Not found"])

        with tab1:
            out_df = pd.DataFrame({"node_id": final_ids})
            out_df["node_name"] = out_df["node_id"].map(
                lambda x: path_map[id_to_path[x]]["name"] if x in id_to_path else ""
            )
            out_df["node_path"] = out_df["node_id"].map(
                lambda x: id_to_path[x].replace("/", " > ") if x in id_to_path else ""
            )
            st.dataframe(out_df, use_container_width=True, height=360)
            st.download_button(
                "⬇️ Download distilled_clean_node_ids.csv",
                data=out_df.to_csv(index=False).encode(),
                file_name="distilled_clean_node_ids.csv",
                mime="text/csv",
                type="primary",
            )

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
                st.warning(f"{len(not_found)} ID(s) were not in the tree and were skipped.")
                st.dataframe(pd.DataFrame({"node_id": not_found}), use_container_width=True)
            else:
                st.success("All input IDs were found in the tree. ✅")
