

import io
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError, NoCredentialsError
from PIL import Image, ImageDraw, ImageFont
import requests
import streamlit as st
import streamlit.components.v1 as components

import uuid
import pandas as pd
import base64
from streamlit_image_comparison import image_comparison
from botocore import UNSIGNED


# --------------------------
# Constants & Page Settings
# --------------------------
INITIAL_IMAGES = 12        # show 12 images (6 pairs) initially
INCREMENT_IMAGES = 12      # load 12 more images per "infinite scroll" or click
IMAGES_PER_ROW = 2         # EO above IR per cell; 3 cells/row -> 6 images/row total (EO+IR)
CELLS_PER_ROW = 3          # 3 columns (each cell will show EO then IR)
PAIRS_PER_CHUNK = INITIAL_IMAGES // 2  # 6 pairs per chunk

IR_LABEL_BY_VALUE = {
    "LLVIP": "LWIR HIKVISION",
    "KAIST": "LWIR FLIR A656sc",
    "FLIR": "LWIR FLIR Tau 2",
}


st.set_page_config(page_title="RADGE EO/IR Data Engine", layout="wide")

st.session_state.pop("job_table_drawn", None)

# --------------------------
# Per-user ID + in-memory persistent store (reload-safe)
# --------------------------
import uuid


def _get_client_id() -> str:
    # 1) try URL param
    cid = None
    try:
        qp = st.query_params
        cid = qp.get("radge_client")
        if isinstance(cid, list):
            cid = cid[0]
    except Exception:
        cid = None

    # 2) fallback to session_state (prevents minting a new ID if URL briefly lacks it)
    if not cid:
        cid = st.session_state.get("_radge_client")

    # 3) mint + persist if still missing
    if not cid:
        cid = str(uuid.uuid4())
        try:
            st.query_params["radge_client"] = cid  # >=1.30
        except Exception:
            st.experimental_set_query_params(radge_client=cid)  # legacy
        st.session_state["_radge_client"] = cid
        st.rerun()

    # keep in session_state for future fallbacks
    st.session_state["_radge_client"] = cid
    return cid


CLIENT_ID = _get_client_id()


@st.cache_resource(show_spinner=False)
def _global_store():
    # { client_id: {"job_history": [], "selected_job_id": None, "seeded": False} }
    return {}

def _user_store():
    store = _global_store()
    if CLIENT_ID not in store:
        store[CLIENT_ID] = {"job_history": [], "selected_job_id": None, "seeded": False}
    return store[CLIENT_ID]











# --------------------------
# Utilities
# --------------------------
def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """Return (bucket, key) for s3://bucket/key"""
    if not s3_uri or not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    _, remainder = s3_uri.split("s3://", 1)
    parts = remainder.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


# @st.cache_resource(show_spinner=False)
# def get_s3_client(region_name: Optional[str]):
#     # Use default credential chain
#     session = boto3.session.Session(region_name=region_name or None)
#     return session.client("s3", config=Config(signature_version="s3v4"))

@st.cache_resource(show_spinner=False)
def get_s3_client(region_name: Optional[str], anonymous: bool = False):
    session = boto3.session.Session(region_name=region_name or None)
    cfg = Config(signature_version=UNSIGNED) if anonymous else Config(signature_version="s3v4")
    return session.client("s3", config=cfg)


# @st.cache_data(show_spinner=False)
# def read_json_from_s3(s3_uri: str, region_name: Optional[str]) -> dict:
#     bucket, key = parse_s3_uri(s3_uri)
#     s3 = get_s3_client(region_name)
#     obj = s3.get_object(Bucket=bucket, Key=key)
#     data = obj["Body"].read()
#     return json.loads(data)
def read_json_from_s3(s3_uri: str, region_name: Optional[str]) -> dict:
    bucket, key = parse_s3_uri(s3_uri)
    try:
        s3 = get_s3_client(region_name, anonymous=False)
        obj = s3.get_object(Bucket=bucket, Key=key)
    except NoCredentialsError:
        s3 = get_s3_client(region_name, anonymous=True)
        obj = s3.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    return json.loads(data)




@st.cache_data(show_spinner=False, max_entries=4096)
def get_image_bytes_from_s3(s3_uri: str, region_name: Optional[str]) -> bytes:
    bucket, key = parse_s3_uri(s3_uri)
    try:
        s3 = get_s3_client(region_name, anonymous=False)
        obj = s3.get_object(Bucket=bucket, Key=key)
    except (NoCredentialsError, ClientError):
        s3 = get_s3_client(region_name, anonymous=True)
        obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def draw_bboxes(img: Image.Image, bboxes: List[Tuple[int,int,int,int]], labels: List[str]) -> Image.Image:
    """Draws rectangle boxes with labels onto an image and returns a NEW image.
       Uses draw.textbbox() (Pillow >=8.0) with fallbacks to font.getsize()."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for (x, y, w, h), label in zip(bboxes, labels):
        # Box
        draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=2)

        if label:
            # --- measure text (prefer textbbox; fallback to font.getsize; last-resort heuristic) ---
            try:
                if hasattr(draw, "textbbox"):
                    l, t, r, b = draw.textbbox((0, 0), label, font=font)
                    tw, th = (r - l), (b - t)
                elif font and hasattr(font, "getsize"):
                    tw, th = font.getsize(label)  # type: ignore[attr-defined]
                else:
                    tw, th = len(label) * 6, 10
            except Exception:
                tw, th = len(label) * 6, 10

            # Label background above the box, clamped within image
            ty = max(0, y - th - 2)
            draw.rectangle([(x, ty), (x + tw + 4, ty + th + 2)], fill="red")
            draw.text((x + 2, ty + 1), label, fill="white", font=font)

    return out



@dataclass
class PairItem:
    file_name: str
    eo_uri: str
    ir_uri: str
    bboxes: List[Tuple[int,int,int,int]]
    labels: List[str]


def build_pairs_from_metadata_annotations(
    metadata: dict, annotations: dict
) -> List[PairItem]:
    """
    Use metadata.json to get the EO/IR folders and annotations.json (COCO-like) to map bboxes.
    Assumes identical filenames in EO and IR folders.
    """
    outputs = metadata.get("outputs", {})
    eo_folder = outputs.get("s3_rgb_folder_uri") or outputs.get("s3_eo_folder_uri")  # EO
    ir_folder = outputs.get("s3_ir_folder_uri") or outputs.get("s3_infrared_folder_uri")  # IR

    if not eo_folder or not ir_folder:
        raise ValueError("Could not find EO/IR folder URIs in metadata.json 'outputs'.")

    # COCO-like structures
    img_entries = annotations.get("images", [])
    ann_entries = annotations.get("annotations", [])
    categories = annotations.get("categories", [])
    cat_map = {c["id"]: c["name"] for c in categories if "id" in c and "name" in c}

    # Map image_id -> list of (bbox, label)
    by_image: Dict[int, List[Tuple[Tuple[int,int,int,int], str]]] = {}
    for a in ann_entries:
        image_id = a.get("image_id")
        bbox = a.get("bbox", [])  # in [x, y, w, h]
        if image_id is None or len(bbox) != 4:
            continue
        bb_tuple = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        label = cat_map.get(a.get("category_id"), "")
        by_image.setdefault(image_id, []).append((bb_tuple, label))

    # Build pairs, assuming identical file_name in EO and IR folders
    pairs: List[PairItem] = []
    for img in img_entries:
        img_id = img.get("id")
        file_name = img.get("file_name", "")
        if img_id is None or not file_name:
            continue
        eo_uri = eo_folder.rstrip("/") + "/" + file_name
        ir_uri = ir_folder.rstrip("/") + "/" + file_name
        ann_list = by_image.get(img_id, [])
        bboxes = [t[0] for t in ann_list]
        labels = [t[1] for t in ann_list]
        pairs.append(PairItem(file_name=file_name, eo_uri=eo_uri, ir_uri=ir_uri, bboxes=bboxes, labels=labels))

    return pairs


def try_fetch_job_status(api_url: str, api_key: str, job_id: str, job_id_param: str = 'job_id', append_job_id_path: bool = False, verify_ssl: bool = True, timeout_s: int = 30) -> dict:
    """
    Flexible status checker: tries GET with params then POST with JSON if GET fails.
    Expects JSON response; returns {} on error.
    """
    headers = {"x-api-key": f"{api_key}", "Content-Type": "application/json"} if api_key else {}

    errors = []

    # Strategy A: GET with params
    try:
        resp = requests.get(api_url, params={job_id_param: job_id}, headers=headers, timeout=timeout_s, verify=verify_ssl)
        if resp.ok:
            return resp.json()
        else:
            errors.append({"strategy": "GET params", "status": resp.status_code, "text": resp.text[:500]})
    except Exception as e:
        errors.append({"strategy": "GET params", "exception": str(e)})

    # Strategy B: POST JSON body
    try:
        resp = requests.post(api_url, json={job_id_param: job_id}, headers=headers, timeout=timeout_s, verify=verify_ssl)
        if resp.ok:
            return resp.json()
        else:
            errors.append({"strategy": "POST json", "status": resp.status_code, "text": resp.text[:500]})
    except Exception as e:
        errors.append({"strategy": "POST json", "exception": str(e)})

    # Strategy C: GET with job_id appended to path
    try:
        url = api_url.rstrip("/") + f"/{job_id}" if append_job_id_path else api_url
        resp = requests.get(url, headers=headers, timeout=timeout_s, verify=verify_ssl)
        if resp.ok:
            return resp.json()
        else:
            errors.append({"strategy": "GET path", "status": resp.status_code, "text": resp.text[:500]})
    except Exception as e:
        errors.append({"strategy": "GET path", "exception": str(e)})

    return {"error": "All strategies failed", "details": errors}


def submit_job(api_url: str, api_key: str, job_id: str, description: str, num_images: int, label_names: List[str], ir_sensor_type: Optional[str]=None) -> dict:
    """
    Posts a job to the create endpoint. Returns JSON (or {} on error).
    The actual endpoint/shape may vary—this is intentionally flexible.
    """
    payload = {
        "job_id": job_id,
        "user_description": description,
        "num_images": num_images,
        "label_names": label_names,
        "ir_sensor_type": ir_sensor_type,
    }
    headers = {"x-api-key": f"{api_key}", "Content-Type": "application/json"} if api_key else { }

    try:
        r = requests.post(api_url, json=payload, headers=headers, timeout=timeout_s, verify=verify_ssl)
        if r.ok:
            return r.json()

        # Fallback for API Gateway route mismatch (common):
        # try the same URL but with a trailing slash
        if r.status_code == 403 and "Missing Authentication Token" in (r.text or ""):
            alt_url = api_url.rstrip("/") + "/"
            r2 = requests.post(alt_url, json=payload, headers=headers, timeout=timeout_s, verify=verify_ssl)
            if r2.ok:
                return r2.json()
            return {"error": f"{r2.status_code} {r2.text}"}

        return {"error": f"{r.status_code} {r.text}"}

    except Exception as e:
        return {"error": str(e)}


def _extract_container(status_json: dict) -> dict:
    """Normalize access whether fields live at top-level or under 'results'/'data' keys."""
    if not isinstance(status_json, dict):
        return {}
    if "results" in status_json and isinstance(status_json["results"], dict):
        return status_json["results"]
    if "data" in status_json and isinstance(status_json["data"], dict):
        return status_json["data"]
    return status_json


def is_ready(status_json: dict) -> bool:
    """
    Heuristic for job readiness:
      - Top-level status in {'SUCCESS','completed','ready','done','succeeded'} (case-insensitive).
      - If response contains 'outputs' with 's3_*_uri' values, consider ready.
      - Or generation_params.successfully_generated >= total_requested.
    """
    if not status_json:
        return False

    # Accept common 'ready' values
    top_status = str(status_json.get("status", "")).lower()
    if top_status in {"success", "completed", "ready", "done", "succeeded"}:
        return True

    container = _extract_container(status_json)
    inner_status = str(container.get("status", "")).lower()
    if inner_status in {"success", "completed", "ready", "done", "succeeded"}:
        return True

    outputs = container.get("outputs")
    if outputs and any(k for k in outputs.keys() if k.startswith("s3_")):
        return True

    gp = container.get("generation_params") or status_json.get("generation_params")
    if gp and gp.get("successfully_generated") is not None and gp.get("total_requested"):
        try:
            return int(gp["successfully_generated"]) >= int(gp["total_requested"])
        except Exception:
            pass

    return False


def derive_json_uris_from_status(status_json: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    From the status payload, derive metadata.json and annotations.json S3 URIs.
    Prefers explicit s3_coco_json_uri; otherwise assumes both under s3_output_location_uri.
    """
    container = _extract_container(status_json)
    outputs = container.get("outputs", {}) if isinstance(container, dict) else {}

    root = outputs.get("s3_output_location_uri")
    ann = outputs.get("s3_coco_json_uri") or (f"{root.rstrip('/')}/annotations.json" if root else None)
    md = f"{root.rstrip('/')}/metadata.json" if root else None

    return md, ann


def attempt_auto_load_from_status(status_json: dict, region_name: Optional[str]) -> Tuple[Optional[dict], Optional[dict], Optional[str]]:
    """
    If status is ready, auto-load metadata & annotations based on outputs.
    Returns (metadata, annotations, error_str)
    """
    if not is_ready(status_json):
        return None, None, None
    md_uri, ann_uri = derive_json_uris_from_status(status_json)
    if not (md_uri and ann_uri):
        return None, None, "Could not derive JSON URIs from status outputs."
    try:
        metadata = read_json_from_s3(md_uri, region_name or None)
        annotations = read_json_from_s3(ann_uri, region_name or None)
        return metadata, annotations, None
    except (ClientError, NoCredentialsError, ValueError) as e:
        return None, None, str(e)



# Keys we’ll persist in the browser for this tab
PERSIST_KEYS = ["job_history", "selected_job_id", "job_id", "seeded_history"]

def _export_state_for_browser() -> dict:
    return {k: st.session_state.get(k) for k in PERSIST_KEYS if k in st.session_state}

def _import_state_from_browser(data: dict):
    for k, v in (data or {}).items():
        st.session_state[k] = v





# --------------------------
# Job History (session) utils
# --------------------------


def _ensure_history():
    # Keep session_state view in sync with the cached store
    st.session_state["job_history"] = _user_store()["job_history"]

def add_job_to_history(job_id: str, description: str, labels: List[str], ir_sensor_type: Optional[str]):
    data = _user_store()
    data["job_history"].append({
        "job_id": job_id,
        "scene_description": description,
        "labels": ", ".join(labels or []),
        "sensor_type": IR_LABEL_BY_VALUE.get(ir_sensor_type, str(ir_sensor_type)),
        "created_at": pd.Timestamp.utcnow().isoformat(timespec="seconds") + "Z",
    })
    # mirror into session_state for your table logic
    st.session_state["job_history"] = data["job_history"]

def get_history_df() -> pd.DataFrame:
    _ensure_history()
    hist = st.session_state.get("job_history", [])
    if not hist:
        return pd.DataFrame(columns=["Select","job_id","scene_description","labels","sensor_type","created_at"])
    df = pd.DataFrame(hist)
    df.insert(0, "Select", False)
    sel_id = st.session_state.get("selected_job_id")
    if sel_id is not None and sel_id in list(df["job_id"]):
        df.loc[df["job_id"] == sel_id, "Select"] = True
    return df


def seed_history_once():
    """Seed the session job history with a few hard-coded jobs (runs only once)."""
    if st.session_state.get("seeded_history"):
        return

    demo_jobs = [
        {
            "job_id": "de443a29-14bf-48cd-ab7c-710d784d0415",
            "scene": "Aerial image of Ukrainian village",
            "labels": ["car"],
            "ir": "LLVIP",
        },
        {
            "job_id": "14013275-9b33-4c92-9bd1-0c8a91be877e",
            "scene": "Aerial image of Ukrainian village at night.",
            "labels": ["car","person"],
            "ir": "LLVIP",
        },
        {
            "job_id": "max_test_9_9_25_1",
            "scene": "Urban environments during the day during the winter.",
            "labels": ["car","person"],
            "ir": "LLVIP",
        },

        
    ]

    for d in demo_jobs:
        add_job_to_history(d["job_id"], d["scene"], d["labels"], d["ir"])

    st.session_state["seeded_history"] = True






def show_pair_slider(eo_img: Image.Image, ir_img: Image.Image, file_name: str, idx: int):
    # Match sizes so the overlay lines up
    if ir_img.size != eo_img.size:
        ir_img = ir_img.resize(eo_img.size, Image.BILINEAR)

    # Pick a safe width for a 3-column layout
    target_w = min(420, eo_img.width)  # tweak to taste (360–480 works well in 3 cols)

    image_comparison(
        img1=eo_img,
        img2=ir_img,
        label1="EO",
        label2="IR",
        width=target_w,          # <-- must be an int, not None
        starting_position=50,
        in_memory=True,          # <-- passing PIL images
        show_labels=True,
        make_responsive=False    # keep the width we chose
    )



# --------------------------
# UI
# --------------------------
st.title("Retrieval-Augmented Data Generation (RADGE)")

default_create = "https://api.yrikka.com/radge/submit-radge-job"
default_status = "https://api.yrikka.com/radge/get-radge-job"
create_url = default_create
status_url = default_status
job_id_param = "job_id"
append_job_id_path = False
verify_ssl =True
timeout_s = 30
aws_region = "us-east-2"

with st.sidebar:
    st.image("assets/yrikka_logo.png", use_column_width=True)   # ← add this
    st.header("API Settings")
    api_key = st.text_input("API Key (Bearer)", type="password")

st.subheader("1) Submit a Job")
colA, colB = st.columns(2)
with colA:
    description = st.text_area("Scene Description", value="Urban environments during the day during the winter.")


with colB:
    num_images = st.number_input("Number of images to generate", min_value=1, max_value=20, value=5, step=1)
    label_choices = st.multiselect(
        "Labels",
        options=["car", "person"],
        default=["person"],
        help="Choose one or more labels"
    )

    # IR sensor type selection
    # 'KAIST', 'FLIR', or 'LLVIP'
    ##LLVIP LWIR HIKVISION
    ##KAIST LWIR FLIR A656sc 
    ##FLIR LWIR FLIR Tau 2
    ir_options = [
        ("LWIR HIKVISION", "LLVIP"),
        ("LWIR FLIR A656sc", "KAIST"),
        ("LWIR FLIR Tau 2", "FLIR"),
    ]
    ir_label_list = [label for label, _ in ir_options]
    ir_choice_label = st.selectbox("IR Sensor Type", ir_label_list, index=0, help="Choose the IR Sensor to simulate;")
    ir_sensor_type = dict(ir_options)[ir_choice_label]

desc_valid = bool(description and description.strip())
labels_valid = bool(label_choices)
can_submit = desc_valid and labels_valid

submit_clicked = st.button("Submit Job", disabled=not can_submit)

if submit_clicked:
    labels = label_choices  # if you're already using the multiselect; otherwise keep your existing parsing
    job_id = str(uuid.uuid4())  # generate a unique UUID
    with st.spinner("Submitting job..."):
        resp = submit_job(create_url, api_key, job_id, description, int(num_images), labels, ir_sensor_type)
    # st.write("Create response:", resp). ##prints out the json response frmo the API
    # st.success(f"Submitted job_id: {job_id}")
    st.session_state["job_id"] = job_id

    try:
        add_job_to_history(job_id, description, labels, ir_sensor_type)
    except Exception:
        pass
    #Reset the table widget and rerun so only one fresh instance renders
    st.session_state.pop("job_history_editor", None)
    st.toast(f"Submitted job {job_id}", icon="✅")
    st.rerun()


st.divider()
# If we already have a non-empty job history (e.g., from restore), skip seeding

_store = _user_store()
if not _store["seeded"]:
    seed_history_once()       # uses add_job_to_history(...), which writes into the store
    _store["seeded"] = True
    st.session_state.pop("job_history_editor", None)  # optional but tidy

# seed_history_once()
st.subheader("2) View Job History")
df_hist = get_history_df()

if df_hist.empty:
    st.info("No jobs submitted in this session yet. Submit a job to see it here.")
else:
    # Force single-select by rebuilding the Select column from session every render
    current_sel = st.session_state.get("selected_job_id")
    if "Select" not in df_hist.columns:
        df_hist.insert(0, "Select", False)
    df_hist["Select"] = df_hist["job_id"].eq(current_sel)

    # Dynamic key resets editor state when selection changes
    hist_len = len(st.session_state.get("job_history", []))
    table_slot = st.empty()

    # Before creating the editor:

    if "job_table_drawn" in st.session_state:
        st.stop()  # prevent a second render in this run
    st.session_state["job_table_drawn"] = True

    edited = table_slot.data_editor(
        df_hist,
        key="job_history_editor",          # <— STABLE
        hide_index=True,
        use_container_width=True,
        disabled=["job_id","scene_description","labels","sensor_type","created_at"],
        column_config={
            "Select": st.column_config.CheckboxColumn(required=False, help="Click to load this job"),
            "scene_description": st.column_config.TextColumn("Scene Description", width="medium"),
            "labels": st.column_config.TextColumn("Labels"),
            "sensor_type": st.column_config.TextColumn("IR Sensor"),
            "created_at": st.column_config.DatetimeColumn("Created (UTC)", format="YYYY-MM-DD HH:mm"),
        },
        num_rows="fixed",
    )



    # Read current selection from the edited table
    sel_rows = edited.index[edited["Select"] == True].tolist() if "Select" in edited.columns else []
    new_sel = edited.loc[sel_rows[-1], "job_id"] if sel_rows else None  # keep only the last checked

    # If the selection changed, update state, load the job, and rerun ONCE to refresh the table
    if new_sel != current_sel:
        st.session_state["selected_job_id"] = new_sel
        st.session_state["job_id"] = new_sel
        # NEW: persist this per-user
        _user_store()["selected_job_id"] = new_sel

        # Reset the editor widget so the single-select checkbox reflects the new selection
        st.session_state.pop("job_history_editor", None)

        if new_sel:
            with st.spinner(f"Loading job {new_sel} ..."):
                status_json = try_fetch_job_status(
                    status_url,
                    api_key,
                    new_sel,
                    job_id_param=job_id_param,
                    append_job_id_path=append_job_id_path,
                    verify_ssl=verify_ssl,
                    timeout_s=int(timeout_s),
                )
                st.session_state["status_json"] = status_json
                md, ann, err = attempt_auto_load_from_status(status_json, aws_region or None)
                if err:
                    st.error(f"Auto-load error: {err}")
                if md and ann:
                    st.session_state["metadata"] = md
                    st.session_state["annotations"] = ann

        # One-time rerun so the editor re-renders with a single checked row
        st.rerun()






# st.subheader("2) Check Status → View Images")

# if "job_id" not in st.session_state:
#     st.info("Enter a Job ID to check status and auto-load data when ready.")
# manual_job_id = st.text_input("Job ID", value=st.session_state.get("job_id", ""))

# cols = st.columns([1,1,2])
# with cols[0]:
#     check_clicked = st.button("🔄 Check Status")
# with cols[1]:
#     pass
# with cols[2]:
#     pass
status_json = st.session_state.get("status_json", {})
# current_job_id = manual_job_id or st.session_state.get("job_id") or ""
current_job_id = st.session_state.get("job_id") or ""

# ---- Check status (manual) ----
# if check_clicked and current_job_id:
if current_job_id:

    with st.spinner("Checking status..."):
        status_json = try_fetch_job_status(status_url, api_key, current_job_id, job_id_param=job_id_param, append_job_id_path=append_job_id_path, verify_ssl=verify_ssl, timeout_s=int(timeout_s))
    st.session_state["status_json"] = status_json

# ---- Auto-load JSONs whenever status indicates ready ----
metadata = st.session_state.get("metadata")
annotations = st.session_state.get("annotations")

if status_json and not (metadata and annotations):
    md, ann, err = attempt_auto_load_from_status(status_json, aws_region or None)
    if err:
        st.error(f"Auto-load error: {err}")
    if md and ann:
        st.session_state["metadata"] = metadata = md
        st.session_state["annotations"] = annotations = ann

# Show status
if status_json:
    # st.write("Status response:", status_json). ##show the API json response
    if isinstance(status_json, dict) and status_json.get("error"):
        st.error("Status call failed.")
    elif is_ready(status_json):
        st.success("Job is ready. Data auto-loaded below.")
    else:
        st.warning("Job is not ready yet. Check again later.")

# If metadata/annotations already in state, proceed to pairs
if metadata and annotations:

    with st.expander("Show annotations.json (truncated)", expanded=False):
        st.json({k: annotations[k] for k in annotations.keys() if k in ("info","images","annotations","categories")})

    # ------------- Build pairs -----------------
    try:
        pairs = build_pairs_from_metadata_annotations(metadata, annotations)
    except Exception as e:
        st.error(f"Error building EO/IR pairs: {e}")
        pairs = []

    if "pairs_total" not in st.session_state or st.session_state.get("pairs_total") != len(pairs):
        st.session_state["pairs_total"] = len(pairs)
        # reset view counts if the dataset changed
        st.session_state["pairs_to_show"] = PAIRS_PER_CHUNK

    # Controls
    draw_boxes = st.checkbox("Draw bounding boxes", value=False)

    total_pairs = len(pairs)
    pairs_to_show = st.session_state.get("pairs_to_show", PAIRS_PER_CHUNK)
    pairs_to_show = min(pairs_to_show, total_pairs)

    # ---------------- Infinite scroll (always on) ----------------

    components.html(
        """
        <script>
        (function() {
        const rootDoc = window.parent.document;
        function nearBottom() {
            const el = rootDoc.scrollingElement || rootDoc.documentElement;
            return (el.scrollTop + el.clientHeight) >= (el.scrollHeight - 48);
        }
        let pending = false;
        rootDoc.addEventListener('scroll', function() {
            if (pending) return;
            if (nearBottom()) {
            pending = true;
            try {
                // Merge existing URL params (e.g., radge_client) with autoload
                const url = new URL(window.location.href);
                const params = Object.fromEntries(url.searchParams.entries());
                params.autoload = Date.now().toString();
                window.parent.postMessage({
                type: "streamlit:setQueryParams",
                queryParams: params
                }, "*");
            } catch (e) {}
            setTimeout(() => { pending = false; }, 800);
            }
        }, true);
        })();
        </script>
        """,
        height=0,
    )




    # On rerun triggered by query param change, bump the number of pairs to show
    qp = st.query_params
    token = qp.get("autoload")
    if token:
        prev = st.session_state.get("autoload_token")
        if token != prev and pairs_to_show < total_pairs:
            st.session_state["autoload_token"] = token
            st.session_state["pairs_to_show"] = min(pairs_to_show + (INCREMENT_IMAGES // 2), total_pairs)
            st.rerun()

    # ---------------- Render grid ----------------
    shown_pairs = pairs[:pairs_to_show]
    st.write(f"Showing {min(pairs_to_show*2, total_pairs*2)} / {total_pairs*2} images "
             f"({pairs_to_show} / {total_pairs} pairs)")

    # Render in CELLS_PER_ROW columns, each cell shows EO then IR
    grid_cols = st.columns(CELLS_PER_ROW)
    for idx, pair in enumerate(shown_pairs):
        col = grid_cols[idx % CELLS_PER_ROW]
        with col:
            # Load EO
            try:
                eo_bytes = get_image_bytes_from_s3(pair.eo_uri, aws_region or None)
                eo_img = Image.open(io.BytesIO(eo_bytes)).convert("RGB")
            except Exception as e:
                eo_img = None
                st.error(f"EO load failed for {pair.eo_uri}: {e}")

            # Optional: draw boxes (don’t fail the whole image if overlay fails)
            if eo_img is not None and draw_boxes and pair.bboxes:
                try:
                    eo_img = draw_bboxes(eo_img, pair.bboxes, pair.labels)
                except Exception as e:
                    st.warning(f"EO overlay failed for {pair.file_name}: {e}")


            try:
                ir_bytes = get_image_bytes_from_s3(pair.ir_uri, aws_region or None)
                ir_img = Image.open(io.BytesIO(ir_bytes)).convert("RGB")
            except Exception as e:
                ir_img = None
                st.error(f"IR load failed for {pair.ir_uri}: {e}")

            # Optional: draw boxes on IR (don’t fail the whole image if overlay fails)
            if ir_img is not None and draw_boxes and pair.bboxes:
                try:
                    ir_img = draw_bboxes(ir_img, pair.bboxes, pair.labels)
                except Exception as e:
                    st.warning(f"IR overlay failed for {pair.file_name}: {e}")


            # if eo_img is not None:
            #     st.image(eo_img, caption=f"EO · {pair.file_name}", use_column_width=True)
            # else:
            #     st.warning(f"EO (failed) — {pair.file_name}")

            # if ir_img is not None:
            #     st.image(ir_img, caption=f"IR · {pair.file_name}", use_column_width=True)
            # else:
            #     st.warning(f"IR (failed) — {pair.file_name}")
            if eo_img is not None and ir_img is not None:
                show_pair_slider(eo_img, ir_img, pair.file_name, idx)
            elif eo_img is not None:
                st.image(eo_img, caption=f"EO · {pair.file_name}", use_column_width=True)
            elif ir_img is not None:
                st.image(ir_img, caption=f"IR · {pair.file_name}", use_column_width=True)
            else:
                st.warning(f"Both EO/IR failed — {pair.file_name}")


    # ------------- "Load more" fallback control -------------
    if pairs_to_show < total_pairs:
        st.divider()
        if st.button("⬇️ Load more images"):
            st.session_state["pairs_to_show"] = min(pairs_to_show + (INCREMENT_IMAGES // 2), total_pairs)
            st.rerun()

else:
    pass



# --- Save to browser sessionStorage so refresh keeps this tab's state ---

# --- Save to browser sessionStorage so refresh keeps this tab's state ---
_state_obj = _export_state_for_browser()
_state_json = json.dumps(_state_obj)

