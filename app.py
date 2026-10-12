import os
import io
import uuid
import sqlite3
import datetime
import streamlit as st
import pandas as pd
from PIL import Image
from streamlit_folium import st_folium
from streamlit_js_eval import get_geolocation

from detector import AutomatedPotholeDetector
from gps_utils import extract_exif_gps
from aggregation import compute_municipal_priorities
from map_utils import build_map

st.set_page_config(page_title="RoadGuard | Automated Pothole Intelligence", page_icon="🛣️", layout="wide")

# Determine whether Supabase is configured, otherwise fallback to local SQLite
USE_SUPABASE = False
supabase_client = None

if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
    try:
        from supabase import create_client
        supabase_client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
        USE_SUPABASE = True
    except Exception as e:
        st.sidebar.warning(f"Supabase connection warning: {e}. Falling back to local storage.")

# Local SQLite initialization (Fallback mode)
DB_PATH = "reports_local.db"
IMG_DIR = "uploaded_images"
os.makedirs(IMG_DIR, exist_ok=True)

def init_local_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id TEXT PRIMARY KEY,
            created_at TEXT,
            image_url TEXT,
            latitude REAL,
            longitude REAL,
            pothole_count INTEGER,
            confidence_avg REAL,
            severity_score REAL,
            status TEXT DEFAULT 'pending'
        )
    ''')
    conn.commit()
    conn.close()

init_local_db()

def get_all_reports():
    if USE_SUPABASE and supabase_client:
        try:
            res = supabase_client.table("reports").select("*").execute()
            return res.data or []
        except Exception:
            pass
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM reports ORDER BY created_at DESC", conn)
    conn.close()
    return df.to_dict(orient="records")

def save_report(image_bytes, file_name, lat, lng, count, conf, severity):
    if USE_SUPABASE and supabase_client:
        try:
            supabase_client.storage.from_("pothole-images").upload(file_name, image_bytes, {"content-type": "image/jpeg"})
            public_url = supabase_client.storage.from_("pothole-images").get_public_url(file_name)
            payload = {
                "image_url": public_url,
                "latitude": lat,
                "longitude": lng,
                "pothole_count": count,
                "confidence_avg": conf,
                "severity_score": severity
            }
            supabase_client.table("reports").insert(payload).execute()
            return public_url
        except Exception as e:
            st.warning(f"Supabase upload failed ({e}). Saving to local database instead.")

    # Local fallback
    local_path = os.path.join(IMG_DIR, file_name)
    with open(local_path, "wb") as f:
        f.write(image_bytes)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO reports (id, created_at, image_url, latitude, longitude, pothole_count, confidence_avg, severity_score, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (str(uuid.uuid4()), datetime.datetime.now().isoformat(), local_path, lat, lng, count, conf, severity, 'pending'))
    conn.commit()
    conn.close()
    return local_path

@st.cache_resource
def load_detector():
    return AutomatedPotholeDetector()

detector = load_detector()

# Initialize session state for deduplication
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# Header
st.title("🛣️ RoadGuard: Automated Pothole Intelligence")
st.caption("AI-powered road defect capture, automated geospatial tracking, and municipal repair dispatch.")

# Navigation
tab_upload, tab_map, tab_admin = st.tabs(["⚡ Auto-Detection", "🗺️ Live Heatmap", "🏛️ Municipal Priority Zones"])

# ----------------- TAB 1: AUTOMATED DETECTION -----------------
with tab_upload:
    st.subheader("Automated Road Defect Capture")
    st.write("Take or upload a photo from your phone. Detection, GPS tagging, and transmission happen automatically.")

    # 1. Fetch Browser GPS automatically
    gps_data = get_geolocation()
    browser_lat, browser_lng = None, None
    if gps_data and "coords" in gps_data and gps_data["coords"]:
        browser_lat = round(gps_data["coords"].get("latitude", 0.0), 6)
        browser_lng = round(gps_data["coords"].get("longitude", 0.0), 6)

    # 2. Camera / Image Input
    img_file = st.file_uploader("Upload or Snap Road Photo", type=["jpg", "jpeg", "png"])

    if img_file is not None:
        pil_image = Image.open(img_file)
        file_signature = f"{img_file.name}_{img_file.size}"

        # Coordinate resolution: Browser -> EXIF fallback
        final_lat, final_lng = browser_lat, browser_lng
        if not final_lat or not final_lng:
            exif_lat, exif_lng = extract_exif_gps(pil_image)
            if exif_lat and exif_lng:
                final_lat, final_lng = exif_lat, exif_lng

        if not final_lat or not final_lng:
            st.info("📍 GPS not auto-detected from browser or EXIF. Enter or confirm coordinates below:")
            c1, c2 = st.columns(2)
            final_lat = c1.number_input("Latitude", value=28.6139, format="%.6f", key="input_lat")
            final_lng = c2.number_input("Longitude", value=77.2090, format="%.6f", key="input_lng")

        # Process only once per uploaded file to prevent re-submitting on tab switch/refresh
        if file_signature not in st.session_state.processed_files:
            with st.spinner("🤖 Running automated YOLOv8 road inspection..."):
                annotated_img, count, conf, severity = detector.detect(pil_image)

                if count > 0:
                    file_id = f"{uuid.uuid4()}.jpg"
                    buf = io.BytesIO()
                    annotated_img.save(buf, format="JPEG")
                    save_report(buf.getvalue(), file_id, final_lat, final_lng, count, conf, severity)

                st.session_state.processed_files.add(file_signature)
                st.session_state.last_result = {
                    "count": count,
                    "conf": conf,
                    "severity": severity,
                    "annotated_img": annotated_img,
                    "original_img": pil_image,
                    "coords": (final_lat, final_lng)
                }

        # Render output
        res = st.session_state.last_result
        if res:
            if res["count"] == 0:
                st.info("✅ Road surface looks clear! No potholes detected in this frame.")
                st.image(res["original_img"], caption="Clean Road Surface", width=400)
            else:
                st.error(f"🚨 Automatically Detected {res['count']} Pothole(s)! Severity: {res['severity']}/10 (Confidence: {int(res['conf']*100)}%)")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.image(res["annotated_img"], caption="Automated YOLOv8 Bounding Boxes", use_column_width=True)
                with col_b:
                    st.success(f"📍 Tagged GPS: ({res['coords'][0]}, {res['coords'][1]})")
                    st.metric("Potholes Detected", res["count"])
                    st.metric("Calculated Road Severity", f"{res['severity']} / 10")
                    st.success("✅ Report automatically logged and transmitted to Municipal Database!")

# ----------------- TAB 2: LIVE MAP -----------------
with tab_map:
    st.subheader("Live Interactive Road Surface Map")
    records = get_all_reports()

    if records:
        df_reports = pd.DataFrame(records)
        hotspots_df = compute_municipal_priorities(df_reports)
        f_map = build_map(df_reports, hotspots_df)
        st_folium(f_map, width=1200, height=600)
    else:
        st.info("No reports logged yet. Upload a pothole photo in Tab 1 to see it appear on the map.")

# ----------------- TAB 3: MUNICIPAL PRIORITIES -----------------
with tab_admin:
    st.subheader("Top Urgent Road Clusters for Municipal Repair Crews")
    records = get_all_reports()

    if records:
        df_reports = pd.DataFrame(records)
        hotspots_df = compute_municipal_priorities(df_reports)

        if not hotspots_df.empty:
            st.dataframe(
                hotspots_df[['cluster_id', 'urgency', 'priority_score', 'report_count', 'total_potholes', 'avg_severity', 'latitude', 'longitude']],
                use_container_width=True
            )
            st.metric("Top Critical Cluster ID", f"Zone #{hotspots_df.iloc[0]['cluster_id']}")
            st.metric("Total Flagged Repair Zones", len(hotspots_df))
        else:
            st.info("Insufficient clustering density to form multi-report zones. Individual reports are visible on the Live Map.")
    else:
        st.info("No data available yet.")
