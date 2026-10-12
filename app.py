import os
import io
import uuid
import sqlite3
import datetime
import urllib.request
import base64
import cv2
import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from streamlit_folium import st_folium
from streamlit_js_eval import get_geolocation
import folium
from folium.plugins import MarkerCluster, HeatMap
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

st.set_page_config(page_title="RoadGuard | Automated Pothole Intelligence", page_icon="🛣️", layout="wide")

# ==================== GPS UTILS ====================
def convert_to_degrees(value):
    try:
        d, m, s = value
        return float(d) + (float(m) / 60.0) + (float(s) / 3600.0)
    except (TypeError, ValueError):
        return float(value)

def extract_exif_gps(image: Image.Image):
    try:
        exif = image._getexif()
        if not exif:
            return None, None
        gps_info = {}
        for tag, val in exif.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                for t in val:
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_info[sub_decoded] = val[t]
        if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
            lat = convert_to_degrees(gps_info["GPSLatitude"])
            if str(gps_info.get("GPSLatitudeRef", "N")).upper() == "S":
                lat = -lat
            lng = convert_to_degrees(gps_info["GPSLongitude"])
            if str(gps_info.get("GPSLongitudeRef", "E")).upper() == "W":
                lng = -lng
            return round(lat, 6), round(lng, 6)
    except Exception:
        pass
    return None, None

# ==================== DETECTOR ====================
class AutomatedPotholeDetector:
    def __init__(self, model_path="pothole_yolov8.pt"):
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            url = "https://huggingface.co/peterhdd/pothole-detection-yolov8/resolve/main/best.pt"
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=30) as resp, open(self.model_path, 'wb') as f:
                    f.write(resp.read())
            except Exception:
                self.model_path = "yolov8n.pt"
        self.model = YOLO(self.model_path)

    def detect(self, image: Image.Image, conf_threshold=0.25):
        img_np = np.array(image.convert("RGB"))
        results = self.model.predict(source=img_np, conf=conf_threshold, verbose=False)
        result = results[0]
        boxes = result.boxes
        count = len(boxes)
        if count == 0:
            return None, 0, 0.0, 0.0
        confidences = boxes.conf.cpu().numpy()
        avg_conf = float(np.mean(confidences))
        img_h, img_w, _ = img_np.shape
        total_area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes.xyxy.cpu().numpy())
        severity = min(10.0, (count * 2.0) + ((total_area / max(1.0, img_h * img_w)) * 20.0))
        annotated_np = result.plot()
        annotated_img = Image.fromarray(cv2.cvtColor(annotated_np, cv2.COLOR_BGR2RGB))
        return annotated_img, count, round(avg_conf, 2), round(severity, 2)

# ==================== AGGREGATION & MAP ====================
def compute_municipal_priorities(reports_df, radius_meters=40.0):
    if reports_df is None or len(reports_df) < 2:
        return pd.DataFrame()
    df = reports_df.copy()
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude'])
    if len(df) < 2:
        return pd.DataFrame()
    df['pothole_count'] = pd.to_numeric(df.get('pothole_count', 1), errors='coerce').fillna(1)
    df['severity_score'] = pd.to_numeric(df.get('severity_score', 1.0), errors='coerce').fillna(1.0)
    coords_rad = np.radians(df[['latitude', 'longitude']].values)
    db = DBSCAN(eps=radius_meters / 6371000.0, min_samples=2, metric='haversine')
    df['cluster_id'] = db.fit_predict(coords_rad)
    clustered = df[df['cluster_id'] != -1]
    if clustered.empty:
        return pd.DataFrame()
    hotspots = []
    for cid, grp in clustered.groupby('cluster_id'):
        rc = len(grp)
        tp = int(grp['pothole_count'].sum())
        av_sev = float(grp['severity_score'].mean())
        score = (rc * 3.0) + (tp * 2.0) + (av_sev * 2.5)
        urg = "CRITICAL" if score >= 30 else ("HIGH" if score >= 15 else "MEDIUM")
        hotspots.append({
            "cluster_id": int(cid),
            "latitude": float(grp['latitude'].mean()),
            "longitude": float(grp['longitude'].mean()),
            "report_count": rc,
            "total_potholes": tp,
            "avg_severity": round(av_sev, 2),
            "priority_score": round(score, 1),
            "urgency": urg
        })
    return pd.DataFrame(hotspots).sort_values(by="priority_score", ascending=False).reset_index(drop=True)

def build_map(reports_df, hotspots_df=None):
    if reports_df is None or len(reports_df) == 0:
        return folium.Map(location=[28.6139, 77.2090], zoom_start=11, tiles="CartoDB positron")
    df = reports_df.copy()
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['severity_score'] = pd.to_numeric(df.get('severity_score', 1.0), errors='coerce').fillna(1.0)
    df = df.dropna(subset=['latitude', 'longitude'])
    if df.empty:
        return folium.Map(location=[28.6139, 77.2090], zoom_start=11, tiles="CartoDB positron")
    m = folium.Map(location=[float(df['latitude'].mean()), float(df['longitude'].mean())], zoom_start=14, tiles="CartoDB positron")
    HeatMap([[float(r['latitude']), float(r['longitude']), float(r['severity_score'])] for _, r in df.iterrows()], radius=18, blur=14).add_to(m)
    cluster = MarkerCluster(name="Reports").add_to(m)
    for _, r in df.iterrows():
        folium.Marker(
            [float(r['latitude']), float(r['longitude'])],
            popup=f"Potholes: {int(r.get('pothole_count', 1))} | Sev: {float(r['severity_score']):.1f}/10",
            icon=folium.Icon(color="red" if float(r['severity_score']) >= 6 else "orange", icon="warning", prefix="fa")
        ).add_to(cluster)
    if hotspots_df is not None and not hotspots_df.empty:
        colors = {"CRITICAL": "#d63031", "HIGH": "#e17055", "MEDIUM": "#fdcb6e"}
        zones = folium.FeatureGroup(name="Priority Zones").add_to(m)
        for _, s in hotspots_df.iterrows():
            folium.Circle([float(s['latitude']), float(s['longitude'])], radius=45, color=colors.get(s['urgency'], "#d63031"), fill=True, fill_opacity=0.45).add_to(zones)
    folium.LayerControl().add_to(m)
    return m

# ==================== DATABASE ====================
DB_PATH = "reports_local.db"
IMG_DIR = "uploaded_images"
os.makedirs(IMG_DIR, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS reports (
        id TEXT PRIMARY KEY, created_at TEXT, image_url TEXT,
        latitude REAL, longitude REAL, pothole_count INTEGER,
        confidence_avg REAL, severity_score REAL, status TEXT DEFAULT 'pending')''')
    conn.commit()
    conn.close()

init_db()

def get_all_reports():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM reports ORDER BY created_at DESC", conn)
    conn.close()
    return df.to_dict(orient="records")

def save_report(image_bytes, file_name, lat, lng, count, conf, sev):
    path = os.path.join(IMG_DIR, file_name)
    with open(path, "wb") as f:
        f.write(image_bytes)
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''INSERT INTO reports VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                 (str(uuid.uuid4()), datetime.datetime.now().isoformat(), path, lat, lng, count, conf, sev, 'pending'))
    conn.commit()
    conn.close()
    return path

@st.cache_resource
def load_detector():
    return AutomatedPotholeDetector()

detector = load_detector()

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ==================== APP UI ====================
st.title("🛣️ RoadGuard: Automated Pothole Intelligence")
tab_upload, tab_map, tab_admin = st.tabs(["⚡ Auto-Detection", "🗺️ Live Heatmap", "🏛️ Municipal Priority Zones"])

with tab_upload:
    st.subheader("Automated Road Defect Capture")
    gps = get_geolocation()
    b_lat, b_lng = None, None
    if gps and "coords" in gps and gps["coords"]:
        b_lat = round(gps["coords"].get("latitude", 0.0), 6)
        b_lng = round(gps["coords"].get("longitude", 0.0), 6)

    img_file = st.file_uploader("Upload or Snap Road Photo", type=["jpg", "jpeg", "png"])
    if img_file is not None:
        pil_img = Image.open(img_file)
        sig = f"{img_file.name}_{img_file.size}"
        f_lat, f_lng = b_lat, b_lng
        if not f_lat or not f_lng:
            ex_lat, ex_lng = extract_exif_gps(pil_img)
            if ex_lat and ex_lng:
                f_lat, f_lng = ex_lat, ex_lng
        if not f_lat or not f_lng:
            st.info("📍 Location not auto-detected. Enter coordinates:")
            c1, c2 = st.columns(2)
            f_lat = c1.number_input("Latitude", value=28.6139, format="%.6f")
            f_lng = c2.number_input("Longitude", value=77.2090, format="%.6f")

        if sig not in st.session_state.processed_files:
            with st.spinner("🤖 Inspecting road surface with YOLOv8..."):
                ann_img, count, conf, sev = detector.detect(pil_img)
                if count > 0:
                    buf = io.BytesIO()
                    ann_img.save(buf, format="JPEG")
                    save_report(buf.getvalue(), f"{uuid.uuid4()}.jpg", f_lat, f_lng, count, conf, sev)
                st.session_state.processed_files.add(sig)
                st.session_state.last_result = {"count": count, "conf": conf, "sev": sev, "ann_img": ann_img, "orig": pil_img, "coords": (f_lat, f_lng)}

        res = st.session_state.last_result
        if res:
            if res["count"] == 0:
                st.info("✅ Road looks clear! No potholes detected.")
                st.image(res["orig"], width=400)
            else:
                st.error(f"🚨 Detected {res['count']} Pothole(s)! Severity: {res['sev']}/10")
                ca, cb = st.columns(2)
                with ca:
                    st.image(res["ann_img"], use_column_width=True)
                with cb:
                    st.success(f"📍 GPS: ({res['coords'][0]}, {res['coords'][1]})")
                    st.metric("Potholes", res["count"])
                    st.metric("Severity", f"{res['sev']} / 10")
                    st.success("✅ Logged to Municipal System!")

with tab_map:
    st.subheader("Live Interactive Road Surface Map")
    records = get_all_reports()
    if records:
        df_rep = pd.DataFrame(records)
        hotspots = compute_municipal_priorities(df_rep)
        f_map = build_map(df_rep, hotspots)
        st_folium(f_map, use_container_width=True, height=550)
    else:
        st.info("No reports yet. Upload a pothole in Tab 1!")

with tab_admin:
    st.subheader("Municipal Priority Dispatch Zones")
    records = get_all_reports()
    if records:
        df_rep = pd.DataFrame(records)
        hotspots = compute_municipal_priorities(df_rep)
        if not hotspots.empty:
            st.dataframe(hotspots, use_container_width=True)
        else:
            st.info("Reports exist, but not clustered tightly enough yet to form a zone.")
