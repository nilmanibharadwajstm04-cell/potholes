import os
import io
import uuid
import sqlite3
import datetime
import urllib.request
import math
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

# Page Setup
st.set_page_config(page_title="RoadGuard | Automated Pothole Intelligence", page_icon="🛣️", layout="wide")

# ==================== 1. GPS & DISTANCE UTILS ====================
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

def calculate_distance_meters(lat1, lon1, lat2, lon2):
    """Calculates Haversine distance in meters between two GPS coordinates."""
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    return round(R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a)), 1)

# ==================== 2. AI MODEL DETECTOR ====================
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

# ==================== 3. SHARED DATABASE SETUP ====================
DB_PATH = "reports_local.db"
IMG_DIR = "uploaded_images"
os.makedirs(IMG_DIR, exist_ok=True)

USE_SUPABASE = False
supabase_client = None
if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
    try:
        from supabase import create_client
        supabase_client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
        USE_SUPABASE = True
    except Exception:
        USE_SUPABASE = False

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS reports (
        id TEXT PRIMARY KEY, created_at TEXT, image_url TEXT,
        latitude REAL, longitude REAL, pothole_count INTEGER,
        confidence_avg REAL, severity_score REAL, status TEXT DEFAULT 'pending')''')
    
    # Pre-seed realistic demo data so the map is never empty
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM reports")
    if c.fetchone()[0] == 0:
        sample_reports = [
            (str(uuid.uuid4()), datetime.datetime.now().isoformat(), "", 28.6139, 77.2090, 3, 0.88, 7.5, 'pending'),
            (str(uuid.uuid4()), datetime.datetime.now().isoformat(), "", 28.6141, 77.2092, 2, 0.92, 6.2, 'pending'),
            (str(uuid.uuid4()), datetime.datetime.now().isoformat(), "", 28.6138, 77.2089, 4, 0.85, 8.8, 'pending'),
            (str(uuid.uuid4()), datetime.datetime.now().isoformat(), "", 28.6300, 77.2200, 1, 0.91, 3.2, 'pending'),
            (str(uuid.uuid4()), datetime.datetime.now().isoformat(), "", 28.6302, 77.2204, 2, 0.84, 4.5, 'pending'),
        ]
        conn.executemany("INSERT INTO reports VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", sample_reports)
    conn.commit()
    conn.close()

init_db()

def get_all_reports():
    if USE_SUPABASE and supabase_client:
        try:
            res = supabase_client.table("reports").select("*").execute()
            if res.data:
                return res.data
        except Exception:
            pass
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM reports ORDER BY created_at DESC", conn)
    conn.close()
    return df.to_dict(orient="records")

def save_report(image_bytes, file_name, lat, lng, count, conf, sev):
    path = os.path.join(IMG_DIR, file_name)
    with open(path, "wb") as f:
        f.write(image_bytes)

    if USE_SUPABASE and supabase_client:
        try:
            supabase_client.storage.from_("pothole-images").upload(file_name, image_bytes, {"content-type": "image/jpeg"})
            public_url = supabase_client.storage.from_("pothole-images").get_public_url(file_name)
            supabase_client.table("reports").insert({
                "image_url": public_url, "latitude": lat, "longitude": lng,
                "pothole_count": count, "confidence_avg": conf, "severity_score": sev
            }).execute()
        except Exception:
            pass

    conn = sqlite3.connect(DB_PATH)
    conn.execute('''INSERT INTO reports VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                 (str(uuid.uuid4()), datetime.datetime.now().isoformat(), path, lat, lng, count, conf, sev, 'pending'))
    conn.commit()
    conn.close()
    return path

# ==================== 4. MUNICIPAL PRIORITIZATION & MAP ====================
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

def build_map(reports_df, hotspots_df=None, user_lat=None, user_lng=None):
    center_lat = user_lat if user_lat else 28.6139
    center_lng = user_lng if user_lng else 77.2090

    if reports_df is not None and not reports_df.empty:
        df = reports_df.copy()
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df = df.dropna(subset=['latitude', 'longitude'])
        if not df.empty and not user_lat:
            center_lat = float(df['latitude'].mean())
            center_lng = float(df['longitude'].mean())

    # Map with 100% Free OpenStreetMap (NO API KEY REQUIRED)
    m = folium.Map(location=[center_lat, center_lng], zoom_start=14, tiles="OpenStreetMap")

    # 1. Plot User's Current GPS Position (Blue Marker)
    if user_lat and user_lng:
        folium.Marker(
            [user_lat, user_lng],
            popup="<b>📍 You Are Here (Current Location)</b>",
            tooltip="Your Current GPS Position",
            icon=folium.Icon(color="blue", icon="user", prefix="fa")
        ).add_to(m)

        # 500-meter detection radar ring around user
        folium.Circle(
            [user_lat, user_lng],
            radius=500,
            color="#0984e3",
            fill=True,
            fill_opacity=0.1,
            tooltip="500m Hazard Detection Range"
        ).add_to(m)

    # 2. Potholes Layer
    if reports_df is not None and not reports_df.empty:
        df = reports_df.copy()
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df['severity_score'] = pd.to_numeric(df.get('severity_score', 1.0), errors='coerce').fillna(1.0)
        df = df.dropna(subset=['latitude', 'longitude'])

        # Heatmap
        HeatMap([[float(r['latitude']), float(r['longitude']), float(r['severity_score'])] for _, r in df.iterrows()], radius=18, blur=14).add_to(m)

        # Individual Pothole Markers
        cluster = MarkerCluster(name="Reported Potholes").add_to(m)
        for _, r in df.iterrows():
            dist_str = ""
            if user_lat and user_lng:
                dist = calculate_distance_meters(user_lat, user_lng, float(r['latitude']), float(r['longitude']))
                dist_str = f"<br/><b>Distance to you:</b> {dist} meters"

            popup_html = f"""
            <div style="font-family: Arial; width: 180px;">
                <b>Potholes Detected:</b> {int(r.get('pothole_count', 1))}<br/>
                <b>Severity:</b> {float(r['severity_score']):.1f} / 10
                {dist_str}
            </div>
            """
            folium.Marker(
                [float(r['latitude']), float(r['longitude'])],
                popup=popup_html,
                icon=folium.Icon(color="red" if float(r['severity_score']) >= 6 else "orange", icon="warning", prefix="fa")
            ).add_to(cluster)

    # 3. High-Priority Municipal Zones
    if hotspots_df is not None and not hotspots_df.empty:
        colors = {"CRITICAL": "#d63031", "HIGH": "#e17055", "MEDIUM": "#fdcb6e"}
        zones = folium.FeatureGroup(name="Priority Action Zones").add_to(m)
        for _, s in hotspots_df.iterrows():
            folium.Circle(
                [float(s['latitude']), float(s['longitude'])],
                radius=45,
                color=colors.get(s['urgency'], "#d63031"),
                fill=True,
                fill_opacity=0.45,
                popup=f"Zone #{s['cluster_id']} [{s['urgency']}]: {s['report_count']} reports"
            ).add_to(zones)

    folium.LayerControl().add_to(m)
    return m

# ==================== 5. LOAD DETECTOR & STATE ====================
@st.cache_resource
def load_detector():
    return AutomatedPotholeDetector()

detector = load_detector()

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ==================== 6. USER INTERFACE ====================
st.title("🛣️ RoadGuard: Automated Pothole Intelligence")
st.caption("Crowdsourced road defect capture, live proximity alerts, and municipal repair dispatch.")

tab_upload, tab_map, tab_admin = st.tabs(["⚡ Auto-Detection", "🗺️ Live Heatmap & Nearby Alerts", "🏛️ Municipal Priority Zones"])

# Fetch User Browser GPS
gps = get_geolocation()
user_curr_lat, user_curr_lng = None, None
if gps and "coords" in gps and gps["coords"]:
    user_curr_lat = round(gps["coords"].get("latitude", 0.0), 6)
    user_curr_lng = round(gps["coords"].get("longitude", 0.0), 6)

with tab_upload:
    st.subheader("Automated Road Defect Capture")
    st.write("Upload or snap a road photo. The pothole is automatically detected, GPS-tagged, and published for all users to see.")

    img_file = st.file_uploader("Upload or Snap Road Photo", type=["jpg", "jpeg", "png"])
    if img_file is not None:
        pil_img = Image.open(img_file)
        sig = f"{img_file.name}_{img_file.size}"
        f_lat, f_lng = user_curr_lat, user_curr_lng
        if not f_lat or not f_lng:
            ex_lat, ex_lng = extract_exif_gps(pil_img)
            if ex_lat and ex_lng:
                f_lat, f_lng = ex_lat, ex_lng
        if not f_lat or not f_lng:
            st.info("📍 Location not auto-detected. Confirm coordinates:")
            c1, c2 = st.columns(2)
            f_lat = c1.number_input("Latitude", value=28.6139, format="%.6f")
            f_lng = c2.number_input("Longitude", value=77.2090, format="%.6f")

        if sig not in st.session_state.processed_files:
            with st.spinner("🤖 Inspecting road surface with YOLOv8..."):
                ann_img, count, conf, sev = detector.detect(pil_img)
                if count > 0:
                    buf = io.BytesIO()
                    ann_img.save(buf, format="JPEG")
                    file_name = str(uuid.uuid4()) + ".jpg"
                    save_report(buf.getvalue(), file_name, f_lat, f_lng, count, conf, sev)
                st.session_state.processed_files.add(sig)
                st.session_state.last_result = {
                    "count": count, "conf": conf, "sev": sev,
                    "ann_img": ann_img, "orig": pil_img,
                    "coords": (f_lat, f_lng)
                }

        res = st.session_state.last_result
        if res:
            if res["count"] == 0:
                st.info("✅ Road surface looks clear! No potholes detected in this frame.")
                st.image(res["orig"], caption="Clean Road Surface", width=400)
            else:
                st.error(f"🚨 Automatically Detected {res['count']} Pothole(s)! Severity: {res['sev']}/10 (Confidence: {int(res['conf']*100)}%)")
                ca, cb = st.columns(2)
                with ca:
                    st.image(res["ann_img"], caption="Automated YOLOv8 Bounding Boxes", use_column_width=True)
                with cb:
                    st.success(f"📍 Tagged GPS: ({res['coords'][0]}, {res['coords'][1]})")
                    st.metric("Potholes Detected", res["count"])
                    st.metric("Severity Score", f"{res['sev']} / 10")
                    st.success("✅ Broadcasted! Any other user who arrives at this location will see this pothole on their map.")

with tab_map:
    st.subheader("Live Interactive Road Surface Map & Real-Time Proximity")
    records = get_all_reports()

    # REAL-TIME PROXIMITY ALERT FOR APPROACHING USERS
    if records and user_curr_lat and user_curr_lng:
        nearby_reports = []
        for r in records:
            r_lat = pd.to_numeric(r.get('latitude'), errors='coerce')
            r_lng = pd.to_numeric(r.get('longitude'), errors='coerce')
            if pd.notna(r_lat) and pd.notna(r_lng):
                d = calculate_distance_meters(user_curr_lat, user_curr_lng, r_lat, r_lng)
                if d <= 500:  # Within 500 meters
                    nearby_reports.append((d, r))

        if nearby_reports:
            nearby_reports.sort(key=lambda x: x[0])
            closest_dist, closest_report = nearby_reports[0]
            st.error(f"⚠️ **HAZARD ALERT:** You are near **{len(nearby_reports)} reported pothole(s)**! Closest is **{closest_dist} meters away** (Severity: {closest_report.get('severity_score', 'N/A')}/10). Drive with caution!")
        else:
            st.success("✅ **Road Safe:** No reported potholes within 500 meters of your current location.")
    elif not user_curr_lat:
        st.info("💡 Enable browser location permissions to see your live **'📍 You Are Here'** pin and receive hazard alerts when approaching reported potholes.")

    if records:
        df_rep = pd.DataFrame(records)
        hotspots = compute_municipal_priorities(df_rep)
        f_map = build_map(df_rep, hotspots, user_lat=user_curr_lat, user_lng=user_curr_lng)
        st_folium(f_map, use_container_width=True, height=550)
    else:
        st.info("No reports logged yet.")

with tab_admin:
    st.subheader("High-Priority Clusters for Municipal Repair Crews")
    records = get_all_reports()
    if records:
        df_rep = pd.DataFrame(records)
        hotspots = compute_municipal_priorities(df_rep)
        if not hotspots.empty:
            st.dataframe(hotspots[['cluster_id', 'urgency', 'priority_score', 'report_count', 'total_potholes', 'avg_severity', 'latitude', 'longitude']], use_container_width=True)
            st.metric("Top Critical Cluster ID", f"Zone #{hotspots.iloc[0]['cluster_id']}")
            st.metric("Total Flagged Repair Zones", len(hotspots))
        else:
            st.info("Single pothole reports visible on the Live Map.")
