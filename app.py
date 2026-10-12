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

# ==================== 1. PAGE SETUP & THEME ====================
st.set_page_config(
    page_title="RoadGuard | Automated Civic Road Intelligence",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Glassmorphism, Clean Cards, and Image Overlays
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 10% 15%, rgba(14, 165, 233, 0.12) 0%, transparent 45%),
                    radial-gradient(circle at 90% 85%, rgba(245, 158, 11, 0.08) 0%, transparent 45%),
                    linear-gradient(180deg, #080d16 0%, #0d1527 50%, #0a0f1d 100%);
        color: #f1f5f9;
    }

    /* Hero Banner Header with Image */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 20px;
        padding: 26px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(16px);
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6);
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(14, 165, 233, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #38bdf8;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.2px;
        padding: 5px 14px;
        border-radius: 9999px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero-title {
        font-size: 36px;
        font-weight: 800;
        letter-spacing: -0.8px;
        margin: 0;
        background: linear-gradient(135deg, #ffffff 30%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-desc {
        color: #94a3b8;
        font-size: 14px;
        margin-top: 6px;
        margin-bottom: 0px;
        line-height: 1.5;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 18px 22px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.35);
    }
    .metric-val {
        font-size: 28px;
        font-weight: 800;
        color: #ffffff;
        margin-top: 4px;
    }
    .metric-lbl {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
    }

    /* Division Feature Image Cards */
    .feature-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        overflow: hidden;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
    .feature-card img {
        width: 100%;
        height: 160px;
        object-fit: cover;
        display: block;
    }
    .feature-card-content {
        padding: 16px 20px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 22px;
        color: #94a3b8;
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.05);
        font-weight: 600;
        font-size: 14px;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.2) 0%, rgba(3, 105, 161, 0.3) 100%) !important;
        border-color: rgba(56, 189, 248, 0.5) !important;
        color: #38bdf8 !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 16px rgba(14, 165, 233, 0.2);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 26px !important;
        box-shadow: 0 4px 18px rgba(2, 132, 199, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(2, 132, 199, 0.55) !important;
    }

    /* Alert Banners */
    .hazard-box {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.25) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 14px;
        padding: 16px 20px;
        color: #fca5a5;
        font-weight: 500;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(239, 68, 68, 0.15);
    }
    .safe-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.2) 100%);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 14px;
        padding: 16px 20px;
        color: #6ee7b7;
        font-weight: 500;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 2. GPS & DISTANCE UTILS ====================
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
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    return round(R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a)), 1)

# ==================== 3. AI MODEL DETECTOR ====================
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

# ==================== 4. SHARED DATABASE SETUP ====================
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

# ==================== 5. MUNICIPAL PRIORITIZATION & MAP ====================
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

    m = folium.Map(location=[center_lat, center_lng], zoom_start=14, tiles="OpenStreetMap")

    if user_lat and user_lng:
        folium.Marker(
            [user_lat, user_lng],
            popup="<b>📍 You Are Here (Current GPS Location)</b>",
            tooltip="Live User Position",
            icon=folium.Icon(color="blue", icon="user", prefix="fa")
        ).add_to(m)

        folium.Circle(
            [user_lat, user_lng],
            radius=500,
            color="#38bdf8",
            fill=True,
            fill_opacity=0.12,
            tooltip="500m Hazard Radar Radius"
        ).add_to(m)

    if reports_df is not None and not reports_df.empty:
        df = reports_df.copy()
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df['severity_score'] = pd.to_numeric(df.get('severity_score', 1.0), errors='coerce').fillna(1.0)
        df = df.dropna(subset=['latitude', 'longitude'])

        HeatMap([[float(r['latitude']), float(r['longitude']), float(r['severity_score'])] for _, r in df.iterrows()], radius=18, blur=14).add_to(m)

        cluster = MarkerCluster(name="Reported Potholes").add_to(m)
        for _, r in df.iterrows():
            dist_str = ""
            if user_lat and user_lng:
                dist = calculate_distance_meters(user_lat, user_lng, float(r['latitude']), float(r['longitude']))
                dist_str = f"<br/><b>Proximity:</b> <span style='color:#0284c7;font-weight:700;'>{dist} meters away</span>"

            popup_html = f"""
            <div style="font-family: Arial, sans-serif; width: 190px; line-height:1.5;">
                <b style="color:#ef4444;font-size:14px;">🚨 Pothole Hazard</b><br/>
                <b>Detected Count:</b> {int(r.get('pothole_count', 1))}<br/>
                <b>Severity Index:</b> {float(r['severity_score']):.1f} / 10
                {dist_str}
            </div>
            """
            folium.Marker(
                [float(r['latitude']), float(r['longitude'])],
                popup=popup_html,
                icon=folium.Icon(color="red" if float(r['severity_score']) >= 6 else "orange", icon="warning", prefix="fa")
            ).add_to(cluster)

    if hotspots_df is not None and not hotspots_df.empty:
        colors = {"CRITICAL": "#ef4444", "HIGH": "#f97316", "MEDIUM": "#eab308"}
        zones = folium.FeatureGroup(name="Priority Action Zones").add_to(m)
        for _, s in hotspots_df.iterrows():
            folium.Circle(
                [float(s['latitude']), float(s['longitude'])],
                radius=45,
                color=colors.get(s['urgency'], "#ef4444"),
                fill=True,
                fill_opacity=0.45,
                popup=f"Zone #{s['cluster_id']} [{s['urgency']}]: {s['report_count']} reports"
            ).add_to(zones)

    folium.LayerControl().add_to(m)
    return m

# ==================== 6. APP ENGINE & STATE ====================
@st.cache_resource
def load_detector():
    return AutomatedPotholeDetector()

detector = load_detector()

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ==================== 7. HERO HEADER WITH ROAD IMAGERY ====================
hero_left, hero_right = st.columns([1.6, 1.0])

with hero_left:
    st.markdown("""
    <div class="hero-banner">
        <div class="badge-pill">⚡ CIVIC ROAD INTELLIGENCE PLATFORM • v2.0</div>
        <h1 class="hero-title">RoadGuard AI</h1>
        <p class="hero-desc">Automated computer vision road defect inspection, crowdsourced hazard broadcasting, and density-based municipal repair prioritization.</p>
    </div>
    """, unsafe_allow_html=True)

with hero_right:
    st.markdown("""
    <div style="border-radius:18px;overflow:hidden;border:1px solid rgba(56,189,248,0.25);box-shadow:0 12px 32px rgba(0,0,0,0.4);height:170px;">
        <img src="https://images.unsplash.com/photo-1508873696983-2df57046475a?auto=format&fit=crop&w=700&q=80" 
             style="width:100%;height:100%;object-fit:cover;" alt="Smart Road Scanning"/>
    </div>
    """, unsafe_allow_html=True)

# Fetch Reports & Calculate Stats
all_records = get_all_reports()
total_potholes_count = sum(int(r.get('pothole_count', 1)) for r in all_records) if all_records else 0
df_stats = pd.DataFrame(all_records) if all_records else pd.DataFrame()
hotspots_stats = compute_municipal_priorities(df_stats) if not df_stats.empty else pd.DataFrame()
critical_zones_count = len(hotspots_stats[hotspots_stats['urgency'] == 'CRITICAL']) if not hotspots_stats.empty else 0

# KPI Stat Cards
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class="metric-card"><div class="metric-lbl">Total Incidents</div><div class="metric-val">{len(all_records)}</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="metric-card"><div class="metric-lbl">Potholes Tracked</div><div class="metric-val">{total_potholes_count}</div></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="metric-card"><div class="metric-lbl">Critical Zones</div><div class="metric-val" style="color:#ef4444;">{critical_zones_count}</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="metric-card"><div class="metric-lbl">Detection Engine</div><div class="metric-val" style="color:#38bdf8;font-size:20px;padding-top:6px;">YOLOv8 Active 🟢</div></div>""", unsafe_allow_html=True)

st.write("")

# Navigation Tabs
tab_upload, tab_map, tab_admin = st.tabs(["📸 Report Pothole (Auto-AI)", "🗺️ Live Map & Proximity Radar", "🏛️ Municipal Action Dispatch"])

# User Geolocation
gps = get_geolocation()
user_curr_lat, user_curr_lng = None, None
if gps and "coords" in gps and gps["coords"]:
    user_curr_lat = round(gps["coords"].get("latitude", 0.0), 6)
    user_curr_lng = round(gps["coords"].get("longitude", 0.0), 6)

# ==================== TAB 1: REPORT POTHOLE DIVISION ====================
with tab_upload:
    st.markdown("### 📷 Automated Road Defect Capture")
    st.caption("Upload or snap a road photo. The AI detects defects, calculates severity, logs GPS, and alerts drivers approaching this road segment.")

    # Educational Guide Cards with Images
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("""
        <div class="feature-card">
            <img src="https://images.unsplash.com/photo-1515162816999-a0c47dc192f7?auto=format&fit=crop&w=500&q=80" alt="Asphalt Defect"/>
            <div class="feature-card-content">
                <b style="color:#38bdf8;">1. Snap Road Surface</b><br/>
                <small style="color:#94a3b8;">Capture asphalt cavity clearly at a 2–4 meter distance.</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with g2:
        st.markdown("""
        <div class="feature-card">
            <img src="https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?auto=format&fit=crop&w=500&q=80" alt="AI Inference"/>
            <div class="feature-card-content">
                <b style="color:#38bdf8;">2. Instant YOLOv8 Inference</b><br/>
                <small style="color:#94a3b8;">Deep neural network bounds cavities and calculates severity.</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with g3:
        st.markdown("""
        <div class="feature-card">
            <img src="https://images.unsplash.com/photo-1524661135-423995f22d0b?auto=format&fit=crop&w=500&q=80" alt="GPS Broadcast"/>
            <div class="feature-card-content">
                <b style="color:#38bdf8;">3. Live Civic Broadcast</b><br/>
                <small style="color:#94a3b8;">Coordinates are broadcasted to warn nearby commuters.</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

    col_up_left, col_up_right = st.columns([1.1, 0.9])

    with col_up_left:
        img_file = st.file_uploader("Upload or Snap Road Photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if img_file is not None:
            pil_img = Image.open(img_file)
            sig = f"{img_file.name}_{img_file.size}"
            f_lat, f_lng = user_curr_lat, user_curr_lng
            if not f_lat or not f_lng:
                ex_lat, ex_lng = extract_exif_gps(pil_img)
                if ex_lat and ex_lng:
                    f_lat, f_lng = ex_lat, ex_lng
            if not f_lat or not f_lng:
                st.info("📍 Location not detected automatically. Confirm coordinates:")
                c1, c2 = st.columns(2)
                f_lat = c1.number_input("Latitude", value=28.6139, format="%.6f")
                f_lng = c2.number_input("Longitude", value=77.2090, format="%.6f")

            if sig not in st.session_state.processed_files:
                with st.spinner("🤖 Running automated YOLOv8 road inspection..."):
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

    with col_up_right:
        res = st.session_state.last_result
        if res:
            if res["count"] == 0:
                st.success("✅ **Clear Surface:** No potholes detected in this frame.")
                st.image(res["orig"], caption="Clean Road Surface", use_column_width=True)
            else:
                st.error(f"🚨 **Detected {res['count']} Pothole(s)** | Severity Index: **{res['sev']}/10**")
                st.image(res["ann_img"], caption="YOLOv8 Automated Bounding Box Analysis", use_column_width=True)
                st.markdown(f"""
                <div style="background:rgba(30,41,59,0.7);padding:14px;border-radius:12px;border:1px solid rgba(56,189,248,0.2);">
                    <b>📍 Tagged Coordinates:</b> {res['coords'][0]}, {res['coords'][1]}<br/>
                    <b>🎯 AI Confidence Score:</b> {int(res['conf']*100)}%<br/>
                    <b>📡 Status:</b> Broadcasted to public hazard map
                </div>
                """, unsafe_allow_html=True)

# ==================== TAB 2: LIVE MAP & PROXIMITY DIVISION ====================
with tab_map:
    st.markdown("### 🗺️ Live Hazard Map & Real-Time Proximity Radar")
    st.caption("Inspect live road condition heatmaps. Approaching drivers are dynamically warned within 500 meters of reported potholes.")

    map_info_left, map_info_right = st.columns([1.8, 1.2])
    with map_info_left:
        records = get_all_reports()

        # Real-Time Proximity Alert Banner
        if records and user_curr_lat and user_curr_lng:
            nearby_reports = []
            for r in records:
                r_lat = pd.to_numeric(r.get('latitude'), errors='coerce')
                r_lng = pd.to_numeric(r.get('longitude'), errors='coerce')
                if pd.notna(r_lat) and pd.notna(r_lng):
                    d = calculate_distance_meters(user_curr_lat, user_curr_lng, r_lat, r_lng)
                    if d <= 500:
                        nearby_reports.append((d, r))

            if nearby_reports:
                nearby_reports.sort(key=lambda x: x[0])
                closest_dist, closest_report = nearby_reports[0]
                st.markdown(f"""
                <div class="hazard-box">
                    ⚠️ <b>HAZARD ALERT:</b> You are approaching <b>{len(nearby_reports)} reported pothole(s)</b>! 
                    Closest defect is <b>{closest_dist} meters ahead</b> (Severity: {closest_report.get('severity_score', 'N/A')}/10). Drive with caution!
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="safe-box">
                    🛡️ <b>Radar Clear:</b> No reported potholes within 500 meters of your current GPS location.
                </div>
                """, unsafe_allow_html=True)
        elif not user_curr_lat:
            st.info("💡 Grant browser location permissions to see your live **'📍 You Are Here'** blue radar pin and receive real-time hazard alerts.")

    with map_info_right:
        st.markdown("""
        <div style="background:rgba(30,41,59,0.5);border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:12px 18px;display:flex;align-items:center;gap:14px;">
            <img src="https://images.unsplash.com/photo-1569336415962-a4bd9f69cd83?auto=format&fit=crop&w=150&q=80" 
                 style="width:55px;height:55px;border-radius:10px;object-fit:cover;" alt="GPS Radar"/>
            <div>
                <b style="color:#38bdf8;font-size:13px;">Spherical Proximity Telemetry</b><br/>
                <small style="color:#94a3b8;">Haversine formula tracking hazards with 500m geofence.</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if records:
        df_rep = pd.DataFrame(records)
        hotspots = compute_municipal_priorities(df_rep)
        f_map = build_map(df_rep, hotspots, user_lat=user_curr_lat, user_lng=user_curr_lng)
        st_folium(f_map, use_container_width=True, height=580)
    else:
        st.info("No reports logged yet.")

# ==================== TAB 3: MUNICIPAL ACTION DIVISION ====================
with tab_admin:
    st.markdown("### 🏛️ Municipal Priority Action Zones")
    st.caption("Density-Based Spatial Clustering (DBSCAN) automatically groups nearby defects into ranked repair work orders.")

    muni_col1, muni_col2 = st.columns([1.7, 1.3])

    with muni_col2:
        st.markdown("""
        <div style="border-radius:16px;overflow:hidden;border:1px solid rgba(255,255,255,0.1);margin-bottom:15px;box-shadow:0 8px 24px rgba(0,0,0,0.3);">
            <img src="https://images.unsplash.com/photo-1504307651254-35680f356dfd?auto=format&fit=crop&w=700&q=80" 
                 style="width:100%;height:140px;object-fit:cover;" alt="Municipal Road Repair"/>
            <div style="background:rgba(30,41,59,0.8);padding:14px 18px;">
                <b style="color:#38bdf8;font-size:13px;">🚜 Civil Engineering Repair Pipeline</b><br/>
                <small style="color:#94a3b8;">High-priority density clusters automatically flag municipal asphalt resurfacing units.</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with muni_col1:
        records = get_all_reports()
        if records:
            df_rep = pd.DataFrame(records)
            hotspots = compute_municipal_priorities(df_rep)
            if not hotspots.empty:
                st.dataframe(
                    hotspots[['cluster_id', 'urgency', 'priority_score', 'report_count', 'total_potholes', 'avg_severity', 'latitude', 'longitude']],
                    use_container_width=True
                )
                st.markdown(f"""
                <div style="background:rgba(30,41,59,0.7);padding:16px;border-radius:12px;border:1px solid rgba(239,68,68,0.3);margin-top:10px;">
                    <b>🚨 Top Ranked Dispatch Zone:</b> Zone #{hotspots.iloc[0]['cluster_id']} (Priority Score: {hotspots.iloc[0]['priority_score']})<br/>
                    <b>👷 Municipal Recommendation:</b> Dispatch rapid asphalt paving unit to coordinates ({hotspots.iloc[0]['latitude']:.4f}, {hotspots.iloc[0]['longitude']:.4f})
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No dense multi-pothole clusters yet. Single reports are plotted on the Live Map.")
