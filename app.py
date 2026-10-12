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

# ==================== 1. PAGE SETUP & MODERN CIVIC-TECH THEME ====================
st.set_page_config(
    page_title="RoadGuard | Automated Civic Road Intelligence",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inject Custom Styling (Dark Glassmorphism + Highway Ambient Glow)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Ambient Modern Asphalt / Smart City Radial Background */
    .stApp {
        background: radial-gradient(circle at 10% 15%, rgba(14, 165, 233, 0.12) 0%, transparent 45%),
                    radial-gradient(circle at 90% 85%, rgba(245, 158, 11, 0.08) 0%, transparent 45%),
                    linear-gradient(180deg, #080d16 0%, #0d1527 50%, #0a0f1d 100%);
        color: #f1f5f9;
    }

    /* Hero Banner Header */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 20px;
        padding: 30px 36px;
        margin-bottom: 24px;
        backdrop-filter: blur(16px);
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.1);
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
        margin-bottom: 14px;
    }

    .hero-title {
        font-size: 38px;
        font-weight: 800;
        letter-spacing: -0.8px;
        margin: 0;
        background: linear-gradient(135deg, #ffffff 30%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-desc {
        color: #94a3b8;
        font-size: 15px;
        margin-top: 8px;
        margin-bottom: 0px;
        max-width: 700px;
        line-height: 1.5;
    }

    /* Metric Glass Cards */
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

    /* Tab Custom Styling */
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
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(51, 65, 85, 0.7);
        color: #ffffff;
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

    /* Custom Alert Banners */
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
            return round(lat, 6), round(lng
