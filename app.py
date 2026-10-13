import os
import io
import uuid
import sqlite3
import datetime
import urllib.request
import math
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

# Custom CSS for Glassmorphism, Road Backgrounds, and High-Tech Badges
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

    /* Hero Banner Header with High-Tech Road Background */
    .hero-banner {
        background: linear-gradient(135deg, rgba(8, 13, 22, 0.90) 0%, rgba(13, 21, 39, 0.85) 100%),
                    url('https://images.unsplash.com/photo-1545459720-aac8509eb02c?auto=format&fit=crop&w=1200&q=80');
        background-size: cover;
        background-position: center;
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 20px;
        padding: 28px 34px;
        margin-bottom: 24px;
        backdrop-filter: blur(16px);
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.7);
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(14, 165, 233, 0.2);
        border: 1px solid rgba(56, 189, 248, 0.5);
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
        font-size: 38px;
        font-weight: 800;
        letter-spacing: -0.8px;
        margin: 0;
        background: linear-gradient(135deg, #ffffff 40%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-desc {
        color: #cbd5e1;
        font-size: 14px;
        margin-top: 6px;
        margin-bottom: 0px;
        line-height: 1.6;
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

    /* Division Feature Cards */
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
