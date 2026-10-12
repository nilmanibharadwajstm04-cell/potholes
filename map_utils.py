import os
import base64
import folium
from folium.plugins import MarkerCluster, HeatMap

def get_image_html_src(image_path: str) -> str:
    """Converts a local path or web URL into a safe HTML img src attribute."""
    if not image_path:
        return ""
    if str(image_path).startswith("http://") or str(image_path).startswith("https://"):
        return str(image_path)
    if os.path.exists(str(image_path)):
        try:
            with open(str(image_path), "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded}"
        except Exception:
            return ""
    return ""

def build_map(reports_df, hotspots_df=None):
    """
    Renders an interactive Folium map with:
    1. Heatmap layer
    2. Marker cluster with popup details & images
    3. Municipal priority zones (circles with color codes)
    """
    if reports_df.empty:
        # Default fallback view
        return folium.Map(location=[28.6139, 77.2090], zoom_start=11, tiles="CartoDB positron")

    center_lat = float(reports_df['latitude'].mean())
    center_lng = float(reports_df['longitude'].mean())
    m = folium.Map(location=[center_lat, center_lng], zoom_start=14, tiles="CartoDB positron")

    # Layer 1: Density Heatmap
    heat_data = [[float(row['latitude']), float(row['longitude']), float(row.get('severity_score', 1.0))] for _, row in reports_df.iterrows()]
    HeatMap(heat_data, radius=18, blur=14, name="Pothole Heatmap").add_to(m)

    # Layer 2: Individual Marker Cluster
    cluster = MarkerCluster(name="Individual Reports").add_to(m)
    for _, row in reports_df.iterrows():
        raw_img = row.get("image_url", "")
        img_src = get_image_html_src(raw_img)
        img_tag = f'<img src="{img_src}" style="width:100%; max-height:140px; object-fit:cover; border-radius:4px; margin-bottom:6px;"/>' if img_src else ""
        
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; width: 200px;">
            {img_tag}
            <b>Potholes Detected:</b> {int(row.get('pothole_count', 1))}<br/>
            <b>Severity:</b> {float(row.get('severity_score', 1.0)):.1f} / 10<br/>
            <b>Status:</b> {str(row.get('status', 'pending')).title()}<br/>
            <small>{str(row.get('created_at', ''))[:16]}</small>
        </div>
        """
        sev = float(row.get('severity_score', 1.0))
        color = "red" if sev >= 6.0 else "orange"
        folium.Marker(
            location=[float(row['latitude']), float(row['longitude'])],
            popup=folium.Popup(popup_html, max_width=240),
            icon=folium.Icon(color=color, icon="warning", prefix="fa")
        ).add_to(cluster)

    # Layer 3: High-Priority Municipal Action Zones
    if hotspots_df is not None and not hotspots_df.empty:
        colors = {"CRITICAL": "#d63031", "HIGH": "#e17055", "MEDIUM": "#fdcb6e", "LOW": "#0984e3"}
        zone_layer = folium.FeatureGroup(name="Municipal Priority Zones").add_to(m)
        for _, spot in hotspots_df.iterrows():
            urg = str(spot.get('urgency', 'MEDIUM'))
            folium.Circle(
                location=[float(spot['latitude']), float(spot['longitude'])],
                radius=45,
                color=colors.get(urg, "#d63031"),
                fill=True,
                fill_opacity=0.45,
                popup=f"Zone #{spot.get('cluster_id', 0)} [{urg}]: {spot.get('report_count', 0)} reports, Score: {spot.get('priority_score', 0)}"
            ).add_to(zone_layer)

    folium.LayerControl().add_to(m)
    return m
