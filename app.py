def build_map(reports_df, hotspots_df=None):
    if reports_df is None or len(reports_df) == 0:
        return folium.Map(location=[28.6139, 77.2090], zoom_start=11, tiles="OpenStreetMap")
    
    df = reports_df.copy()
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['severity_score'] = pd.to_numeric(df.get('severity_score', 1.0), errors='coerce').fillna(1.0)
    df = df.dropna(subset=['latitude', 'longitude'])
    
    if df.empty:
        return folium.Map(location=[28.6139, 77.2090], zoom_start=11, tiles="OpenStreetMap")

    # Center map on the reported potholes
    m = folium.Map(
        location=[float(df['latitude'].mean()), float(df['longitude'].mean())],
        zoom_start=14,
        tiles="OpenStreetMap"  # <-- 100% free, NO API key required
    )

    # 1. Pothole Heatmap Layer
    heat_data = [[float(r['latitude']), float(r['longitude']), float(r['severity_score'])] for _, r in df.iterrows()]
    HeatMap(heat_data, radius=18, blur=14).add_to(m)

    # 2. Individual Pothole Markers
    cluster = MarkerCluster(name="Individual Reports").add_to(m)
    for _, r in df.iterrows():
        folium.Marker(
            [float(r['latitude']), float(r['longitude'])],
            popup=f"Potholes: {int(r.get('pothole_count', 1))} | Severity: {float(r['severity_score']):.1f}/10",
            icon=folium.Icon(color="red" if float(r['severity_score']) >= 6 else "orange", icon="warning", prefix="fa")
        ).add_to(cluster)

    # 3. High-Priority Municipal Zones
    if hotspots_df is not None and not hotspots_df.empty:
        colors = {"CRITICAL": "#d63031", "HIGH": "#e17055", "MEDIUM": "#fdcb6e"}
        zones = folium.FeatureGroup(name="Priority Zones").add_to(m)
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
