import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

def compute_municipal_priorities(reports_df: pd.DataFrame, radius_meters: float = 40.0):
    """
    Groups reports within `radius_meters` into high-risk clusters using DBSCAN with Haversine distance.
    Returns a DataFrame of prioritized zones.
    """
    if reports_df is None or reports_df.empty:
        return pd.DataFrame()

    df = reports_df.copy()
    
    # Ensure types are numeric and valid
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude'])

    if len(df) < 2:
        return pd.DataFrame()

    df['pothole_count'] = pd.to_numeric(df.get('pothole_count', 1), errors='coerce').fillna(1)
    df['severity_score'] = pd.to_numeric(df.get('severity_score', 1.0), errors='coerce').fillna(1.0)

    # Scikit-learn haversine requires [latitude, longitude] in radians
    coords_rad = np.radians(df[['latitude', 'longitude']].values)
    epsilon = radius_meters / 6371000.0  # Earth radius in meters

    db = DBSCAN(eps=epsilon, min_samples=2, metric='haversine')
    df['cluster_id'] = db.fit_predict(coords_rad)

    # Filter out unclustered individual points (-1)
    clustered = df[df['cluster_id'] != -1]
    if clustered.empty:
        return pd.DataFrame()

    hotspots = []
    for cluster_id, group in clustered.groupby('cluster_id'):
        report_count = len(group)
        total_potholes = int(group['pothole_count'].sum())
        avg_severity = float(group['severity_score'].mean())

        # Priority metric calculation
        priority_score = (report_count * 3.0) + (total_potholes * 2.0) + (avg_severity * 2.5)

        urgency = "LOW"
        if priority_score >= 30.0:
            urgency = "CRITICAL"
        elif priority_score >= 15.0:
            urgency = "HIGH"
        elif priority_score >= 8.0:
            urgency = "MEDIUM"

        first_img = group['image_url'].dropna().iloc[0] if 'image_url' in group.columns and not group['image_url'].dropna().empty else ""

        hotspots.append({
            "cluster_id": int(cluster_id),
            "latitude": float(group['latitude'].mean()),
            "longitude": float(group['longitude'].mean()),
            "report_count": report_count,
            "total_potholes": total_potholes,
            "avg_severity": round(avg_severity, 2),
            "priority_score": round(priority_score, 1),
            "urgency": urgency,
            "sample_image": first_img
        })

    return pd.DataFrame(hotspots).sort_values(by="priority_score", ascending=False).reset_index(drop=True)
