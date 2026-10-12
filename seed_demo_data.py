import sqlite3
import datetime
import uuid

# Seeds 8 realistic pothole coordinates clustered around an area
SAMPLE_COORDS = [
    (28.6139, 77.2090, 3, 0.88, 7.5),
    (28.6141, 77.2092, 2, 0.92, 6.2),
    (28.6138, 77.2089, 4, 0.85, 8.8),
    (28.6143, 77.2095, 2, 0.79, 5.5),
    (28.6300, 77.2200, 1, 0.91, 3.2),
    (28.6302, 77.2204, 2, 0.84, 4.5),
    (28.6500, 77.1900, 1, 0.75, 2.8),
]

def seed():
    conn = sqlite3.connect("reports_local.db")
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
    for lat, lng, count, conf, sev in SAMPLE_COORDS:
        c.execute('''
            INSERT INTO reports (id, created_at, image_url, latitude, longitude, pothole_count, confidence_avg, severity_score, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (str(uuid.uuid4()), datetime.datetime.now().isoformat(), "https://images.unsplash.com/photo-1515162816999-a0c47dc192f7?w=500", lat, lng, count, conf, sev, 'pending'))
    conn.commit()
    conn.close()
    print("Demo data seeded successfully into reports_local.db!")

if __name__ == "__main__":
    seed()
