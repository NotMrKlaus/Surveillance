import psycopg2
import json

conn = psycopg2.connect("dbname=face_db user=guri password=1004")
cur = conn.cursor()

cur.execute("""
DROP TABLE IF EXISTS faces;
CREATE TABLE faces (
    id SERIAL PRIMARY KEY,
    channel VARCHAR(50),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    embedding vector(512),
    confidence FLOAT,
    image_path VARCHAR(255)
);
""")
conn.commit()
cur.close()
conn.close()
print("Database setup complete - JSONB")