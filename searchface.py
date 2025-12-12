# search_by_face.py – FINAL, beautiful, and fully correct

import sys
import os
from dotenv import load_dotenv
import psycopg2
import cv2
from insightface.app import FaceAnalysis

load_dotenv()

if len(sys.argv) < 2:
    print("Usage: python search_by_face.py <image1.jpg> [image2.png ...]")
    sys.exit(1)

images = sys.argv[1:]

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def get_face_embedding(path):
    img = cv2.imread(path)
    if img is None:
        return None
    faces = app.get(img)
    return faces[0].normed_embedding if faces else None

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)
cur = conn.cursor()

matched_ids = set()

print(f"Searching {len(images)} face(s) in database...\n")
print(f"{'Image':<35} {'Person ID':>10} {'Similarity':>12} {'Result'}")
print("─" * 78)

for img_path in images:
    emb = get_face_embedding(img_path)
    filename = os.path.basename(img_path)

    if emb is None:
        print(f"{filename:<35} {'—':>10} {'—':>12} {'No face'}")
        continue

    emb_str = '[' + ','.join(map(str, emb.tolist())) + ']'

    cur.execute("""
        SELECT person_id, embedding <-> %s::vector
        FROM people
        WHERE embedding IS NOT NULL
        ORDER BY embedding <-> %s::vector
        LIMIT 1
    """, (emb_str, emb_str))

    pid, dist = cur.fetchone()
    similarity = 1 - dist

    if dist < 0.6:
        result = "MATCH"
        matched_ids.add(pid)
    else:
        result = "No match"

    print(f"{filename:<35} {pid:>10} {similarity:>10.1%}     {result}")

cur.close()
conn.close()

print("\n" + "═" * 78)
if matched_ids:
    print("STRONG MATCHES FOUND (similarity > 40% / distance < 0.6)")
    print("   Person IDs → " + ", ".join(map(str, sorted(matched_ids))))
    print(f"   Total unique people: {len(matched_ids)}")
else:
    print("No strong matches found.")
print("═" * 78)