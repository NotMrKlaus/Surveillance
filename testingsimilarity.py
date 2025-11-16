import insightface
import cv2
import numpy as np
import psycopg2
import logging
import os
conn = psycopg2.connect("dbname=face_db user=guri password=1004")


app = insightface.app.FaceAnalysis(name='buffalo_l')
os.environ['ONNX_LOG_SEVERITY_LEVEL'] = '3'  # 3 = ERROR only

app.prepare(ctx_id=0, det_size=(320, 320))

def get_facial_embedding(image_path):
    img = cv2.imread(image_path)
    faces = app.get(img)
    if len(faces) == 0:
        return None
    return faces[0].normed_embedding

# Search similar faces
query_emb = get_facial_embedding('/home/guri/Desktop/Surveillance/sensei_martian_high_confidence/sensei_martian_face_(0)1763207793.jpg')
if query_emb is None:
    print("No face found")
else:
    emb_str = '[' + ','.join(map(str, query_emb.tolist())) + ']'
    cur = conn.cursor()
    cur.execute("""
    SELECT 
        person_id, 
        image_path, 
        1 - (embedding <-> %s::vector) / 2 AS similarity
    FROM people 
    ORDER BY embedding <-> %s::vector 
    LIMIT 50
""", (emb_str, emb_str))
    #cur.execute("""
    #   SELECT image_path 
    #   FROM faces 
    #   ORDER BY embedding <-> %s::vector 
    #   LIMIT 20
    #""", (emb_str,))
    results = cur.fetchall()
    cur.close()
    print(results)



