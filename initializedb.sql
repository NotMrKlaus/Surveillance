
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE people (
  person_id SERIAL PRIMARY KEY,
  embedding VECTOR(512),
  confidence FLOAT,
  image_path VARCHAR(255),
  image BYTEA, 
  count INTEGER DEFAULT 0
);

-- Faces (all detections)
CREATE TABLE faces (
  id SERIAL PRIMARY KEY,
  channel VARCHAR(50),
  embedding VECTOR(512),
  confidence FLOAT,
  image_path VARCHAR(255),
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  person_id BIGINT
);

CREATE INDEX IF NOT EXISTS people_embedding_idx 
ON people USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 200);