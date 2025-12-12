# save_person_images.py
# Usage:
#   python save_person_images.py <folder> 29 7 44
#   python save_person_images.py ./suspects 1 5-15 29
#   python save_person_images.py /tmp/faces all

import sys
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
    print("Usage: python save_person_images.py <output_folder> <id1> [id2 id3 range ...] [all]")
    print("Examples:")
    print("  python save_person_images.py ./matches 29 7 44")
    print("  python save_person_images.py ./found 1 10-20 35")
    print("  python save_person_images.py /tmp/faces all")
    sys.exit(1)

output_folder = sys.argv[1].rstrip("/")
os.makedirs(output_folder, exist_ok=True)

# Parse IDs from remaining args
ids_to_save = set()
has_all = False

for arg in sys.argv[2:]:
    arg = arg.strip()
    if arg.lower() == "all":
        has_all = True
        break
    elif "-" in arg and not arg.startswith("-"):
        try:
            start, end = map(int, arg.split("-"))
            ids_to_save.update(range(start, end + 1))
        except:
            print(f"Invalid range: {arg}")
            sys.exit(1)
    elif arg.isdigit():
        ids_to_save.add(int(arg))

# Connect
conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)
cur = conn.cursor()

print(f"Saving to: {os.path.abspath(output_folder)}\n")

if has_all:
    print("Exporting ALL people with images...")
    cur.execute("SELECT person_id, image FROM people WHERE image IS NOT NULL")
else:
    if not ids_to_save:
        print("No person IDs provided.")
        sys.exit(1)
    placeholders = ",".join(["%s"] * len(ids_to_save))
    cur.execute(f"""
        SELECT person_id, image 
        FROM people 
        WHERE person_id IN ({placeholders}) AND image IS NOT NULL
        ORDER BY person_id
    """, tuple(sorted(ids_to_save)))

rows = cur.fetchall()
conn.close()

if not rows:
    print("No images found for the given person IDs.")
    sys.exit(0)

saved = 0
for person_id, img_data in rows:
    person_dir = os.path.join(output_folder, str(person_id))
    os.makedirs(person_dir, exist_ok=True)
    path = os.path.join(person_dir, f"person_{person_id}.jpg")

    with open(path, "wb") as f:
        f.write(img_data)

    print(f"Saved → {path}")
    saved += 1

print(f"\nDone! {saved} image(s) saved into:")
print(f"   {os.path.abspath(output_folder)}/")