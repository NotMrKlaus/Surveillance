import sys
import os
from dotenv import load_dotenv
import psycopg2
from datetime import date, timedelta

load_dotenv()

def parse_date(text):
    text = text.strip().lower()
    today = date.today()

    if text == "today":
        return today
    if text == "yesterday":
        return today - timedelta(days=1)
    if text == "week":
        return today - timedelta(days=6)  # last 7 days including today

    # Only accept MM-DD-YY format
    try:
        month, day, year = map(int, text.split('-'))
        if year < 100:
            year += 2000  # 25 → 2025
        return date(year, month, day)
    except:
        raise ValueError(f"Invalid date: {text}. Use MM-DD-YY, today, yesterday, or week")

# === Usage ===
if len(sys.argv) not in (3, 4):
    print("Usage: python streamsearch.py <channel> <start> [end]")
    print("  Dates: MM-DD-YY  |  today  |  yesterday  |  week")
    print("Examples:")
    print("  python streamsearch.py jinnytty today")
    print("  python streamsearch.py karii 12-10-25")
    print("  python streamsearch.py fanfan 11-01-25 11-30-25")
    print("  python streamsearch.py michaaam week")
    sys.exit(1)

channel = sys.argv[1].lower()

try:
    if len(sys.argv) == 4:
        start = parse_date(sys.argv[2])
        end = parse_date(sys.argv[3])
    else:
        start = end = parse_date(sys.argv[2])
except Exception as e:
    print(e)
    sys.exit(1)

if start > end:
    start, end = end, start

print(f"\nSearching '{channel}' from {start} to {end}\n")

conn = psycopg2.connect(**{
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
})

cur = conn.cursor()
cur.execute("""
    SELECT 
        p.person_id,
        COUNT(*) AS seen,
        ROUND(CAST(AVG(f.confidence) AS NUMERIC), 3) AS conf,
        MIN(f.timestamp)::date AS first,
        MAX(f.timestamp)::date AS last
    FROM faces f
    JOIN people p ON f.person_id = p.person_id
    WHERE f.channel = %s
      AND f.timestamp::date BETWEEN %s AND %s
    GROUP BY p.person_id
    ORDER BY seen DESC, conf DESC;
""", (channel, start, end))

rows = cur.fetchall()

if not rows:
    print("No known people found.")
else:
    print(f"{'ID':>4} {'Seen':>7} {'Conf':>7} {'First':<12} {'Last'}")
    print("-" * 52)
    for pid, seen, conf, first, last in rows:
        print(f"{pid:4d} {seen:7d} {conf:7.3f} {first}    {last}")

conn.close()