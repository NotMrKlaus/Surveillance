# searchpersonid.py
# Usage: python searchpersonid.py <person_id> <start_date> [end_date]
# Dates: MM-DD-YY | today | yesterday | week

import sys
import os
from dotenv import load_dotenv
import psycopg2
from datetime import date, timedelta

# Load environment variables from .env file (DB credentials)
load_dotenv()

def parse_date(text):
    """Convert user input into a proper date object."""
    text = text.strip().lower()
    today = date.today()

    # Handle keyword shortcuts
    if text == "today":
        return today
    if text == "yesterday":
        return today - timedelta(days=1)
    if text == "week":
        return today - timedelta(days=6)  # last 7 days including today

    # Handle MM-DD-YY format
    try:
        m, d, y = map(int, text.split('-'))
        # Support 2-digit years (e.g. 25 → 2025)
        y = y if y >= 100 else y + 2000
        return date(y, m, d)
    except:
        print(f"Invalid date: {text} → use MM-DD-YY, today, yesterday, week")
        sys.exit(1)

# Validate command-line arguments
if len(sys.argv) not in (3, 4):
    print("Usage: python searchpersonid.py <person_id> <start> [end]")
    print("Example:")
    print(" python searchpersonid.py 29 today")
    print(" python searchpersonid.py 7 12-01-25 12-10-25")
    print(" python searchpersonid.py 44 week")
    sys.exit(1)

# Parse person ID from first argument
try:
    person_id = int(sys.argv[1])
except:
    print("Person ID must be a number")
    sys.exit(1)

# Parse date range
start = parse_date(sys.argv[2])
end = parse_date(sys.argv[3]) if len(sys.argv) == 4 else start

# Ensure start ≤ end
if start > end:
    start, end = end, start

# Connect to PostgreSQL database using credentials from .env
conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)
cur = conn.cursor()

# Query all appearances of the given person in the date range
cur.execute("""
    SELECT
        channel,
        timestamp AT TIME ZONE 'UTC' AS ts,
        confidence
    FROM faces
    WHERE person_id = %s
      AND timestamp::date BETWEEN %s AND %s
    ORDER BY timestamp DESC
""", (person_id, start, end))

rows = cur.fetchall()
conn.close()

# Pretty output header
print(f"\nSearching appearances of Person {person_id}")
print(f"From {start} → {end}\n")
print(f"{'Channel':<20} {'Date':<12} {'Time':<8} {'Confidence':<10}")
print("─" * 60)

# Show results or no-data message
if not rows:
    print("No appearances found in this time period.")
else:
    for channel, ts, conf in rows:
        date_str = ts.strftime("%m-%d-%Y")
        time_str = ts.strftime("%H:%M:%S")
        print(f"{channel:<20} {date_str:<12} {time_str:<8} {conf:.3f}")

    print(f"\nTotal: {len(rows)} appearance(s)")