import csv
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tempfile
import shutil
import logging
import plotly.express as px
import datetime
import sys
import json
from garminconnect import (
    Garmin,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
    GarminConnectAuthenticationError,
)
import re

# --- CONFIGURATION ---
CSV_FILE = "garmin_activities.csv"
STEPS_CSV_FILE = "garmin_steps.csv"
LAPS_CSV_FILE = "garmin_activities_laps.csv"
LOG_FILE = "garmin_export.log"
FETCH_LIMIT = 100
EMAIL = os.getenv("GARMIN_EMAIL")
PASSWORD = os.getenv("GARMIN_PASSWORD")

# --- LOGGING ---
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)



# --- CSV Cleaning Helper ---
def clean_garmin_csv(input_file):
    if not os.path.exists(input_file):
        return
    with open(input_file, "rb") as f:
        raw_bytes = f.read()
    text = raw_bytes.decode("utf-8", errors="ignore")
    with open(input_file, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print(f" Cleaned {input_file} for proper UTF-8 parsing")

# --- Load existing data ---
def load_existing_data():
    if not os.path.exists(CSV_FILE):
        return [], set()
    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        existing_ids = {row["activityId"] for row in reader if "activityId" in row}
        return reader, existing_ids

# --- Extract typeKey from activityType ---
def extract_typekey(activity_type_value):
    try:
        if isinstance(activity_type_value, dict):
            return activity_type_value.get("typeKey", "")
        elif isinstance(activity_type_value, str) and activity_type_value.strip():
            fixed_str = activity_type_value.replace("'", "\"")
            data = json.loads(fixed_str)
            return data.get("typeKey", "")
    except Exception as e:
        logging.warning(f"Failed to extract typeKey from: {activity_type_value} ({e})")
    return ""

# --- Format startTimeLocal consistently ---
def format_start_time(row, key="startTimeLocal"):
    raw = row.get(key)
    if not raw:
        return ""
    try:
        dt = None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                dt = datetime.datetime.strptime(raw, fmt)
                break
            except Exception:
                continue
        if not dt:
            dt = pd.to_datetime(raw, errors="coerce")
        return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else ""
    except Exception:
        return ""

# --- Add "activity" and clean columns ---
def inject_activity_column(rows):
    updated_rows = []
    for row in rows:
        if not row.get("activity"):
            raw_value = row.get("activityType", "")
            typekey = extract_typekey(raw_value)
            row["activity"] = typekey

        # Clean elevationGain
        raw_gain = str(row.get("elevationGain", "")).strip()
        match = re.search(r"[-+]?\d*\.?\d+", raw_gain)
        if match:
            try:
                row["elevationGain"] = round(float(match.group()), 2)
            except ValueError:
                row["elevationGain"] = 0.0
        else:
            row["elevationGain"] = 0.0

        # Format startTimeLocal consistently
        row["startTimeLocal"] = format_start_time(row, "startTimeLocal")
        updated_rows.append(row)
    return updated_rows

# --- Write combined data to CSV ---
def write_all_activities(new_activities, existing_activities):
    all_rows = inject_activity_column(new_activities + existing_activities)
    all_keys = set()
    for row in all_rows:
        all_keys.update(row.keys())
    fieldnames = sorted(all_keys)

    if "activityType" in fieldnames and "activity" in fieldnames:
        fieldnames.remove("activity")
        index = fieldnames.index("activityType") + 1
        fieldnames.insert(index, "activity")

    dir_name = os.path.dirname(CSV_FILE) or "."
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=dir_name, delete=False) as tmpfile:
        writer = csv.DictWriter(tmpfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
        temp_name = tmpfile.name

    shutil.move(temp_name, CSV_FILE)
    clean_garmin_csv(CSV_FILE)

# --- Fetch laps and write to CSV ---
def write_activity_laps(client, activities):
    all_laps = []
    for act in activities:
        activity_id = act.get("activityId")
        try:
            details = client.get_activity(activity_id, include_laps=True)
            laps = details.get("laps", [])
            for i, lap in enumerate(laps, start=1):
                all_laps.append({
                    "activityId": activity_id,
                    "activityName": act.get("activityName"),
                    "activity": act.get("activity", ""),
                    "lapNumber": i,
                    "lapDistance": lap.get("distance", 0.0),
                    "lapDuration": lap.get("duration", 0.0),
                    "lapElevationGain": lap.get("elevationGain", 0.0),
                    "lapAvgHR": lap.get("averageHR", None),
                    "lapMaxHR": lap.get("maxHR", None),
                })
        except Exception as e:
            logging.warning(f"Failed to fetch laps for activity {activity_id}: {e}")

    if all_laps:
        pd.DataFrame(all_laps).to_csv(LAPS_CSV_FILE, index=False)
        print(f" {LAPS_CSV_FILE} written with {len(all_laps)} lap entries")
    else:
        print("No laps fetched.")

# --- Fetch wellness data ---
def fetch_sleep_data(client, dates):
    data = []
    for date in dates:
        sleep = client.get_sleep_data(date)
        if sleep:
            dto = sleep.get("dailySleepDTO", {})
            data.append({
                "date": date,
                "sleepTimeSeconds": dto.get("sleepTimeSeconds"),
                "napTimeSeconds": dto.get("napTimeSeconds"),
                "deepSleepSeconds": dto.get("deepSleepSeconds"),
                "lightSleepSeconds": dto.get("lightSleepSeconds"),
                "remSleepSeconds": dto.get("remSleepSeconds"),
                "awakeSleepSeconds": dto.get("awakeSleepSeconds"),
                "avgRespirationValue": dto.get("averageRespirationValue"),
                "avgSpO2": dto.get("averageSpo2Value"),
            })
    return data

def fetch_hr_data(client, dates):
    data = []
    for date in dates:
        hr = client.get_heart_rates(date)
        if hr:
            data.append({
                "date": date,
                "restingHR": hr.get("restingHeartRate"),
                "maxHR": hr.get("maxHeartRate"),
                "minHR": hr.get("minHeartRate"),
                "averageHR": hr.get("averageHeartRate"),
            })
    return data

def fetch_hill_score_data(client, dates):
    hill_scores = []
    for date_str in dates:
        try:
            raw_data = client.get_hill_score(date_str)
            row = {
                "date": date_str,
                "strengthScore": None,
                "enduranceScore": None,
                "overallScore": None,
                "hillScoreClassificationId": None,
                "hillScoreFeedbackPhraseId": None,
                "vo2Max": None,
                "vo2MaxPreciseValue": None,
            }
            if raw_data:
                row.update({
                    "date": raw_data.get("calendarDate", date_str),
                    "strengthScore": raw_data.get("strengthScore"),
                    "enduranceScore": raw_data.get("enduranceScore"),
                    "overallScore": raw_data.get("overallScore"),
                    "hillScoreClassificationId": raw_data.get("hillScoreClassificationId"),
                    "hillScoreFeedbackPhraseId": raw_data.get("hillScoreFeedbackPhraseId"),
                    "vo2Max": raw_data.get("vo2Max"),
                    "vo2MaxPreciseValue": raw_data.get("vo2MaxPreciseValue"),
                })
            hill_scores.append(row)
        except Exception as e:
            print(f" Error fetching hill score for {date_str}: {e}")
            hill_scores.append({"date": date_str})
    return hill_scores

def fetch_steps_data(client, dates):
    steps_data = []
    for date in dates:
        try:
            raw = client.get_stats(date)  # or client.get_user_summary(date), both contain totalSteps
            steps = 0
            if raw and "totalSteps" in raw:
                steps = raw["totalSteps"]

            steps_data.append({
                "date": date,
                "steps": steps
            })
        except Exception as e:
            logging.warning(f"Failed to fetch steps for {date}: {e}")
            steps_data.append({
                "date": date,
                "steps": 0
            })
    return steps_data


# --- Get missing dates ---
def get_missing_dates(filename, start_date, end_date, key="date"):
    existing_dates = set()
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        if key in df.columns:
            try:
                existing_dates = set(pd.to_datetime(df[key], errors="coerce").dt.strftime("%Y-%m-%d"))
            except Exception as e:
                print(f"⚠️ Failed to parse dates in {filename}: {e}")
    all_dates = pd.date_range(start=start_date, end=end_date).strftime("%Y-%m-%d")
    return [d for d in all_dates if d not in existing_dates]

# --- Smart daily data writer ---
def write_daily_data(filename, new_data, key="date", days_back=30):
    if os.path.exists(filename):
        df_existing = pd.read_csv(filename)
    else:
        df_existing = pd.DataFrame()

    df_new = pd.DataFrame(new_data)
    if key in df_new.columns:
        df_new[key] = pd.to_datetime(df_new[key], errors="coerce").dt.strftime("%Y-%m-%d")
    if not df_existing.empty and key in df_existing.columns:
        df_existing[key] = pd.to_datetime(df_existing[key], errors="coerce").dt.strftime("%Y-%m-%d")

    cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=days_back)).strftime("%Y-%m-%d")
    df_recent_existing = df_existing[df_existing[key] >= cutoff_date] if not df_existing.empty else pd.DataFrame()
    df_older_existing = df_existing[df_existing[key] < cutoff_date] if not df_existing.empty else pd.DataFrame()
    df_recent_merged = pd.concat([df_recent_existing, df_new], ignore_index=True)
    df_recent_merged = df_recent_merged.sort_values(by=[key]).drop_duplicates(subset=[key], keep="last")
    df_final = pd.concat([df_older_existing, df_recent_merged], ignore_index=True)
    df_final.to_csv(filename, index=False)
    print(f" {filename} updated — {len(df_new)} new/updated rows merged (last {days_back} days checked).")

# --- Export Garmin Data ---
def export_garmin_data():
    try:
        logging.info("Logging into Garmin Connect...")
        client = Garmin(EMAIL, PASSWORD)
        client.login()

        test_date = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"\n Testing steps fetch for {test_date}")

        try:
            summary = client.get_stats(test_date)
            print("get_stats:", json.dumps(summary, indent=2))
        except Exception as e:
            print("Error get_stats:", e)

        try:
            user = client.get_user_summary(test_date)
            print("get_user_summary:", json.dumps(user, indent=2))
        except Exception as e:
            print("Error get_user_summary:", e)

        # some versions also have get_daily_steps
        try:
            steps_only = client.get_daily_steps(test_date)
            print("get_daily_steps:", json.dumps(steps_only, indent=2))
        except Exception as e:
            print("Error get_daily_steps:", e)

        existing_activities, existing_ids = load_existing_data()
        new_activities = []
        start = 0

        logging.info("Fetching activities...")
        while True:
            batch = client.get_activities(start, FETCH_LIMIT)
            if not batch:
                break
            for activity in batch:
                if str(activity["activityId"]) not in existing_ids:
                    new_activities.append(activity)
                else:
                    logging.info("Reached known activity. Stopping.")
                    break
            else:
                start += FETCH_LIMIT
                continue
            break

        if new_activities:
            logging.info(f"Adding {len(new_activities)} new activities at top of CSV.")
            write_all_activities(new_activities, existing_activities)
            print(f"{len(new_activities)} new activities added at top.")

            # --- Fetch and write laps ---
            write_activity_laps(client, new_activities)
        else:
            print("No new activities found.")


        print("\nFetching daily wellness data...")
        days_back_full = 1500
        start_date = datetime.datetime.now() - datetime.timedelta(days=days_back_full)
        end_date = datetime.datetime.now()

        wellness_sources = [
            ("garmin_sleep.csv", fetch_sleep_data, "Sleep"),
            ("garmin_hr.csv", fetch_hr_data, "Heart Rate"),
            ("garmin_hill_score.csv", fetch_hill_score_data, "Hill Score"),
            (STEPS_CSV_FILE, fetch_steps_data, "Steps"),
        ]

        for filename, fetch_func, label in wellness_sources:
            print(f"\nChecking {label} data...")
            missing_dates = get_missing_dates(filename, start_date, end_date, key="date")

            if not os.path.exists(filename) or os.stat(filename).st_size == 0:
                dates_to_fetch = missing_dates
            else:
                recent_30_days = pd.date_range(end=end_date, periods=30).strftime("%Y-%m-%d")
                dates_to_fetch = sorted(set(missing_dates) | set(recent_30_days))

            if dates_to_fetch:
                print(f"{label}: fetching {len(dates_to_fetch)} days of data...")
                try:
                    data = fetch_func(client, dates_to_fetch)
                    write_daily_data(filename, data, key="date", days_back=30)
                except Exception as e:
                    print(f"Failed to fetch {label}: {e}")
            else:
                print(f"{label} data is up-to-date.")

    except (
        GarminConnectConnectionError,
        GarminConnectAuthenticationError,
        GarminConnectTooManyRequestsError
    ) as e:
        logging.error(f"Garmin API error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

def fill_missing_activity_dates(filename, start_date="2020-01-01", end_date=None):
    if not os.path.exists(filename):
        print(f"{filename} does not exist yet.")
        return

    df = pd.read_csv(filename)

    if "startTimeLocal" not in df.columns:
        print("No startTimeLocal column in activities file.")
        return

    # Parse startTimeLocal as datetime
    df["startTimeLocal"] = pd.to_datetime(df["startTimeLocal"], errors="coerce")
    df = df.dropna(subset=["startTimeLocal"])

    # Determine start and end dates
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.Timestamp.now().normalize() if end_date is None else pd.to_datetime(end_date).normalize()

    all_days = pd.date_range(start=start_date_dt, end=end_date_dt, freq="D")

    # Identify missing dates
    existing_dates = set(df["startTimeLocal"].dt.strftime("%Y-%m-%d"))
    missing_dates = [d for d in all_days if d.strftime("%Y-%m-%d") not in existing_dates]

    if not missing_dates:
        print("No new Rest Day rows to add.")
        return

    # Build Rest Day rows for missing dates
    rest_day_rows = []
    for day in missing_dates:
        row = {}
        for col in df.columns:
            if col == "startTimeLocal":
                row[col] = pd.to_datetime(f"{day.strftime('%Y-%m-%d')} 00:00:00")
            elif col == "startTimeGMT":
                # Match the same value as startTimeLocal
                row[col] = pd.to_datetime(f"{day.strftime('%Y-%m-%d')} 00:00:00")
            elif col == "activityId":
                row[col] = 0
            elif col == "activity":
                row[col] = "Rest Day"
            else:
                # Numeric columns → 0, text columns → "nothing to add"
                row[col] = 0 if df[col].dtype in [int, float] else "nothing to add"
        rest_day_rows.append(row)

    # Prepend new Rest Day rows and sort newest → oldest
    df = pd.concat([pd.DataFrame(rest_day_rows), df], ignore_index=True)
    df = df.sort_values(by="startTimeLocal", ascending=False).reset_index(drop=True)
    df.to_csv(filename, index=False)

    print(f"Added {len(rest_day_rows)} Rest Day rows from {start_date_dt.date()} to {end_date_dt.date()} (newest → oldest).")

# --- Main ---
if __name__ == "__main__":
    print(f"\nRunning Garmin export at {datetime.datetime.now()}")
    export_garmin_data()
    fill_missing_activity_dates(CSV_FILE)
