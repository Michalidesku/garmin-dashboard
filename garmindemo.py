import csv
import pandas as pd
import matplotlib.pyplot as plt
from dash import Dash, dcc, html, Input, Output
import seaborn as sns
import os
import logging
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import datetime
import sys
import json
from garminconnect import (
    Garmin,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
    GarminConnectAuthenticationError,
)

# --- CONFIGURATION ---
CSV_FILE = "garmin_activities.csv"
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

# --- Load existing data and activity IDs ---
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
            # Try to decode the string
            fixed_str = activity_type_value.replace("'", "\"")
            data = json.loads(fixed_str)
            return data.get("typeKey", "")
    except Exception as e:
        logging.warning("Failed to extract typeKey from: {activity_type_value} ({e})")
    return ""

# --- Add "activity" column to every row ---
def inject_activity_column(rows):
    updated_rows = []
    for row in rows:
        if not row.get("activity"):
            raw_value = row.get("activityType", "")
            typekey = extract_typekey(raw_value)
            row["activity"] = typekey
        updated_rows.append(row)
    return updated_rows

# --- Write combined data to CSV ---
def write_all_activities(new_activities, existing_activities):
    all_rows = inject_activity_column(new_activities + existing_activities)

    # Build all fieldnames
    all_keys = set()
    for row in all_rows:
        all_keys.update(row.keys())
    fieldnames = sorted(all_keys)

    # Move 'activity' after 'activityType'
    if "activityType" in fieldnames and "activity" in fieldnames:
        fieldnames.remove("activity")
        index = fieldnames.index("activityType") + 1
        fieldnames.insert(index, "activity")

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

# --- Export Garmin Data ---
def export_garmin_data():
    try:
        logging.info("Logging into Garmin Connect...")
        client = Garmin(EMAIL, PASSWORD)
        client.login()

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
        else:
            print("No new activities found.")

    except (
        GarminConnectConnectionError,
        GarminConnectAuthenticationError,
        GarminConnectTooManyRequestsError
    ) as e:
        logging.error(f"Garmin API error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

# --- Main ---
if __name__ == "__main__":
    print(f"\Running Garmin export at {datetime.datetime.now()}")
    export_garmin_data()

    # Load CSV
    df = pd.read_csv("garmin_activities.csv")

    # Convert startTimeLocal to datetime
    df["startTimeLocal"] = pd.to_datetime(df["startTimeLocal"], errors='coerce')

    # Convert distance to km and duration to minutes
    df["distance_km"] = pd.to_numeric(df["distance"], errors="coerce") / 1000
    df["duration_min"] = pd.to_numeric(df["duration"], errors="coerce") / 60

    # Calculate pace (min/km)
    df["pace_min_per_km"] = df["duration_min"] / df["distance_km"]

    # Filter out invalid entries
    df = df[df["distance_km"] > 0]

    # --- Plot 1: Number of Activities Over Time ---
    plt.figure(figsize=(10, 5))
    df.set_index("startTimeLocal").resample("W")["activityId"].count().plot()
    plt.title("Number of Activities Per Week")
    plt.xlabel("Week")
    plt.ylabel("Activities")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("someplot.png")
    plt.close()

    # --- Plot 2: Total Distance per Activity Type ---
    plt.figure(figsize=(8, 5))
    df.groupby("activity")["distance_km"].sum().sort_values().plot(kind="barh")
    plt.title("Total Distance by Activity Type")
    plt.xlabel("Distance (km)")
    plt.tight_layout()
    plt.savefig("someplot.png")
    plt.close()

    # --- Plot 3: Average Pace Over Time ---
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="startTimeLocal", y="pace_min_per_km", hue="activity", alpha=0.7)
    plt.title("Pace Over Time")
    plt.xlabel("Date")
    plt.ylabel("Pace (min/km)")
    plt.legend(title="Activity Type")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("someplot.png")
    plt.close()


    def generate_graphs():
        import pandas as pd
        import plotly.express as px
        from dash import Dash, dcc, html, Input, Output

        try:
            df = pd.read_csv(CSV_FILE)

            if "startTimeLocal" not in df or "distance" not in df or "duration" not in df:
                logging.warning("CSV is missing required columns for plotting.")
                return

            df["startTimeLocal"] = pd.to_datetime(df["startTimeLocal"], errors="coerce")

            # Clean and calculate additional columns
            df["distance_km"] = pd.to_numeric(df["distance"], errors="coerce") / 1000
            df["duration_min"] = pd.to_numeric(df["duration"], errors="coerce") / 60
            df["pace_min_per_km"] = df["duration_min"] / df["distance_km"]

            # Filter invalid rows
            df = df[df["distance_km"] > 0]

            sns.set(style="whitegrid")
            plt.rcParams.update({'figure.max_open_warning': 0})  # avoid warning on many plots

            # Plot 1: Activities Per Week
            plt.figure(figsize=(10, 5))
            df.set_index("startTimeLocal").resample("W")["activityId"].count().plot()
            plt.title("Number of Activities Per Week")
            plt.xlabel("Week")
            plt.ylabel("Activities")
            plt.tight_layout()
            plt.savefig("activities_per_week.png")
            plt.close()

            # Plot 2: Distance by Activity Type
            plt.figure(figsize=(8, 5))
            df.groupby("activity")["distance_km"].sum().sort_values().plot(kind="barh")
            plt.title("Total Distance by Activity Type")
            plt.xlabel("Distance (km)")
            plt.tight_layout()
            plt.savefig("distance_by_activity.png")
            plt.close()

            # Plot 3: Pace Over Time by Activity
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=df, x="startTimeLocal", y="pace_min_per_km", hue="activity", alpha=0.7)
            plt.title("Pace Over Time")
            plt.xlabel("Date")
            plt.ylabel("Pace (min/km)")
            plt.tight_layout()
            plt.savefig("pace_over_time.png")
            plt.close()
            print("Graphs generated and saved as PNG files.")

        except Exception as e:
            logging.error(f"Failed to generate graphs: {e}")
            print("Error while generating graphs. See log for details.")

            # --- Web Dashboard ---

            def generate_dashboard():
                import pandas as pd
                import plotly.express as px
                from dash import Dash, dcc, html, Input, Output

            # Generate graphs from CSV
            generate_graphs()

            # Load and clean data
            df = pd.read_csv("garmin_activities.csv")
            df["startTimeLocal"] = pd.to_datetime(df["startTimeLocal"], errors="coerce")
            df["year"] = df["startTimeLocal"].dt.year
            df["distance_km"] = pd.to_numeric(df["distance"], errors="coerce") / 1000
            df["duration_min"] = pd.to_numeric(df["duration"], errors="coerce") / 60
            df["pace_min_per_km"] = df["duration_min"] / df["distance_km"]
            df = df[df["distance_km"] > 0]
            df["activity_lower"] = df["activity"].str.lower()

            # Filter supported activities
            supported = ["running", "trail_running", "ultra_running"]
            df = df[df["activity_lower"].isin(supported)]

            if df.empty:
                print("No running activities available for dashboard.")
                return

            # App init
            app = Dash(__name__)
            app.title = "Garmin Dashboard"

            app.layout = html.Div([
                html.H1("Garmin Activities Dashboard 🏃‍♂️"),

                html.Label("Filter by Year:"),
                dcc.Dropdown(
                    options=[{"label": y, "value": y} for y in sorted(df["year"].unique())],
                    value=sorted(df["year"].unique())[-1],
                    id="year-dropdown",
                    clearable=False
                ),

                dcc.Graph(id="distance-bar"),
                dcc.Graph(id="pace-line"),
                dcc.Graph(id="activity-count")
            ])

            @app.callback(
                Output("distance-bar", "figure"),
                Output("pace-line", "figure"),
                Output("activity-count", "figure"),
                Input("year-dropdown", "value")
            )
            def update_graphs(selected_year):
                dff = df[df["year"] == selected_year]

                fig1 = px.bar(
                    dff.groupby("activity")["distance_km"].sum().reset_index(),
                    x="activity", y="distance_km",
                    title="Total Distance per Activity (km)"
                )

                fig2 = px.line(
                    dff, x="startTimeLocal", y="pace_min_per_km", color="activity",
                    title="Pace Over Time (min/km)"
                )

                fig3 = px.histogram(
                    dff, x="startTimeLocal", color="activity",
                    title="Activity Count Over Time"
                )

                return fig1, fig2, fig3

            print("Starting Garmin dashboard at http://127.0.0.1:8051 ...")
            app.run_server(debug=True)

        except Exception as e:
            logging.error(f"Error running dashboard: {e}")
            print("Failed to start dashboard.")

            if __name__ == "__main__":
                export_garmin_data()
                generate_graphs()
                generate_dashboard()
                app.run_server(debug=True, port=8051)

garmindemo = Dash(__name__)
server = app.garmindemo




