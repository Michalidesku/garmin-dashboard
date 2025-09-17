import pandas as pd
import re
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import datetime
import logging
import dash_mantine_components as dmc
import plotly.graph_objects as go

hr_zone_cols = [f"hrTimeInZone_{i}" for i in range(1, 6)]
power_zone_cols = [f"powerTimeInZone_{i}" for i in range(1, 6)]
sleep_zone_cols = [f"SleepTimeInZone_{i}" for i in range(1, 6)]

# --- Constants ---
CSV_FILE = "C:/Users/skriv/Desktop/Garmin-demo/garmin_activities.csv"
HR_CSV_FILE = "C:/Users/skriv/Desktop/Garmin-demo/garmin_hr.csv"
SLEEP_CSV_FILE = "C:/Users/skriv/Desktop/Garmin-demo/garmin_sleep.csv"
HILL_CSV_FILE = "C:/Users/skriv/Desktop/Garmin-demo/garmin_hill_score.csv"
STEP_CSV_FILE ="C:/Users/skriv/Desktop/Garmin-demo/garmin_steps.csv"
APP_PORT = 8051


# --- Load and preprocess data ---
def load_data():
    try:
        with open(CSV_FILE, "rb") as f:  # force fresh read of raw bytes
            df = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)

            # Clean all string columns
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip()  # Remove leading/trailing spaces
            df[col] = df[col].apply(lambda x: re.sub(r'[\x00-\x1F\x7F\xa0]', '', x))  # Remove invisible chars

        if "startTimeLocal" in df.columns:
            df["startTimeLocal"] = pd.to_datetime(df["startTimeLocal"].astype(str), errors="coerce",
                                                  infer_datetime_format=True)

            df = df.dropna(subset=["startTimeLocal"])

            df["month"] = df["startTimeLocal"].dt.month
            df["year"] = df["startTimeLocal"].dt.isocalendar().year
            df["day"] = df["startTimeLocal"].dt.isocalendar().day
            df["week"] = df["startTimeLocal"].dt.isocalendar().week
        if "distance" in df.columns:
            df["distance"] = pd.to_numeric(df["distance"], errors="coerce") / 1000  # meters to km
        else:
            df["distance"] = 0.0
        if "activity" in df.columns:
            df["activity"] = df["activity"].astype(str)
        else:
            df["activity"] = "Unknown"
        if "duration" in df.columns:
            df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0)
            df["duration_hours"] = df["duration"] / 3600  # convert seconds → hours
        else:
            df["duration_hours"] = 0.0

        if "vO2MaxValue" not in df.columns:
            df["vO2MaxValue"] = None
        else:
            df["vO2MaxValue"] = pd.to_numeric(df["vO2MaxValue"], errors="coerce")
        if "elevationGain" not in df.columns:
            df["elevationGain"] = 0
        else:
            df["elevationGain"] = pd.to_numeric(df["elevationGain"], errors="coerce").fillna(0)

        if "steps" in df.columns:
            df["steps"] = pd.to_numeric(df["steps"], errors="coerce").fillna(0)
        else:
            df["steps"] = 0

            for col in hr_zone_cols:
                if col not in df.columns:
                    df[col] = 0  # or NaN if you prefer
                else:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            for col in power_zone_cols:
                if col not in df.columns:
                    df[col] = 0  # or NaN if you prefer
                else:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            for col in sleep_zone_cols:
                if col not in df.columns:
                    df[col] = 0  # or NaN if you prefer
                else:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df

    except Exception as e:
        print(f"Error loading CSV: {e}")
        return pd.DataFrame()


def load_hr_data():
    try:
        hr_daily_df = pd.read_csv(HR_CSV_FILE)
        if "date" in hr_daily_df.columns:
            hr_daily_df["date"] = pd.to_datetime(hr_daily_df["date"], errors="coerce")
        # Make sure HR columns are numeric
        hr_cols = ["restingHR", "maxHR", "minHR", "averageHR"]
        for col in hr_cols:
            if col in hr_daily_df.columns:
                hr_daily_df[col] = pd.to_numeric(hr_daily_df[col], errors="coerce")
        hr_daily_df = hr_daily_df.dropna(subset=["date"]).sort_values("date")
        return hr_daily_df

    except Exception as e:
        print(f"Error loading HR CSV: {e}")
        return pd.DataFrame()


hr_daily_df = load_hr_data()


def load_sleep_data():
    try:
        hr_sleep_df = pd.read_csv(SLEEP_CSV_FILE)
        if "date" in hr_sleep_df.columns:
            hr_sleep_df["date"] = pd.to_datetime(hr_sleep_df["date"], errors="coerce")
        # Make sure Sleep columns are numeric
        sleep_cols = ["sleepTimeSeconds", "deepSleepSeconds", "lightSleepSeconds", "remSleepSeconds",
                      "awakeSleepSeconds", "avgRespirationValue", "avgSpO2"]
        for col in sleep_cols:
            if col in hr_sleep_df.columns:
                hr_sleep_df[col] = pd.to_numeric(hr_sleep_df[col], errors="coerce")
        hr_sleep_df = hr_sleep_df.dropna(subset=["date"]).sort_values("date")
        return hr_sleep_df

    except Exception as e:
        print(f"Error loading SLEEP CSV: {e}")
        return pd.DataFrame()


def load_hill_data():
    try:
        hill_df = pd.read_csv(HILL_CSV_FILE)
        if "date" in hill_df.columns:
            hill_df["date"] = pd.to_datetime(hill_df["date"], errors="coerce")
        if "date" in hill_df.columns:
            hill_df["date"] = pd.to_datetime(hill_df["date"], errors="coerce")
        score_cols = ["strengthScore", "enduranceScore", "overallScore"]
        for col in score_cols:
            if col in hill_df.columns:
                hill_df[col] = pd.to_numeric(hill_df[col], errors="coerce")
        hill_df = hill_df.dropna(subset=["date"]).sort_values("date")
        return hill_df
    except Exception as e:
        print(f"Error loading HILL CSV: {e}")
        return pd.DataFrame()

def load_steps_file():
    try:
        steps_df = pd.read_csv(STEP_CSV_FILE)
        steps_df["date"] = pd.to_datetime(steps_df["date"], errors="coerce")
        steps_df["steps"] = pd.to_numeric(steps_df["steps"], errors="coerce").fillna(0)
        steps_df = steps_df.dropna(subset=["date"]).sort_values("date")
        steps_df["year"] = steps_df["date"].dt.year
        steps_df["month"] = steps_df["date"].dt.month
        steps_df["week"] = steps_df["date"].dt.isocalendar().week
        steps_df["month_year"] = steps_df["month"].apply(lambda x: f"{x:02d}") + "-" + steps_df["year"].astype(str)
        steps_df["year_week"] = "W" + steps_df["week"].astype(str).str.zfill(2) + "-" + steps_df["year"].astype(str)
        return steps_df
    except Exception as e:
        print(f"Error loading STEPS CSV: {e}")
        return pd.DataFrame()


hill_df = load_hill_data()
hr_sleep_df = load_sleep_data()
hr_df = load_hr_data()
df = load_data()
steps_file_df = load_steps_file()

activity_options = df["activity"].dropna().unique()
elevationGain_options = df["elevationGain"].dropna().unique()
year_options = sorted(df["year"].dropna().unique().astype(int))
month_options = sorted(df["month"].dropna().unique().astype(int))
week_options = sorted(df["week"].dropna().unique().astype(int))
day_options = sorted(df["day"].dropna().unique().astype(int))

app = Dash(__name__)
app.layout = html.Div([dcc.Interval(
    id="interval-component",
    interval=60 * 1000,  # 10 seconds in milliseconds
    n_intervals=0
),
    html.H1("Garmin Activities Dashboard"),

    html.Label("Select Activity:"),
    dcc.Dropdown(
        id="activity-dropdown",
        options=[{"label": a.title(), "value": a} for a in activity_options],
        value=[a for a in activity_options],
        multi=True,
        placeholder="Select activity/activities",
        style={"width": "300px"}
    ),

    html.Label("Select Year:"),
    dcc.Dropdown(
        id="year-dropdown",
        options=[{"label": str(y), "value": y} for y in year_options],
        value=[],
        multi=True,
        placeholder="Select year(s)",
        style={"width": "300px"}
    ),

    html.Label("Select Month:"),
    dcc.Dropdown(
        id="month-dropdown",
        options=[{"label": str(m), "value": m} for m in month_options],
        value=[],
        multi=True,
        placeholder="Select month(s)",
        style={"width": "300px"}
    ),

    html.Label("Select Week:"),
    dcc.Dropdown(
        id="week-dropdown",
        options=[{"label": str(w), "value": w} for w in week_options],
        value=[],
        multi=True,
        placeholder="Select week(s)",
        style={"width": "300px"}
    ),

    html.Label("Select Day:"),
    dcc.Dropdown(
        id="day-dropdown",
        options=[{"label": str(d), "value": d} for d in day_options],
        value=[],
        multi=True,
        placeholder="Select day(s)",
        style={"width": "300px"}
    ),

    html.Label("Select Date Range:"),
    dcc.DatePickerRange(
        id="date-range-picker",
        min_date_allowed=df["startTimeLocal"].min().date(),
        max_date_allowed=df["startTimeLocal"].max().date(),
        start_date=df["startTimeLocal"].min().date(),
        end_date=df["startTimeLocal"].max().date(),
        display_format="DD.MM.YYYY",
        style={"margin-bottom": "20px"},
    ),




    html.Label("X-axis:"),
    dcc.Dropdown(
        id="xaxis-dropdown",
        options=[
            {"label": "Distance (km)", "value": "distance"},
            {"label": "Duration (hours)", "value": "duration_hours"},
        ],
        value="distance",
        clearable=False,
        style={"width": "200px"},
    ),

    html.Label("Select Activity ID:"),
    dcc.Dropdown(
        id="activity-id-dropdown",
        options=[{"label": f"{row['activityId']} ({row['activity']})", "value": row['activityId']}
                 for _, row in df.iterrows()],
        value=[],
        multi=True,
        placeholder="Select specific activity IDs",
        style={"width": "400px"}
    ),

    html.Label("Filter by Distance (km)"),
    dcc.RangeSlider(
        id="distance-slider",
        min=df["distance"].min(),
        max=df["distance"].max(),
        step=0.1,
        value=[df["distance"].min(), df["distance"].max()],
        tooltip={"placement": "bottom", "always_visible": True},
    ),

    html.Label("Filter by Elevation Gain (m)"),
    dcc.RangeSlider(
        id="elevation-slider",
        min=df["elevationGain"].min(),
        max=df["elevationGain"].max(),
        step=1,
        value=[df["elevationGain"].min(), df["elevationGain"].max()],
        tooltip={"placement": "bottom", "always_visible": True},
    ),

    dcc.Graph(id="yearly-duration-by-activity"),
    dcc.Graph(id="yearly-distance-by-activity"),
    dcc.Graph(id="weekly-duration-elevation"),
    dcc.Graph(id="monthly-duration-elevation"),
    dcc.Graph(id="distance-over-week"),
    dcc.Graph(id="monthly-distance-elevation"),
    dcc.Graph(id="total-distance-by-activity"),
    dcc.Graph(id="ytd-distance-by-activity"),
    dcc.Graph(id="ytd-elevation-by-activity"),
    dcc.Graph(id="vo2max-over-time"),
    dcc.Graph(id="hr-zone-summary"),
    dcc.Graph(id="power-zone-summary"),
    dcc.Graph(id="hr-daily-summary"),
    dcc.Graph(id="sleep-daily-summary"),
    dcc.Graph(id="hill-score-summary"),
    dcc.Graph(id="activity-comparison-graph"),
dcc.Graph(id="yearly-steps-by-activity"),
dcc.Graph(id="monthly-steps-by-activity"),
dcc.Graph(id="weekly-steps-by-activity"),
dcc.Graph(id="ytd-steps-all"),
dcc.Graph(id="yearly-steps-file"),
dcc.Graph(id="monthly-steps-file"),
dcc.Graph(id="weekly-steps-file"),


])
app.title = "Garmin Activities Dashboard"


@app.callback(
    Output("activity-comparison-graph", "figure"),
    Output("yearly-duration-by-activity", "figure"),
    Output("yearly-distance-by-activity", "figure"),
    Output("weekly-duration-elevation", "figure"),
    Output("monthly-duration-elevation", "figure"),
    Output("distance-over-week", "figure"),
    Output("monthly-distance-elevation", "figure"),
    Output("total-distance-by-activity", "figure"),
    Output("ytd-distance-by-activity", "figure"),
    Output("ytd-elevation-by-activity", "figure"),
    Output("vo2max-over-time", "figure"),
    Output("hr-zone-summary", "figure"),
    Output("power-zone-summary", "figure"),
    Output("hr-daily-summary", "figure"),
    Output("sleep-daily-summary", "figure"),
    Output("hill-score-summary", "figure"),
    Output("yearly-steps-by-activity", "figure"),
    Output("monthly-steps-by-activity", "figure"),
    Output("weekly-steps-by-activity", "figure"),
Output("ytd-steps-all", "figure"),
Output("yearly-steps-file", "figure"),
    Output("monthly-steps-file", "figure"),
    Output("weekly-steps-file", "figure"),
    Input("activity-dropdown", "value"),
    Input("year-dropdown", "value"),
    Input("month-dropdown", "value"),
    Input("week-dropdown", "value"),
    Input("day-dropdown", "value"),
    Input("date-range-picker", "start_date"),
    Input("date-range-picker", "end_date"),
    Input("interval-component", "n_intervals"),
    Input("xaxis-dropdown", "value")

)
def update_graphs(selected_activities, selected_years, selected_months, selected_weeks, selected_days, start_date,
                  end_date, n_intervals, xaxis_dropdown, ):
    df = load_data()
    hr_sleep_df = load_sleep_data()
    hr_daily_df = load_hr_data()
    hr_filtered = hr_daily_df.copy()
    sleep_filtered = hr_sleep_df.copy()
    hill_filtered = hill_df.copy()

    # --- Initialize step figures to empty plots ---
    yearly_steps_fig = px.bar(title="No Data for Yearly Steps")
    monthly_steps_fig = px.bar(title="No Data for Monthly Steps")
    weekly_steps_fig = px.bar(title="No Data for Weekly Steps")

    if df.empty:
        empty_fig = px.bar(title="No Data")
        return px.bar(title="No Data"), px.bar(title="No Data"), px.bar(title="No Data"), px.bar(
            title="No Data"), px.bar(title="No Data"), px.bar(title="No Data"), px.bar(title="No Data"), px.bar(
            title="No Data"), px.bar(title="No Data"), px.bar(title="No Data"), px.bar(title="No Data"), px.bar(
            title="No Data"), px.bar(title="No Data"), px.bar(title="No Data"), px.bar(title="No Data"), px.bar(title="No Data"), px.bar(title="No Data"), px.bar(title="No Data"), px.bar(title="No Data"), px.bar(title="No Data"), px.bar(title="No Data"), px.bar(title="No Data"),

        # Default to all activities if none selected
    if not selected_activities:
        selected_activities = df["activity"].dropna().unique().tolist()

        # --- Filter data ---
    filtered_df = df.dropna(subset=["startTimeLocal", "distance", "activity", "elevationGain"]).copy()
    filtered_df.set_index("startTimeLocal", inplace=True)

    if selected_activities:
        filtered_df = filtered_df[filtered_df["activity"].isin(selected_activities)]

    if start_date and end_date:
        start_date = pd.to_datetime(start_date).date()
        end_date = pd.to_datetime(end_date).date()
        filtered_df = filtered_df[(filtered_df.index.date >= start_date) & (filtered_df.index.date <= end_date)]
    if selected_years:
        filtered_df = filtered_df[filtered_df.index.year.isin(selected_years)]
    if selected_months:
        filtered_df = filtered_df[filtered_df.index.month.isin(selected_months)]
    if selected_weeks:
        filtered_df = filtered_df[filtered_df.index.isocalendar().week.isin(selected_weeks)]
    if selected_days:
        filtered_df = filtered_df[filtered_df.index.isocalendar().day.isin(selected_days)]

    if start_date and end_date:
        start_date = pd.to_datetime(start_date).date()
        end_date = pd.to_datetime(end_date).date()
        hr_filtered = hr_filtered[
            (hr_filtered["date"].dt.date >= start_date) & (hr_filtered["date"].dt.date <= end_date)]
        sleep_filtered = sleep_filtered[
            (sleep_filtered["date"].dt.date >= start_date) & (sleep_filtered["date"].dt.date <= end_date)]
        hill_filtered = hill_filtered[
            (hill_filtered["date"].dt.date >= start_date) & (hill_filtered["date"].dt.date <= end_date)]

    if selected_years:
        hr_filtered = hr_filtered[hr_filtered["date"].dt.year.isin(selected_years)]
        sleep_filtered = sleep_filtered[sleep_filtered["date"].dt.year.isin(selected_years)]
        hill_filtered = hill_filtered[hill_filtered["date"].dt.year.isin(selected_years)]

    if selected_months:
        hr_filtered = hr_filtered[hr_filtered["date"].dt.month.isin(selected_months)]
        sleep_filtered = sleep_filtered[sleep_filtered["date"].dt.month.isin(selected_months)]
        hill_filtered = hill_filtered[hill_filtered["date"].dt.month.isin(selected_months)]

    if selected_weeks:
        hr_filtered = hr_filtered[hr_filtered["date"].dt.isocalendar().week.isin(selected_weeks)]
        sleep_filtered = sleep_filtered[sleep_filtered["date"].dt.isocalendar().week.isin(selected_weeks)]
        hill_filtered = hill_filtered[hill_filtered["date"].dt.isocalendar().week.isin(selected_weeks)]

    if selected_days:
        hr_filtered = hr_filtered[hr_filtered["date"].dt.isocalendar().day.isin(selected_days)]
        sleep_filtered = sleep_filtered[sleep_filtered["date"].dt.isocalendar().day.isin(selected_days)]
        hill_filtered = hill_filtered[hill_filtered["date"].dt.isocalendar().day.isin(selected_days)]

    if filtered_df.empty:
        empty_fig = px.bar(title="No Data")
        return px.bar(title="No Data"), px.bar(title="No Data"), px.bar(title="No Data"), px.bar(
            title="No Data"), px.bar(title="No Data"), px.bar(title="No Data"), px.bar(title="No Data"), px.bar(
            title="No Data"), px.bar(title="No Data"), px.bar(title="No Data"), px.bar(title="No Data"), px.bar(title="No Data"),

    # --- Filter steps file data by selected date range, years, months, weeks ---
    steps_file_filtered = steps_file_df.copy()
    if start_date and end_date:
        steps_file_filtered = steps_file_filtered[
            (steps_file_filtered["date"].dt.date >= pd.to_datetime(start_date).date()) &
            (steps_file_filtered["date"].dt.date <= pd.to_datetime(end_date).date())
            ]
    if selected_years:
        steps_file_filtered = steps_file_filtered[steps_file_filtered["year"].isin(selected_years)]
    if selected_months:
        steps_file_filtered = steps_file_filtered[steps_file_filtered["month"].isin(selected_months)]
    if selected_weeks:
        steps_file_filtered = steps_file_filtered[steps_file_filtered["week"].isin(selected_weeks)]

    # --- Time grouping and formatting ---
    filtered_df["week"] = filtered_df.index.isocalendar().week
    filtered_df["year"] = filtered_df.index.year

    filtered_df["month"] = filtered_df.index.month  # <<< Also ensure month is defined here
    filtered_df["month_year"] = filtered_df["month"].apply(lambda x: f"{x:02d}") + "-" + filtered_df[
        "year"].astype(str)

    filtered_df["year_week"] = (
            "W" + filtered_df["week"].astype(str).str.zfill(2) + "-" + filtered_df["year"].astype(str)
    )
    # --- Group by week and activity for distance ---
    weekly_dist = (
        filtered_df.groupby(["year_week", "activity", "week", "year"])["distance"]
        .sum()
        .reset_index()
        .sort_values(by=["week", "year"])
    )

    # --- Group by week (all activities) for elevation gain ---
    weekly_elev = (
        filtered_df.groupby(["year_week", "week", "year"])["elevationGain"]
        .sum()
        .reset_index()
        .sort_values(by=["week", "year"])
    )
    # Group monthly data (unchanged)
    monthly_dist = (
        filtered_df.groupby(["month_year", "activity", "month", "year"])["distance"]
        .sum()
        .reset_index()
        .sort_values(by=["month", "year"])
    )

    # Define custom category order for proper sorting
    ordered_months = monthly_dist[["month_year", "month", "year"]].drop_duplicates()
    ordered_months = ordered_months.sort_values(by=["month", "year"])

    monthly_dist["month_year"] = pd.Categorical(
        monthly_dist["month_year"],
        categories=ordered_months["month_year"].tolist(),
        ordered=True,
    )

    monthly_elev = (
        filtered_df.groupby(["month_year", "month", "year"])["elevationGain"]
        .sum()
        .reset_index()
        .sort_values(by=["month", "year"])
    )

    monthly_elev["month_year"] = pd.Categorical(
        monthly_elev["month_year"],
        categories=ordered_months["month_year"].tolist(),
        ordered=True,
    )

    # --- Weekly Duration ---
    weekly_duration = (
        filtered_df.groupby(["year_week", "activity", "week", "year"])["duration_hours"]
        .sum()
        .reset_index()
        .sort_values(by=["week", "year"])
    )

    weekly_duration_totals = weekly_duration.groupby("year_week")["duration_hours"].sum().reset_index()

    # --- Monthly Duration ---
    monthly_duration = (
        filtered_df.groupby(["month_year", "activity", "month", "year"])["duration_hours"]
        .sum()
        .reset_index()
        .sort_values(by=["month", "year"])
    )

    monthly_duration_totals = monthly_duration.groupby("month_year")["duration_hours"].sum().reset_index()

    monthly_duration["month_year"] = pd.Categorical(
        monthly_duration["month_year"],
        categories=ordered_months["month_year"].tolist(),
        ordered=True,
    )

    # --- Yearly Duration ---
    yearly_duration = (
        filtered_df.groupby(["year", "activity"])["duration_hours"]
        .sum()
        .reset_index()
        .sort_values(by=["year"])
    )

    yearly_duration_totals = yearly_duration.groupby("year")["duration_hours"].sum().reset_index()

    yearly_duration["year"] = pd.Categorical(
        yearly_duration["year"],
        categories=year_options,
        ordered=True,
    )

    # --- Group Steps by Year ---
    yearly_steps = (
        filtered_df.groupby(["activity", "year"])["steps"]
        .sum()
        .reset_index()
    )
    yearly_steps["year"] = pd.Categorical(
        yearly_steps["year"], categories=year_options, ordered=True
    )

    # --- Group Steps by Month ---
    monthly_steps = (
        filtered_df.groupby(["month_year", "activity", "month", "year"])["steps"]
        .sum()
        .reset_index()
    )
    monthly_steps["month_year"] = pd.Categorical(
        monthly_steps["month_year"],
        categories=ordered_months["month_year"].tolist(),
        ordered=True,
    )

    # --- Group Steps by Week ---
    weekly_steps = (
        filtered_df.groupby(["year_week", "activity", "week", "year"])["steps"]
        .sum()
        .reset_index()
    )
    weekly_steps = weekly_steps.sort_values(by=["year", "week"])

    # --- Chart 1: Yearly Duration by Activity ---
    yearly_duration_summary = (
        filtered_df.groupby(["activity", "year"])["duration_hours"]
        .sum()
        .reset_index()
    )

    yearly_duration_fig = px.bar(
        yearly_duration_summary,
        x="year",
        y="duration_hours",
        color="activity",
        barmode="relative",
        text="duration_hours",
        title="Yearly Duration by Activity",
        labels={"year": "Year", "duration_hours": "Total Durations (hours)", "activity": "Activity"},
    )
    yearly_duration_fig.update_traces(texttemplate='%{y:.1f}', textposition='inside')

    # Add total on top of each stacked bar
    yearly_totals = yearly_duration_summary.groupby("year")["duration_hours"].sum().reset_index()
    yearly_duration_fig.add_scatter(
        x=yearly_totals["year"],
        y=yearly_totals["duration_hours"],
        mode="text",
        text=yearly_totals["duration_hours"].round(1).astype(str),
        textposition="top center",
        showlegend=False
    )

    # --- New Chart: Compare Activities by Elevation ---
    compare_df = filtered_df.copy()  # always define it first

    # Ensure only 1-3 activities are selected
    if selected_activities:
        compare_df = compare_df[compare_df["activity"].isin(selected_activities[:3])]
    else:
        compare_df = compare_df.head(0)  # empty DataFrame if no activity selected

    # Reset index so startTimeLocal is a column
    compare_df = compare_df.reset_index()

    if compare_df.empty:
        compare_fig = px.scatter(title="No Data for Selected Activities")
    else:
        compare_fig = px.scatter(
            compare_df,
            x=xaxis_dropdown,  # 'distance' or 'duration_hours'
            y="elevationGain",
            color="activity",
            hover_data=["startTimeLocal", "activityId", "distance", "duration_hours", "elevationGain"],
            title=f"Compare Activities ({', '.join(compare_df['activity'].unique())})",
            facet_col="activity" if len(compare_df['activity'].unique()) > 1 else None,
            facet_col_wrap=3,
        )
        compare_fig.update_traces(mode="markers+lines")
        compare_fig.update_layout(
            xaxis_title=xaxis_dropdown.replace("_", " ").title(),
            yaxis_title="Elevation Gain (m)",
            legend_title="Activity",
        )

        # --- Chart 1: Yearly Distance by Activity ---
        yearly_summary = (
            filtered_df.groupby(["activity", "year"])["distance"]
            .sum()
            .reset_index()
        )

        yearly_fig = px.bar(
            yearly_summary,
            x="year",
            y="distance",
            color="activity",
            barmode="relative",
            text="distance",
            title="Yearly Distance by Activity",
            labels={"year": "Year", "distance": "Total Distance (km)", "activity": "Activity"},
        )
        yearly_fig.update_traces(texttemplate='%{y:.1f}', textposition='inside')

        # Add total on top of each stacked bar
        yearly_totals = yearly_summary.groupby("year")["distance"].sum().reset_index()
        yearly_fig.add_scatter(
            x=yearly_totals["year"],
            y=yearly_totals["distance"],
            mode="text",
            text=yearly_totals["distance"].round(1).astype(str),
            textposition="top center",
            showlegend=False
        )



    # --- Chart 2: Weekly Distance + Elevation Gain ---
    time_fig = px.bar(
        weekly_dist,
        x="year_week",
        y="distance",
        color="activity",
        barmode="relative",
        title="Weekly Distance by Activity with Elevation Gain",
        labels={"year_week": "Week-Year", "distance": "Distance (km)", "activity": "Activity"},
        category_orders={"year_week": weekly_dist["year_week"].tolist()}
    )
    time_fig.update_traces(texttemplate='%{y:.1f}', textposition='inside')

    # Add secondary y-axis line for elevation gain
    time_fig.add_scatter(
        x=weekly_elev["year_week"],
        y=weekly_elev["elevationGain"],
        mode="lines+markers",
        name="Elevation Gain (m)",
        yaxis="y2",
        line=dict(color="black", width=2, dash="dot")
    )

    weekly_totals = weekly_dist.groupby("year_week")["distance"].sum().reset_index()

    weekly_totals = weekly_dist.groupby("year_week")["distance"].sum().reset_index()
    time_fig.add_scatter(
        x=weekly_totals["year_week"],
        y=weekly_totals["distance"],
        mode="text",
        text=weekly_totals["distance"].round(1).astype(str),
        textposition="top center",
        showlegend=False
    )
    # Dual Y-axis layout
    time_fig.update_layout(
        yaxis=dict(title="Distance (km)"),
        yaxis2=dict(
            title="Elevation Gain (m)",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        xaxis=dict(title="Week-Year"),
        legend=dict(x=1.1, y=1, orientation="v")
    )
    # --- Chart 3: Monthly Distance + Elevation Gain ---
    month_fig = px.bar(
        monthly_dist,
        x="month_year",
        y="distance",
        color="activity",
        barmode="relative",
        title="Monthly Distance by Activity with Elevation Gain",
        labels={"month_year": "Month-Year", "distance": "Distance (km)", "activity": "Activity"},
        category_orders={"month_year": ordered_months["month_year"].tolist()}  # <== ADD THIS
    )
    month_fig.update_layout(xaxis_type='category')  # Force categorical, no extra ticks

    month_fig.add_scatter(
        x=monthly_elev["month_year"],
        y=monthly_elev["elevationGain"],
        mode="lines+markers",
        name="Elevation Gain (m)",
        yaxis="y2",
        line=dict(color="black", width=2, dash="dot")
    )
    monthly_totals = monthly_dist.groupby("month_year")["distance"].sum().reset_index()
    month_fig.add_scatter(
        x=monthly_totals["month_year"],
        y=monthly_totals["distance"],
        mode="text",
        text=monthly_totals["distance"].round(1).astype(str),
        textposition="top center",
        showlegend=False
    )

    month_fig.update_layout(
        yaxis=dict(title="Distance (km)"),
        yaxis2=dict(
            title="Elevation Gain (m)",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        xaxis=dict(title="Month-Year"),
        legend=dict(x=1.1, y=1, orientation="v")
    )

    # --- Chart: Weekly Duration + Elevation Gain ---
    weekly_duration_fig = px.bar(
        weekly_duration,
        x="year_week",
        y="duration_hours",
        color="activity",
        barmode="relative",
        title="Weekly Duration by Activity with Elevation Gain",
        labels={"year_week": "Week-Year", "duration_hours": "Duration (hours)", "activity": "Activity"},
        category_orders={"year_week": weekly_duration["year_week"].tolist()}
    )

    weekly_duration_fig.add_scatter(
        x=weekly_elev["year_week"],
        y=weekly_elev["elevationGain"],
        mode="lines+markers",
        name="Elevation Gain (m)",
        yaxis="y2",
        line=dict(color="black", width=2, dash="dot")
    )

    weekly_duration_fig.add_scatter(
        x=weekly_duration_totals["year_week"],
        y=weekly_duration_totals["duration_hours"],
        mode="text",
        text=weekly_duration_totals["duration_hours"].round(1).astype(str),
        textposition="top center",
        showlegend=False
    )

    weekly_duration_fig.update_layout(
        yaxis=dict(title="Duration (hours)"),
        yaxis2=dict(
            title="Elevation Gain (m)",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        xaxis=dict(title="Week-Year"),
        legend=dict(x=1.1, y=1, orientation="v")
    )

    # --- Chart: Monthly Duration + Elevation Gain ---
    monthly_duration_fig = px.bar(
        monthly_duration,
        x="month_year",
        y="duration_hours",
        color="activity",
        barmode="relative",
        title="Monthly Duration by Activity with Elevation Gain",
        labels={"month_year": "Month-Year", "duration_hours": "Duration (hours)", "activity": "Activity"},
        category_orders={"month_year": ordered_months["month_year"].tolist()}
    )

    monthly_duration_fig.add_scatter(
        x=monthly_elev["month_year"],
        y=monthly_elev["elevationGain"],
        mode="lines+markers",
        name="Elevation Gain (m)",
        yaxis="y2",
        line=dict(color="black", width=2, dash="dot")
    )

    monthly_duration_fig.add_scatter(
        x=monthly_duration_totals["month_year"],
        y=monthly_duration_totals["duration_hours"],
        mode="text",
        text=monthly_duration_totals["duration_hours"].round(1).astype(str),
        textposition="top center",
        showlegend=False
    )

    monthly_duration_fig.update_layout(
        yaxis=dict(title="Duration (hours)"),
        yaxis2=dict(
            title="Elevation Gain (m)",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        xaxis=dict(title="Month-Year"),
        legend=dict(x=1.1, y=1, orientation="v")
    )

    # --- Chart 4: Total Distance by Activity (All Time) ---
    if selected_activities:
        summary_df = df[df["activity"].isin(selected_activities)]
    else:
        summary_df = df.copy()

    if "activity" in summary_df.columns and not summary_df.empty:
        summary = summary_df.groupby("activity")["distance"].sum().reset_index()
    else:
        summary = pd.DataFrame(columns=["activity", "distance"])

    summary_fig = px.bar(
        summary,
        x="activity",
        y="distance",
        title="Total Distance by Activity (All Time)",
        labels={"distance": "Total Distance (km)", "activity": "Activity"},
    )

    summary_fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')

    # --- Chart 5: YTD Distance per Year by Activity ---
    today = datetime.datetime.now()
    ytd_cutoff = (today.month, today.day)

    df["month"] = df["startTimeLocal"].dt.month
    df["day"] = df["startTimeLocal"].dt.day
    df["year"] = df["startTimeLocal"].dt.year

    ytd_df = df[
        ((df["month"] < ytd_cutoff[0]) |
         ((df["month"] == ytd_cutoff[0]) & (df["day"] <= ytd_cutoff[1])))
        & (df["activity"].isin(selected_activities))
        ]

    ytd_summary = ytd_df.groupby(["activity", "year"])["distance"].sum().reset_index()
    ytd_summary["year"] = ytd_summary["year"].astype(str)

    ytd_fig = px.bar(
        ytd_summary,
        x="distance",
        y="year",
        color="activity",
        barmode="relative",
        text="distance",
        title=f"YTD Distance by Activity per Year (as of {today.strftime('%B %d')})",
        labels={"distance": "Distance (km)", "year": "Year", "activity": "Activity"},
    )
    ytd_fig.update_traces(texttemplate='%{x:.1f}', textposition='inside')

    # Add total distance text on top of each year
    ytd_totals = ytd_summary.groupby("year")["distance"].sum().reset_index()
    ytd_fig.add_scatter(
        x=ytd_totals["distance"],
        y=ytd_totals["year"],
        mode="text",
        text=ytd_totals["distance"].round(1).astype(str),
        textposition="top right",
        showlegend=False
    )

    # --- Chart 6: YTD Elevation Gain per Year by Activity ---
    today = datetime.datetime.now()
    ytd_cutoff = (today.month, today.day)

    df["month"] = df["startTimeLocal"].dt.month
    df["day"] = df["startTimeLocal"].dt.day
    df["year"] = df["startTimeLocal"].dt.year

    ytd_elev_df = df[
        ((df["month"] < ytd_cutoff[0]) |
         ((df["month"] == ytd_cutoff[0]) & (df["day"] <= ytd_cutoff[1])))
        & (df["activity"].isin(selected_activities))
        ]

    ytd_elev_summary = ytd_elev_df.groupby(["activity", "year"])["elevationGain"].sum().reset_index()
    ytd_elev_summary["year"] = ytd_elev_summary["year"].astype(str)

    ytd_elev_fig = px.bar(
        ytd_elev_summary,
        x="elevationGain",
        y="year",
        color="activity",
        barmode="relative",
        text="elevationGain",
        title=f"YTD Elevation Gain by Activity per Year (as of {today.strftime('%B %d')})",
        labels={"elevationGain": "Ascent (m)", "activity": "Activity"},
    )
    ytd_elev_fig.update_traces(texttemplate='%{x:.1f}', textposition='inside')

    # Add total elevation gain text on top of each year
    ytd_elev_totals = ytd_elev_summary.groupby("year")["elevationGain"].sum().reset_index()
    ytd_elev_fig.add_scatter(
        x=ytd_elev_totals["elevationGain"],
        y=ytd_elev_totals["year"],
        mode="text",
        text=ytd_elev_totals["elevationGain"].round(0).astype(int).astype(str),
        textposition="top right",
        showlegend=False
    )

    # --- Chart 7: VO2Max Over Time (from Hill Score CSV) ---
    if not hill_filtered.empty and "vo2MaxPreciseValue" in hill_filtered.columns:
        vo2_df = hill_filtered.dropna(subset=["vo2MaxPreciseValue"]).copy()
        vo2_df = vo2_df.sort_values("date")

        vo2_fig = px.line(
            vo2_df,
            x="date",
            y="vo2MaxPreciseValue",
            title="VO₂Max Over Time",
            labels={"vo2MaxPreciseValue": "VO₂ Max", "date": "Date"},
            markers=True
        )
        vo2_fig.update_traces(mode="lines+markers")

        # Optional: smoothing line
        vo2_df["vO2MaxSmooth"] = vo2_df["vo2MaxPreciseValue"].rolling(window=7).mean()
        vo2_fig.add_scatter(
            x=vo2_df["date"],
            y=vo2_df["vO2MaxSmooth"],
            mode="lines",
            name="Smoothed (7-day avg)",
            line=dict(dash="dot", color="red")
        )
    else:
        vo2_fig = px.line(title="No VO₂Max Data")

    # --- Chart 8: HR Zones with separate bars per activity ---
    hr_zone_melt = filtered_df.melt(
        id_vars=["activity"],
        value_vars=hr_zone_cols,
        var_name="Heart Rate Zone",
        value_name="Time (seconds)"
    )

    zone_label_map = {f"hrTimeInZone_{i}": f"HR Zone {i}" for i in range(1, 6)}
    hr_zone_melt["Heart Rate Zone"] = hr_zone_melt["Heart Rate Zone"].map(zone_label_map)

    # Convert seconds to hours
    hr_zone_melt["Time (hours)"] = hr_zone_melt["Time (seconds)"] / 3600

    # Group by HR zone and activity summing time
    hr_zone_summary = (
        hr_zone_melt.groupby(["Heart Rate Zone", "activity"])["Time (hours)"]
        .sum()
        .reset_index()
    )

    # Plot grouped bar chart
    hr_fig = px.bar(
        hr_zone_summary,
        x="Heart Rate Zone",
        y="Time (hours)",
        color="activity",
        barmode="group",  # side-by-side bars
        title="Time Spent in Heart Rate Zones by Activity",
        labels={"Heart Rate Zone": "Heart Rate Zone", "Time (hours)": "Time (hours)", "activity": "Activity"},
        text=hr_zone_summary["Time (hours)"].round(2),
    )

    hr_fig.update_traces(textposition="outside")
    hr_fig.update_layout(xaxis_title="Heart Rate Zone", yaxis_title="Time (hours)")

    # --- Chart 9: Power Zones with separate bars per activity ---
    power_zone_melt = filtered_df.melt(
        id_vars=["activity"],
        value_vars=power_zone_cols,
        var_name="Power Rate Zone",
        value_name="Time (seconds)"
    )

    zone_label_map = {f"powerTimeInZone_{i}": f"Power Zone {i}" for i in range(1, 6)}
    power_zone_melt["Power Rate Zone"] = power_zone_melt["Power Rate Zone"].map(zone_label_map)

    # Convert seconds to hours
    power_zone_melt["Time (hours)"] = power_zone_melt["Time (seconds)"] / 3600

    # Group by HR zone and activity summing time
    power_zone_summary = (
        power_zone_melt.groupby(["Power Rate Zone", "activity"])["Time (hours)"]
        .sum()
        .reset_index()
    )

    # Plot grouped bar chart
    power_fig = px.bar(
        power_zone_summary,
        x="Power Rate Zone",
        y="Time (hours)",
        color="activity",
        barmode="group",  # side-by-side bars
        title="Time Spent in Power Rate Zones by Activity",
        labels={"Power Rate Zone": "Power Rate Zone", "Time (hours)": "Time (hours)", "activity": "Activity"},
        text=power_zone_summary["Time (hours)"].round(2),
    )

    power_fig.update_traces(textposition="outside")
    power_fig.update_layout(xaxis_title="Power Rate Zone", yaxis_title="Time (hours)")

    # --- Chart 10: Daily HR Summary ---
    hr_cols = {
        "restingHR": "Rest HR",
        "maxHR": "Max HR",
        "minHR": "Min HR",
        "averageHR": "Avg HR"
    }
    hr_daily_df_renamed = hr_filtered.copy()
    for old_col, new_col in hr_cols.items():
        if old_col in hr_daily_df_renamed:
            hr_daily_df_renamed.rename(columns={old_col: new_col}, inplace=True)

    hr_daily_fig = px.line(
        hr_daily_df_renamed,
        x="date",
        y=list(hr_cols.values()),
        title="Daily Heart Rate Summary",
        labels={"value": "Heart Rate (bpm)", "date": "Date", "variable": "HR Metric"}
    )
    hr_daily_fig.update_xaxes(tickformat="%Y-%m-%d")

    hr_daily_fig.update_layout(
        legend_title_text="HR Metric",
        hovermode="x unified"
    )

    # --- Chart 11: Daily Sleep Summary ---
    sleep_cols = {
        "deepSleepSeconds": "Deep Sleep 1 hour & 15 minutes",
        "lightSleepSeconds": "Light Sleep 4 hours & 15 minutes",
        "remSleepSeconds": "REM Sleep 1 hour & 27 minutes",
        "awakeSleepSeconds": "Awake Time 1 hour & 3 minutes"
    }

    # Convert seconds to minutes
    sleep_df_minutes = sleep_filtered.copy()
    for old_col, new_col in sleep_cols.items():
        if old_col in sleep_df_minutes:
            sleep_df_minutes[new_col] = sleep_df_minutes[old_col] / 60

    # Create stacked bar chart
    sleep_daily_fig = go.Figure()

    colors = {
        "Total Sleep 8 hours": "black",
        "Light Sleep 4 hours & 15 minutes": "blue",
        "Deep Sleep 1 hour & 15 minutes": "green",
        "REM Sleep 1 hour & 27 minutes": "purple",
        "Awake Time 1 hour & 3 minutes": "red",

    }

    for stage in sleep_cols.values():
        sleep_daily_fig.add_bar(
            x=sleep_df_minutes["date"],
            y=sleep_df_minutes[stage],
            name=stage,
            marker_color=colors[stage]
        )

    # --- Add guideline traces (min only, based on 8h = 480 min) ---
    total_sleep = 480
    guidelines = {
        "Total Sleep 8 hours":  1 * total_sleep,
        "Light Sleep 4 hours & 15 minutes": 0.53125 * total_sleep,  # 240 min
        "Deep Sleep 1 hour & 15 minutes": 0.15625 * total_sleep,  # 48 min
        "REM Sleep 1 hour & 27 minutes": 0.18125 * total_sleep,  # 48 min
        "Awake Time 1 hour & 3 minutes": 0.13125 * total_sleep  # 24 min
    }

    for stage, need in guidelines.items():
        sleep_daily_fig.add_trace(go.Scatter(
            x=sleep_df_minutes["date"],
        y=[need] * len(sleep_df_minutes),
        mode="lines",
        line=dict(color=colors[stage], dash="dot"),
        name=f"{stage} Recommended - middle value (min)",
        visible=True  # will auto-toggle with legend
        ))

    # Layout
    sleep_daily_fig.update_layout(
        barmode="relative",
        title="Daily Sleep Time Summary",
        xaxis=dict(tickformat="%Y-%m-%d"),
        yaxis_title="Sleep time (minutes)",
        legend_title_text="Sleep Metric",
        hovermode="x unified"
    )

    # --- Chart 12: Hill Strength/Endurance/Overall Scores ---
    if not hill_filtered.empty:
        hill_fig = px.line(
            hill_filtered,
            x="date",
            y=["strengthScore", "enduranceScore", "overallScore"],
            title="Hill Strength, Endurance, and Overall Score Over Time",
            labels={"value": "Score", "date": "Date", "variable": "Score Type"},
            markers=True
        )
        hill_fig.update_traces(mode="lines+markers")
        hill_fig.update_layout(hovermode="x unified")
    else:
        hill_fig = px.line(title="No Hill Score Data")

    # --- Yearly Steps by Activity ---
    yearly_steps_fig = px.bar(
        yearly_steps,
        x="year",
        y="steps",
        color="activity",
        barmode="relative",
        text="steps",
        title="Yearly Steps by Activity",
        labels={"year": "Year", "steps": "Total Steps", "activity": "Activity"},
    )
    yearly_steps_fig.update_traces(texttemplate='%{y:.0f}', textposition='inside')
    yearly_totals = yearly_steps.groupby("year")["steps"].sum().reset_index()
    yearly_steps_fig.add_scatter(
        x=yearly_totals["year"],
        y=yearly_totals["steps"],
        mode="text",
        text=yearly_totals["steps"],
        textposition="top center",
        showlegend=False
    )


    # --- Monthly Steps by Activity ---
    monthly_steps_fig = px.bar(monthly_steps,
        x="month_year",
        y="steps",
        color="activity",
        barmode="relative",
        text="steps",
        title="Monthly Steps by Activity",
        labels={"month_year": "Month-Year", "steps": "Total Steps", "activity": "Activity"},
        category_orders={"month_year": ordered_months["month_year"].tolist()},
    )
    monthly_steps_fig.update_traces(texttemplate='%{y}', textposition='inside')
    monthly_totals = monthly_steps.groupby("month_year")["steps"].sum().reset_index()
    monthly_steps_fig.add_scatter(
        x=monthly_totals["month_year"],
        y=monthly_totals["steps"],
        mode="text",
        text=monthly_totals["steps"],
        textposition="top center",
        showlegend=False
    )

    # --- Weekly Steps by Activity ---
    weekly_steps_fig = px.bar(
        weekly_steps,
        x="year_week",
        y="steps",
        color="activity",
        barmode="relative",
        text="steps",
        title="Weekly Steps by Activity",
        labels={"year_week": "Week-Year", "steps": "Total Steps", "activity": "Activity"},
        category_orders={"year_week": weekly_steps["year_week"].tolist()},
    )
    weekly_steps_fig.update_traces(texttemplate='%{y:.1f}', textposition='inside')
    weekly_totals = weekly_steps.groupby("year_week")["steps"].sum().reset_index()
    weekly_steps_fig.add_scatter(
        x=weekly_totals["year_week"],
        y=weekly_totals["steps"],
        mode="text",
        text=weekly_totals["steps"],
        textposition="top center",
        showlegend=False
    )

    # Load steps CSV
    steps_df = pd.read_csv(STEP_CSV_FILE)
    steps_df['date'] = pd.to_datetime(steps_df['date'], errors='coerce')

    today = pd.Timestamp.now()
    today_month_day = (today.month, today.day)

    # For each year, sum steps up to the same month/day as today
    def ytd_steps_for_year(group):
        return group[(group['date'].dt.month < today_month_day[0]) |
                     ((group['date'].dt.month == today_month_day[0]) &
                      (group['date'].dt.day <= today_month_day[1]))]['steps'].sum()

    ytd_steps = steps_df.groupby(steps_df['date'].dt.year).apply(ytd_steps_for_year).reset_index()
    ytd_steps.columns = ['year', 'steps']
    ytd_steps = ytd_steps.sort_values('year', ascending=False)  # latest year on top

    ytd_steps_fig = px.bar(
        ytd_steps,
        y='year',
        x='steps',
        orientation='h',
        title=f"YTD Steps per Year (up to {today.strftime('%d %b')})",
        labels={'year': 'Year', 'steps': 'Total Steps'}
    )

    ytd_steps_fig.update_traces(texttemplate='%{x:.0f}', textposition='inside')

    # Yearly total steps (from garmin_steps.csv)
    yearly_steps_file_fig = px.bar(
        steps_file_filtered.groupby("year")["steps"].sum().reset_index(),
        x="year",
        y="steps",
        title="Yearly Steps (All Data)",
        text="steps"
    )
    yearly_steps_file_fig.update_traces(texttemplate='%{y:.0f}', textposition='inside')

    # Monthly total steps
    monthly_steps_file_fig = px.bar(
        steps_file_filtered.groupby("month_year")["steps"].sum().reset_index(),
        x="month_year",
        y="steps",
        title="Monthly Steps (All Data)",
        text="steps"
    )
    monthly_steps_file_fig.update_traces(texttemplate='%{y:.0f}', textposition='inside')

    # Weekly total steps
    weekly_steps_file_fig = px.bar(
        steps_file_filtered.groupby("year_week")["steps"].sum().reset_index(),
        x="year_week",
        y="steps",
        title="Weekly Steps (All Data)",
        text="steps"
    )
    weekly_steps_file_fig.update_traces(texttemplate='%{y:.0f}', textposition='inside')

    return compare_fig, yearly_duration_fig, yearly_fig, weekly_duration_fig, time_fig, monthly_duration_fig, month_fig, summary_fig, ytd_fig, ytd_elev_fig, vo2_fig, hr_fig, power_fig, hr_daily_fig, sleep_daily_fig, hill_fig, yearly_steps_fig, monthly_steps_fig, weekly_steps_fig,ytd_steps_fig, yearly_steps_file_fig, monthly_steps_file_fig, weekly_steps_file_fig,




if __name__ == "__main__":
    print("Starting Garmin dashboard at http://127.0.0.1:8051 ...")
    try:
        app.run(debug=True, port=8051)
    except Exception as e:
        logging.error(f"Error running dashboard: {e}")

