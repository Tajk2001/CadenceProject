import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, dash_table, Input, Output
from fitparse import FitFile

# ── USER CONFIG ────────────────────────────────────────────────────────────────
FIT_FILE        = r"C:\Users\User\Desktop\New folder\Job_s_not_done_typa_day.fit"
CRANK_LENGTH_M  = 0.1725   # meters
MASS_KG         = 68.0     # kg (tagged only)
HEIGHT_M        = 1.75     # m  (tagged only)
# ────────────────────────────────────────────────────────────────────────────────

# which metrics to display
METRICS = [
    "Power", "Cadence", "Speed", "Elevation", "Heart_Rate",
    "Slope", "Gear_Ratio", "Torque_Nm", "Tangential_Force_N"
]

# ── 1) Parse FIT + build raw DataFrame ────────────────────────────────────────
fit = FitFile(FIT_FILE)
records = {}
last_fg = last_rg = None

for msg in fit.get_messages("record"):
    rec = {f.name: f.value for f in msg
           if f.name in (
               "timestamp","power","cadence","speed","distance",
               "altitude","heart_rate","front_gear","rear_gear"
           )}
    t = rec.get("timestamp")
    if not t: 
        continue
    e = records.setdefault(t, {})
    e["Power"]      = rec.get("power", e.get("Power", np.nan))
    e["Cadence"]    = rec.get("cadence", e.get("Cadence", np.nan))
    e["Speed_mps"]  = rec.get("speed", e.get("Speed_mps", np.nan))
    e["Distance_m"] = rec.get("distance", e.get("Distance_m", np.nan))
    e["Elevation"]  = rec.get("altitude", e.get("Elevation", np.nan))
    e["Heart_Rate"] = rec.get("heart_rate", e.get("Heart_Rate", np.nan))
    fg = rec.get("front_gear")
    rg = rec.get("rear_gear")
    if fg is not None: last_fg = fg
    if rg is not None: last_rg = rg
    e["Front_Gear"] = last_fg
    e["Rear_Gear"]  = last_rg

# also parse event messages for gear if missed
for msg in fit.get_messages("event"):
    ev = {f.name: f.value for f in msg}
    t = ev.get("timestamp")
    if not t:
        continue
    e = records.setdefault(t, {})
    if ev.get("front_gear") is not None:
        last_fg = ev["front_gear"]
    if ev.get("rear_gear") is not None:
        last_rg = ev["rear_gear"]
    e["Front_Gear"] = last_fg
    e["Rear_Gear"]  = last_rg

df = pd.DataFrame.from_dict(records, orient="index").sort_index()
df.index.name = "Time"
df.reset_index(inplace=True)
df["Time"] = (df["Time"] - df["Time"].iloc[0]).dt.total_seconds()

# ── 2) Clean & validate ───────────────────────────────────────────────────────
# ensure columns exist
for c in ("Front_Gear","Rear_Gear","Distance_m"):
    if c not in df:
        df[c] = np.nan
# forward-fill
df[["Front_Gear","Rear_Gear","Distance_m"]] = (
    df[["Front_Gear","Rear_Gear","Distance_m"]].ffill().fillna(0)
)
# convert speed to km/h
df["Speed"] = df["Speed_mps"] * 3.6

# drop rows with nonsensical values
df = df[
    df["Power"].between(0, 2000) &
    df["Cadence"].between(0, 200) &
    df["Speed"].between(0, 100) &
    df["Heart_Rate"].between(0, 220) &
    df["Elevation"].between(-100, 5000)
].copy()

# ── 3) Compute derived metrics ────────────────────────────────────────────────
# gear ratio & shift
df["Gear_Ratio"] = df["Front_Gear"] / df["Rear_Gear"].replace(0, np.nan)
df["Shift"]      = df["Gear_Ratio"].diff().abs().gt(0).astype(int)
# slope (%) based on raw distance in meters
df["Slope"] = (
    df["Elevation"].diff() / df["Distance_m"].diff().replace(0, np.nan)
) * 100
df["Slope"].fillna(0, inplace=True)
# torque & force
omega = df["Cadence"] * 2 * np.pi / 60
df["Torque_Nm"]         = df["Power"] / omega.replace(0, np.nan)
df["Tangential_Force_N"] = df["Torque_Nm"] / CRANK_LENGTH_M

# tag anthropometrics
df["Crank_Length_m"] = CRANK_LENGTH_M
df["Mass_kg"]        = MASS_KG
df["Height_m"]       = HEIGHT_M

# ── 4) Build Dash app ─────────────────────────────────────────────────────────
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Cycling Data Dashboard"),
    html.Button("Toggle X-Axis: Time/Distance", id="toggle-axis", n_clicks=0),
    dcc.Graph(id="cycling-graph"),
    html.H2("Summary Statistics"),
    dash_table.DataTable(
        id="summary-table",
        style_table={'overflowX': 'auto'},
        style_cell={'padding': '5px', 'textAlign': 'left'}
    )
])

@app.callback(
    [Output("cycling-graph","figure"),
     Output("summary-table","data"),
     Output("summary-table","columns")],
    [Input("toggle-axis","n_clicks")]
)
def update(n_clicks):
    x = "Time" if n_clicks % 2 == 0 else "Distance_m"
    xlabel = "Time (s)" if x=="Time" else "Distance (m)"

    fig = go.Figure()
    # generate multiple y-axes
    for i, m in enumerate(METRICS, start=1):
        axis_name = "y" if i == 1 else f"y{i}"
        trace = go.Scatter(x=df[x], y=df[m],
                           mode='lines', name=m,
                           yaxis=axis_name)
        fig.add_trace(trace)
    # layout axes
    layout = {"xaxis":{"title":xlabel}, "hovermode":"x unified"}
    for i, m in enumerate(METRICS, start=1):
        name = "yaxis" if i == 1 else f"yaxis{i}"
        side = "left" if i % 2 else "right"
        pos  = 0.0 + (i-1)*0.05
        layout[name] = dict(
            title=m, overlaying="y", side=side,
            anchor="free", position=pos
        )
    fig.update_layout(
        title=f"Metrics vs {xlabel}", **layout
    )

    # summary stats
    summary = df[METRICS].describe().loc[["min","mean","max"]].round(2)
    summary = summary.reset_index().rename(columns={"index":"Statistic"})
    cols = [{"name":c,"id":c} for c in summary.columns]

    return fig, summary.to_dict("records"), cols

if __name__=="__main__":
    app.run(debug=True)

