import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import fitparse
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np


def read_fit_file(file_path):
    fitfile = fitparse.FitFile(file_path)

    # Data storage
    data = {field: [] for field in
            ['Timestamp', 'Heart Rate', 'Power', 'Speed', 'Cadence', 'Altitude', 'Distance', 'Temperature']}
    gear_data = {'Timestamp': [], 'Front Gear': [], 'Rear Gear': []}

    current_gear = {'front': None, 'rear': None}

    for record in fitfile.get_messages():
        record_data = {field.name: field.value for field in record}
        event = record_data.get('event')
        timestamp = record_data.get('timestamp')

        if record.name == 'event':
            if event == 'front_gear_change':
                current_gear['front'] = record_data.get('front_gear')
            elif event == 'rear_gear_change':
                current_gear['rear'] = record_data.get('rear_gear')
            gear_data['Timestamp'].append(timestamp)
            gear_data['Front Gear'].append(current_gear['front'])
            gear_data['Rear Gear'].append(current_gear['rear'])
        elif record.name == 'record':
            for key in data.keys():
                data[key].append(record_data.get(key.lower().replace(' ', '_')))

    # Convert to DataFrame
    df = pd.DataFrame(data)
    df_gears = pd.DataFrame(gear_data)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df_gears['Timestamp'] = pd.to_datetime(df_gears['Timestamp'], errors='coerce')

    if df['Timestamp'].isnull().all():
        raise ValueError("No valid Timestamp data found in the file.")

    df.set_index('Timestamp', inplace=True)
    df_gears.set_index('Timestamp', inplace=True)
    df = df.merge(df_gears, how='outer', left_index=True, right_index=True).fillna(method='ffill').reset_index()

    df_resampled_time = df.resample('1s', on='Timestamp').mean().reset_index()
    df_resampled_time['Duration'] = pd.to_timedelta(
        df_resampled_time['Timestamp'] - df_resampled_time['Timestamp'].iloc[0]).dt.total_seconds().astype(int)
    df_resampled_time['Duration'] = pd.to_timedelta(df_resampled_time['Duration'], unit='s').astype(str).str[-8:]
    df['Distance'] = df['Distance'].interpolate(method='linear')

    return df_resampled_time, df.reset_index()


def convert_units(df, unit_system):
    conversion_factors = {
        'Imperial': {'Speed': 2.23694, 'Distance': 0.000621371, 'Temperature': (9 / 5, 32), 'Altitude': 3.28084},
        'Metric': {'Speed': 3.6, 'Distance': 0.001, 'Temperature': (1, 0), 'Altitude': 1}
    }
    factor = conversion_factors[unit_system]
    df['Speed'] *= factor['Speed']
    df['Distance'] *= factor['Distance']
    df['Temperature'] = df['Temperature'] * factor['Temperature'][0] + factor['Temperature'][1]
    df['Altitude'] *= factor['Altitude']

    labels = {
        'Speed': f"Speed ({'mph' if unit_system == 'Imperial' else 'kph'})",
        'Distance': f"Distance ({'miles' if unit_system == 'Imperial' else 'kilometers'})",
        'Temperature': f"Temperature ({'°F' if unit_system == 'Imperial' else '°C'})",
        'Altitude': f"Altitude ({'feet' if unit_system == 'Imperial' else 'meters'})",
        'Heart Rate': 'Heart Rate (bpm)',
        'Power': 'Power (Watts)',
        'Cadence': 'Cadence (rpm)'
    }
    return df, labels['Speed'], labels['Distance'], labels['Temperature'], labels['Altitude'], labels['Heart Rate'], \
        labels['Power'], labels['Cadence']


def smooth_data(df, window_size):
    if window_size > 0:
        numeric_cols = df.select_dtypes(include='number').columns
        df[numeric_cols] = df[numeric_cols].rolling(window=window_size, min_periods=1).mean()
    return df


def create_summary_table(df, unit_system):
    # Define metric labels
    metrics = {
        'Heart Rate (bpm)': 'Heart Rate',
        'Power (Watts)': 'Power',
        'Cadence (rpm)': 'Cadence',
        f"Speed ({'mph' if unit_system == 'Imperial' else 'kph'})": 'Speed',
        f"Temperature ({'°F' if unit_system == 'Imperial' else '°C'})": 'Temperature',
        f"Altitude ({'feet' if unit_system == 'Imperial' else 'meters'})": 'Altitude'
    }

    # Create a summary dictionary
    summary = {
        metric: [round(df[col].min()), round(df[col].mean()), round(df[col].max())]
        for metric, col in metrics.items()
        if col in df.columns
    }

    # Convert to DataFrame
    summary_df = pd.DataFrame(summary, index=['Min', 'Avg', 'Max']).T.reset_index().rename(columns={'index': 'Metric'})

    return summary_df


def create_figure(df_time, df_distance, x_axis, unit_system, smoothing):
    df_time, speed_label, distance_label, temperature_label, altitude_label, heart_rate_label, power_label, cadence_label = convert_units(
        df_time, unit_system)
    df_distance, _, _, _, _, _, _, _ = convert_units(df_distance, unit_system)

    df_time_smooth = smooth_data(df_time, smoothing)
    df_distance_smooth = smooth_data(df_distance, smoothing)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    x_data = df_time_smooth if x_axis == 'Duration' else df_distance_smooth
    x_data_col = 'Duration' if x_axis == 'Duration' else 'Distance'

    tick_spacing = 20 * 60 if x_axis == 'Duration' else 10
    tickvals = pd.to_timedelta(
        np.arange(0, pd.to_timedelta(x_data[x_data_col]).dt.total_seconds().max() + tick_spacing, tick_spacing),
        unit='s').astype(str).str[-8:] if x_axis == 'Duration' else np.arange(0,
                                                                              x_data['Distance'].max() + tick_spacing,
                                                                              tick_spacing)

    for col, name, secondary_y in [
        ('Heart Rate', heart_rate_label, False),
        ('Power', power_label, False),
        ('Speed', speed_label, True),
        ('Cadence', cadence_label, True),
        ('Altitude', altitude_label, True),
        ('Temperature', temperature_label, True)
    ]:
        fig.add_trace(go.Scatter(x=x_data[x_data_col], y=x_data[col], mode='lines', name=name), secondary_y=secondary_y)

    if 'Front Gear' in df_time_smooth.columns and 'Rear Gear' in df_time_smooth.columns:
        valid_gear_data = df_time_smooth.dropna(subset=['Front Gear', 'Rear Gear'])
        fig.add_trace(go.Scatter(x=valid_gear_data[x_data_col], y=valid_gear_data['Front Gear'], mode='lines+markers',
                                 name='Front Gear', line=dict(dash='dash')), secondary_y=False)
        fig.add_trace(go.Scatter(x=valid_gear_data[x_data_col], y=valid_gear_data['Rear Gear'], mode='lines+markers',
                                 name='Rear Gear', line=dict(dash='dash')), secondary_y=False)

    fig.update_layout(
        title='Workout Data',
        xaxis_title=x_axis,
        yaxis_title=f'{heart_rate_label}, {power_label}, {temperature_label}',
        yaxis2_title=f'{speed_label}, {cadence_label}, {altitude_label}',
        hovermode='x',
        legend=dict(y=0.5, traceorder='reversed', font_size=16),
        xaxis=dict(tickmode='array', tickvals=tickvals, range=[x_data[x_data_col].min(), x_data[x_data_col].max()])
    )

    return fig


def calculate_highest_averages(df, window_sizes):
    metrics = ['Power', 'Cadence', 'Heart Rate']
    highest_averages = {metric: [] for metric in metrics}

    # Ensure 'Timestamp' is in datetime format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df.set_index('Timestamp', inplace=True)

    for window_size in window_sizes:
        for metric in metrics:
            if metric in df.columns:
                # Calculate rolling mean
                rolling_mean = df[metric].rolling(window=window_size, min_periods=1).mean()
                # Find the maximum of the rolling mean
                highest_avg = rolling_mean.max()
                highest_averages[metric].append(highest_avg)
            else:
                # If the metric is not in the DataFrame, append NaN
                highest_averages[metric].append(float('nan'))

    return highest_averages


def create_highest_averages_figure(highest_averages, intervals):
    fig = go.Figure()

    for metric, averages in highest_averages.items():
        fig.add_trace(go.Bar(
            x=intervals,
            y=averages,
            name=metric
        ))

    fig.update_layout(
        title='Highest Average Metrics for Different Time Intervals',
        xaxis_title='Time Interval',
        yaxis_title='Highest Average Value',
        barmode='group',
        xaxis=dict(
            tickvals=list(range(len(intervals))),
            ticktext=intervals
        ),
        yaxis=dict(
            title='Highest Average Value'
        )
    )

    return fig


def create_highest_averages_figure(highest_averages, window_labels):
    fig = go.Figure()

    for metric, averages in highest_averages.items():
        # Round the averages to the nearest whole number
        rounded_averages = [round(avg) if not pd.isna(avg) else 0 for avg in averages]

        fig.add_trace(go.Bar(
            x=window_labels,
            y=rounded_averages,
            name=metric
        ))

    fig.update_layout(
        title='Max Over Time',
        xaxis_title='Time',
        yaxis_title='Value',
        barmode='group',
        xaxis=dict(
            tickvals=list(range(len(window_labels))),
            ticktext=window_labels
        ),
        yaxis=dict(
            title='Highest Average Value',
            tickformat='d'  # Format y-axis ticks as integers
        )
    )

    return fig


app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Workout Data Visualization"),
    dcc.Dropdown(id='x-axis-dropdown',
                 options=[{'label': 'Time', 'value': 'Duration'}, {'label': 'Distance', 'value': 'Distance'}],
                 value='Duration'),
    dcc.Dropdown(id='unit-dropdown',
                 options=[{'label': 'Metric', 'value': 'Metric'}, {'label': 'Imperial', 'value': 'Imperial'}],
                 value='Metric'),
    dcc.Slider(id='smoothing-slider', min=0, max=100, step=1, value=0, marks={},
               tooltip={'always_visible': True, 'placement': 'bottom'}),
    dcc.Graph(id='graph'),
    html.Div(id='summary-table'),
    dcc.Graph(id='highest-averages-graph')  # New graph for highest averages
])


@app.callback(
    [Output('graph', 'figure'),
     Output('summary-table', 'children'),
     Output('highest-averages-graph', 'figure')],
    [Input('x-axis-dropdown', 'value'),
     Input('unit-dropdown', 'value'),
     Input('smoothing-slider', 'value')]
)
def update_graph(x_axis, unit_system, smoothing):
    file_path = r"C:\Users\tajkr\Downloads\TestData.fit"
    df_time, df_distance = read_fit_file(file_path)

    # Create main graph
    fig_main = create_figure(df_time, df_distance, x_axis, unit_system, smoothing)

    # Create summary table
    summary_df = create_summary_table(df_time, unit_system)
    summary_table = html.Table([
        html.Thead(html.Tr([html.Th(col) for col in summary_df.columns])),
        html.Tbody([
            html.Tr([html.Td(summary_df.iloc[i][col]) for col in summary_df.columns])
            for i in range(len(summary_df))
        ])
    ])

    # Define window sizes in seconds and corresponding labels
    window_sizes_seconds = [3, 10, 30, 60, 300, 600, 1200, 3600]
    window_labels = ['3 sec', '10 sec', '30 sec', '1 min', '5 min', '10 min', '20 min', '1 hr']

    # Calculate highest averages for defined rolling windows
    highest_averages = calculate_highest_averages(df_time, window_sizes_seconds)

    # Create highest averages graph
    fig_highest_averages = create_highest_averages_figure(highest_averages, window_labels)

    return fig_main, summary_table, fig_highest_averages


if __name__ == '__main__':
    app.run_server(debug=True)

