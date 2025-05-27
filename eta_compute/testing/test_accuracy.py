import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import os
import math
import json
import matplotlib.pyplot as plt
import seaborn as sns    
from eta_compute.data import AMNEX_DATA_PATH, ROUTE_STOPS_PATH, TEST_SCHEDULE_PATH, FLEET_DEVICE_PATH
import numpy as np

# Set up logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f'logs/test_logs/test_accuracy_{timestamp}.log'
os.makedirs('logs/test_logs', exist_ok=True)
os.makedirs('output', exist_ok=True)
os.makedirs('output/actual_times', exist_ok=True)
os.makedirs('output/comparison_results', exist_ok=True)

# Create a dedicated test_logger for accuracy testing
test_logger = logging.getLogger('eta_accuracy_test')
test_logger.setLevel(logging.INFO)

# Create handlers
file_handler = logging.FileHandler(log_file)
console_handler = logging.StreamHandler()

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to test_logger
test_logger.addHandler(file_handler)
test_logger.addHandler(console_handler)

# Prevent propagation to root test_logger
test_logger.propagate = False

class RouteTimeAnalyzer:
    def __init__(self, gps_data_path: str, route_stops_path: str, schedule_path: str, fleet_device_path: str, 
             fleet_id: str, date: str):
        """
        Initialize the analyzer
        
        Args:
            gps_data_path: Path to GPS data CSV
            route_stops_path: Path to route stops mapping CSV
            schedule_path: Path to schedule CSV with columns: route_id, start_time, end_time
            fleet_id: Fleet ID to analyze
            date: Date to analyze in YYYY-MM-DD format
        """
        self.fleet_id = fleet_id
        self.date = date

        test_logger.info(f"Loading data for fleet {fleet_id} on {date}")


        self.gps_data = pd.read_csv(gps_data_path)
        self.route_stops = pd.read_csv(route_stops_path)
        self.schedule = pd.read_csv(schedule_path)
        self.fleet_device_df = pd.read_csv(FLEET_DEVICE_PATH)

        self.device_id = self.fleet_device_df[self.fleet_device_df['Fleet'] == self.fleet_id][['Obu Iemi' ,'Chalo DeviceID']]
        if self.device_id['Obu Iemi'].isna().all():
            self.device_id = int(self.device_id['Chalo DeviceID'].values[0])
        else:
            self.device_id = int(self.device_id['Obu Iemi'].values[0])
        
        # Convert timestamps
        self.gps_data['Timestamp'] = pd.to_datetime(self.gps_data['Timestamp'])
        self.gps_data = self.gps_data.sort_values('Timestamp')
        
        # Convert schedule times from IST to UTC
        self.schedule['start_time'] = pd.to_datetime(self.date + 'T' + self.schedule['start_time'], format='%Y-%m-%dT%H:%M')
        self.schedule['end_time'] = pd.to_datetime(self.date + 'T' + self.schedule['end_time'], format='%Y-%m-%dT%H:%M')
        
        # Convert to UTC by subtracting 5:30 hours
        self.schedule['start_time'] = self.schedule['start_time'] - pd.Timedelta(hours=5, minutes=30)
        self.schedule['end_time'] = self.schedule['end_time'] - pd.Timedelta(hours=5, minutes=30)
        
        test_logger.info("Data loading complete")

    def analyze_single_route(self, route_id: str, start_time: str, end_time: str) -> pd.DataFrame:
        """
        Analyze travel times between stops for a specific route and time period
        
        Args:
            route_id: The route ID to analyze
            start_time: Start time in format 'HH:MM'
            end_time: End time in format 'HH:MM'
        """
        test_logger.info(f"Analyzing route {route_id} from {start_time} to {end_time}")
        
        # Add one hour buffer on either side
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
        buffered_start = start_dt - pd.Timedelta(hours=1)
        buffered_end = end_dt + pd.Timedelta(hours=1)
        
        test_logger.info(f"Using buffered time window: {buffered_start} to {buffered_end}")
        
        # Filter GPS data for this time period with buffer
        route_gps = self.gps_data[
            (self.gps_data['Timestamp'] >= buffered_start) &
            (self.gps_data['Timestamp'] <= buffered_end)
        ]
        
        test_logger.info(f"Found {len(route_gps)} GPS points for this time period")
        
        # Get stops for this route
        route_stops = self.route_stops[
            self.route_stops['TUMMOC Route ID'] == int(route_id)
        ].sort_values('Sequence')
        
        test_logger.info(f"Found {len(route_stops)} stops for route {route_id}")
        
        # Find closest GPS point to each stop
        stop_visits = []
        for _, stop in route_stops.iterrows():
            # Calculate distance to stop for each GPS point
            route_gps['distance_to_stop'] = route_gps.apply(
                lambda row: self._haversine_distance(
                    row['Lat'], row['Long'], 
                    stop['LAT'], stop['LON']
                ), axis=1
            )
            
            # Find point where bus was closest to stop
            closest_point = route_gps.loc[route_gps['distance_to_stop'].idxmin()]
            distance_to_stop = closest_point['distance_to_stop'] * 1000  # Convert to meters
            
            # Skip if the closest point is too far away (more than 100 meters)
            if distance_to_stop > 100:
                test_logger.warning(
                    f"Skipping stop {stop['Sequence']} {stop['Stop ID']} ({stop['Name']}): "
                    f"closest GPS point is {distance_to_stop:.1f} meters away"
                )
                continue
            
            stop_visits.append({
                'stop_id': stop['Stop ID'],
                'stop_name': stop['Name'],
                'sequence': stop['Sequence'],
                'timestamp': closest_point['Timestamp'],
                'distance': distance_to_stop
            })
        
        # Sort visits by timestamp
        stop_visits.sort(key=lambda x: x['timestamp'])
        
        # Calculate travel times between stops, ensuring sequence only goes up
        results = []
        for i in range(len(stop_visits) - 1):
            current_stop = stop_visits[i]
            next_stop = stop_visits[i + 1]
            
            # Skip if sequence doesn't go up or if there's a gap in sequence
            if (next_stop['sequence'] <= current_stop['sequence'] or 
                next_stop['sequence'] - current_stop['sequence'] > 1):
                test_logger.warning(
                    f"Skipping segment: sequence {current_stop['sequence']} to {next_stop['sequence']} "
                    f"is not consecutive"
                )
                continue
            
            travel_time = (next_stop['timestamp'] - current_stop['timestamp']).total_seconds()
            
            results.append({
                'route_id': route_id,
                'from_stop': current_stop['stop_name'],
                'to_stop': next_stop['stop_name'],
                'from_stop_id': current_stop['stop_id'],
                'to_stop_id': next_stop['stop_id'],
                'from_sequence': current_stop['sequence'],
                'to_sequence': next_stop['sequence'],
                'start_time': current_stop['timestamp'],
                'end_time': next_stop['timestamp'],
                'travel_time_seconds': travel_time,
                'distance_to_stop_meters': current_stop['distance']
            })
        
        test_logger.info(f"Calculated {len(results)} valid segment times for route {route_id}")
        return pd.DataFrame(results)

    def analyze_all_routes(self) -> pd.DataFrame:
        """Analyze all routes in the schedule"""
        all_results = []
        
        for _, row in self.schedule.iterrows():
            test_logger.info(f"Analyzing route {row['route_id']} from {row['start_time']} to {row['end_time']}")
            results = self.analyze_single_route(
                row['route_id'],
                row['start_time'],
                row['end_time']
            )
            all_results.append(results)
        
        return pd.concat(all_results, ignore_index=True)

    def compare_with_predictions(self, actual_times: pd.DataFrame, predictions_path: str) -> pd.DataFrame:
        """
        Compare actual travel times with ETA predictions
        
        Args:
            actual_times: DataFrame of actual travel times
            predictions_path: Path to JSON file with ETA predictions
        """
        test_logger.info(f"Loading predictions from {predictions_path}")
        
        # Load predictions
        with open(predictions_path, 'r') as f:
            predictions = json.load(f)
        
        comparison_results = []
        
        # For each actual segment
        for _, actual in actual_times.iterrows():
            # Find matching prediction using stop IDs
            segment_key = f"{actual['from_stop_id']}:{actual['to_stop_id']}:eta"
            
            if segment_key in predictions:
                # Get all predictions for this segment
                segment_predictions = predictions[segment_key]
                
                # Find the prediction closest to our actual time
                actual_timestamp = actual['end_time'].timestamp()
                closest_prediction = None
                min_time_diff = float('inf')
                
                for pred_key, pred_time in segment_predictions.items():
                    fleet_id, exit_timestamp = pred_key.split(':', 1)
                    time_diff = abs(int(pd.to_datetime(exit_timestamp, format='%Y-%m-%d %H:%M:%S').timestamp()) - actual_timestamp)
                    
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        closest_prediction = pred_time
                
                if closest_prediction is not None:
                    actual_time = actual['travel_time_seconds']
                    pred_time = closest_prediction
                    
                        
                    error = pred_time - actual_time
                    if actual_time > 0:
                        error_percent = (error / actual_time) * 100
                    else:
                        error_percent = None
                    
                    comparison_results.append({
                        'route_id': actual['route_id'],
                        'from_stop': actual['from_stop'],
                        'to_stop': actual['to_stop'],
                        'from_stop_id': actual['from_stop_id'],
                        'to_stop_id': actual['to_stop_id'],
                        'from_sequence': actual['from_sequence'],
                        'to_sequence': actual['to_sequence'],
                        'exit_time': actual['end_time'].strftime('%H:%M:%S'),
                        'actual_time': actual_time,
                        'predicted_time': pred_time,
                        'error_seconds': error,
                        'error_percent': error_percent,
                        'distance_to_stop': actual['distance_to_stop_meters'],
                        'time_diff_to_prediction': min_time_diff
                    })
        
        test_logger.info(f"Compared {len(comparison_results)} segments with predictions")
        return pd.DataFrame(comparison_results)

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points in kilometers"""
        from math import radians, sin, cos, sqrt, asin
        
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        return R * c

def plot_error_histograms(comparison_df: pd.DataFrame, output_dir: str = 'output/plots'):
    """
    Plot histograms of prediction errors, removing outliers
    
    Args:
        comparison_df: DataFrame containing comparison results
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Log high percentage errors (>100%)
    high_percent_errors = comparison_df[comparison_df['error_percent'] > 100]
    if len(high_percent_errors) > 0:
        test_logger.info("High Percentage Errors (>100%):")
        test_logger.info(f"Found {len(high_percent_errors)} segments with >100% error")
        for _, row in high_percent_errors.iterrows():
            test_logger.info(
                f" Route: {row['route_id']}, "
                f" From Stop: {row['from_stop']} (ID: {row['from_stop_id']}, Seq: {row['from_sequence']}), "
                f" To Stop: {row['to_stop']} (ID: {row['to_stop_id']}, Seq: {row['to_sequence']}), "
                f" Exit Time: {row['exit_time']}, "
                f" Actual Time: {row['actual_time']:.1f}s, "
                f" Predicted Time: {row['predicted_time']:.1f}s, "
                f" Error: {row['error_seconds']:.1f}s, "
                f" Error %: {row['error_percent']:.1f}%"
            )
    
    # Remove outliers using IQR method
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Get the outliers
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        
        # Log detailed information about outliers
        test_logger.info(f"Detailed Outlier Information for {column}:")
        test_logger.info(f" Outlier bounds: [{lower_bound:.1f}, {upper_bound:.1f}]")
        test_logger.info(f" Number of outliers: {len(outliers)}")
        
        if len(outliers) > 0:
            test_logger.info("Outlier Details:")
            for _, row in outliers.iterrows():
                test_logger.info(
                    f" Route: {row['route_id']}, "
                    f"From Stop: {row['from_stop']} (ID: {row['from_stop_id']}, Seq: {row['from_sequence']}), "
                    f"To Stop: {row['to_stop']} (ID: {row['to_stop_id']}, Seq: {row['to_sequence']}), "
                    f"Exit Time: {row['exit_time']}, "
                    f"Actual Time: {row['actual_time']:.1f}s, "
                    f"Predicted Time: {row['predicted_time']:.1f}s, "
                    f"Error: {row[column]:.1f}"
                )
        
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    # Remove outliers based on error_seconds only
    clean_df = remove_outliers(comparison_df, 'error_seconds')
    
    # Remove first and last segments of each route for histogram
    route_groups = clean_df.groupby('route_id')
    histogram_df = pd.concat([
        group.iloc[1:-1] if len(group) > 2 else group 
        for _, group in route_groups
    ])
    
    test_logger.info(f"Histogram Statistics:")
    test_logger.info(f" Total segments after outlier removal: {len(clean_df)}")
    test_logger.info(f" Segments used in histogram (excluding first/last): {len(histogram_df)}")
    
    # Log outlier removal statistics
    removed = len(comparison_df) - len(clean_df)
    test_logger.info(f"Outlier Removal Statistics:")
    test_logger.info(f" Removed {removed} outliers ({removed/len(comparison_df)*100:.1f}% of total)")
    
    # Save outliers to CSV file
    outliers = comparison_df[~comparison_df.index.isin(clean_df.index)]
    outliers.to_csv(f'output/comparison_results/outliers_{timestamp}.csv', index=False)
    
    # Save high percentage errors to separate CSV
    high_percent_errors.to_csv(f'output/comparison_results/high_percent_errors_{timestamp}.csv', index=False)
    
    # Set style
    plt.style.use(plt.style.available[0])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Calculate appropriate bin edges
    # For seconds: use bins that go from negative to positive
    max_abs_error = max(abs(histogram_df['error_seconds'].min()), abs(histogram_df['error_seconds'].max()))
    if max_abs_error <= 300:  # 5 minutes
        bin_size_seconds = 5
        bin_edges_seconds = np.arange(-max_abs_error, max_abs_error + bin_size_seconds, bin_size_seconds)
    else:
        bin_size_seconds = 25
        bin_edges_seconds = np.arange(-max_abs_error, max_abs_error + bin_size_seconds, bin_size_seconds)
    
    # For percentage: use 2.5% bins
    bin_size_percent = 2.5
    max_percent = histogram_df['error_percent'].max()
    bin_edges_percent = np.arange(0, max_percent + bin_size_percent, bin_size_percent)
    
    # Plot error in seconds (without outliers and first/last segments)
    sns.histplot(data=histogram_df, x='error_seconds', bins=bin_edges_seconds, ax=ax1)
    ax1.set_title(f'Distribution of Prediction Error (Seconds) - Outliers Removed\nBin Size: {bin_size_seconds} seconds')
    ax1.set_xlabel('Error (seconds)')
    ax1.set_ylabel('Count')
    
    # Add mean and median lines
    mean_error = histogram_df['error_seconds'].mean()
    median_error = histogram_df['error_seconds'].median()
    ax1.axvline(mean_error, color='red', linestyle='--', label=f'Mean: {mean_error:.1f}s')
    ax1.axvline(median_error, color='green', linestyle='--', label=f'Median: {median_error:.1f}s')
    ax1.legend()
    
    # Add bin size annotation
    ax1.text(0.95, 0.95, f'Bin Size: {bin_size_seconds}s',
             transform=ax1.transAxes, ha='right', va='top',
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Plot error percentage (without outliers and first/last segments)
    sns.histplot(data=histogram_df, x='error_percent', bins=bin_edges_percent, ax=ax2)
    ax2.set_title(f'Distribution of Prediction Error (Percentage) - Outliers Removed\nBin Size: {bin_size_percent}%')
    ax2.set_xlabel('Error (%)')
    ax2.set_ylabel('Count')
    
    # Add mean and median lines
    mean_percent = histogram_df['error_percent'].mean()
    median_percent = histogram_df['error_percent'].median()
    ax2.axvline(mean_percent, color='red', linestyle='--', label=f'Mean: {mean_percent:.1f}%')
    ax2.axvline(median_percent, color='green', linestyle='--', label=f'Median: {median_percent:.1f}%')
    ax2.legend()
    
    # Add bin size annotation
    ax2.text(0.95, 0.95, f'Bin Size: {bin_size_percent}%',
             transform=ax2.transAxes, ha='right', va='top',
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'error_histograms_{timestamp}.png')
    plt.savefig(plot_path)
    plt.close()
    
    test_logger.info(f"Saved error histograms to {plot_path}")
    
    # Log bin information
    test_logger.info("Histogram Bin Information:")
    test_logger.info(f" Seconds histogram: {len(bin_edges_seconds)-1} bins of {bin_size_seconds} seconds each")
    test_logger.info(f" Percentage histogram: {len(bin_edges_percent)-1} bins of {bin_size_percent}% each")
    
    # Print summary statistics (without outliers and first/last segments)
    test_logger.info("Error Distribution Statistics (Outliers Removed, First/Last Segments Excluded):")
    test_logger.info(f" Seconds - Mean: {mean_error:.1f}, Median: {median_error:.1f}")
    test_logger.info(f" Percent - Mean: {mean_percent:.1f}, Median: {median_percent:.1f}")
    
    # Calculate percentiles (without outliers and first/last segments)
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        error_p = histogram_df['error_seconds'].quantile(p/100)
        test_logger.info(f"{p}th percentile error: {error_p:.1f} seconds")
    
    # Save cleaned data
    clean_df.to_csv(f'output/comparison_results/cleaned_comparison_results_{timestamp}.csv', index=False)
    
    return clean_df  # Return cleaned data for further analysis if needed

def get_latest_cache_file(directory: str = 'output', prefix: str = 'travel_time_cache_') -> str:
    """
    Find the most recent travel time cache JSON file
    
    Args:
        directory: Directory to search in
        prefix: Prefix of the cache files
    
    Returns:
        Path to the most recent cache file
    """
    # Get all files matching the pattern
    cache_files = [f for f in os.listdir(directory) 
                  if f.startswith(prefix) and f.endswith('.json')]
    
    if not cache_files:
        raise FileNotFoundError(f"No cache files found in {directory}")
    
    # Sort by timestamp in filename (assuming format: prefix_YYYYMMDD_HHMMSS.json)
    latest_file = sorted(cache_files)[-1]
    
    return os.path.join(directory, latest_file)

if __name__ == "__main__":
    # Configure for testing  
    fleet_id = 'K0377'
    date = '2025-05-02'

    test_logger.info(f"Testing for fleet {fleet_id} on {date}")
    # Example usage
    analyzer = RouteTimeAnalyzer(
        gps_data_path=AMNEX_DATA_PATH,
        route_stops_path=ROUTE_STOPS_PATH,
        schedule_path=TEST_SCHEDULE_PATH.format(fleet_id=fleet_id, date=date),
        fleet_device_path=FLEET_DEVICE_PATH,
        fleet_id=fleet_id,
        date=date
    )

    test_logger.info("Analyzer set up")

    # Analyze all routes
    actual_times = analyzer.analyze_all_routes()
    
    actual_times.to_csv(f'output/actual_times/actual_times_{fleet_id}_{date}.csv', index=False)
    
    test_logger.info("Analyzed all routes actual times")

    latest_cache = get_latest_cache_file()
    test_logger.info(f"Using cache file: {latest_cache}")
    
    # Compare with predictions
    comparison = analyzer.compare_with_predictions(
        actual_times,
        latest_cache
    )
    
    # Plot error histograms and get cleaned data
    cleaned_comparison = plot_error_histograms(comparison)
    
    # Print final summary statistics using cleaned data
    test_logger.info("Final Accuracy Statistics (Outliers Removed):")
    test_logger.info(f"Total segments analyzed: {len(comparison)}")
    test_logger.info(f"Segments after outlier removal: {len(cleaned_comparison)}")
    
    # Time range statistics
    test_logger.info("Time Range Statistics:")
    test_logger.info("Actual Times:")
    test_logger.info(f"  Mean: {cleaned_comparison['actual_time'].mean():.1f} seconds")
    test_logger.info(f"  Median: {cleaned_comparison['actual_time'].median():.1f} seconds")
    test_logger.info(f"  Min: {cleaned_comparison['actual_time'].min():.1f} seconds")
    test_logger.info(f"  Max: {cleaned_comparison['actual_time'].max():.1f} seconds")
    test_logger.info(f"  Standard deviation: {cleaned_comparison['actual_time'].std():.1f} seconds")
    
    test_logger.info("Predicted Times:")
    test_logger.info(f"  Mean: {cleaned_comparison['predicted_time'].mean():.1f} seconds")
    test_logger.info(f"  Median: {cleaned_comparison['predicted_time'].median():.1f} seconds")
    test_logger.info(f"  Min: {cleaned_comparison['predicted_time'].min():.1f} seconds")
    test_logger.info(f"  Max: {cleaned_comparison['predicted_time'].max():.1f} seconds")
    test_logger.info(f"  Standard deviation: {cleaned_comparison['predicted_time'].std():.1f} seconds")
    
    # Signed error statistics (shows bias)
    test_logger.info("Signed Error Statistics (shows prediction bias):")
    test_logger.info(f" Mean signed error: {cleaned_comparison['error_seconds'].mean():.1f} seconds")
    test_logger.info(f" Median signed error: {cleaned_comparison['error_seconds'].median():.1f} seconds")
    test_logger.info(f" Mean signed error percentage: {cleaned_comparison['error_percent'].mean():.1f}%")
    
    # Absolute error statistics (shows magnitude of error)
    test_logger.info("Absolute Error Statistics (shows error magnitude):")
    test_logger.info(f" Mean absolute error: {cleaned_comparison['error_seconds'].abs().mean():.1f} seconds")
    test_logger.info(f" Median absolute error: {cleaned_comparison['error_seconds'].abs().median():.1f} seconds")
    test_logger.info(f" Mean absolute error percentage: {cleaned_comparison['error_percent'].abs().mean():.1f}%")
    
    # Additional statistics
    test_logger.info("Error Distribution (Outliers Removed):")
    test_logger.info(f" Standard deviation: {cleaned_comparison['error_seconds'].std():.1f} seconds")
    test_logger.info(f" Min error: {cleaned_comparison['error_seconds'].min():.1f} seconds")
    test_logger.info(f" Max error: {cleaned_comparison['error_seconds'].max():.1f} seconds")
    
    # Split statistics for over/under/perfect predictions
    over_predictions = cleaned_comparison[cleaned_comparison['error_seconds'] > 0]
    under_predictions = cleaned_comparison[cleaned_comparison['error_seconds'] < 0]
    perfect_predictions = cleaned_comparison[cleaned_comparison['error_seconds'] == 0]
    
    test_logger.info("Prediction Bias Analysis:")
    test_logger.info(f" Total segments: {len(cleaned_comparison)}")
    test_logger.info(f" Over-predictions (too slow): {len(over_predictions)} segments ({len(over_predictions)/len(cleaned_comparison)*100:.1f}%)")
    test_logger.info(f"   Mean over-prediction: {over_predictions['error_seconds'].mean():.1f} seconds")
    test_logger.info(f"   Mean over-prediction percentage: {over_predictions['error_percent'].mean():.1f}%")
    
    test_logger.info(f" Under-predictions (too fast): {len(under_predictions)} segments ({len(under_predictions)/len(cleaned_comparison)*100:.1f}%)")
    test_logger.info(f"   Mean under-prediction: {under_predictions['error_seconds'].mean():.1f} seconds")
    test_logger.info(f"   Mean under-prediction percentage: {under_predictions['error_percent'].mean():.1f}%")
    
    test_logger.info(f" Perfect predictions (error = 0): {len(perfect_predictions)} segments ({len(perfect_predictions)/len(cleaned_comparison)*100:.1f}%)")
    
    # Percentile statistics for absolute errors
    percentiles = [50, 75, 90, 95, 99]
    test_logger.info("Percentile Statistics (Absolute Errors):")
    for p in percentiles:
        error_p = cleaned_comparison['error_seconds'].abs().quantile(p/100)
        test_logger.info(f" {p}th percentile absolute error: {error_p:.1f} seconds")