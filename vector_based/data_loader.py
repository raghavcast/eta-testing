import pandas as pd
import os

def load_bus_route_source():
    """Load bus route source data into a DataFrame"""
    try:
        df = pd.read_csv('data/bus_route_source.csv')
        print(f"Loaded bus_route_source.csv with {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading bus_route_source.csv: {e}")
        return None

def load_waybill_report():
    """Load waybill report data into a DataFrame"""
    try:
        file_path = 'data/waybill_metabase.csv'
        df = pd.read_csv(file_path)
        print(f"Loaded {file_path} with {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading waybill report: {e}")
        return None

def load_route_stop_mapping():
    """Load route stop mapping data into a DataFrame"""
    try:
        df = pd.read_csv('data/route_stop_mapping.csv')
        print(f"Loaded route_stop_mapping.csv with {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading route_stop_mapping.csv: {e}")
        return None

def load_all_data():
    """Load all required dataframes"""
    results = {
        'bus_route_source': load_bus_route_source(),
        'waybill_report': load_waybill_report(),
        'route_stop_mapping': load_route_stop_mapping()
    }
    
    # Check if any dataframes failed to load
    failed = [name for name, df in results.items() if df is None]
    if failed:
        print(f"Failed to load the following dataframes: {', '.join(failed)}")
    
    return results


if __name__ == "__main__":
    # Test loading all data
    data = load_all_data()
    
    # Print first few rows of each dataframe
    for name, df in data.items():
        if df is not None:
            print(f"\nPreview of {name}:")
            print(df.head())
            print(f"Columns: {df.columns.tolist()}") 