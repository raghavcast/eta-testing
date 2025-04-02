import clickhouse_connect
import pandas as pd
from datetime import datetime
from bus_eta_service.config.settings import settings
import urllib3
import socket

def test_connection(host, port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"Socket test error: {str(e)}")
        return False

def read_locations_to_csv():
    print(f"Connecting to ClickHouse at {settings.CLICKHOUSE_HOST}:{settings.CLICKHOUSE_PORT}")
    print(f"Using database: {settings.CLICKHOUSE_DB_LOCATIONS}")
    
    # Test basic connectivity first
    print("Testing basic connectivity...")
    if not test_connection(settings.CLICKHOUSE_HOST, settings.CLICKHOUSE_PORT):
        print(f"Warning: Cannot connect to {settings.CLICKHOUSE_HOST}:{settings.CLICKHOUSE_PORT}")
        return
    
    try:
        # Initialize ClickHouse client with debug logging
        print("Initializing ClickHouse client...")
        print(f"Username: {settings.CLICKHOUSE_USERNAME}")
        print(f"Database: {settings.CLICKHOUSE_DB_LOCATIONS}")
        
        client = clickhouse_connect.get_client(
            host=settings.CLICKHOUSE_HOST,
            port=settings.CLICKHOUSE_PORT,
            username=settings.CLICKHOUSE_USERNAME,
            password=settings.CLICKHOUSE_PASSWORD,
            database='atlas_kafka'
        )
        
        # Test connection with a simple query
        print("Testing connection with simple query...")
        test_result = client.query('SELECT 1')
        print("Connection test successful!")
        
        # Check table schema first
        print("Checking table schema...")
        schema_query = "DESCRIBE amnex_direct_data"
        schema_result = client.query(schema_query)
        print("\nTable schema:")
        for row in schema_result.result_rows:
            print(f"Column: {row[0]}, Type: {row[1]}")
        
        # Query to get recent location data
        print('\nExecuting main query...')
        query = """
        SELECT 
            *
        FROM amnex_direct_data
        WHERE date >= now() - INTERVAL 24 HOUR
        AND provider = 'chalo'
        AND dataState LIKE '%L%'
        LIMIT 10000  -- Increased limit for more data points
        """
        
        print("Executing query...")
        result = client.query(query)
        
        print("Converting results to DataFrame...")
        df = pd.DataFrame(result.result_rows, columns=result.column_names)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"locations_{timestamp}.csv"
        
        print(f"Saving {len(df)} location records to {filename}...")
        df.to_csv(filename, index=False)
        print(f"Successfully saved locations to {filename}")
        print(f"Total records: {len(df)}")
        
    except clickhouse_connect.driver.exceptions.OperationalError as e:
        print(f"ClickHouse Operational Error: {str(e)}")
        print(f"Error type: {type(e)}")
        if hasattr(e, '__cause__'):
            print(f"Caused by: {e.__cause__}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(f"Error type: {type(e)}")
        if hasattr(e, '__cause__'):
            print(f"Caused by: {e.__cause__}")
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    read_locations_to_csv() 