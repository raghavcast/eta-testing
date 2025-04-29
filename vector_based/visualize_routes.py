#!/usr/bin/env python3
"""
Script to launch the Bus Route Vector Visualizer

This script checks if the necessary data files exist and launches
the Node.js application for route visualization.
"""

import os
import sys
import subprocess
import webbrowser
import time
import argparse

def check_data_files():
    """Check if the required data files exist"""
    required_files = [
        'data/generated_bus_route_data.csv', 
        'data/waybill_metabase.csv', 
        'data/route_stop_mapping.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    return missing_files

def build_visualizer():
    """Build the visualization application"""
    print("Building the visualization application...")
    
    try:
        result = subprocess.run(
            ['npm', 'run', 'build', '--legacy-peer-deps'], 
            cwd='bus-route-visualizer',
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Build successful!")
            return True
        else:
            print("❌ Build failed.")
            print(result.stderr)
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error building the application: {e}")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"❌ Unexpected error during build: {e}")
        return False

def launch_visualizer(rebuild=False):
    """Launch the visualization application"""
    # Check if Node.js is installed
    try:
        subprocess.run(['node', '--version'], 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE, 
                       check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: Node.js is not installed. Please install Node.js to use this visualizer.")
        return False
    
    # Check if the required data files exist
    missing_files = check_data_files()
    if missing_files:
        print("❌ Error: The following required data files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    # Check if the visualizer app exists
    if not os.path.exists('bus-route-visualizer'):
        print("❌ Error: The visualizer application directory is missing.")
        return False
    
    # Check if the project has node_modules
    if not os.path.exists('bus-route-visualizer/node_modules'):
        print("⚠️ Node modules not found. Installing dependencies...")
        try:
            subprocess.run(
                ['npm', 'install', '--legacy-peer-deps'], 
                cwd='bus-route-visualizer',
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    # Build the application if requested
    if rebuild and not build_visualizer():
        return False
        
    # Launch the server
    print("🚀 Starting Bus Route Visualizer server...")
    server_process = subprocess.Popen(
        ['npm', 'run', 'server'], 
        cwd='bus-route-visualizer',
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    time.sleep(2)
    
    # Check if server started successfully
    if server_process.poll() is not None:
        print("❌ Error: Failed to start the server.")
        return False
    
    # Open browser
    print("🌐 Opening visualization in your web browser...")
    webbrowser.open('http://localhost:3001')
    
    print("\n🟢 Bus Route Vector Visualizer is running.")
    print("🔴 Press Ctrl+C to stop the server when you're done.")
    
    try:
        # Keep the server running until user interrupts
        server_process.wait()
    except KeyboardInterrupt:
        print("\n⏹️ Stopping the server...")
        server_process.terminate()
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Bus Route Vector Visualizer')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild the application before launching')
    args = parser.parse_args()
    
    print("🚍 Bus Route Vector Visualizer")
    print("===========================\n")
    
    success = launch_visualizer(rebuild=args.rebuild)
    
    if not success:
        print("\n❌ Failed to launch the Bus Route Vector Visualizer.")
        sys.exit(1)

if __name__ == "__main__":
    main() 