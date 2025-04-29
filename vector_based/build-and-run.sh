#!/bin/bash

# Script to build and run the Bus Route Visualizer

cd bus-route-visualizer

# Build the app
echo "Building the application..."
npm run build --legacy-peer-deps

if [ $? -eq 0 ]; then
  echo "Build successful!"
  
  # Run the server
  echo "Starting the server..."
  npm run server
else
  echo "Build failed. Please check the errors above."
  exit 1
fi 