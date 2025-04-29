# Bus Route Vector Visualizer

This application visualizes bus routes and the vectors used for direction determination. It allows users to select different bus devices and see their routes, locations, and the vector calculations used to determine the direction of travel.

## Features

- Interactive map display of bus routes
- Visualization of both bus vectors and stop vectors with arrow representations
- Selection of different bus devices
- Display of route details including direction determination
- Marker indicators for bus locations and stops
- Visual comparison of bus movement vector and stop-based route vector
- Direction similarity analysis based on vector dot product

## Prerequisites

- Node.js (v14 or higher)
- npm (v6 or higher)

## Data Requirements

The application requires the following CSV files to be in the `../data/` directory:

- `generated_bus_route_data.csv`: Contains bus location data
- `waybill_metabase.csv`: Contains device to route mapping information
- `route_stop_mapping.csv`: Contains route stop information

## Installation

1. Clone the repository
2. Navigate to the project directory
3. Install dependencies

```bash
cd bus-route-visualizer
npm install --legacy-peer-deps
```

## Running the Application

The application consists of two parts:

1. A React frontend for the visualization
2. An Express server for serving data files

You can run both together using:

```bash
npm run dev
```

Or individually:

- Frontend only: `npm run start`
- Server only: `npm run server`

The application will be available at:
- Frontend: http://localhost:3000
- Server API: http://localhost:3001

## How It Works

1. The application loads data from CSV files using the Express server
2. When a device is selected, it calculates the vectors using the same logic as in `direction_determination.py`
3. The map displays:
   - The bus route as a blue polyline
   - Bus locations as blue markers
   - Stops as red markers
   - The bus vector as a blue arrow with arrowhead
   - The stop vector as a red arrow with arrowhead
   - The nearest stop is highlighted with a red circle
   - The vector analysis point (where vectors are compared) is highlighted with a purple circle

## Vector Visualization

The application uses the following visual elements to represent vectors:

- **Blue Arrow**: Represents the Bus Vector - direction calculated from bus location history
- **Red Arrow**: Represents the Stop Vector - direction calculated from stop locations
- **Purple Circle**: The vector analysis point where vectors are compared
- **Red Circle**: Marks the nearest stop to current bus location

The similarity between vectors is calculated using the dot product, which gives values:
- Close to 1: Vectors point in the same direction
- Close to 0: Vectors are perpendicular
- Close to -1: Vectors point in opposite directions

## Vector Calculation Logic

The application uses the following steps to determine direction:

1. Calculate a normalized vector from the bus's location points
2. Find the nearest stop to the current bus location
3. Get neighboring stops to calculate a route vector
4. Compare the bus vector and route vector using dot product (similarity)
5. Determine the direction based on the similarity value

## API Endpoints

- `/api/files`: Lists all available CSV files
- `/api/data/:filename`: Serves a specific CSV file
- `/api/vectors/:deviceId`: Placeholder for direct vector computation

## Technologies Used

- React.js for the frontend
- Leaflet for mapping (via react-leaflet)
- Express.js for the backend server
- Webpack for building
- Chakra UI for the interface components

## Credits

This application is based on the direction determination logic from the `direction_determination.py` script, adapted for JavaScript and visualized using React and Leaflet. 