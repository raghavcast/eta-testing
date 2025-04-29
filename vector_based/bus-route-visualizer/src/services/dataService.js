import axios from 'axios';

// Base URL for the API
const API_BASE_URL = 'http://localhost:3001/api';

// Function to read CSV file and convert to JSON
const readCsvFile = async (filename) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/data/${filename}`);
    const csvText = response.data;
    
    // Simple CSV parser
    const lines = typeof csvText === 'string' ? csvText.split('\n') : [];
    if (lines.length === 0) {
      throw new Error('CSV file is empty or not properly formatted');
    }
    
    const headers = lines[0].split(',');
    
    return lines.slice(1).filter(line => line.trim() !== '').map(line => {
      const values = line.split(',');
      return headers.reduce((obj, header, index) => {
        obj[header] = values[index];
        return obj;
      }, {});
    });
  } catch (error) {
    console.error('Error reading CSV file:', error);
    throw new Error(`Failed to read CSV file: ${error.message}`);
  }
};

// Fetch all unique device IDs from the bus data
export const fetchDeviceIds = async () => {
  try {
    const busData = await readCsvFile('generated_bus_route_data.csv');
    
    // Extract unique device IDs
    const deviceIds = [...new Set(busData.map(item => item.deviceId))];
    
    return deviceIds;
  } catch (error) {
    console.error('Error fetching device IDs:', error);
    throw new Error(`Failed to fetch device IDs: ${error.message}`);
  }
};

// Fetch all data necessary for route visualization
export const fetchRouteData = async (deviceId) => {
  try {
    // Load all required data
    const busData = await readCsvFile('generated_bus_route_data.csv');
    const waybillData = await readCsvFile('waybill_metabase.csv');
    const routeStopData = await readCsvFile('route_stop_mapping.csv');
    
    // Filter bus data for the selected device
    const deviceBusData = busData.filter(item => item.deviceId === deviceId);
    
    if (deviceBusData.length === 0) {
      throw new Error(`No data found for device ID: ${deviceId}`);
    }
    
    // Convert string values to proper types
    const processedBusData = deviceBusData.map(item => ({
      deviceId: item.deviceId,
      lat: parseFloat(item.lat),
      long: parseFloat(item.long),
      date: new Date(item.date),
      provider: item.provider,
      dataState: item.dataState
    }));
    
    // Sort by date
    processedBusData.sort((a, b) => a.date - b.date);
    
    // Extract route information from waybill
    const waybillMatch = waybillData.find(item => item['Device Serial Number'] === deviceId);
    
    if (!waybillMatch) {
      throw new Error(`No waybill entry found for device ID: ${deviceId}`);
    }
    
    // Extract route number from schedule
    const scheduleNo = waybillMatch['Schedule No'];
    const routeNumberMatch = scheduleNo.match(/^.*?-(.+?)-/);
    const routeNumber = routeNumberMatch ? routeNumberMatch[1] : null;
    
    if (!routeNumber) {
      throw new Error(`Could not extract route number from schedule: ${scheduleNo}`);
    }
    
    // Get route stops
    const routeStops = routeStopData.filter(item => item['MTC ROUTE NO'] === routeNumber);
    
    if (routeStops.length === 0) {
      throw new Error(`No stops found for route number: ${routeNumber}`);
    }
    
    // Process stop data
    const processedStopData = routeStops.map(item => ({
      tummoc_id: item['TUMMOC Route ID'],
      route_num: item['MTC ROUTE NO'],
      stop_id: item['Stop ID'],
      stop_sequence: parseInt(item['Sequence']),
      stop_name: item['Name'],
      stop_latitude: parseFloat(item['LAT']),
      stop_longitude: parseFloat(item['LON']),
      direction: item['DIRECTION'],
      source: item['SOURCE'],
      destination: item['DESTIN']
    }));
    
    // Calculate bus vector
    const busVector = calculateVector(processedBusData);
    
    // Find the nearest stop to the last bus location
    const lastBusPoint = processedBusData[processedBusData.length - 1];
    
    // Find stops within some distance
    const nearbyStops = findStopsWithinDistance(lastBusPoint, processedStopData);
    
    // Group by tummoc_id
    const tummocGroups = {};
    processedStopData.forEach(stop => {
      if (!tummocGroups[stop.tummoc_id]) {
        tummocGroups[stop.tummoc_id] = [];
      }
      tummocGroups[stop.tummoc_id].push(stop);
    });
    
    // Prepare matches array
    const matches = [];
    
    for (const tummocId in tummocGroups) {
      const routeStops = tummocGroups[tummocId];
      const direction = routeStops[0].direction;
      
      // Sort stops by sequence
      routeStops.sort((a, b) => a.stop_sequence - b.stop_sequence);
      
      // Find the nearest stop
      let nearestStop = null;
      let minDistance = Infinity;
      
      routeStops.forEach(stop => {
        const distance = haversine(
          lastBusPoint.long, lastBusPoint.lat,
          stop.stop_longitude, stop.stop_latitude
        );
        
        if (distance < minDistance) {
          minDistance = distance;
          nearestStop = stop;
        }
      });
      
      if (nearestStop) {
        // Get neighboring stops
        const neighbors = getNeighboringStops(routeStops, nearestStop.stop_id);
        
        console.log("Neighbors for stop:", nearestStop.stop_id, neighbors);
        
        // Calculate stop vector
        const stopsForVector = [nearestStop, ...neighbors];
        const stopVector = calculateStopVector(stopsForVector);
        
        console.log("Calculated stop vector:", stopVector, "from stops:", stopsForVector.map(s => s.stop_id));
        
        // Calculate similarity
        const similarity = vectorSimilarity(busVector, stopVector);
        
        matches.push({
          tummoc_id: tummocId,
          direction,
          similarity,
          nearest_stop_id: nearestStop.stop_id,
          nearest_stop_name: nearestStop.stop_name,
          stop_vector: stopVector
        });
      }
    }
    
    // Sort matches by similarity (highest first)
    matches.sort((a, b) => b.similarity - a.similarity);
    
    return {
      device_id: deviceId,
      route_number: routeNumber,
      bus_vector: busVector,
      num_points_used: processedBusData.length,
      location_data: processedBusData,
      stop_data: processedStopData,
      matches
    };
  } catch (error) {
    console.error('Error fetching route data:', error);
    throw new Error(`Failed to fetch route data: ${error.message}`);
  }
};

// Helper functions from direction_determination.py
function haversine(lon1, lat1, lon2, lat2) {
  // Convert decimal degrees to radians
  lon1 = toRadians(lon1);
  lat1 = toRadians(lat1);
  lon2 = toRadians(lon2);
  lat2 = toRadians(lat2);
  
  // Haversine formula
  const dlon = lon2 - lon1;
  const dlat = lat2 - lat1;
  const a = Math.sin(dlat/2)**2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dlon/2)**2;
  const c = 2 * Math.asin(Math.sqrt(a));
  const r = 6371; // Radius of earth in kilometers
  
  return c * r;
}

function toRadians(degrees) {
  return degrees * (Math.PI / 180);
}

function calculateVector(points) {
  // Need at least 2 points to calculate a direction
  if (!points || points.length < 2) {
    console.log("Warning: Not enough points to calculate bus vector", points);
    // Default to east direction if we can't calculate
    return [1, 0];
  }
  
  // For bus vectors, we use the first few points to get a more accurate initial direction
  // Take the first 5 points or all points if less than 5
  const pointsToUse = points.slice(0, Math.min(5, points.length));
  
  if (pointsToUse.length >= 2) {
    // Use the first and last of these points to get direction
    const firstPoint = pointsToUse[0];
    const lastPoint = pointsToUse[pointsToUse.length - 1];
    
    const dx = lastPoint.long - firstPoint.long;
    const dy = lastPoint.lat - firstPoint.lat;
    
    // Normalize the vector
    const magnitude = Math.sqrt(dx**2 + dy**2);
    if (magnitude > 0) {
      console.log("Bus vector calculated from first-last points:", [dx/magnitude, dy/magnitude]);
      return [dx/magnitude, dy/magnitude];
    }
  }
  
  // If we get here, try the old approach (calculating across all points)
  let dx = 0;
  let dy = 0;
  
  for (let i = 1; i < points.length; i++) {
    dy += points[i].lat - points[i-1].lat;
    dx += points[i].long - points[i-1].long;
  }
  
  // Normalize the vector
  const magnitude = Math.sqrt(dx**2 + dy**2);
  if (magnitude > 0) {
    console.log("Bus vector calculated across all points:", [dx/magnitude, dy/magnitude]);
    return [dx/magnitude, dy/magnitude];
  } else {
    console.log("Warning: Zero magnitude in bus vector calculation. Using default.");
    // Default to east direction if we can't calculate
    return [1, 0];
  }
}

function vectorSimilarity(v1, v2) {
  // Calculate dot product
  const dotProduct = v1[0]*v2[0] + v1[1]*v2[1];
  return dotProduct;
}

function findStopsWithinDistance(busPoint, stopsData, maxDistance = 1.0) {
  return stopsData.filter(stop => {
    const distance = haversine(
      busPoint.long, busPoint.lat,
      stop.stop_longitude, stop.stop_latitude
    );
    return distance <= maxDistance;
  }).sort((a, b) => {
    const distA = haversine(
      busPoint.long, busPoint.lat,
      a.stop_longitude, a.stop_latitude
    );
    const distB = haversine(
      busPoint.long, busPoint.lat,
      b.stop_longitude, b.stop_latitude
    );
    return distA - distB;
  });
}

function getNeighboringStops(stopsData, stopId) {
  // Sort by stop sequence to ensure correct ordering
  const sortedStops = [...stopsData].sort((a, b) => a.stop_sequence - b.stop_sequence);
  const neighbors = [];
  
  // Find the index of the stop
  const stopIndex = sortedStops.findIndex(stop => stop.stop_id === stopId);
  
  if (stopIndex === -1) {
    return [];
  }
  
  // Get previous stop (if exists)
  if (stopIndex > 0) {
    neighbors.push(sortedStops[stopIndex - 1]);
  }
  
  // Get next stop (if exists)
  if (stopIndex < sortedStops.length - 1) {
    neighbors.push(sortedStops[stopIndex + 1]);
  }
  
  return neighbors;
}

function calculateStopVector(stops) {
  if (!stops || stops.length < 2) {
    console.log("Warning: Not enough stops to calculate vector", stops);
    // Default to east direction if we can't calculate
    return [1, 0];
  }
  
  // Sort stops by sequence to ensure correct ordering
  const sortedStops = [...stops].sort((a, b) => a.stop_sequence - b.stop_sequence);
  
  // Use the first and last stop to get direction if we have multiple stops
  if (sortedStops.length >= 2) {
    const firstStop = sortedStops[0];
    const lastStop = sortedStops[sortedStops.length - 1];
    
    const dx = lastStop.stop_longitude - firstStop.stop_longitude;
    const dy = lastStop.stop_latitude - firstStop.stop_latitude;
    
    // Normalize the vector
    const magnitude = Math.sqrt(dx**2 + dy**2);
    if (magnitude > 0) {
      console.log("Stop vector calculated from first-last stops:", [dx/magnitude, dy/magnitude]);
      return [dx/magnitude, dy/magnitude];
    }
  }
  
  // If we get here, try the old approach (calculating across all stops)
  let dx = 0;
  let dy = 0;
  
  for (let i = 1; i < sortedStops.length; i++) {
    dy += sortedStops[i].stop_latitude - sortedStops[i-1].stop_latitude;
    dx += sortedStops[i].stop_longitude - sortedStops[i-1].stop_longitude;
  }
  
  // Normalize the vector
  const magnitude = Math.sqrt(dx**2 + dy**2);
  if (magnitude > 0) {
    console.log("Stop vector calculated across all stops:", [dx/magnitude, dy/magnitude]);
    return [dx/magnitude, dy/magnitude];
  } else {
    console.log("Warning: Zero magnitude in stop vector calculation. Using default.");
    // Default to east direction if we can't calculate
    return [1, 0];
  }
} 