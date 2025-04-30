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
    
    console.log(`Parsing CSV file ${filename} with ${lines.length} lines`);
    
    // Get headers and trim whitespace
    const headers = lines[0].split(',').map(header => header.trim());
    console.log(`CSV headers: ${headers.join(', ')}`);
    
    // Parse data rows
    const results = lines.slice(1)
      .filter(line => line.trim() !== '')
      .map(line => {
        const values = line.split(',');
        
        // Create object with trimmed values
        return headers.reduce((obj, header, index) => {
          // Trim values and handle missing values
          const value = index < values.length ? values[index].trim() : '';
          obj[header] = value;
          return obj;
        }, {});
      });
    
    console.log(`Successfully parsed ${results.length} data rows from ${filename}`);
    
    // Log first row for debugging
    if (results.length > 0) {
      console.log(`First row sample: ${JSON.stringify(results[0])}`);
    }
    
    return results;
  } catch (error) {
    console.error(`Error reading CSV file ${filename}:`, error);
    throw new Error(`Failed to read CSV file ${filename}: ${error.message}`);
  }
};

// Fetch all unique device IDs from the bus data
export const fetchDeviceIds = async () => {
  try {
    const busData = await readCsvFile('generated_bus_route_data.csv');
    const mappingData = await readCsvFile('fleet_device_mapping.csv');
    
    console.log(`Loaded ${busData.length} bus data records and ${mappingData.length} mapping records`);
    
    // Create a map of deviceId -> fleetNumber using the mapping data
    const deviceMap = new Map();
    
    // Extract unique device IDs - use Number for consistency
    const deviceIds = [...new Set(busData.map(item => Number(item.deviceId)))].map(id => String(id));
    console.log(`Found ${deviceIds.length} unique device IDs`);
    
    // Find fleet numbers for each device ID
    deviceIds.forEach(deviceId => {
      // Find the matching mapping entry (use Number for comparison)
      const mappingEntry = mappingData.find(item => Number(item['Chalo DeviceID']) === Number(deviceId));
      
      if (mappingEntry) {
        deviceMap.set(deviceId, mappingEntry['Fleet']);
        console.log(`Mapped device ID ${deviceId} to fleet number ${mappingEntry['Fleet']}`);
      } else {
        deviceMap.set(deviceId, null);
        console.log(`Warning: No fleet mapping found for device ID ${deviceId}`);
      }
    });
    
    // Convert the map to an array of objects
    const deviceObjects = Array.from(deviceMap).map(([id, fleetNumber]) => ({
      id,
      fleetNumber
    }));
    
    console.log(`Returning ${deviceObjects.length} device objects`);
    return deviceObjects;
  } catch (error) {
    console.error('Error fetching device IDs:', error);
    throw new Error(`Failed to fetch device IDs: ${error.message}`);
  }
};

// Find the fleet number from mapping data
const findFleetNumber = (mappingData, deviceId) => {
  if (!deviceId) {
    console.error('findFleetNumber called with null or undefined deviceId');
    return null;
  }
  
  // Convert to string for comparison and clean the input
  const deviceIdStr = String(deviceId).trim();
  
  console.log(`Looking for fleet number for device ID: '${deviceIdStr}'`);
  
  // First try exact string match
  let mappingEntry = mappingData.find(item => Number(item['Chalo DeviceID']) === Number(deviceId));
  
  if (!mappingEntry) {
    console.log(`No exact match found for device ID: ${deviceIdStr}, trying numeric match`);
    
    // // Try numeric matching in case of formatting differences
    // const deviceIdNum = Number(deviceId);
    // if (!isNaN(deviceIdNum)) {
    //   mappingEntry = mappingData.find(item => {
    //     const mappingIdStr = String(item['Chalo DeviceID']).trim();
    //     const mappingIdNum = Number(mappingIdStr);
    //     return !isNaN(mappingIdNum) && mappingIdNum === deviceIdNum;
    //   });
    // }
  }
  
  if (mappingEntry) {
    const fleetNumber = mappingEntry['Fleet'];
    console.log(`Found fleet mapping: Device ID ${deviceIdStr} -> Fleet ${fleetNumber}`);
    return fleetNumber;
  }
  
  // If we got here, no match was found
  console.error(`No mapping found for device ID: ${deviceIdStr}`);
  
  // Last resort: dump all mapping entries for debugging
  console.log("Available mappings (first 10):");
  mappingData.slice(0, 10).forEach(entry => {
    console.log(`  Chalo DeviceID: '${entry['Chalo DeviceID']}' (${typeof entry['Chalo DeviceID']}) -> Fleet: '${entry['Fleet']}'`);
  });
  
  return null;
};

// Find matching waybill entry
const findWaybillMatch = (waybillData, fleetNumber) => {
  if (!fleetNumber) {
    console.error('findWaybillMatch called with null or undefined fleetNumber');
    return null;
  }
  
  // Normalize fleet number
  const fleetStr = String(fleetNumber).trim();
  
  console.log(`Looking for waybill match with fleet number: '${fleetStr}'`);
  
  // Try exact match first (case sensitive)
  let match = waybillData.find(item => String(item['Vehicle No']).trim() === fleetStr);
  
  if (!match) {
    console.log(`No exact match found for fleet number: ${fleetStr}, trying case-insensitive match`);
    
    // Try case-insensitive match
    // match = waybillData.find(item => 
    //   String(item['Vehicle No']).trim().toLowerCase() === fleetStr.toLowerCase()
    // );
  }
  
  if (!match) {
    console.log(`No case-insensitive match found for fleet number: ${fleetStr}, trying numeric match`);
    
    // Try numeric matching if the fleet number appears to be numeric
    // const fleetNum = Number(
  }
  
  if (match) {
    console.log(`Found waybill match for fleet number ${fleetStr}: Vehicle No: '${match['Vehicle No']}', Route: ${match['Route No']}`);
    return match;
  }
  
  // Debug information if no match found
  console.error(`No waybill found for fleet number: ${fleetStr}`);
  console.log("Sample waybill Vehicle No entries (first 10):");
  waybillData.slice(0, 10).forEach(item => {
    console.log(`  Vehicle No: '${item['Vehicle No']}' (${typeof item['Vehicle No']})`);
  });
  
  return null;
};

// Fetch all data necessary for route visualization
export const fetchRouteData = async (deviceId) => {
  try {
    // Load all required data
    const busData = await readCsvFile('generated_bus_route_data.csv');
    const waybillData = await readCsvFile('waybill_metabase.csv');
    const routeStopData = await readCsvFile('route_stop_mapping.csv');
    const mappingData = await readCsvFile('fleet_device_mapping.csv');
    
    // Filter bus data for the selected device
    const deviceBusData = busData.filter(item => Number(item.deviceId) === Number(deviceId));
    
    if (deviceBusData.length === 0) {
      throw new Error(`No data found for device ID: ${deviceId}`);
    }
    
    // Use the helper function to find fleet number
    const fleetNumber = findFleetNumber(mappingData, deviceId);
    
    if (!fleetNumber) {
      console.error(`Could not find fleet number for device ID: ${deviceId} in mapping data`);
      throw new Error(`Could not find fleet number for device ID: ${deviceId} in mapping data`);
    }
    
    console.log(`Using fleet number ${fleetNumber} for device ID ${deviceId}`);
    
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
    
    // Filter clustered points (like in the Python implementation)
    const filteredBusData = filterClusteredPoints(processedBusData);
    console.log(`Filtered bus data from ${processedBusData.length} to ${filteredBusData.length} points`);
    
    // Calculate bus vector using the filtered data (first 5 points max)
    // Matching Python implementation exactly
    const busVector = calculateVector(filteredBusData);
    
    // Extract route information from waybill using fleet number
    const waybillMatch = findWaybillMatch(waybillData, fleetNumber);
    
    if (!waybillMatch) {
      throw new Error(`No waybill entry found for fleet number: ${fleetNumber}`);
    }
    
    // Extract route number from Schedule No
    const scheduleNo = waybillMatch['Schedule No'];
    let routeNumber = null;
    
    if (scheduleNo) {
      // Try to extract route number using the pattern: "X-ROUTE-Y-Z"
      const routeNumberMatch = scheduleNo.match(/^.*?-(.+?)-/);
      if (routeNumberMatch && routeNumberMatch[1]) {
        routeNumber = routeNumberMatch[1];
        console.log(`Extracted route number ${routeNumber} from schedule: ${scheduleNo}`);
      }
    }
    
    if (!routeNumber) {
      throw new Error(`Could not extract route number from waybill entry: ${JSON.stringify(waybillMatch)}`);
    }
    
    // Get route stops
    const routeStops = routeStopData.filter(item => 
      String(item['MTC ROUTE NO']).trim() === String(routeNumber).trim()
    );
    
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
    
    // Find the last bus point (most recent location)
    const lastBusPoint = filteredBusData[filteredBusData.length - 1];
    
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
      
      // Find stops within distance (like in Python)
      const nearbyStops = findStopsWithinDistance(lastBusPoint, routeStops);
      
      if (nearbyStops.length > 0) {
        // Get the nearest stop
        const nearestStop = nearbyStops[0];
        
        // Get neighboring stops
        const neighbors = getNeighboringStops(routeStops, nearestStop.stop_id);
        
        if (neighbors.length > 0) {
          // Combine nearest stop with neighbors for vector calculation
          const stopsForVector = [nearestStop, ...neighbors];
          const stopVector = calculateStopVector(stopsForVector);
          
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
    }
    
    // Sort matches by similarity (highest first)
    matches.sort((a, b) => b.similarity - a.similarity);

    return {
      device_id: deviceId,
      fleet_number: fleetNumber,
      route_number: routeNumber,
      bus_vector: busVector,
      num_points_used: filteredBusData.length,
      location_data: filteredBusData,
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
    return [0, 0];
  }
  
  console.log('==================== CALCULATING BUS VECTOR ====================');
  console.log('Total points available:', points.length);
  
  // For bus vectors, we use the first few points to get a more accurate initial direction
  // Take the first 4 points or all points if less than 4 (to match direction_determination.py)
  const pointsToUse = points.slice(0, Math.min(4, points.length));
  console.log('Using points:', pointsToUse);
  
  // Use the first and last of these points to get direction - matching Python implementation
  const firstPoint = pointsToUse[0];
  const lastPoint = pointsToUse[pointsToUse.length - 1];
  console.log('First point:', firstPoint);
  console.log('Last point:', lastPoint);
  
  const dx = lastPoint.long - firstPoint.long;
  const dy = lastPoint.lat - firstPoint.lat;
  console.log('dx:', dx, 'dy:', dy);
  
  // Normalize the vector
  const magnitude = Math.sqrt(dx**2 + dy**2);
  console.log('Magnitude:', magnitude);
  
  if (magnitude > 0) {
    const vector = [dx/magnitude, dy/magnitude];
    console.log("Bus vector calculated:", vector);
    console.log('==============================================================');
    return vector;
  }
  
  // Default to no direction if we can't calculate
  console.log("Using default [0, 0] vector due to zero magnitude");
  console.log('==============================================================');
  return [0, 0];
}

function calculateStopVector(stops) {
  if (!stops || stops.length < 2) {
    console.log("Warning: Not enough stops to calculate vector", stops);
    // Default to no direction if we can't calculate
    return [0, 0];
  }
  
  console.log('==================== CALCULATING STOP VECTOR ====================');
  console.log('Total stops available:', stops.length);
  
  // Sort stops by sequence to ensure correct ordering
  const sortedStops = [...stops].sort((a, b) => Number(a.stop_sequence) - Number(b.stop_sequence));
  console.log('Sorted stops:', sortedStops.map(s => ({ id: s.stop_id, seq: s.stop_sequence, name: s.stop_name })));
  
  // Use the first and last stop to get direction - matching Python implementation
  const firstStop = sortedStops[0];
  const lastStop = sortedStops[sortedStops.length - 1];
  console.log('First stop:', { id: firstStop.stop_id, seq: firstStop.stop_sequence, name: firstStop.stop_name, lat: firstStop.stop_latitude, long: firstStop.stop_longitude });
  console.log('Last stop:', { id: lastStop.stop_id, seq: lastStop.stop_sequence, name: lastStop.stop_name, lat: lastStop.stop_latitude, long: lastStop.stop_longitude });
  
  const dx = lastStop.stop_longitude - firstStop.stop_longitude;
  const dy = lastStop.stop_latitude - firstStop.stop_latitude;
  console.log('dx:', dx, 'dy:', dy);
  
  // Normalize the vector
  const magnitude = Math.sqrt(dx**2 + dy**2);
  console.log('Magnitude:', magnitude);
  
  if (magnitude > 0) {
    const vector = [dx/magnitude, dy/magnitude];
    console.log("Stop vector calculated:", vector);
    console.log('===============================================================');
    return vector;
  }
  
  // Default to no direction if we can't calculate
  console.log("Using default [0, 0] vector due to zero magnitude");
  console.log('===============================================================');
  return [0, 0];
}

function vectorSimilarity(v1, v2) {
  // Calculate dot product - a value between -1 and 1
  // 1 means same direction, -1 means opposite direction
  // Exactly matching Python implementation
  const result = v1[0]*v2[0] + v1[1]*v2[1];
  
  // Debug logging
  console.log('==================== VECTOR SIMILARITY ====================');
  console.log('Vector 1:', v1);
  console.log('Vector 2:', v2);
  console.log('Dot product:', result);
  console.log('===========================================================');
  
  return result;
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

// Function to filter clustered points (matching Python implementation)
function filterClusteredPoints(points, distanceThresholdKm = 0.02) {
  if (points.length <= 1) {
    return points;
  }
  
  const filteredIndices = [0]; // Always keep the first point
  
  for (let i = 1; i < points.length; i++) {
    const prevIdx = filteredIndices[filteredIndices.length - 1];
    
    // Check distance
    const distance = haversine(
      points[prevIdx].long, points[prevIdx].lat,
      points[i].long, points[i].lat
    );
    
    // If distance is significant, keep the point
    if (distance > distanceThresholdKm) {
      filteredIndices.push(i);
    }
  }
  
  return filteredIndices.map(idx => points[idx]);
} 