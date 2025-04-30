import React, { useState, useEffect, useMemo, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, Circle, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix for marker icon in Leaflet with webpack
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

// Custom marker icons
const DefaultIcon = L.icon({
  iconUrl: icon,
  shadowUrl: iconShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41]
});

const BusIcon = L.divIcon({
  html: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="36" height="36" fill="#4299E1">
          <path d="M12 2C7.58 2 4 2.5 4 6v9.5c0 1.38.62 2.62 1.58 3.5H4v1h3.5c1.38 0 2.5-1.12 2.5-2.5S8.88 15 7.5 15H7v-2h10v2h-.5c-1.38 0-2.5 1.12-2.5 2.5s1.12 2.5 2.5 2.5H20v-1h-1.58c.96-.88 1.58-2.12 1.58-3.5V6c0-3.5-3.58-4-8-4zM7.5 17c.83 0 1.5.67 1.5 1.5S8.33 20 7.5 20H6v-3h1.5zm9 0H18v3h-1.5c-.83 0-1.5-.67-1.5-1.5s.67-1.5 1.5-1.5zM12 4.5c2.97 0 5.5.24 6.5.74V7h-13V5.24c1-.5 3.53-.74 6.5-.74zM6 9h12v3H6V9z"/>
        </svg>`,
  className: '',
  iconSize: [36, 36],
  iconAnchor: [18, 18]
});

// Function to generate a stop icon with a specific color
const createStopIcon = (color) => {
  return L.divIcon({
    html: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24" fill="${color}">
            <circle cx="12" cy="12" r="8"/>
          </svg>`,
    className: '',
    iconSize: [24, 24],
    iconAnchor: [12, 12]
  });
};

// Vector arrow component
const VectorArrow = ({ start, vector, color, scale = 0.01, label }) => {
  if (!start || !vector) {
    console.log(`Missing data for ${label} vector:`, { start, vector });
    return null;
  }
  
  // Ensure vector has non-zero magnitude
  const magnitude = Math.sqrt(vector[0]**2 + vector[1]**2);
  if (magnitude === 0) {
    console.log(`Zero magnitude for ${label} vector`);
    // Use a default vector if magnitude is zero (pointing north)
    vector = [0, 1];
  }
  
  const end = [
    start[0] + vector[0] * scale,
    start[1] + vector[1] * scale
  ];
  
  // Create arrow head points
  const createArrowHead = (start, end, headLength = 0.002) => {
    const dx = end[0] - start[0];
    const dy = end[1] - start[1];
    const angle = Math.atan2(dy, dx);
    
    return [
      [
        end[0] - headLength * Math.cos(angle - Math.PI/6),
        end[1] - headLength * Math.sin(angle - Math.PI/6)
      ],
      end,
      [
        end[0] - headLength * Math.cos(angle + Math.PI/6),
        end[1] - headLength * Math.sin(angle + Math.PI/6)
      ]
    ];
  };
  
  const arrowHead = createArrowHead(start, end);
  const arrowOptions = { color, weight: 5 };
  const arrowHeadOptions = { color, weight: 5, opacity: 1.0 };
  
  return (
    <>
      <Polyline positions={[start, end]} {...arrowOptions}>
        <Popup>
          {label}: [{vector[0].toFixed(4)}, {vector[1].toFixed(4)}]
        </Popup>
      </Polyline>
      <Polyline positions={arrowHead} {...arrowHeadOptions} />
      
      {/* Add a small dot at the origin of the vector */}
      <Circle 
        center={start} 
        radius={5} 
        pathOptions={{ 
          fillColor: color, 
          fillOpacity: 1.0, 
          color: 'white', 
          weight: 1 
        }} 
      />
    </>
  );
};

// Map bounds setter component
const MapBoundsSetter = ({ positions }) => {
  const map = useMap();
  
  useEffect(() => {
    if (positions && positions.length > 0) {
      const bounds = L.latLngBounds(positions);
      map.fitBounds(bounds, { padding: [50, 50] });
    }
  }, [map, positions]);
  
  return null;
};

// Color generator for tummoc routes
const getRouteColor = (index) => {
  const colors = [
    '#F56565', // red
    '#38A169', // green
    '#3182CE', // blue
    '#D69E2E', // yellow
    '#805AD5', // purple
    '#DD6B20', // orange
    '#319795', // teal
    '#ED64A6', // pink
    '#718096', // gray
    '#2C5282', // dark blue
    '#9B2C2C', // dark red
    '#285E61'  // dark teal
  ];
  return colors[index % colors.length];
};

// Resizable panel component
const ResizablePanel = ({ children, defaultPosition, defaultSize, minWidth = 200, minHeight = 100, title }) => {
  const [position, setPosition] = useState(defaultPosition);
  const [size, setSize] = useState(defaultSize);
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const [resizeStartPos, setResizeStartPos] = useState({ x: 0, y: 0 });
  const [dragStartPos, setDragStartPos] = useState({ x: 0, y: 0 });
  const [isPanelCollapsed, setIsPanelCollapsed] = useState(false);
  
  const panelRef = useRef(null);
  
  const handleResizeStart = (e) => {
    e.stopPropagation();
    setIsResizing(true);
    setResizeStartPos({
      x: e.clientX,
      y: e.clientY
    });
  };
  
  const handleDragStart = (e) => {
    e.stopPropagation();
    if (e.target.className.includes('resize-handle') || e.target.className.includes('collapse-button')) {
      return;
    }
    setIsDragging(true);
    setDragStartPos({
      x: e.clientX - position.x,
      y: e.clientY - position.y
    });
  };

  const handleMouseMove = (e) => {
    if (isDragging) {
      const newX = e.clientX - dragStartPos.x;
      const newY = e.clientY - dragStartPos.y;
      
      // Ensure panel stays within map bounds
      const mapWidth = window.innerWidth;
      const mapHeight = window.innerHeight;
      
      const boundedX = Math.max(0, Math.min(newX, mapWidth - size.width));
      const boundedY = Math.max(0, Math.min(newY, mapHeight - size.height));
      
      setPosition({ x: boundedX, y: boundedY });
    }
    
    if (isResizing) {
      const deltaX = e.clientX - resizeStartPos.x;
      const deltaY = e.clientY - resizeStartPos.y;
      
      const newWidth = Math.max(minWidth, size.width + deltaX);
      const newHeight = Math.max(minHeight, size.height + deltaY);
      
      setSize({ width: newWidth, height: newHeight });
      setResizeStartPos({ x: e.clientX, y: e.clientY });
    }
  };
  
  const handleMouseUp = () => {
    setIsDragging(false);
    setIsResizing(false);
  };
  
  const toggleCollapse = (e) => {
    e.stopPropagation();
    setIsPanelCollapsed(!isPanelCollapsed);
  };
  
  useEffect(() => {
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, isResizing, dragStartPos, resizeStartPos]);
  
  const panelHeight = isPanelCollapsed ? '30px' : `${size.height}px`;
  
  return (
    <div
      ref={panelRef}
      style={{
        position: 'absolute',
        left: `${position.x}px`,
        top: `${position.y}px`,
        width: `${size.width}px`,
        height: panelHeight,
        backgroundColor: 'rgba(255, 255, 255, 0.9)',
        border: '1px solid #ccc',
        borderRadius: '5px',
        boxShadow: '0 2px 5px rgba(0, 0, 0, 0.2)',
        overflow: 'hidden',
        zIndex: 1000,
        transition: 'height 0.3s ease-in-out',
        cursor: isDragging ? 'grabbing' : 'grab'
      }}
      onMouseDown={handleDragStart}
    >
      <div 
        style={{ 
          padding: '5px 10px',
          backgroundColor: '#f0f0f0',
          borderBottom: isPanelCollapsed ? 'none' : '1px solid #ccc',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          cursor: 'move',
          userSelect: 'none'
        }}
      >
        <div style={{ fontWeight: 'bold', fontSize: '12px' }}>{title}</div>
        <button 
          className="collapse-button"
          onClick={toggleCollapse}
          style={{ 
            backgroundColor: 'transparent',
            border: 'none',
            cursor: 'pointer',
            fontSize: '16px',
            color: '#666'
          }}
        >
          {isPanelCollapsed ? '▼' : '▲'}
        </button>
      </div>
      
      <div style={{ padding: '10px', overflowY: 'auto', height: 'calc(100% - 30px)' }}>
        {children}
      </div>
      
      <div
        className="resize-handle"
        style={{
          position: 'absolute',
          right: '0',
          bottom: '0',
          width: '16px',
          height: '16px',
          cursor: 'nwse-resize',
          backgroundColor: 'transparent'
        }}
        onMouseDown={handleResizeStart}
      >
        <svg width="16" height="16" viewBox="0 0 16 16">
          <path d="M11 11 L16 16 M8 11 L16 3 M14 8 L16 6" stroke="#666" strokeWidth="1.5" />
        </svg>
      </div>
    </div>
  );
};

const BusRouteMap = ({ routeData, visibleTummocIds = {}, colorMap = {} }) => {
  const [busPositions, setBusPositions] = useState([]);
  const [stops, setStops] = useState([]);
  const [center, setCenter] = useState([13.0, 80.0]); // Default center (Chennai)
  const [allPositions, setAllPositions] = useState([]);
  const [routeGroups, setRouteGroups] = useState({});
  
  useEffect(() => {
    if (!routeData) return;
    
    // Log for debugging
    console.log("RouteData:", routeData);
    if (routeData.matches && routeData.matches.length > 0) {
      console.log("Stop Vector:", routeData.matches[0].stop_vector);
    }
    
    // Extract bus positions
    const { location_data, stop_data } = routeData;
    
    if (location_data && location_data.length > 0) {
      const positions = location_data.map(point => [point.lat, point.long]);
      setBusPositions(positions);
      
      // Set the center to the last bus position
      const lastPosition = positions[positions.length - 1];
      if (lastPosition) {
        setCenter(lastPosition);
      }
    }
    
    // Extract stops and group by tummoc_id
    if (stop_data && stop_data.length > 0) {
      // Process all stops
      const allStops = stop_data.map(stop => ({
        id: stop.stop_id,
        name: stop.stop_name,
        position: [stop.stop_latitude, stop.stop_longitude],
        sequence: stop.stop_sequence,
        tummoc_id: stop.tummoc_id,
        direction: stop.direction
      }));
      
      setStops(allStops);
      
      // Group stops by tummoc_id
      const groupedStops = {};
      
      allStops.forEach(stop => {
        if (!groupedStops[stop.tummoc_id]) {
          groupedStops[stop.tummoc_id] = [];
        }
        groupedStops[stop.tummoc_id].push(stop);
      });
      
      setRouteGroups(groupedStops);
    }
    
    // Combine all positions for map bounds
    const allPos = [
      ...(location_data ? location_data.map(point => [point.lat, point.long]) : []),
      ...(stop_data ? stop_data.map(stop => [stop.stop_latitude, stop.stop_longitude]) : [])
    ];
    
    setAllPositions(allPos);
    
  }, [routeData]);

  if (!routeData) {
    return <div>No route data available</div>;
  }

  // Get the last bus position for vector visualization
  const lastBusPosition = busPositions.length > 0 ? busPositions[busPositions.length - 1] : null;
  
  // Get bus vector from routeData
  const busVector = routeData.bus_vector;
  
  // Get stop vector from matches
  const stopVector = routeData.matches && 
                    routeData.matches.length > 0 && 
                    routeData.matches[0].stop_vector ? 
                      routeData.matches[0].stop_vector : 
                      null;

  // If stop vector is missing but we have match data, create a default
  if (!stopVector && routeData.matches && routeData.matches.length > 0) {
    console.warn("Stop vector missing in match data - creating default east vector");
    // Default to east direction
    routeData.matches[0].stop_vector = [1, 0];
  }
  
  // Get best match if available
  const bestMatch = routeData.matches && routeData.matches.length > 0 ? 
                    routeData.matches[0] : null;
                    
  // Get the color for best match route
  const bestMatchColor = bestMatch && colorMap[bestMatch.tummoc_id] ? 
                         colorMap[bestMatch.tummoc_id] : '#F56565';

  return (
    <MapContainer center={center} zoom={13} style={{ height: '100%', width: '100%' }}>
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      
      {/* Set map bounds to include all points */}
      <MapBoundsSetter positions={allPositions} />
      
      {/* Bus route polyline */}
      {busPositions.length > 1 && (
        <Polyline 
          positions={busPositions} 
          color="blue" 
          weight={3} 
          opacity={0.7}
        >
          <Popup>Bus Route Path</Popup>
        </Polyline>
      )}
      
      {/* Bus markers (first and last position) */}
      {busPositions.length > 0 && (
        <>
          <Marker 
            position={busPositions[0]} 
            icon={BusIcon}
          >
            <Popup>Start Position</Popup>
          </Marker>
          
          <Marker 
            position={busPositions[busPositions.length - 1]} 
            icon={BusIcon}
          >
            <Popup>
              Current Position<br />
              Fleet Number: {routeData.fleet_number}<br />
              {routeData.matches && routeData.matches.length > 0 && (
                <>
                  Direction: {routeData.matches[0].direction}<br />
                  Similarity: {routeData.matches[0].similarity.toFixed(4)}
                </>
              )}
            </Popup>
          </Marker>
        </>
      )}
      
      {/* Route groups - stops colored by tummoc_id */}
      {Object.entries(routeGroups).map(([tummoc_id, routeStops]) => {
        // Skip rendering if this tummoc_id is not visible
        if (!visibleTummocIds[tummoc_id]) return null;
        
        const color = colorMap[tummoc_id];
        const StopIcon = createStopIcon(color);
        
        // Sort stops by sequence
        const sortedStops = [...routeStops].sort((a, b) => a.sequence - b.sequence);
        
        return (
          <React.Fragment key={tummoc_id}>
            {/* Stops for this tummoc_id */}
            {sortedStops.map(stop => (
              <Marker 
                key={`${tummoc_id}-${stop.id}`} 
                position={stop.position}
                icon={StopIcon}
              >
                <Popup>
                  <div>
                    <strong>{stop.name}</strong><br />
                    Stop ID: {stop.id}<br />
                    Sequence: {stop.sequence}<br />
                    Tummoc ID: {tummoc_id}<br />
                    Direction: {stop.direction}
                  </div>
                </Popup>
              </Marker>
            ))}
            
            {/* Stop sequence line */}
            {sortedStops.length > 1 && (
              <Polyline 
                positions={sortedStops.map(stop => stop.position)}
                color={color}
                weight={2}
                dashArray="5,10"
                opacity={0.6}
              >
                <Popup>
                  Route: {tummoc_id}<br />
                  Direction: {sortedStops[0].direction}<br />
                  Stops: {sortedStops.length}
                </Popup>
              </Polyline>
            )}
          </React.Fragment>
        );
      })}
      
      {/* Bus vector arrow */}
      {lastBusPosition && busVector && (
        <VectorArrow 
          start={lastBusPosition} 
          vector={busVector} 
          color="blue" 
          scale={0.03}
          label="Bus Vector"
        />
      )}
      
      {/* Stop vector arrows for all visible routes */}
      {lastBusPosition && routeData.matches && routeData.matches.map(match => {
        if (!visibleTummocIds[match.tummoc_id] || !match.stop_vector) return null;
        const color = colorMap[match.tummoc_id] || '#F56565';
        return (
          <VectorArrow 
            key={`vector-${match.tummoc_id}`}
            start={lastBusPosition} 
            vector={match.stop_vector} 
            color={color} 
            scale={0.03}
            label={`Stop Vector (ID ${match.tummoc_id})`}
          />
        );
      })}
      
      {/* Vector labels */}
      {lastBusPosition && busVector && (
        <>
          <Circle 
            center={lastBusPosition}
            radius={30}
            color="purple"
            fillColor="purple"
            fillOpacity={0.2}
          >
            <Popup>
              <strong>Vector Analysis Point</strong><br />
              Bus Vector: [{busVector[0].toFixed(4)}, {busVector[1].toFixed(4)}]<br />
              {routeData.matches && routeData.matches.length > 0 && (
                <>
                  Similarity: {routeData.matches[0].similarity.toFixed(4)}<br />
                  Best Match: {routeData.matches[0].tummoc_id} ({routeData.matches[0].direction})
                </>
              )}
            </Popup>
          </Circle>
        </>
      )}
      
      {/* Highlight the nearest stop with a circle for each visible route */}
      {routeData.matches && routeData.matches.map(match => {
        if (!visibleTummocIds[match.tummoc_id] || !match.nearest_stop_id) return null;
        const color = colorMap[match.tummoc_id] || '#F56565';
        
        return stops
          .filter(stop => stop.id === match.nearest_stop_id && stop.tummoc_id === match.tummoc_id)
          .map(stop => (
            <Circle 
              key={`circle-${match.tummoc_id}-${stop.id}`}
              center={stop.position} 
              radius={50} 
              color={color}
              fillColor={color}
              fillOpacity={0.2}
            />
          ));
      })}
    </MapContainer>
  );
};

export default BusRouteMap;