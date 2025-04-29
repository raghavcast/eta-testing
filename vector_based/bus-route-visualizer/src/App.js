import React, { useState, useEffect, useRef } from 'react';
import { Box, Flex, Heading, Select, Text, Button, VStack, HStack, Stack, Divider, Badge, Spinner } from '@chakra-ui/react';
import BusRouteMap from './components/BusRouteMap';
import { fetchDeviceIds, fetchRouteData } from './services/dataService';

// Resizable panel component
const ResizablePanel = ({ children, initialWidth, initialHeight, minWidth = 200, minHeight = 150 }) => {
  const [width, setWidth] = useState(initialWidth);
  const [height, setHeight] = useState(initialHeight);
  const [isResizing, setIsResizing] = useState(false);
  const [resizeStartDimensions, setResizeStartDimensions] = useState({ width: 0, height: 0 });
  const [resizeStartPos, setResizeStartPos] = useState({ x: 0, y: 0 });
  const panelRef = useRef(null);

  const handleResizeStart = (e) => {
    e.preventDefault();
    setIsResizing(true);
    setResizeStartDimensions({ width, height });
    setResizeStartPos({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e) => {
    if (!isResizing) return;
    
    const deltaX = e.clientX - resizeStartPos.x;
    const deltaY = e.clientY - resizeStartPos.y;
    
    const newWidth = Math.max(minWidth, resizeStartDimensions.width + deltaX);
    const newHeight = Math.max(minHeight, resizeStartDimensions.height + deltaY);
    
    setWidth(newWidth);
    setHeight(newHeight);
  };

  const handleMouseUp = () => {
    setIsResizing(false);
  };

  useEffect(() => {
    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    } else {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    }
    
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing, resizeStartDimensions, resizeStartPos]);

  return (
    <Box 
      ref={panelRef}
      position="relative" 
      width={`${width}px`}
      height={`${height}px`}
      overflow="auto"
    >
      {children}
      <Box
        position="absolute"
        bottom="2px"
        right="2px"
        width="14px"
        height="14px"
        cursor="nwse-resize"
        zIndex="1"
        onMouseDown={handleResizeStart}
      >
        <svg width="14" height="14" viewBox="0 0 14 14">
          <path d="M11 11 L14 14 M6 11 L14 3 M10 6 L14 2" stroke="#666" strokeWidth="1.5" />
        </svg>
      </Box>
    </Box>
  );
};

// Vector visualization legend component
const VectorLegend = ({ routeData }) => {
  // Extract best match tummoc_id and its color if available
  const bestMatch = routeData.matches && routeData.matches.length > 0 
    ? routeData.matches[0] 
    : null;
    
  return (
    <ResizablePanel initialWidth={320} initialHeight={300} minWidth={250} minHeight={200}>
      <Box p={4} bg="gray.50" borderRadius="md" borderWidth={1} borderColor="gray.200" mb={4} height="100%">
        <Heading as="h3" size="sm" mb={3}>Visualization Guide</Heading>
        <Stack spacing={2} fontSize="sm">
          <Heading as="h4" size="xs" mt={1}>Vectors</Heading>
          <Flex align="center">
            <Box bg="blue.500" w="20px" h="4px" mr={2}></Box>
            <Text><strong>Blue Arrow:</strong> Bus Vector - direction calculated from bus location history</Text>
          </Flex>
          <Flex align="center">
            <Box bg="red.500" w="20px" h="4px" mr={2}></Box>
            <Text><strong>Stop Vector:</strong> Direction calculated from stop locations (colored by route)</Text>
          </Flex>
          
          <Heading as="h4" size="xs" mt={2}>Markers</Heading>
          <Flex align="center">
            <Box bg="purple.200" w="15px" h="15px" borderRadius="full" mr={2}></Box>
            <Text><strong>Purple Circle:</strong> Vector analysis point (location where vectors are compared)</Text>
          </Flex>
          <Flex align="center">
            <Box bg="red.200" w="15px" h="15px" borderRadius="full" mr={2}></Box>
            <Text><strong>Red Circle:</strong> Nearest stop to current bus location</Text>
          </Flex>
          
          <Heading as="h4" size="xs" mt={2}>Routes</Heading>
          <Text fontSize="xs">Each tummoc route ID is displayed with a unique color. The stops on each route are connected with dashed lines of the same color.</Text>
          {bestMatch && (
            <Text fontSize="xs" fontWeight="medium" mt={1}>
              Best matching route: TUMMOC ID {bestMatch.tummoc_id} ({bestMatch.direction} direction)
            </Text>
          )}
        </Stack>
      </Box>
    </ResizablePanel>
  );
};

// Vector similarity visualization
const SimilarityIndicator = ({ similarity }) => {
  let color = "gray.500";
  let status = "Neutral";
  
  if (similarity > 0.7) {
    color = "green.500";
    status = "Same Direction";
  } else if (similarity > 0) {
    color = "yellow.500";
    status = "Similar Direction";
  } else if (similarity > -0.7) {
    color = "orange.500";
    status = "Different Direction";
  } else {
    color = "red.500";
    status = "Opposite Direction";
  }
  
  return (
    <Badge colorScheme={color.split('.')[0]} px={2} py={1} borderRadius="md">
      {status}: {similarity.toFixed(4)}
    </Badge>
  );
};

const App = () => {
  const [deviceIds, setDeviceIds] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState('');
  const [routeData, setRouteData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [visibleTummocIds, setVisibleTummocIds] = useState({});

  // Fetch all available device IDs when the component mounts
  useEffect(() => {
    const loadDeviceIds = async () => {
      try {
        const ids = await fetchDeviceIds();
        setDeviceIds(ids);
        if (ids.length > 0) {
          setSelectedDeviceId(ids[0]);
        }
      } catch (err) {
        setError('Failed to load device IDs: ' + err.message);
      }
    };

    loadDeviceIds();
  }, []);

  // Handle device selection change
  const handleDeviceChange = (e) => {
    setSelectedDeviceId(e.target.value);
    setRouteData(null); // Clear previous route data
  };

  // Fetch route data for the selected device
  const handleViewRoute = async () => {
    if (!selectedDeviceId) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const data = await fetchRouteData(selectedDeviceId);
      setRouteData(data);
      
      // Initialize visibility of tummoc IDs
      if (data.matches && data.matches.length > 0) {
        const initialVisibility = {};
        data.matches.forEach(match => {
          initialVisibility[match.tummoc_id] = true;
        });
        setVisibleTummocIds(initialVisibility);
      }
    } catch (err) {
      setError('Failed to load route data: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Function to format vector for display
  const formatVector = (vector) => {
    if (!vector) return 'N/A';
    return `[${vector[0].toFixed(4)}, ${vector[1].toFixed(4)}]`;
  };

  // Toggle visibility for a tummoc ID
  const toggleTummocVisibility = (tummocId) => {
    setVisibleTummocIds(prev => ({
      ...prev,
      [tummocId]: !prev[tummocId]
    }));
  };

  // Get color for tummoc IDs
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

  // Create color map for tummoc IDs
  const getColorMap = () => {
    if (!routeData || !routeData.matches || routeData.matches.length === 0) {
      return {};
    }
    
    const colorMap = {};
    routeData.matches.forEach((match, index) => {
      colorMap[match.tummoc_id] = getRouteColor(index);
    });
    
    return colorMap;
  };

  const colorMap = getColorMap();

  return (
    <Box p={4}>
      <Heading as="h1" size="xl" mb={6}>Bus Route Vector Visualizer</Heading>
      
      <Flex 
        direction={{ base: 'column', md: 'row' }} 
        gap={4} 
        mb={4}
        flexWrap="wrap"
        alignItems="flex-start"
      >
        <VStack align="flex-start" spacing={4} flex={{ base: '1', md: '0 0 320px' }}>
          <Box width="full" p={4} bg="gray.50" borderRadius="md" borderWidth={1}>
            <Text fontWeight="bold" mb={2}>Select Device ID:</Text>
            <Select 
              placeholder="Select a device ID" 
              value={selectedDeviceId} 
              onChange={handleDeviceChange}
              disabled={deviceIds.length === 0 || loading}
              width="full"
              mb={3}
            >
              {deviceIds.map(id => (
                <option key={id} value={id}>{id}</option>
              ))}
            </Select>
            
            <Button 
              colorScheme="blue" 
              onClick={handleViewRoute} 
              isLoading={loading}
              loadingText="Loading"
              isDisabled={!selectedDeviceId}
              width="full"
            >
              View Route
            </Button>
          </Box>
          
          {routeData && routeData.matches && routeData.matches.length > 0 && (
            <Box width="full" p={4} bg="gray.50" borderRadius="md" borderWidth={1}>
              <Heading as="h3" size="sm" mb={3}>Matched Routes</Heading>
              <VStack align="flex-start" spacing={2} maxHeight="600px" overflowY="auto" width="full">
                {routeData.matches.map((match, index) => {
                  const color = colorMap[match.tummoc_id] || '#F56565';
                  return (
                    <Box 
                      key={match.tummoc_id} 
                      p={3} 
                      borderWidth={1} 
                      borderRadius="md" 
                      width="full"
                      borderColor={visibleTummocIds[match.tummoc_id] ? color : 'gray.200'}
                      opacity={visibleTummocIds[match.tummoc_id] ? 1 : 0.6}
                    >
                      <HStack justifyContent="space-between" mb={2}>
                        <HStack>
                          <Box 
                            width="12px" 
                            height="12px" 
                            bg={color} 
                            borderRadius="full" 
                          />
                          <Text fontWeight="bold">Tummoc ID: {match.tummoc_id}</Text>
                        </HStack>
                        <Badge colorScheme={match.similarity > 0.7 ? 'green' : match.similarity > 0 ? 'yellow' : 'red'}>
                          {match.similarity.toFixed(2)}
                        </Badge>
                      </HStack>
                      
                      <Text fontSize="sm">Direction: {match.direction}</Text>
                      <Text fontSize="sm">Nearest Stop: {match.nearest_stop_name}</Text>
                      <Text fontSize="sm">Stop Vector: {match.stop_vector ? formatVector(match.stop_vector) : 'N/A'}</Text>
                      
                      <Button 
                        size="sm" 
                        mt={2} 
                        width="full"
                        variant={visibleTummocIds[match.tummoc_id] ? "outline" : "solid"}
                        colorScheme={visibleTummocIds[match.tummoc_id] ? "gray" : "blue"}
                        onClick={() => toggleTummocVisibility(match.tummoc_id)}
                      >
                        {visibleTummocIds[match.tummoc_id] ? "Hide Route" : "Show Route"}
                      </Button>
                    </Box>
                  );
                })}
              </VStack>
            </Box>
          )}
          
          {routeData && (
            <Box width="full" p={4} bg="gray.50" borderRadius="md" borderWidth={1}>
              <Heading as="h3" size="sm" mb={3}>Vector Legend</Heading>
              <VStack align="flex-start" spacing={2}>
                <Flex align="center">
                  <Box bg="blue.500" w="30px" h="5px" mr={2}></Box>
                  <Text fontSize="sm">Bus Vector: {formatVector(routeData.bus_vector)}</Text>
                </Flex>
                
                {routeData.matches && routeData.matches.map((match, index) => {
                  if (!visibleTummocIds[match.tummoc_id]) return null;
                  const color = colorMap[match.tummoc_id] || '#F56565';
                  return (
                    <Flex key={`legend-${match.tummoc_id}`} align="center">
                      <Box bg={color} w="30px" h="5px" mr={2}></Box>
                      <Text fontSize="sm">ID {match.tummoc_id} Vector: {formatVector(match.stop_vector)}</Text>
                    </Flex>
                  );
                })}
                
                <Divider my={2} />
                
                <Flex align="center">
                  <Box bg="purple.200" w="16px" h="16px" borderRadius="full" mr={2}></Box>
                  <Text fontSize="sm">Vector Analysis Point</Text>
                </Flex>
                
                <Flex align="center">
                  <Box bg="red.200" w="16px" h="16px" borderRadius="full" mr={2}></Box>
                  <Text fontSize="sm">Nearest Stop</Text>
                </Flex>
              </VStack>
            </Box>
          )}
          
          {routeData && (
            <ResizablePanel initialWidth={320} initialHeight={300} minWidth={250} minHeight={200}>
              <Box p={4} bg="gray.50" borderRadius="md" borderWidth={1} borderColor="gray.200" mb={4} height="100%">
                <Heading as="h3" size="sm" mb={3}>Route Details</Heading>
                
                <Box width="full">
                  <Text fontSize="lg" fontWeight="medium">Device: {routeData.device_id}</Text>
                  <Text>Route Number: {routeData.route_number}</Text>
                  <Text>Points Used: {routeData.num_points_used}</Text>
                </Box>
                
                <Divider my={3} />
                
                <Box width="full">
                  <Heading as="h4" size="xs" mb={2}>Vector Analysis</Heading>
                  
                  <VStack align="start" spacing={1} mt={4}>
                    <Text fontWeight="bold">Vector Similarity</Text>
                    <SimilarityIndicator similarity={routeData.matches && routeData.matches.length > 0 ? routeData.matches[0].similarity : 0} />
                  </VStack>
                </Box>
              </Box>
            </ResizablePanel>
          )}
        </VStack>
        
        {error && (
          <Box p={4} bg="red.100" color="red.800" borderRadius="md" mb={4}>
            {error}
          </Box>
        )}
        
        <Box flex="1" height={{base: "500px", md: "calc(100vh - 160px)"}} borderRadius="md" overflow="hidden" borderWidth={1} borderColor="gray.200">
          {loading ? (
            <Flex height="100%" align="center" justify="center">
              <Spinner size="xl" thickness="4px" color="blue.500" />
            </Flex>
          ) : routeData ? (
            <BusRouteMap 
              routeData={routeData} 
              visibleTummocIds={visibleTummocIds}
              colorMap={colorMap}
            />
          ) : (
            <Flex height="100%" align="center" justify="center" bg="gray.50">
              <Text color="gray.500">Select a device and click "View Route" to see the visualization</Text>
            </Flex>
          )}
        </Box>
      </Flex>
    </Box>
  );
};

export default App; 