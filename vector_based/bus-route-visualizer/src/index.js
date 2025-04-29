import React from 'react';
import { createRoot } from 'react-dom/client';
import { ChakraProvider } from '@chakra-ui/react';
import App from './App';

// Create a root for the application
const rootElement = document.getElementById('root');
const root = createRoot(rootElement);

// Render the application
root.render(
  <ChakraProvider>
    <App />
  </ChakraProvider>
); 