const express = require('express');
const path = require('path');
const fs = require('fs');
const app = express();
const cors = require('cors');
const port = 3001;

// Enable CORS for all routes
app.use(cors());

// Increase the limit for JSON and URL-encoded bodies
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// Serve static files from the 'dist' directory
app.use(express.static(path.join(__dirname, 'dist')));

// Data directory path
const dataPath = path.join(__dirname, '..', 'data');

// Endpoint to get the list of CSV files
app.get('/api/files', (req, res) => {
  fs.readdir(dataPath, (err, files) => {
    if (err) {
      console.error('Error reading data directory:', err);
      return res.status(500).json({ error: 'Failed to read data directory' });
    }
    
    const csvFiles = files.filter(file => file.endsWith('.csv'));
    res.json({ files: csvFiles });
  });
});

// Endpoint to serve a specific CSV file
app.get('/api/data/:filename', (req, res) => {
  const filename = req.params.filename;
  const filePath = path.join(dataPath, filename);
  
  // Check if file exists
  if (!fs.existsSync(filePath)) {
    return res.status(404).json({ error: `File not found: ${filename}` });
  }
  
  // Get file stats
  try {
    const stats = fs.statSync(filePath);
    const fileSizeInMB = stats.size / (1024 * 1024);
    
    console.log(`Serving file: ${filename} (${fileSizeInMB.toFixed(2)} MB)`);
    
    // Serve the file
    res.sendFile(filePath);
  } catch (err) {
    console.error(`Error serving file ${filename}:`, err);
    res.status(500).json({ error: `Failed to serve file: ${err.message}` });
  }
});

// Endpoint for vector visualization data
app.get('/api/vectors/:deviceId', (req, res) => {
  const deviceId = req.params.deviceId;
  
  // This is a placeholder. In a real app, you would compute vectors here
  // or use a Python bridge to call the direction_determination.py methods
  res.json({
    message: `Vector data for device ${deviceId} would be computed here`
  });
});

// Fallback route for SPA
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Server error:', err);
  res.status(500).json({ error: 'Internal server error', message: err.message });
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
  console.log(`Serving data from: ${dataPath}`);
}); 