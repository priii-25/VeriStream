// frontend/src/App.js
import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import VideoAnalytics from './components/VideoAnalytics';
import RealTimeAnalysis from './components/RealTimeAnalysis';

function App() {
  return (
    <Router>
      <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
        <h1>VeriStream</h1>
        <nav>
          <Link to="/" style={{ marginRight: '20px' }}>File Upload</Link>
          <Link to="/realtime">Real-Time Stream Analysis</Link>
        </nav>
        <Routes>
          <Route path="/" element={<VideoAnalytics />} />
          <Route path="/realtime" element={<RealTimeAnalysis />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;