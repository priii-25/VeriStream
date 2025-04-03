import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import VideoAnalytics from './components/VideoAnalytics';
import RealTimeAnalysis from './components/RealTimeAnalysis';
import './styles/App.css';

function App() {
  return (
    <Router>
      <div className="app-container">
        <h1>VeriStream</h1>
        <nav className="nav-bar">
          <Link to="/">File Upload</Link>
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