import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Line, Bar } from 'react-chartjs-2';
import Papa from 'papaparse';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

// Fix Leaflet marker icon issue
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
});

const VideoAnalytics = () => {
  const [file, setFile] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [videoUrl, setVideoUrl] = useState(null);
  const [translation, setTranslation] = useState(null);
  const [language, setLanguage] = useState('en');
  const wsRef = useRef(null);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);
    setAnalysisResult(null);
    setError(null);
    setProgress(0);
    setTranslation(null);
    if (selectedFile) setVideoUrl(URL.createObjectURL(selectedFile));
    else setVideoUrl(null);
  };

  useEffect(() => {
    wsRef.current = new WebSocket('ws://127.0.0.1:5000/api/video/progress');
    wsRef.current.onopen = () => console.log('WebSocket connected for progress');
    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.progress) setProgress(data.progress);
    };
    wsRef.current.onerror = (error) => console.error('WebSocket error:', error);
    wsRef.current.onclose = () => console.log('WebSocket closed');
    return () => wsRef.current.close();
  }, []);

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      setError('Please select a video file.');
      return;
    }

    setLoading(true);
    setError(null);
    setProgress(0);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(
        'http://127.0.0.1:5000/api/video/analyze',
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' }, timeout: 180000 }
      );
      setAnalysisResult(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred while analyzing the video.');
      setAnalysisResult(null);
    } finally {
      setLoading(false);
    }
  };

  const handleTranslate = async () => {
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('language', language);

    try {
      const response = await axios.post(
        'http://127.0.0.1:5000/api/video/translate',
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' }, timeout: 30000 }
      );
      setTranslation(response.data.translation);
    } catch (err) {
      setError(err.response?.data?.error || 'Translation failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!analysisResult?.frames_data) return;
    const csvData = analysisResult.frames_data.timestamps.map((timestamp, index) => ({
      Timestamp: timestamp.toFixed(2),
      DeepfakeScore: (analysisResult.frames_data.max_scores[index] || 0).toFixed(2),
      FacesDetected: analysisResult.frames_data.faces_detected[index] ? 'Yes' : 'No',
    }));
    const csv = Papa.unparse(csvData);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'video_analysis_results.csv';
    link.click();
  };

  const calculateSummaryMetrics = () => {
    if (!analysisResult?.frames_data) return null;
    const scores = analysisResult.frames_data.max_scores;
    const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length || 0;
    const peakScore = Math.max(...scores) || 0;
    const frameCount = scores.length;
    const alert = analysisResult.final_score > 0.7 ? 'High probability of deepfake detected' : 'Low deepfake probability';
    return { avgScore, peakScore, frameCount, alert };
  };

  const summaryMetrics = calculateSummaryMetrics();

  const lineChartData = analysisResult?.frames_data ? {
    labels: analysisResult.frames_data.timestamps.map((t) => t.toFixed(2)),
    datasets: [{ label: 'Deepfake Score', data: analysisResult.frames_data.max_scores, borderColor: 'blue', fill: false }],
  } : null;

  const barChartData = analysisResult?.frames_data ? {
    labels: analysisResult.frames_data.max_scores.map((_, i) => i),
    datasets: [{ label: 'Score Distribution', data: analysisResult.frames_data.max_scores, backgroundColor: 'rgba(55, 83, 109, 0.5)' }],
  } : null;

  const chartOptions = {
    responsive: true,
    plugins: { legend: { position: 'top' }, title: { display: true } },
    scales: { x: { title: { display: true, text: 'Time (s)' } }, y: { title: { display: true, text: 'Score' }, beginAtZero: true, max: 1 } },
  };

  // Dummy coordinates for locations (replace with actual geocoding if needed)
  const locations = analysisResult?.text_analysis?.locations?.map(loc => ({
    name: loc.text,
    latitude: 51.505,  // Placeholder; use a geocoding API in production
    longitude: -0.09
  })) || [];

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h1>Video Analytics</h1>

      <form onSubmit={handleSubmit}>
        <input type="file" accept="video/*" onChange={handleFileChange} disabled={loading} />
        <button type="submit" disabled={loading || !file}>{loading ? 'Analyzing...' : 'Analyze Video'}</button>
      </form>

      {videoUrl && (
        <div style={{ marginTop: '20px' }}>
          <h3>Uploaded Video</h3>
          <video src={videoUrl} controls style={{ maxWidth: '640px', height: 'auto' }} />
        </div>
      )}

      {loading && (
        <div style={{ marginTop: '10px' }}>
          <p>Processing video... ({(progress * 100).toFixed(0)}%)</p>
          <progress value={progress} max="1" style={{ width: '100%' }} />
        </div>
      )}

      {error && (
        <div style={{ color: 'red', marginTop: '10px' }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {analysisResult && (
        <div style={{ marginTop: '20px' }}>
          <h2>Analysis Results</h2>

          <div>
            <strong>Transcription:</strong>{' '}
            {analysisResult.transcription || 'No transcription available'}
            <div>
              <select value={language} onChange={(e) => setLanguage(e.target.value)} style={{ margin: '10px' }}>
                <option value="en">English</option>
                <option value="es">Spanish</option>
                <option value="fr">French</option>
                <option value="hi">Hindi</option>
              </select>
              <button onClick={handleTranslate} disabled={loading}>Translate</button>
              {translation && <p><strong>Translated ({language}):</strong> {translation}</p>}
            </div>
          </div>

          <div>
            <strong>Final Deepfake Score:</strong>{' '}
            {analysisResult.final_score ? analysisResult.final_score.toFixed(2) : 'N/A'}
          </div>

          {summaryMetrics && (
            <div style={{ marginTop: '10px' }}>
              <h3>Summary Metrics</h3>
              <p>Average Score: {summaryMetrics.avgScore.toFixed(2)}</p>
              <p>Peak Score: {summaryMetrics.peakScore.toFixed(2)}</p>
              <p>Total Frames Analyzed: {summaryMetrics.frameCount}</p>
              <p style={{ color: summaryMetrics.alert.includes('High') ? 'red' : 'green' }}>
                {summaryMetrics.alert}
              </p>
            </div>
          )}

          {lineChartData && (
            <div style={{ marginTop: '20px' }}>
              <h3>Deepfake Detection Over Time</h3>
              <Line data={lineChartData} options={{ ...chartOptions, plugins: { ...chartOptions.plugins, title: { text: 'Deepfake Detection Over Time' } } }} />
            </div>
          )}

          {barChartData && (
            <div style={{ marginTop: '20px' }}>
              <h3>Score Distribution</h3>
              <Bar data={barChartData} options={{ ...chartOptions, plugins: { ...chartOptions.plugins, title: { text: 'Score Distribution' } }, scales: { x: { title: { text: 'Frame Index' } } } }} />
            </div>
          )}

          {analysisResult.frames_data?.timestamps?.length > 0 && (
            <div>
              <strong>Frame Analysis ({analysisResult.frames_data.timestamps.length} frames):</strong>
              <ul>
                {analysisResult.frames_data.timestamps.map((timestamp, index) => (
                  <li key={index}>
                    Timestamp: {timestamp.toFixed(2)}s,
                    Deepfake Score: {(analysisResult.frames_data.max_scores[index] || 0).toFixed(2)},
                    Face Detected: {analysisResult.frames_data.faces_detected[index] ? 'Yes' : 'No'}
                  </li>
                ))}
              </ul>
              <button onClick={handleDownload} style={{ marginTop: '10px' }}>Download Results as CSV</button>
            </div>
          )}

          {analysisResult.text_analysis?.knowledge_graph && (
            <div style={{ marginTop: '20px' }}>
              <h3>Knowledge Graph</h3>
              <iframe
                src="http://127.0.0.1:5000/knowledge_graph"
                style={{ width: '100%', height: '400px', border: 'none' }}
                title="Knowledge Graph"
              />
            </div>
          )}

          {locations.length > 0 && (
            <div style={{ marginTop: '20px' }}>
              <h3>Geospatial Map</h3>
              <MapContainer center={[51.505, -0.09]} zoom={2} style={{ height: '400px', width: '100%' }}>
                <TileLayer
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  attribution='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                />
                {locations.map((loc, idx) => (
                  <Marker key={idx} position={[loc.latitude, loc.longitude]}>
                    <Popup>{loc.name}</Popup>
                  </Marker>
                ))}
              </MapContainer>
            </div>
          )}

          {analysisResult.text_analysis && (
            <div>
              <strong>Text Analysis:</strong>
              <ul>
                <li>Sentiment: {analysisResult.text_analysis.sentiment?.label || 'N/A'} (Score: {analysisResult.text_analysis.sentiment?.score?.toFixed(2) || 'N/A'})</li>
                <li>Manipulation Score: {analysisResult.text_analysis.manipulation_score?.toFixed(2) || 'N/A'}</li>
                <li>Emotional Triggers: {analysisResult.text_analysis.emotional_triggers?.join(', ') || 'None'}</li>
                <li>Stereotypes: {analysisResult.text_analysis.stereotypes?.join(', ') || 'None'}</li>
                <li>
                  Entities:
                  {analysisResult.text_analysis.entities?.length > 0 ? (
                    <ul>
                      {analysisResult.text_analysis.entities.map((entity, idx) => (
                        <li key={idx}>{entity.text} ({entity.type})</li>
                      ))}
                    </ul>
                  ) : ' None'}
                </li>
                <li>Fact Checks: {analysisResult.text_analysis.fact_checks?.length > 0 ? JSON.stringify(analysisResult.text_analysis.fact_checks) : 'None'}</li>
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default VideoAnalytics;