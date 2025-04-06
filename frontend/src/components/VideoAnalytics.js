// frontend/src/components/VideoAnalytics.js
import React, { useState, useEffect, useRef, useMemo } from 'react'; // Added useMemo
import axios from 'axios';
import { Line, Bar } from 'react-chartjs-2'; // Ensure Bar is imported
import Papa from 'papaparse';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet'; // Keep imports even if commented out in JSX
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement, // Ensure BarElement is registered
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import '../styles/VideoAnalytics.css'; // Ensure CSS file exists

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement, // Register BarElement
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

// --- Constants ---
// Ensure this matches the backend host and port exactly
const BACKEND_URL = 'http://127.0.0.1:5001';

const VideoAnalytics = () => {
  // --- State ---
  const [file, setFile] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [videoUrl, setVideoUrl] = useState(null); // Local preview URL
  const [translation, setTranslation] = useState(null);
  const [language, setLanguage] = useState('en');
  const [isFactCheckExpanded, setIsFactCheckExpanded] = useState(false);
  const [showFrameDetails, setShowFrameDetails] = useState(false); // Toggle for detailed frame list

  // --- Refs ---
  const wsRef = useRef(null); // WebSocket for progress

  // --- Handlers ---

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);
    // Reset everything on new file selection
    setAnalysisResult(null);
    setError(null);
    setProgress(0);
    setTranslation(null);
    setIsFactCheckExpanded(false);
    setShowFrameDetails(false);
    if (videoUrl) URL.revokeObjectURL(videoUrl); // Clean up previous preview URL
    if (selectedFile) setVideoUrl(URL.createObjectURL(selectedFile));
    else setVideoUrl(null);
  };

  // Setup WebSocket for progress updates
  useEffect(() => {
    wsRef.current = new WebSocket(`${BACKEND_URL.replace('http', 'ws')}/api/video/progress`); // Use BACKEND_URL base
    wsRef.current.onopen = () => console.log('Progress WebSocket connected');
    wsRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (typeof data.progress === 'number') {
          setProgress(Math.min(Math.max(data.progress, 0), 1));
        }
      } catch (e) { console.error("Progress WS message error:", e); }
    };
    wsRef.current.onerror = (error) => console.error('Progress WebSocket error:', error);
    wsRef.current.onclose = (event) => console.log(`Progress WebSocket closed (Code: ${event.code})`);
    // Cleanup on unmount
    return () => { wsRef.current?.close(1000, "Component unmounting"); };
  }, []); // Run only on mount

  // Handle analysis submission
  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) { setError('Please select a video file.'); return; }

    setLoading(true);
    setError(null);
    setProgress(0);
    setAnalysisResult(null);
    setTranslation(null);
    setIsFactCheckExpanded(false);
    setShowFrameDetails(false);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(
        `${BACKEND_URL}/api/video/analyze`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' }, timeout: 600000 } // 10 min timeout
      );
      setAnalysisResult(response.data);
      console.log("Analysis Result Received:", response.data);
      setProgress(1); // Mark complete
    } catch (err) {
      console.error("Analysis API Error:", err);
      setError(err.response?.data?.detail || `Analysis failed: ${err.message}`);
      setAnalysisResult(null);
      setProgress(0);
    } finally {
      setLoading(false);
    }
  };

  // Handle translation request
  const handleTranslate = async () => {
    if (!analysisResult?.original_transcription) { /*...*/ return; }
    setLoading(true); setError(null); setTranslation(null);
    try {
      const response = await axios.post(
        `${BACKEND_URL}/api/video/translate`,
        { transcription: analysisResult.original_transcription, language: language },
        { headers: { 'Content-Type': 'application/json' }, timeout: 60000 }
      );
      setTranslation(response.data.translation);
    } catch (err) { /* ... error handling ... */ } finally { setLoading(false); }
  };

  // Handle CSV download
  const handleDownload = () => {
    // *** CORRECTED PATH ***
    const frameData = analysisResult?.deepfake_frames_data;
    if (!frameData?.timestamps || frameData.timestamps.length === 0) { /*...*/ return; }
    const csvData = frameData.timestamps.map((timestamp, index) => ({
      'Timestamp (s)': timestamp?.toFixed(3) ?? 'N/A',
      'Deepfake Score': (frameData.max_scores?.[index] ?? 0).toFixed(4),
      // 'Faces Detected': frameData.faces_detected?.[index] ? 'Yes' : 'No', // Keep commented if not available
    }));
    try {
        const csv = Papa.unparse(csvData);
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const csvFilename = file ? `${file.name.split('.').slice(0, -1).join('.')}_analysis.csv` : 'video_analysis_results.csv';
        link.href = URL.createObjectURL(blob);
        link.download = csvFilename;
        document.body.appendChild(link); link.click(); document.body.removeChild(link);
        URL.revokeObjectURL(link.href); setError(null);
    } catch (csvErr) { /* ... error handling ... */ }
  };

  // --- Derived Data for UI (Memoized) ---

  const summaryMetrics = useMemo(() => {
    // *** CORRECTED PATH ***
    const scores = analysisResult?.deepfake_frames_data?.max_scores;
    if (!scores || scores.length === 0) return null;
    const validScores = scores.filter(s => typeof s === 'number' && !isNaN(s));
    if (validScores.length === 0) return { avgScore: 0, peakScore: 0, frameCount: scores.length, alert: 'No valid scores.' };
    const avgScore = validScores.reduce((a, b) => a + b, 0) / validScores.length;
    const peakScore = Math.max(...validScores);
    // *** CORRECTED PATH ***
    const finalScore = analysisResult?.deepfake_final_score ?? 0;
    const alert = finalScore > 0.7 ? 'High deepfake probability detected' : 'Low deepfake probability';
    return { avgScore, peakScore, frameCount: scores.length, alert };
  }, [analysisResult]);

  const lineChartData = useMemo(() => {
    // *** CORRECTED PATH ***
    const frameData = analysisResult?.deepfake_frames_data;
    if (!frameData?.timestamps || frameData.timestamps.length === 0) return null;
    const labels = frameData.timestamps.map((t) => t?.toFixed(1) ?? '');
    const maxLabels = 50;
    const skipInterval = labels.length > maxLabels ? Math.ceil(labels.length / maxLabels) : 1;
    const chartLabels = labels.map((label, index) => index % skipInterval === 0 ? label : '');
    return {
        labels: chartLabels,
        datasets: [{
            label: 'Deepfake Score', data: frameData.max_scores ?? [],
            borderColor: 'rgb(255, 99, 132)', backgroundColor: 'rgba(255, 99, 132, 0.5)',
            tension: 0.2, pointRadius: 0, borderWidth: 1.5,
        }],
    };
  }, [analysisResult]);

  // Bar Chart Data
  const barChartData = useMemo(() => {
    // *** CORRECTED PATH ***
    const frameData = analysisResult?.deepfake_frames_data;
    if (!frameData?.max_scores || frameData.max_scores.length === 0) return null;
    const maxBars = 100;
    const scores = frameData.max_scores;
    const data = scores.length > maxBars ? scores.slice(0, maxBars) : scores;
    const labels = data.map((_, i) => i);
    return {
        labels: labels,
        datasets: [{
            label: `Score Distribution (First ${data.length} Frames)`, data: data,
            backgroundColor: 'rgba(54, 162, 235, 0.6)', borderColor: 'rgba(54, 162, 235, 1)', borderWidth: 1,
        }],
    };
  }, [analysisResult]);

  // Chart Options (common)
  const chartOptions = useMemo(() => ({
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { position: 'top' }, title: { display: true, text: 'Chart' } },
        scales: { x: { title: { display: true, text: 'Time (s) or Frame Index' } }, y: { title: { display: true, text: 'Score' }, beginAtZero: true, max: 1 } },
        animation: false, parsing: false,
  }), []);

  // Location Data (Placeholder)
  const locations = useMemo(() =>
    analysisResult?.text_analysis?.locations?.map((loc) => ({
      name: loc.text,
      latitude: 51.505, // Replace with actual geocoded data if available
      longitude: -0.09, // Replace with actual geocoded data if available
    })) || [],
  [analysisResult]);


  // --- JSX ---
  return (
    <div className="video-analytics-container">
      <h1>Video File Analysis</h1>

      {/* Upload Form */}
      <form onSubmit={handleSubmit} className="upload-form">
         <label htmlFor="video-upload-input" className="upload-label">
            {file ? file.name : "Choose Video File"}
        </label>
        <input id="video-upload-input" type="file" accept="video/*" onChange={handleFileChange} disabled={loading} style={{ display: 'none' }} />
        <button type="submit" disabled={loading || !file}>
          {loading ? `Analyzing... (${(progress * 100).toFixed(0)}%)` : 'Analyze Video'}
        </button>
      </form>

      {/* Loading/Progress/Error Display */}
      {loading && <div className="progress-section"> /* ... progress bar ... */ </div>}
      {error && <div className="error-message"><strong>Error:</strong> {error}</div>}

      {/* Content Area: Preview + Results */}
      <div className={`content-area ${analysisResult ? 'show-results' : ''}`}>
          {videoUrl && <div className="uploaded-video card"><h3>Preview</h3><video src={videoUrl} controls width="100%" /></div>}

          {analysisResult && (
            <div className="results-display">
              <h2>Analysis Results</h2>

              {/* Transcription Card */}
              <div className="card">
                <h4>Transcription</h4>
                <p><strong>Detected Language:</strong> {analysisResult.detected_language || 'N/A'}</p>
                <p className="transcription-text"><strong>Original:</strong> {analysisResult.original_transcription || 'N/A'}</p>
                {analysisResult.detected_language !== "en" && analysisResult.english_transcription && (
                    <p className="transcription-text"><strong>English:</strong> {analysisResult.english_transcription}</p>
                )}
                {analysisResult.original_transcription && (
                    <div className="translate-section">
                        <select value={language} onChange={(e) => setLanguage(e.target.value)} disabled={loading}>{/* Options */}</select>
                        <button onClick={handleTranslate} disabled={loading}>{loading ? '...' : `Translate`}</button>
                        {translation && <p><strong>Translation ({language}):</strong> {translation}</p>}
                    </div>
                )}
              </div>

              {/* Deepfake Summary Card */}
              <div className="card">
                <h4>Deepfake Summary</h4>
                <p><strong>Overall Score:</strong> <span className="highlight score-value">{analysisResult.deepfake_final_score?.toFixed(3) ?? 'N/A'}</span></p>
                {summaryMetrics && (
                  <>
                    <p>Peak Frame Score: <span className="highlight">{summaryMetrics.peakScore.toFixed(3)}</span></p>
                    <p>Average Frame Score: <span className="highlight">{summaryMetrics.avgScore.toFixed(3)}</span></p>
                    <p className={`alert ${summaryMetrics.alert.includes('High') ? 'alert-high' : 'alert-low'}`}>{summaryMetrics.alert}</p>
                  </>
                )}
              </div>

              {/* Explanation Images Card */}
              {/* *** CORRECTED PATH *** */}
              {analysisResult.deepfake_frames_data?.explanations?.length > 0 && (
                <div className="card explanation-section">
                  <h4>Deepfake Explanation (High-Scoring Frames)</h4>
                  <div className="explanation-images">
                    {analysisResult.deepfake_frames_data.explanations.map((exp, idx) => (
                      <div key={idx} className="explanation-item">
                        <p>Frame @ {exp.timestamp?.toFixed(2) ?? '?'}s (Score: <span className="highlight">{exp.score?.toFixed(3) ?? 'N/A'}</span>)</p>
                        <img
                          src={`${BACKEND_URL}${exp.url}`} // Use BACKEND_URL constant
                          alt={`Explanation frame at ${exp.timestamp?.toFixed(2) ?? '?'}s`}
                          className="explanation-image"
                          loading="lazy"
                          onError={(e) => { e.target.style.display = 'none'; console.error(`Failed to load explanation: ${exp.url}`) }}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Line Chart Card */}
              {lineChartData && (
                <div className="card chart-container">
                  <h4>Deepfake Score Over Time</h4>
                  <div className="chart-wrapper" style={{ height: '300px' }}>
                     <Line data={lineChartData} options={{ ...chartOptions, plugins: { ...chartOptions.plugins, title: { text: 'Deepfake Score Over Time' } }, scales: { ...chartOptions.scales, x: { ...chartOptions.scales.x, title: { text: 'Time (s)' } } } }} />
                  </div>
                </div>
              )}

              {/* Bar Chart Card */}
              {barChartData && (
                <div className="card chart-container">
                  <h4>Score Distribution</h4>
                  <div className="chart-wrapper" style={{ height: '300px' }}>
                     <Bar data={barChartData} options={{ ...chartOptions, plugins: { ...chartOptions.plugins, title: { text: `Score Distribution (First ${barChartData.datasets[0].data.length} Frames)` } }, scales: { ...chartOptions.scales, x: { ...chartOptions.scales.x, title: { text: 'Frame Index' } } } }} />
                  </div>
                </div>
              )}

              {/* Frame Data Card */}
              {/* *** CORRECTED PATH *** */}
              {analysisResult.deepfake_frames_data?.timestamps?.length > 0 && (
                  <div className="card frame-data-section">
                      <h4>Frame Analysis Details</h4>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                          <span>Total frames analyzed: {analysisResult.deepfake_frames_data.timestamps.length}</span>
                          <div>
                            <button onClick={handleDownload} className="download-button" style={{ marginRight: '10px' }}>Download CSV</button>
                            <button onClick={() => setShowFrameDetails(!showFrameDetails)} className="expand-button">{showFrameDetails ? 'Hide' : 'Show'} Frame List</button>
                          </div>
                      </div>
                      {/* Scrollable list */}
                       {showFrameDetails && (
                            <div className="frame-list-scrollable">
                                <ul>
                                    {/* *** CORRECTED PATH *** */}
                                    {analysisResult.deepfake_frames_data.timestamps.map((timestamp, index) => (
                                    <li key={index}>
                                        Time: {timestamp?.toFixed(2) ?? '?'}s,
                                        Score: <span className="highlight">{(analysisResult.deepfake_frames_data.max_scores?.[index] ?? 0).toFixed(3)}</span>
                                        {/* Face Detected: {analysisResult.deepfake_frames_data.faces_detected?.[index] ? 'Yes' : 'No'} */}
                                    </li>
                                    ))}
                                </ul>
                            </div>
                       )}
                  </div>
              )}

              {/* Fact Check Card */}
              {/* *** CORRECTED PATH *** */}
              {analysisResult.fact_check_analysis && (
                <div className="card fact-check-section">
                  <h4>Fact Check Analysis</h4>
                  <button onClick={() => setIsFactCheckExpanded(!isFactCheckExpanded)} className="expand-button">
                    {isFactCheckExpanded ? 'Hide Details' : 'Show Details'}
                  </button>
                  {isFactCheckExpanded && (
                      <div className="fact-check-details">
                           <h5>Processed Claims</h5>
                           {/* *** CORRECTED PATH *** */}
                            {analysisResult.fact_check_analysis.processed_claims?.length > 0 ? (
                                analysisResult.fact_check_analysis.processed_claims.map((claim, idx) => (
                                <div key={idx} className="claim-detail">
                                    <p><strong>Claim {idx + 1}:</strong> "{claim.original_claim || 'N/A'}"</p>
                                    <p>Verdict: <span className="highlight">{claim.final_verdict || 'N/A'}</span></p>
                                    <p>Explanation: {claim.final_explanation || 'N/A'}</p>
                                    <p><em>(Source: {claim.source || 'N/A'})</em></p>
                                </div>
                                ))
                            ) : <p>No claims processed.</p>}
                            {/* *** CORRECTED PATH *** */}
                             {analysisResult.fact_check_analysis.summary && (
                                 <><h5>Summary</h5><pre>{analysisResult.fact_check_analysis.summary}</pre></>
                             )}
                      </div>
                  )}
                </div>
              )}

              {/* Text Analysis Card */}
              {/* *** CORRECTED PATH *** */}
              {analysisResult.text_analysis && (
                <div className="card">
                  <h4>Text Analysis</h4>
                  <ul>
                    <li>Bias: {analysisResult.text_analysis.political_bias?.label || 'N/A'} (Score: <span className="highlight">{analysisResult.text_analysis.political_bias?.score?.toFixed(2) || 'N/A'}</span>)</li>
                    <li>Manipulation Score: <span className="highlight">{analysisResult.text_analysis.manipulation_score?.toFixed(2) || 'N/A'}</span></li>
                    <li>Triggers: {analysisResult.text_analysis.emotional_triggers?.join(', ') || 'None'}</li>
                    <li>Stereotypes: {analysisResult.text_analysis.stereotypes?.join(', ') || 'None'}</li>
                     {analysisResult.text_analysis.entities?.length > 0 && (
                         <li>Entities: <ul className="entity-list">{analysisResult.text_analysis.entities.map((e, i) => <li key={i}>{e.text} ({e.type})</li>)}</ul></li>
                     )}
                  </ul>
                </div>
              )}

              {/* Knowledge Graph Card */}
              {/* Renders iframe pointing to backend endpoint */}
              {/* *** CORRECTED PATH (Check if KG path is in analysisResult) - Assuming fixed path now *** */}
              {analysisResult && ( // Render if analysis ran, assuming KG endpoint is stable
                <div className="card knowledge-graph-section">
                    <h3>Knowledge Graph</h3>
                    <iframe
                        src={`${BACKEND_URL}/knowledge_graph`} // Use constant
                        title="Knowledge Graph"
                        className="knowledge-graph-iframe"
                        sandbox="allow-scripts allow-same-origin"
                        onError={(e) => console.error("KG iframe error:", e)}
                    />
                    <p><a href={`${BACKEND_URL}/knowledge_graph`} target="_blank" rel="noopener noreferrer">Open graph in new tab</a></p>
                </div>
              )}

              {/* Map Card - Remains commented */}
              {/* {locations.length > 0 && ( <div className="card"><h3>Map</h3> ... </div>)} */}

            </div> // End results-display
          )}
      </div> {/* End content-area */}
    </div> // End video-analytics-container
  );
};

export default VideoAnalytics;