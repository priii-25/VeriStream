// frontend/src/components/VideoAnalytics.js
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import Papa from 'papaparse';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement,
  Title, Tooltip, Legend, Filler
} from 'chart.js';
import '../styles/VideoAnalytics.css'; // Import updated CSS

// Chart.js Registration and Leaflet fix
ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler
);
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
});

// Define backend URL constant
const BACKEND_URL = 'http://127.0.0.1:5001'; // Ensure this matches your backend

const VideoAnalytics = () => {
  // --- State Variables ---
  const [file, setFile] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [videoUrl, setVideoUrl] = useState(null);
  const [translation, setTranslation] = useState(null);
  const [language, setLanguage] = useState('en');
  const [isFactCheckExpanded, setIsFactCheckExpanded] = useState(false);
  const [suspiciousFrames, setSuspiciousFrames] = useState([]); // Stores { index, timestamp, score, url }
  const [visibleHeatmapFrameIndex, setVisibleHeatmapFrameIndex] = useState(null); // Track visible heatmap

  // --- Refs ---
  const wsRef = useRef(null);

  // --- Handlers ---
  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);
    // Reset everything on new file selection
    setAnalysisResult(null); setError(null); setProgress(0);
    setTranslation(null); setIsFactCheckExpanded(false);
    setSuspiciousFrames([]); setVisibleHeatmapFrameIndex(null); // Reset heatmap state
    if (videoUrl) URL.revokeObjectURL(videoUrl); // Clean up previous blob URL
    setVideoUrl(selectedFile ? URL.createObjectURL(selectedFile) : null);
  };

  // WebSocket for progress updates
  useEffect(() => {
    const wsUrl = `ws://${BACKEND_URL.split('//')[1]}/api/video/progress`; // Construct WS URL
    wsRef.current = new WebSocket(wsUrl);

    wsRef.current.onopen = () => console.log('Progress WebSocket connected');
    wsRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data && typeof data.progress === 'number') {
           setProgress(Math.min(1, Math.max(0, data.progress)));
           if (data.progress < 0) { // Check for backend error signal
              setError("Analysis failed during processing (received error signal). Check backend logs.");
              setLoading(false);
           }
        }
      } catch (e) { console.error("Failed to parse progress message:", e); }
    };
    wsRef.current.onerror = (error) => { console.error('Progress WebSocket error:', error); /* Optionally set error state */ };
    wsRef.current.onclose = (event) => console.log(`Progress WebSocket closed (Code: ${event.code})`);

    // Cleanup function
    return () => {
      if (wsRef.current) wsRef.current.close(1000, "Component unmounting");
      if (videoUrl) URL.revokeObjectURL(videoUrl); // Also cleanup video URL on unmount
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Empty dependency array means run only on mount and unmount

  // Handle analysis submission
  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) { setError('Please select a video file.'); return; }
    if (loading) return; // Prevent double submissions

    setLoading(true); setError(null); setProgress(0);
    setAnalysisResult(null); setTranslation(null); setIsFactCheckExpanded(false);
    setSuspiciousFrames([]); setVisibleHeatmapFrameIndex(null); // Reset heatmap state

    const formData = new FormData();
    formData.append('file', file);

    try {
      const apiUrl = `${BACKEND_URL}/api/video/analyze`;
      const response = await axios.post( apiUrl, formData,
        { headers: { 'Content-Type': 'multipart/form-data' }, timeout: 480000 } // Increased to 8 minutes
      );
      console.log("Analysis Response Received:", response.data);
      setAnalysisResult(response.data);
      setProgress(1); // Ensure progress completes

      // Process overlay URLs from response
      if (response.data?.frames_data?.overlay_urls) {
        const overlays = response.data.frames_data.overlay_urls;
        const frames = response.data.frames_data;
        const foundSuspicious = [];
        // Use Object.keys and map to build the array safely
        Object.keys(overlays).forEach(frameIndexStr => {
            const frameIndex = parseInt(frameIndexStr, 10);
            // Find the corresponding data point using frame_indices
            const dataIndex = frames.frame_indices?.findIndex(idx => idx === frameIndex); // Check if frame_indices exists

            if (dataIndex !== -1 && frames.timestamps && frames.scores) { // Ensure required arrays exist
                const relativeUrl = overlays[frameIndexStr];
                foundSuspicious.push({
                    index: frameIndex,
                    timestamp: frames.timestamps[dataIndex],
                    score: frames.scores[dataIndex],
                    // Construct full URL using BACKEND_URL
                    url: relativeUrl.startsWith('/') ? `${BACKEND_URL}${relativeUrl}` : relativeUrl
                });
            } else {
                console.warn(`Could not find matching data for overlay frame index: ${frameIndex}`);
            }
        });
        // Sort by score descending to show most suspicious first
        foundSuspicious.sort((a, b) => b.score - a.score);
        setSuspiciousFrames(foundSuspicious);
        console.log("Processed suspicious frames with overlays:", foundSuspicious);
      } else {
         console.log("No overlay URLs found in the analysis response.");
      }

    } catch (err) {
      console.error("Analysis Submission Error:", err);
      // Provide more specific error message if possible
      let errMsg = 'An error occurred during analysis.';
      if (err.code === 'ECONNABORTED') {
          errMsg = `Analysis timed out after ${err.config.timeout / 1000} seconds. Please try a shorter video or check backend performance.`;
      } else {
          errMsg = err.response?.data?.detail || err.message || errMsg;
      }
      setError(errMsg);
      setAnalysisResult(null);
      setProgress(0); // Reset progress on error
    } finally {
      setLoading(false);
    }
  };

  // Handle translation request
  const handleTranslate = async () => {
    if (!analysisResult?.original_transcription) {
      setError('No transcription available to translate.'); return;
    }
    if (loading) return;
    setLoading(true); setError(null); setTranslation(null);
    try {
      const apiUrl = `${BACKEND_URL}/api/video/translate`;
      const response = await axios.post( apiUrl,
        { transcription: analysisResult.original_transcription, language: language },
        { headers: { 'Content-Type': 'application/json' }, timeout: 60000 } // 60 sec timeout
      );
      setTranslation(response.data.translation);
    } catch (err) {
      console.error("Translation Request Error:", err);
      setError(err.response?.data?.detail || 'Translation failed.');
    } finally { setLoading(false); }
  };

  // Handle CSV download
  const handleDownload = () => {
    const frameData = analysisResult?.frames_data;
    // Check existence of all required arrays
    if (!frameData?.timestamps || !frameData?.scores || !frameData?.frame_indices) {
      alert("Frame data is incomplete or missing for download."); return;
    }
    try {
      // Map data, checking for undefined values at each step
      const csvData = frameData.frame_indices.map((frameIndex, i) => ({
        FrameIndex: frameIndex ?? 'N/A',
        Timestamp: frameData.timestamps[i]?.toFixed(3) ?? 'N/A',
        DeepfakeScore: frameData.scores[i]?.toFixed(3) ?? 'N/A',
        OverlayGenerated: frameData.overlay_urls?.[frameIndex] ? 'Yes' : 'No', // Check using frameIndex
      }));

      const csv = Papa.unparse(csvData);
      const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      // Sanitize filename
      const safeFilename = file?.name.replace(/[^a-z0-9._-]/gi, '_').toLowerCase() || 'video_analysis';
      link.download = `${safeFilename}_frames.csv`;
      document.body.appendChild(link); link.click(); document.body.removeChild(link);
      URL.revokeObjectURL(link.href); // Clean up blob URL
    } catch (e) {
      console.error("Error creating/downloading CSV:", e);
      setError("Failed to generate CSV data.");
    }
  };

  // Handler for the "Show/Hide Heatmap" button
  const handleToggleHeatmap = (frameIndex) => {
    setVisibleHeatmapFrameIndex(prevIndex =>
      prevIndex === frameIndex ? null : frameIndex // Toggle logic
    );
  };

  // --- Derived Data for Display (Chart data, metrics, locations - remain the same logic) ---
  const calculateSummaryMetrics = () => {
    const scores = analysisResult?.frames_data?.scores;
    const finalScore = analysisResult?.final_score;
    if (!scores?.length || finalScore === undefined || finalScore === null) return null;
    const validScores = scores.filter(s => typeof s === 'number');
    if (!validScores.length) return { avgScore: 0, peakScore: 0, frameCount: scores.length, alert: 'No valid scores', finalScore };
    const avgScore = validScores.reduce((a, b) => a + b, 0) / validScores.length;
    const peakScore = Math.max(...validScores);
    const frameCount = scores.length;
    const alert = finalScore > 0.7 ? 'High Deepfake Probability Detected!' : (finalScore >= 0 ? 'Low Deepfake Probability' : 'Score Error');
    return { avgScore, peakScore, frameCount, alert, finalScore };
  };
  const summaryMetrics = calculateSummaryMetrics();

  const lineChartData = analysisResult?.frames_data?.timestamps?.length > 0 && analysisResult?.frames_data?.scores?.length > 0
    ? {
        labels: analysisResult.frames_data.timestamps.map((t) => t?.toFixed(2) ?? '?'), // Handle potential nulls
        datasets: [ {
            label: 'Deepfake Score',
            data: analysisResult.frames_data.scores,
            borderColor: 'rgb(0, 123, 255)', // Blue line
            backgroundColor: 'rgba(0, 123, 255, 0.1)', // Light blue fill
            fill: true,
            tension: 0.1, pointRadius: 1, pointHoverRadius: 5, borderWidth: 1.5,
        } ],
      } : null;

  const chartOptions = {
    responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { position: 'top', labels: { color: '#e0e0e0' } }, // Light text for dark theme
      title: { display: true, text: 'Deepfake Score Over Time', color: '#ffffff' },
       tooltip: {
            bodyColor: '#e0e0e0', titleColor: '#ffffff', backgroundColor: 'rgba(0, 0, 0, 0.8)',
            callbacks: {
               label: function(context) {
                   let label = context.dataset.label || ''; if (label) { label += ': '; }
                   if (context.parsed.y !== null) { label += context.parsed.y.toFixed(3); }
                   const frameIndex = analysisResult?.frames_data?.frame_indices?.[context.dataIndex];
                   if (frameIndex !== undefined) { label += ` (Frame ${frameIndex})`; }
                   return label;
               }
           }
       }
    },
    scales: {
      x: { title: { display: true, text: 'Time (s)', color: '#ccc' }, ticks: { color: '#ccc', maxRotation: 0, autoSkip: true, maxTicksLimit: 15 } },
      y: { title: { display: true, text: 'Score', color: '#ccc' }, ticks: { color: '#ccc' }, beginAtZero: true, max: 1, grid: { color: 'rgba(255, 255, 255, 0.1)' } }, // Lighter grid lines
    },
    animation: { duration: 300 }
  };

  // Dummy locations - replace if real geocoding is added
  const locations = analysisResult?.text_analysis?.locations?.map((loc, index) => ({
      name: loc.text, latitude: 51.505 + (Math.random() - 0.5) * 0.2, longitude: -0.09 + (Math.random() - 0.5) * 0.2,
  })) || [];


  // --- JSX Rendering ---
  return (
    <div className="video-analytics-container">
      <h1>Video File Analysis</h1>

      {/* Upload Form */}
      <form onSubmit={handleSubmit} className="upload-form">
        <label htmlFor="video-upload-input" className="file-label" title={file ? file.name : "Choose Video File"}>
            {file ? file.name : "Choose Video File"}
        </label>
        <input id="video-upload-input" type="file" accept="video/*" onChange={handleFileChange} disabled={loading} style={{ display: 'none' }} />
        <button type="submit" disabled={loading || !file}>
          {loading ? `Analyzing... (${(progress * 100).toFixed(0)}%)` : 'Analyze Video'}
        </button>
      </form>

      {/* Loading/Progress Indicator */}
      {loading && <div className="progress-section"><progress value={progress} max="1" /><span>{(progress * 100).toFixed(0)}%</span></div>}
      {error && <div className="error-message"><strong>Error:</strong> {error}</div>}

      {/* Video Preview Area */}
      {videoUrl && !analysisResult && !loading && <div className="uploaded-video card"><h3>Video Preview</h3><video src={videoUrl} controls width="720" /></div>}

      {/* Results Display Area */}
      {analysisResult && (
        <div className="results-display">
          <h2>Analysis Results: {file?.name}</h2>

          {/* Summary Card */}
          {summaryMetrics && (
             <div className="card summary-card"><h3>Summary</h3>
                <p>Overall Max Deepfake Score: <span className="highlight">{summaryMetrics.finalScore?.toFixed(3) ?? 'N/A'}</span></p>
                <p className={`alert ${summaryMetrics.alert.includes('High') ? 'alert-high' : 'alert-low'}`}>{summaryMetrics.alert}</p>
                <p>Average Frame Score: {summaryMetrics.avgScore?.toFixed(3) ?? 'N/A'}</p>
                <p>Frames Analyzed: {summaryMetrics.frameCount ?? 'N/A'}</p>
             </div>
          )}

          {/* Transcription Card */}
          <div className="card">
             <h3>Transcription</h3>
             <p><strong>Detected Language:</strong> {analysisResult.detected_language || 'N/A'}</p>
             <p><strong>Original Text:</strong></p>
             <textarea readOnly value={analysisResult.original_transcription || '[No transcription]'} rows={5}></textarea>
             {analysisResult.english_transcription && analysisResult.detected_language !== 'en' && (
                <> <p><strong>English Text:</strong></p> <textarea readOnly value={analysisResult.english_transcription} rows={5}></textarea> </>
             )}
             {/* Translation Controls */}
             {analysisResult.original_transcription && (
                 <div className="translate-section">
                    <label htmlFor="language-select">Translate Original to:</label>
                    <select id="language-select" value={language} onChange={(e) => setLanguage(e.target.value)} disabled={loading}>
                    <option value="en">English</option>
                        <option value="hi">Hindi (हिन्दी)</option>
                        <option value="bn">Bengali (বাংলা)</option>
                        <option value="mr">Marathi (मराठी)</option>
                        <option value="te">Telugu (తెలుగు)</option>
                        <option value="ta">Tamil (தமிழ்)</option>
                        <option value="gu">Gujarati (ગુજરાતી)</option>
                        <option value="kn">Kannada (ಕನ್ನಡ)</option>
                        <option value="ml">Malayalam (മലയാളം)</option>
                        <option value="pa">Punjabi (ਪੰਜਾਬੀ)</option>
                        <option value="or">Odia (ଓଡ଼ିଆ)</option>
                        <option value="as">Assamese (অসমীয়া)</option>
                        <option value="ur">Urdu (اردو)</option>
                    </select>
                    <button onClick={handleTranslate} disabled={loading}> {loading ? 'Translating...' : 'Translate'} </button>
                    {translation && ( <div className="translated-text"><p><strong>Translated Text ({language}):</strong></p><textarea readOnly value={translation} rows={3}></textarea></div> )}
                </div>
             )}
          </div>

          {/* Deepfake Chart Card */}
          <div className="card">
             <h3>Deepfake Score Over Time</h3>
             {lineChartData ? (
                <div className="chart-container"> <Line data={lineChartData} options={chartOptions} /> </div>
             ) : <p>No frame data available for chart.</p>}
             {analysisResult?.frames_data?.timestamps?.length > 0 && ( <button onClick={handleDownload} className="download-button">Download Frame Data (CSV)</button> )}
          </div>

          {/* Suspicious Frames / Heatmaps Card (Alternative 1 Implementation) */}
          {suspiciousFrames.length > 0 && (
            <div className="card suspicious-frames-card">
              <h3>Suspicious Frames (Score {'>'} 0.7)</h3>
              <p>Frames with high deepfake scores. Click button to see attention heatmap.</p>
              <ul className="suspicious-frames-list">
                  {suspiciousFrames.map(frame => (
                      <li key={frame.index} className="suspicious-frame-item-detailed">
                          {/* Frame Info + Button */}
                          <div className="frame-info">
                              <p>Frame: {frame.index}</p>
                              <p>Time: {frame.timestamp?.toFixed(2)}s</p>
                              <p>Score: <span className="highlight">{frame.score?.toFixed(3)}</span></p>
                              <button onClick={() => handleToggleHeatmap(frame.index)} className="toggle-heatmap-button">
                                  {visibleHeatmapFrameIndex === frame.index ? 'Hide Heatmap' : 'Show Heatmap'}
                              </button>
                          </div>
                          {/* Conditionally Rendered Heatmap */}
                          {visibleHeatmapFrameIndex === frame.index && (
                              <div className="heatmap-display-area">
                                  <img src={frame.url} alt={`Heatmap Overlay Frame ${frame.index}`} loading="lazy" className="heatmap-image-large"/>
                              </div>
                          )}
                      </li>
                  ))}
              </ul>
            </div>
          )}
          {/* End Suspicious Frames Card */}


          {/* Text Analysis Card */}
          {analysisResult.text_analysis && (
            <div className="card">
              <h3>Textual Content Analysis</h3>
              <p> Political Bias: {analysisResult.text_analysis.political_bias?.label || 'N/A'} (Score: <span className="highlight">{analysisResult.text_analysis.political_bias?.score?.toFixed(2) || 'N/A'}</span>)</p>
              <p> Manipulation Score: <span className="highlight">{analysisResult.text_analysis.manipulation_score?.toFixed(2) || 'N/A'}</span></p>
              <p> Emotional Triggers: {analysisResult.text_analysis.emotional_triggers?.join(', ') || 'None'}</p>
              <p> Stereotypes: {analysisResult.text_analysis.stereotypes?.join(', ') || 'None'} </p>
               {!!analysisResult.text_analysis.entities?.length && ( <div> <strong>Entities:</strong> <ul> {analysisResult.text_analysis.entities.map((entity, idx) => (<li key={idx}> {entity.text} ({entity.type}) </li> ))} </ul> </div> )}
            </div>
          )}

          {/* Fact Check Card */}
          {analysisResult.text_analysis?.fact_check_result && (
            <div className="card fact-check-section">
              <h3>Fact Check Details</h3>
              <button onClick={() => setIsFactCheckExpanded(!isFactCheckExpanded)} className="toggle-button">
                {isFactCheckExpanded ? 'Hide Details' : 'Show Details'}
              </button>
              {isFactCheckExpanded && (
                <div className="fact-check-content">
                    <h4>Processed Claims</h4>
                    {analysisResult.text_analysis.fact_check_result.processed_claims?.length > 0 ? (
                      analysisResult.text_analysis.fact_check_result.processed_claims.map((claim, idx) => (
                        <div key={idx} className="claim-detail">
                          <p><strong>Claim {idx + 1}:</strong> "{claim.original_claim || 'N/A'}"</p>
                          <p>Verdict: <span className="highlight">{claim.final_label || 'N/A'}</span></p>
                          <p>Explanation: {claim.final_explanation || 'N/A'}</p>
                          <p><em>(Source: {claim.source || 'N/A'})</em></p>
                        </div>
                      ))
                    ) : (<p>No claims processed or found in this text.</p>)}
                     <h4>Processing Summary</h4>
                     <pre className="summary-box">{analysisResult.text_analysis.fact_check_result.summary || 'No summary provided.'}</pre>
                </div>
              )}
            </div>
          )}

            {/* Map Card - REMOVE IF UNSTABLE OR NOT NEEDED */}
            {locations.length > 0 && (
                <div className="card">
                <h3>Mentioned Locations (Map - Placeholder Coords)</h3>
                <MapContainer center={[20, 0]} zoom={2} scrollWheelZoom={false} className="map-container">
                    <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" attribution='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' />
                    {locations.map((loc, idx) => ( <Marker key={idx} position={[loc.latitude, loc.longitude]}><Popup>{loc.name}</Popup></Marker> ))}
                </MapContainer>
                <p><em>Note: Map locations use placeholder coordinates.</em></p>
                </div>
            )}
            {/* End Map Card */}

        </div> // End results-display
      )}
    </div> // End video-analytics-container
  );
};

export default VideoAnalytics;