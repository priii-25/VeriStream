// frontend/src/components/VideoAnalytics.js
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2'; // Using only Line chart
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
  // BarElement, // Removed as Bar chart is not used
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import '../styles/VideoAnalytics.css'; // Ensure this CSS file exists and is styled

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  // BarElement, // Removed
  Title,
  Tooltip,
  Legend
);

// Fix Leaflet marker icon issue (keep if using Leaflet)
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
});

const VideoAnalytics = () => {
  // --- State Variables ---
  const [file, setFile] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null); // Stores the entire response
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false); // Tracks if analysis/translation is in progress
  const [progress, setProgress] = useState(0); // Upload/analysis progress (0 to 1)
  const [videoUrl, setVideoUrl] = useState(null); // Object URL for video preview
  const [translation, setTranslation] = useState(null); // Stores translated text
  const [language, setLanguage] = useState('en'); // Target language for translation
  const [isFactCheckExpanded, setIsFactCheckExpanded] = useState(false); // Toggle for fact check card

  // --- Refs ---
  const wsRef = useRef(null); // WebSocket for progress

  // --- Handlers ---
  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);
    // Reset everything when a new file is selected
    setAnalysisResult(null);
    setError(null);
    setProgress(0);
    setTranslation(null);
    setIsFactCheckExpanded(false);
    if (videoUrl) {
        URL.revokeObjectURL(videoUrl); // Clean up previous object URL
    }
    if (selectedFile) {
        setVideoUrl(URL.createObjectURL(selectedFile));
    } else {
        setVideoUrl(null);
    }
  };

  // WebSocket connection for progress updates
  useEffect(() => {
    // Establish WebSocket connection
    // Ensure the URL matches your backend WebSocket endpoint for progress
    wsRef.current = new WebSocket('ws://127.0.0.1:5001/api/video/progress'); // PORT 5001

    wsRef.current.onopen = () => console.log('Progress WebSocket connected');

    wsRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        // Check if progress data exists and is a number
        if (data && typeof data.progress === 'number') {
           setProgress(Math.min(1, Math.max(0, data.progress))); // Clamp progress 0-1
           // Check for backend error signal (-1)
           if (data.progress < 0) {
              setError("Analysis failed during processing (received error signal).");
              setLoading(false); // Stop loading indicator
           }
        }
      } catch (e) {
        console.error("Failed to parse progress message:", e);
      }
    };

    wsRef.current.onerror = (error) => {
        console.error('Progress WebSocket error:', error);
        // Optionally set an error state here
        // setError("Progress connection error.");
    };

    wsRef.current.onclose = (event) => {
        console.log(`Progress WebSocket closed (Code: ${event.code})`);
        // Attempt to reconnect if closed unexpectedly? Usually not needed for progress WS.
    };

    // Cleanup function: close WebSocket and revoke object URL
    return () => {
      if (wsRef.current) {
        wsRef.current.close(1000, "Component unmounting"); // Clean closure
        wsRef.current = null;
      }
      if (videoUrl) {
        URL.revokeObjectURL(videoUrl);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Empty array means run only on mount and unmount

  // Handle analysis submission
  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      setError('Please select a video file.');
      return;
    }

    setLoading(true);
    setError(null);
    setProgress(0);
    setAnalysisResult(null); // Clear previous results
    setTranslation(null);
    setIsFactCheckExpanded(false);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // API endpoint for video analysis
      const response = await axios.post(
        'http://127.0.0.1:5001/api/video/analyze', // PORT 5001
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 300000 // Increased timeout to 5 minutes
        }
      );
      console.log("Analysis Response Received:", response.data);
      setAnalysisResult(response.data);
      setProgress(1); // Ensure progress bar completes on success
    } catch (err) {
      console.error("Analysis Submission Error:", err);
      // Try to get detailed error from backend response
      setError(err.response?.data?.detail || err.message || 'An error occurred during analysis.');
      setAnalysisResult(null); // Clear results on error
      setProgress(0); // Reset progress
    } finally {
      setLoading(false); // Stop loading indicator
    }
  };

  // Handle translation request - FIXED
  const handleTranslate = async () => {
    // Check if there's a result and transcription to translate
    if (!analysisResult?.original_transcription) {
      setError('No transcription available to translate.');
      return;
    }

    setLoading(true); // Use loading state for translation too
    setError(null);
    setTranslation(null);

    try {
      // API endpoint for translation - Sends JSON, not FormData
      const response = await axios.post(
        'http://127.0.0.1:5001/api/video/translate', // PORT 5001
        {
          // Payload expected by the backend endpoint
          transcription: analysisResult.original_transcription,
          language: language, // Target language from state
        },
        {
          headers: { 'Content-Type': 'application/json' },
          timeout: 45000, // 45 second timeout for translation
        }
      );
      setTranslation(response.data.translation);
    } catch (err) {
      console.error("Translation Request Error:", err);
      setError(err.response?.data?.detail || 'Translation failed.');
      setTranslation(null);
    } finally {
      setLoading(false);
    }
  };

  // Handle CSV download - Ensure keys match response
  const handleDownload = () => {
    const frameData = analysisResult?.frames_data; // Use correct key from backend
    if (!frameData?.timestamps || !frameData?.max_scores || !frameData?.faces_detected) {
      alert("Frame data is incomplete or missing for download.");
      return;
    }

    try {
      const csvData = frameData.timestamps.map((timestamp, index) => ({
        Timestamp: timestamp !== undefined ? timestamp.toFixed(3) : 'N/A',
        DeepfakeScore: frameData.max_scores?.[index] !== undefined ? frameData.max_scores[index].toFixed(3) : 'N/A',
        // Assuming faces_detected is boolean array
        FacesDetected: typeof frameData.faces_detected?.[index] === 'boolean'
            ? (frameData.faces_detected[index] ? 'Yes' : 'No')
            : 'N/A',
      }));
      const csv = Papa.unparse(csvData);
      const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      const safeFilename = file?.name.replace(/[^a-z0-9.]/gi, '_').toLowerCase() || 'video';
      link.download = `${safeFilename}_analysis_frames.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(link.href); // Clean up blob URL
    } catch (e) {
      console.error("Error creating/downloading CSV:", e);
      setError("Failed to generate CSV data.");
    }
  };

  // --- Derived Data for Display ---
  // Calculate Summary Metrics - Check for necessary data
  const calculateSummaryMetrics = () => {
    const scores = analysisResult?.frames_data?.max_scores;
    const finalScore = analysisResult?.final_score; // Use the overall score from backend

    if (!scores || scores.length === 0 || finalScore === undefined || finalScore === null) return null;

    const validScores = scores.filter(s => typeof s === 'number');
    const avgScore = validScores.reduce((a, b) => a + b, 0) / validScores.length || 0;
    const peakScore = Math.max(...validScores) || 0;
    const frameCount = scores.length; // Total frames attempted
    const alert = finalScore > 0.7 ? 'High Deepfake Probability Detected!' : 'Low Deepfake Probability';

    return { avgScore, peakScore, frameCount, alert, finalScore };
  };

  const summaryMetrics = calculateSummaryMetrics();

  // Chart Data - Check for necessary data
  const lineChartData = analysisResult?.frames_data?.timestamps?.length > 0 && analysisResult?.frames_data?.max_scores?.length > 0
    ? {
        labels: analysisResult.frames_data.timestamps.map((t) => (t !== undefined ? t.toFixed(2) : '')),
        datasets: [
          {
            label: 'Deepfake Score per Frame',
            data: analysisResult.frames_data.max_scores,
            borderColor: 'rgb(255, 99, 132)', // Red for alerts often
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            fill: false,
            tension: 0.1,
            pointRadius: 2, // Smaller points
          },
        ],
      }
    : null;

  // Chart Options
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'top', labels: { color: '#333' } },
      title: { display: true, text: 'Deepfake Score Over Time', color: '#333' },
    },
    scales: {
      x: { title: { display: true, text: 'Time (s)', color: '#333' }, ticks: { color: '#333', maxRotation: 0, autoSkip: true, maxTicksLimit: 20 } }, // Auto skip labels if too many
      y: { title: { display: true, text: 'Score', color: '#333' }, ticks: { color: '#333' }, beginAtZero: true, max: 1 },
    },
    animation: { duration: 500 } // Add subtle animation
  };

  // Location Data (Placeholder coordinates)
  const locations = analysisResult?.text_analysis?.locations?.map((loc, index) => ({
      name: loc.text,
      // Replace with actual geocoding results if available
      latitude: 51.505 + (index * 0.01), // Dummy latitude variation
      longitude: -0.09 + (index * 0.01), // Dummy longitude variation
  })) || [];


  // --- JSX Rendering ---
  return (
    <div className="video-analytics-container">
      <h1>Video File Analysis</h1>

      {/* Upload Form */}
      <form onSubmit={handleSubmit} className="upload-form">
        <label htmlFor="video-upload-input" className="file-label">
            {file ? file.name : "Choose Video File"}
        </label>
        <input
            id="video-upload-input"
            type="file"
            accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/*"
            onChange={handleFileChange}
            disabled={loading}
            style={{ display: 'none' }} // Hide default input, use label instead
        />
        <button type="submit" disabled={loading || !file}>
          {loading ? `Analyzing... (${(progress * 100).toFixed(0)}%)` : 'Analyze Video'}
        </button>
      </form>

      {/* Loading/Progress Indicator */}
      {loading && (
        <div className="progress-section">
          <progress value={progress} max="1" />
          <span>{(progress * 100).toFixed(0)}%</span>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Video Preview Area */}
      {videoUrl && !analysisResult && !loading && (
        <div className="uploaded-video card">
          <h3>Video Preview</h3>
          <video src={videoUrl} controls width="80%" />
        </div>
      )}

      {/* Results Display Area */}
      {analysisResult && (
        <div className="results-display">
          <h2>Analysis Results for {file?.name}</h2>

          {/* Overall Score & Summary */}
          {summaryMetrics && (
            <div className="card summary-card">
              <h3>Summary</h3>
               <p>Overall Max Deepfake Score: <span className="highlight">{summaryMetrics.finalScore.toFixed(3)}</span></p>
               <p className={`alert ${summaryMetrics.alert.includes('High') ? 'alert-high' : 'alert-low'}`}>
                   {summaryMetrics.alert}
               </p>
               <p>Average Frame Score: {summaryMetrics.avgScore.toFixed(3)}</p>
               <p>Peak Frame Score: {summaryMetrics.peakScore.toFixed(3)}</p>
               <p>Frames Analyzed: {summaryMetrics.frameCount}</p>
            </div>
          )}

          {/* Transcription Card */}
          <div className="card">
            <h3>Transcription</h3>
            <p><strong>Detected Language:</strong> {analysisResult.detected_language || 'N/A'}</p>
            <p><strong>Original Text:</strong></p>
            <textarea readOnly value={analysisResult.original_transcription || '[No transcription]'} rows={5}></textarea>

            {analysisResult.english_transcription && analysisResult.detected_language !== 'en' && (
               <>
                <p><strong>English Text:</strong></p>
                <textarea readOnly value={analysisResult.english_transcription} rows={5}></textarea>
               </>
             )}

             {/* Translation Controls */}
             {analysisResult.original_transcription && (
                 <div className="translate-section">
                    <label htmlFor="language-select">Translate Original to:</label>
                    <select id="language-select" value={language} onChange={(e) => setLanguage(e.target.value)} disabled={loading}>
                        {/* Add more common languages */}
                        <option value="en">English</option><option value="es">Spanish</option>
                        <option value="fr">French</option><option value="de">German</option>
                        <option value="zh-CN">Chinese (Simp)</option><option value="ja">Japanese</option>
                        <option value="ko">Korean</option><option value="ru">Russian</option>
                        <option value="ar">Arabic</option><option value="hi">Hindi</option>
                        <option value="bn">Bengali</option><option value="pt">Portuguese</option>
                         <option value="ta">Tamil</option><option value="te">Telugu</option>
                    </select>
                    <button onClick={handleTranslate} disabled={loading}>
                        {loading ? 'Translating...' : 'Translate'}
                    </button>
                    {translation && (
                        <div className="translated-text">
                            <p><strong>Translated Text ({language}):</strong></p>
                            <textarea readOnly value={translation} rows={3}></textarea>
                        </div>
                    )}
                </div>
             )}
          </div>

          {/* Deepfake Chart Card */}
          <div className="card">
             <h3>Deepfake Score Over Time</h3>
             {lineChartData ? (
                <div className="chart-container" style={{ height: '300px', position: 'relative' }}>
                    <Line data={lineChartData} options={chartOptions} />
                </div>
             ) : <p>No frame data available for chart.</p>}
              {/* Download Button */}
             {analysisResult?.frames_data?.timestamps?.length > 0 && (
                <button onClick={handleDownload} className="download-button">
                    Download Frame Data (CSV)
                </button>
             )}
          </div>

          {/* Text Analysis Card */}
          {analysisResult.text_analysis && (
            <div className="card">
              <h3>Textual Content Analysis</h3>
              <p>
                  Political Bias: {analysisResult.text_analysis.political_bias?.label || 'N/A'}
                  (Score: <span className="highlight">{analysisResult.text_analysis.political_bias?.score?.toFixed(2) || 'N/A'}</span>)
              </p>
              <p>
                  Manipulation Score: <span className="highlight">{analysisResult.text_analysis.manipulation_score?.toFixed(2) || 'N/A'}</span>
              </p>
              <p>
                  Emotional Triggers: {analysisResult.text_analysis.emotional_triggers?.length > 0 ? analysisResult.text_analysis.emotional_triggers.join(', ') : 'None'}
              </p>
              <p>
                  Stereotypes: {analysisResult.text_analysis.stereotypes?.length > 0 ? analysisResult.text_analysis.stereotypes.join(', ') : 'None'}
              </p>
               {analysisResult.text_analysis.entities?.length > 0 && (
                 <div>
                    <strong>Entities:</strong>
                    <ul>
                        {analysisResult.text_analysis.entities.map((entity, idx) => (
                            <li key={idx}> {entity.text} ({entity.type}) </li>
                        ))}
                    </ul>
                 </div>
               )}
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
                          <p>Verdict: <span className="highlight">{claim.final_verdict || 'N/A'}</span></p>
                          <p>Explanation: {claim.final_explanation || 'N/A'}</p>
                          <p><em>(Source: {claim.source || 'N/A'})</em></p>
                          <hr/>
                        </div>
                      ))
                    ) : (<p>No claims processed.</p>)}

                     <h4>Summary</h4>
                     <pre className="summary-box">{analysisResult.text_analysis.fact_check_result.summary || 'No summary provided.'}</pre>
                </div>
              )}
            </div>
          )}

            {/* Map Card - REMOVE IF LEAFLET CAUSES ISSUES */}
            {locations.length > 0 && (
                <div className="card">
                <h3>Mentioned Locations (Map - Placeholder Coords)</h3>
                <MapContainer center={[20, 0]} zoom={2} scrollWheelZoom={false} className="map-container">
                    <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" attribution='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' />
                    {locations.map((loc, idx) => (
                    <Marker key={idx} position={[loc.latitude, loc.longitude]}>
                        <Popup>{loc.name}</Popup>
                    </Marker>
                    ))}
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