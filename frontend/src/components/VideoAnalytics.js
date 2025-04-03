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
import '../styles/VideoAnalytics.css';

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
  const [expanded, setExpanded] = useState(false); // For fact-check collapse
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
    wsRef.current = new WebSocket('ws://127.0.0.1:5001/api/video/progress');
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
        'http://127.0.0.1:5001/api/video/analyze',
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' }, timeout: 180000 }
      );
      setAnalysisResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred while analyzing the video.');
      setAnalysisResult(null);
    } finally {
      setLoading(false);
    }
  };

  const handleTranslate = async () => {
    if (!analysisResult || !analysisResult.original_transcription) {
      setError('No transcription available to translate.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(
        'http://127.0.0.1:5001/api/video/translate',
        {
          transcription: analysisResult.original_transcription,
          language: language,
        },
        {
          headers: { 'Content-Type': 'application/json' },
          timeout: 30000,
        }
      );
      setTranslation(response.data.translation);
    } catch (err) {
      setError(err.response?.data?.detail || 'Translation failed.');
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

  const lineChartData = analysisResult?.frames_data
    ? {
        labels: analysisResult.frames_data.timestamps.map((t) => t.toFixed(2)),
        datasets: [
          {
            label: 'Deepfake Score',
            data: analysisResult.frames_data.max_scores,
            borderColor: '#007bff',
            backgroundColor: 'rgba(0, 123, 255, 0.2)',
            fill: false,
          },
        ],
      }
    : null;

  const barChartData = analysisResult?.frames_data
    ? {
        labels: analysisResult.frames_data.max_scores.map((_, i) => i),
        datasets: [
          {
            label: 'Score Distribution',
            data: analysisResult.frames_data.max_scores,
            backgroundColor: 'rgba(0, 123, 255, 0.5)',
            borderColor: '#007bff',
            borderWidth: 1,
          },
        ],
      }
    : null;

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: { position: 'top', labels: { color: '#ffffff' } },
      title: { display: true, color: '#ffffff' },
    },
    scales: {
      x: { title: { display: true, text: 'Time (s)', color: '#ffffff' }, ticks: { color: '#ffffff' } },
      y: { title: { display: true, text: 'Score', color: '#ffffff' }, ticks: { color: '#ffffff' }, beginAtZero: true, max: 1 },
    },
  };

  const locations =
    analysisResult?.text_analysis?.locations?.map((loc) => ({
      name: loc.text,
      latitude: 51.505, // Placeholder
      longitude: -0.09,
    })) || [];

  return (
    <div className="video-analytics-container">
      <h1>Video Analytics</h1>

      <form onSubmit={handleSubmit} className="upload-form">
        <input type="file" accept="video/*" onChange={handleFileChange} disabled={loading} />
        <button type="submit" disabled={loading || !file}>
          {loading ? 'Analyzing...' : 'Analyze Video'}
        </button>
      </form>

      {videoUrl && (
        <div className="uploaded-video">
          <h3>Uploaded Video</h3>
          <video src={videoUrl} controls />
        </div>
      )}

      {loading && (
        <div className="progress-section">
          <p>Processing video... ({(progress * 100).toFixed(0)}%)</p>
          <progress value={progress} max="1" />
        </div>
      )}

      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
        </div>
      )}

      {analysisResult && (
        <div className="results-section">
          <h2>Analysis Results</h2>

          <div className="card">
            <strong>Original Transcription ({analysisResult.detected_language}):</strong>{' '}
            {analysisResult.original_transcription || 'No transcription available'}
            {analysisResult.detected_language !== "en" && (
              <div>
                <strong>English Transcription:</strong>{' '}
                {analysisResult.english_transcription}
              </div>
            )}
            <div className="translate-section">
              <select value={language} onChange={(e) => setLanguage(e.target.value)}>
                <option value="en">English</option>
                <option value="es">Spanish</option>
                <option value="fr">French</option>
                <option value="hi">Hindi</option>
                <option value="bn">Bengali</option>
                <option value="ta">Tamil</option>
                <option value="te">Telugu</option>
                <option value="mr">Marathi</option>
                <option value="gu">Gujarati</option>
                <option value="kn">Kannada</option>
                <option value="ml">Malayalam</option>
                <option value="pa">Punjabi</option>
                <option value="de">German</option>
                <option value="it">Italian</option>
                <option value="zh">Chinese (Simplified)</option>
                <option value="ja">Japanese</option>
                <option value="ko">Korean</option>
              </select>
              <button onClick={handleTranslate} disabled={loading || !analysisResult.original_transcription}>
                {loading ? 'Translating...' : 'Translate'}
              </button>
              {translation && (
                <p>
                  <strong>Translated ({language}):</strong> {translation}
                </p>
              )}
            </div>
          </div>

          <div className="card">
            <strong>Final Deepfake Score:</strong>{' '}
            <span className="highlight">{analysisResult.final_score ? analysisResult.final_score.toFixed(2) : 'N/A'}</span>
          </div>

          {summaryMetrics && (
            <div className="card">
              <h3>Summary Metrics</h3>
              <p>Average Score: <span className="highlight">{summaryMetrics.avgScore.toFixed(2)}</span></p>
              <p>Peak Score: <span className="highlight">{summaryMetrics.peakScore.toFixed(2)}</span></p>
              <p>Total Frames Analyzed: {summaryMetrics.frameCount}</p>
              <p className={summaryMetrics.alert.includes('High') ? 'alert-high' : 'alert-low'}>
                {summaryMetrics.alert}
              </p>
            </div>
          )}

          {lineChartData && (
            <div className="chart-container">
              <h3>Deepfake Detection Over Time</h3>
              <Line
                data={lineChartData}
                options={{
                  ...chartOptions,
                  plugins: { ...chartOptions.plugins, title: { text: 'Deepfake Detection Over Time' } },
                }}
              />
            </div>
          )}

          {barChartData && (
            <div className="chart-container">
              <h3>Score Distribution</h3>
              <Bar
                data={barChartData}
                options={{
                  ...chartOptions,
                  plugins: { ...chartOptions.plugins, title: { text: 'Score Distribution' } },
                  scales: { x: { title: { text: 'Frame Index' } } },
                }}
              />
            </div>
          )}

          {analysisResult.frames_data?.timestamps?.length > 0 && (
            <div className="card">
              <strong>Frame Analysis ({analysisResult.frames_data.timestamps.length} frames):</strong>
              <ul>
                {analysisResult.frames_data.timestamps.map((timestamp, index) => (
                  <li key={index}>
                    Timestamp: {timestamp.toFixed(2)}s, Deepfake Score:{' '}
                    <span className="highlight">{(analysisResult.frames_data.max_scores[index] || 0).toFixed(2)}</span>, Face Detected:{' '}
                    {analysisResult.frames_data.faces_detected[index] ? 'Yes' : 'No'}
                  </li>
                ))}
              </ul>
              <button onClick={handleDownload} className="download-button">
                Download Results as CSV
              </button>
            </div>
          )}

          {analysisResult.text_analysis?.fact_check_result && (
            <div className="card fact-check-section">
              <h3>Fact Check Analysis</h3>
              <button onClick={() => setExpanded(!expanded)}>
                {expanded ? 'Hide Details' : 'Show Details'}
              </button>
              {expanded && (
                <>
                  <div>
                    <h4>Raw Google Fact Check API Results</h4>
                    {analysisResult.text_analysis.fact_check_result.raw_fact_checks &&
                    Object.keys(analysisResult.text_analysis.fact_check_result.raw_fact_checks).length > 0 ? (
                      Object.entries(analysisResult.text_analysis.fact_check_result.raw_fact_checks).map(([claim, results], idx) => (
                        <div key={idx}>
                          <p><strong>Claim:</strong> "{claim}"</p>
                          <ul>
                            {results.map((res, i) => (
                              <li key={i}>Verdict: {res.verdict} | Evidence: {res.evidence}</li>
                            ))}
                          </ul>
                        </div>
                      ))
                    ) : (
                      <p>No raw fact check data available.</p>
                    )}
                  </div>

                  <div>
                    <h4>Filtered Non-Checkable Sentences</h4>
                    {analysisResult.text_analysis.fact_check_result.non_checkable_claims?.length > 0 ? (
                      <ul>
                        {analysisResult.text_analysis.fact_check_result.non_checkable_claims.map((sentence, idx) => (
                          <li key={idx}>"{sentence}"</li>
                        ))}
                      </ul>
                    ) : (
                      <p>No sentences were filtered out.</p>
                    )}
                  </div>

                  <div>
                    <h4>Processed Claim Details</h4>
                    {analysisResult.text_analysis.fact_check_result.processed_claims?.length > 0 ? (
                      analysisResult.text_analysis.fact_check_result.processed_claims.map((claim, idx) => (
                        <div key={idx}>
                          <p><strong>Claim {idx + 1} (Original):</strong> "{claim.original_claim}" [Source: {claim.source}]</p>
                          <p>Preprocessed: "{claim.preprocessed_claim}"</p>
                          {claim.source === "Knowledge Graph" ? (
                            <>
                              <p>Final Verdict (From KG): <span className="highlight">{claim.final_verdict}</span></p>
                              <p>KG Explanation: {claim.final_explanation}</p>
                              {claim.kg_timestamp && (
                                <p>KG Timestamp: {new Date(claim.kg_timestamp * 1000).toLocaleString()}</p>
                              )}
                            </>
                          ) : claim.source === "Full Pipeline" ? (
                            <>
                              <p>NER Entities: {claim.ner_entities.length > 0 ? claim.ner_entities.map(e => `${e.text} (${e.label})`).join(', ') : 'None'}</p>
                              <p>Factual Score: {claim.factual_score?.toFixed(2) || 'N/A'}</p>
                              <p>Initial Check: {claim.initial_verdict_raw}</p>
                              <p>RAG Status: {claim.rag_status}</p>
                              {claim.top_rag_snippets.length > 0 && (
                                <div>
                                  <p>Top RAG Snippets:</p>
                                  <ul>
                                    {claim.top_rag_snippets.map((snippet, i) => (
                                      <li key={i}>{snippet}</li>
                                    ))}
                                  </ul>
                                </div>
                              )}
                              <p>Final Verdict (RAG+LLM): <span className="highlight">{claim.final_verdict}</span></p>
                              <p>LLM Justification: {claim.final_explanation}</p>
                            </>
                          ) : (
                            <p className="alert-high">Error: {claim.final_explanation}</p>
                          )}
                        </div>
                      ))
                    ) : (
                      <p>No processed claims available.</p>
                    )}
                  </div>

                  <div>
                    <h4>XAI (SHAP) Summary</h4>
                    {analysisResult.text_analysis.fact_check_result.shap_explanations?.length > 0 ? (
                      <ul>
                        {analysisResult.text_analysis.fact_check_result.shap_explanations.map((ex, idx) => (
                          <li key={idx}>
                            "{ex.claim}": {typeof ex.shap_values === 'string' ? ex.shap_values : '[SHAP Values Available]'}
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p>SHAP analysis skipped or no results.</p>
                    )}
                  </div>

                  <div>
                    <h4>Chain of Thought Summary</h4>
                    <pre>{analysisResult.text_analysis.fact_check_result.summary || 'No summary available.'}</pre>
                  </div>
                </>
              )}
            </div>
          )}

          {analysisResult.text_analysis?.knowledge_graph && (
            <div className="card knowledge-graph-section">
              <h3>Knowledge Graph</h3>
              <iframe
                src="http://127.0.0.1:5001/knowledge_graph"
                title="Knowledge Graph"
              />
            </div>
          )}

          {locations.length > 0 && (
            <div className="card">
              <h3>Geospatial Map</h3>
              <MapContainer center={[51.505, -0.09]} zoom={2} className="map-container">
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
            <div className="card">
              <strong>Text Analysis:</strong>
              <ul>
                <li>
                  Political Bias: {analysisResult.text_analysis.political_bias?.label || 'N/A'} (Score:{' '}
                  <span className="highlight">{analysisResult.text_analysis.political_bias?.score?.toFixed(2) || 'N/A'}</span>)
                </li>
                <li>
                  Manipulation Score:{' '}
                  <span className="highlight">{analysisResult.text_analysis.manipulation_score?.toFixed(2) || 'N/A'}</span>
                </li>
                <li>
                  Emotional Triggers:{' '}
                  {analysisResult.text_analysis.emotional_triggers?.join(', ') || 'None'}
                </li>
                <li>
                  Stereotypes: {analysisResult.text_analysis.stereotypes?.join(', ') || 'None'}
                </li>
                <li>
                  Entities:
                  {analysisResult.text_analysis.entities?.length > 0 ? (
                    <ul>
                      {analysisResult.text_analysis.entities.map((entity, idx) => (
                        <li key={idx}>
                          {entity.text} ({entity.type})
                        </li>
                      ))}
                    </ul>
                  ) : (
                    ' None'
                  )}
                </li>
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default VideoAnalytics;