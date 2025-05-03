import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import Papa from 'papaparse';
// Uncomment if adding Leaflet map back
// import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
// import 'leaflet/dist/leaflet.css';
// import L from 'leaflet';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement,
  Title, Tooltip, Legend, Filler
} from 'chart.js';
// Ensure you have a corresponding CSS file or reuse VideoAnalytics.css
import '../styles/VerifyUpload.css'; // Or '../styles/VideoAnalytics.css'

// --- Register Chart.js elements ---
ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler
);
// --- Leaflet Icon Fix (Uncomment if using map) ---
// delete L.Icon.Default.prototype._getIconUrl;
// L.Icon.Default.mergeOptions({
//   iconRetinaUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png',
//   iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
//   shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
// });

const BACKEND_URL = 'http://127.0.0.1:5001'; // Ensure this matches your backend

function VerifyUpload() {
  // --- State Variables (Combined from VideoAnalytics and original VerifyUpload) ---
  const [file, setFile] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null); // Holds ALL results
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  // const [progress, setProgress] = useState(0); // Progress state (requires backend endpoint modification)
  const [videoUrl, setVideoUrl] = useState(null);
  const [translation, setTranslation] = useState(null);
  const [language, setLanguage] = useState('en');
  const [isFactCheckExpanded, setIsFactCheckExpanded] = useState(false);
  const [suspiciousFrames, setSuspiciousFrames] = useState([]);
  const [visibleHeatmapFrameIndex, setVisibleHeatmapFrameIndex] = useState(null);
  const [isSpeaking, setIsSpeaking] = useState(false); // For TTS
  const [ttsSupported, setTtsSupported] = useState(true); // Assume supported initially
  const [voices, setVoices] = useState([]); // Optional: for voice selection
  const [selectedVoiceURI, setSelectedVoiceURI] = useState(null); // Optional: store selected voice

  // --- Refs ---
  // const wsRef = useRef(null); // Ref for progress WebSocket (if implemented)
  const currentUtteranceRef = useRef(null); // For TTS state management

  // --- Check TTS Support and Load Voices ---
  useEffect(() => {
    const synth = window.speechSynthesis;
    if (!synth) {
      console.warn("Speech Synthesis not supported by this browser.");
      setTtsSupported(false);
      return;
    }
    const loadVoices = () => {
        try {
            const availableVoices = synth.getVoices();
            if (availableVoices.length > 0) {
                setVoices(availableVoices);
                console.log("Voices loaded:", availableVoices.length);
            } else {
                 console.log("getVoices() initial call returned empty list, waiting for event...");
            }
        } catch (e) {
             console.error("Error getting voices:", e);
             setTtsSupported(false);
        }
    };
    loadVoices();
    if (synth.onvoiceschanged !== undefined) synth.onvoiceschanged = loadVoices;
    return () => {
        if (synth && synth.onvoiceschanged !== undefined) synth.onvoiceschanged = null;
        if (synth && synth.speaking) synth.cancel();
    };
  }, []);

  // --- TTS Handlers ---
  const speakText = useCallback((textToSpeak) => {
    const synth = window.speechSynthesis;
    if (!ttsSupported || !synth || !textToSpeak || typeof textToSpeak !== 'string' || textToSpeak.trim() === '') return;

    const wasSpeaking = synth.speaking;
    if (wasSpeaking) synth.cancel();

    const startSpeaking = () => {
        if (synth.speaking) { setIsSpeaking(false); return; } // Abort if still speaking

        const utterance = new SpeechSynthesisUtterance(textToSpeak.trim());
        currentUtteranceRef.current = utterance;

        if (selectedVoiceURI) {
            const selectedVoice = voices.find(v => v.voiceURI === selectedVoiceURI);
            if (selectedVoice) utterance.voice = selectedVoice;
        }

        utterance.onstart = () => { if (currentUtteranceRef.current === utterance) setIsSpeaking(true); };
        utterance.onend = () => { if (currentUtteranceRef.current === utterance) { setIsSpeaking(false); currentUtteranceRef.current = null; }};
        utterance.onerror = (event) => { console.error("TTS Error:", event.error); setError(`Speech error: ${event.error}`); if (currentUtteranceRef.current === utterance) { setIsSpeaking(false); currentUtteranceRef.current = null; }};

        try { synth.speak(utterance); }
        catch(e) { console.error("TTS synth.speak Error:", e); setError("Failed to initiate speech."); setIsSpeaking(false); currentUtteranceRef.current = null; }
    };

    if (wasSpeaking) setTimeout(startSpeaking, 150);
    else startSpeaking();
  }, [ttsSupported, voices, selectedVoiceURI, setError]);

  const handleStopSpeaking = useCallback(() => {
    const synth = window.speechSynthesis;
    if (ttsSupported && synth && synth.speaking) {
        synth.cancel();
        setIsSpeaking(false);
        currentUtteranceRef.current = null;
    }
  }, [ttsSupported]);

  // --- File Handling ---
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setAnalysisResult(null); // Reset combined results
    setError(null);
    // setProgress(0); // Reset progress if using
    setTranslation(null);
    setIsFactCheckExpanded(false);
    setSuspiciousFrames([]);
    setVisibleHeatmapFrameIndex(null);
    handleStopSpeaking(); // Stop TTS
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoUrl(selectedFile ? URL.createObjectURL(selectedFile) : null);
  };

  // --- Form Submission ---
  const handleSubmit = async (event) => {
    event.preventDefault(); // Prevent default form submission
    if (!file || loading) return;

    // Reset state
    setLoading(true); setError(null); setAnalysisResult(null); setTranslation(null);
    setIsFactCheckExpanded(false); setSuspiciousFrames([]); setVisibleHeatmapFrameIndex(null);
    handleStopSpeaking();
    // setProgress(0);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // *** Call the combined analysis+verification endpoint ***
      const response = await axios.post(
        `${BACKEND_URL}/api/video/analyze-verify`, // Target the endpoint doing ALL the work
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' }, timeout: 600000 } // Extended timeout (10 min)
      );

      console.log("Combined Analysis & Verification Response Received:", response.data);
      setAnalysisResult(response.data); // Store the full combined results

      // --- Process overlays (copied from VideoAnalytics) ---
       if (response.data?.frames_data?.overlay_urls) {
          const overlays = response.data.frames_data.overlay_urls;
          const frames = response.data.frames_data;
          const foundSuspicious = Object.keys(overlays).map(frameIndexStr => {
                const frameIndex = parseInt(frameIndexStr, 10);
                // Ensure frame_indices exists before searching
                const dataIndex = frames.frame_indices?.findIndex(idx => idx === frameIndex);
                // Ensure other data arrays exist and index is valid
                if (dataIndex !== undefined && dataIndex !== -1 && frames.timestamps?.[dataIndex] !== undefined && frames.scores?.[dataIndex] !== undefined) {
                    const relativeUrl = overlays[frameIndexStr];
                    return { index: frameIndex, timestamp: frames.timestamps[dataIndex], score: frames.scores[dataIndex], url: relativeUrl.startsWith('/') ? `${BACKEND_URL}${relativeUrl}` : relativeUrl };
                }
                console.warn(`Could not find matching data for overlay frame index: ${frameIndex}`);
                return null;
          }).filter(item => item !== null); // Filter out entries where data was missing

          foundSuspicious.sort((a, b) => b.score - a.score); // Sort by score desc
          setSuspiciousFrames(foundSuspicious);
          console.log("Processed suspicious frames with overlays:", foundSuspicious);
      } else {
          console.log("No overlay URLs found in the analysis response.");
          setSuspiciousFrames([]); // Ensure it's empty if not found
      }
      // --- (End overlay processing) ---

      // setProgress(1); // If using progress
    } catch (err) {
      console.error("Combined Analysis/Verification Submission Error:", err);
      let errMsg = 'An error occurred during analysis & verification.';
       if (err.code === 'ECONNABORTED') { errMsg = `Analysis/Verification timed out. Please try a shorter video or check the backend.`; }
       else { errMsg = err.response?.data?.detail || err.message || errMsg; }
      setError(errMsg);
      setAnalysisResult(null);
      // setProgress(0); // Reset progress if using
    } finally {
      setLoading(false);
    }
  };

  // --- Other Handlers (Translate, Download, Toggle Heatmap) ---
  const handleTranslate = async () => {
      if (!analysisResult?.original_transcription || loading) return;
      setLoading(true); setError(null); setTranslation(null); handleStopSpeaking();
      try {
          const apiUrl = `${BACKEND_URL}/api/video/translate`;
          const response = await axios.post(apiUrl, {
              transcription: analysisResult.original_transcription,
              language: language
          }, { headers: { 'Content-Type': 'application/json' }, timeout: 60000 });
          setTranslation(response.data.translation);
      } catch (err) {
          setError(err.response?.data?.detail || 'Translation failed.');
      } finally { setLoading(false); }
  };

  const handleDownload = () => {
    const frameData = analysisResult?.frames_data;
    if (!frameData?.timestamps || !frameData?.scores || !frameData?.frame_indices) {
        alert("Frame data is incomplete or missing for download."); return;
    }
    try {
        const csvData = frameData.frame_indices.map((frameIndex, i) => ({
            FrameIndex: frameIndex ?? 'N/A',
            Timestamp: frameData.timestamps[i]?.toFixed(3) ?? 'N/A',
            DeepfakeScore: frameData.scores[i]?.toFixed(3) ?? 'N/A',
            OverlayGenerated: frameData.overlay_urls?.[frameIndex] ? 'Yes' : 'No',
        }));
        const csv = Papa.unparse(csvData);
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        const safeFilename = file?.name.replace(/[^a-z0-9._-]/gi, '_').toLowerCase() || 'video_analysis_verify';
        link.download = `${safeFilename}_frames.csv`;
        document.body.appendChild(link); link.click(); document.body.removeChild(link);
        URL.revokeObjectURL(link.href);
    } catch (e) {
        console.error("Error creating/downloading CSV:", e);
        setError("Failed to generate CSV data.");
    }
  };

  const handleToggleHeatmap = (frameIndex) => {
      setVisibleHeatmapFrameIndex(prev => prev === frameIndex ? null : frameIndex);
  };

  // --- Derived Data for Display ---
  const calculateSummaryMetrics = () => {
    const scores = analysisResult?.frames_data?.scores;
    const finalScore = analysisResult?.final_score; // This is the max deepfake score
    if (!scores?.length || finalScore === undefined || finalScore === null) return null;
    const validScores = scores.filter(s => typeof s === 'number');
    if (!validScores.length) return { avgScore: 0, peakScore: finalScore, frameCount: scores.length, alert: 'No valid deepfake scores', finalScore };
    const avgScore = validScores.reduce((a, b) => a + b, 0) / validScores.length;
    const alert = finalScore > 0.7 ? 'High Deepfake Probability Detected!' : (finalScore >= 0 ? 'Low Deepfake Probability' : 'Score Error');
    return { avgScore, peakScore: finalScore, frameCount: scores.length, alert, finalScore };
  };
  const summaryMetrics = calculateSummaryMetrics();

  const lineChartData = analysisResult?.frames_data?.timestamps?.length > 0 && analysisResult?.frames_data?.scores?.length > 0
    ? {
        labels: analysisResult.frames_data.timestamps.map((t) => t?.toFixed(2) ?? '?'),
        datasets: [{
            label: 'Deepfake Score', data: analysisResult.frames_data.scores,
            borderColor: 'rgb(0, 123, 255)', backgroundColor: 'rgba(0, 123, 255, 0.1)',
            fill: true, tension: 0.1, pointRadius: 1, pointHoverRadius: 5, borderWidth: 1.5,
        }],
      } : null;

  const chartOptions = { /* (Keep existing options from VideoAnalytics) */
        responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
        plugins: { legend: { position: 'top', labels: { color: '#e0e0e0' } }, title: { display: true, text: 'Deepfake Score Over Time', color: '#ffffff' }, tooltip: { bodyColor: '#e0e0e0', titleColor: '#ffffff', backgroundColor: 'rgba(0, 0, 0, 0.8)', callbacks: { label: function(context) { let label = context.dataset.label || ''; if (label) { label += ': '; } if (context.parsed.y !== null) { label += context.parsed.y.toFixed(3); } const frameIndex = analysisResult?.frames_data?.frame_indices?.[context.dataIndex]; if (frameIndex !== undefined) { label += ` (Frame ${frameIndex})`; } return label; } } } },
        scales: { x: { title: { display: true, text: 'Time (s)', color: '#ccc' }, ticks: { color: '#ccc', maxRotation: 0, autoSkip: true, maxTicksLimit: 15 } }, y: { title: { display: true, text: 'Score', color: '#ccc' }, ticks: { color: '#ccc' }, beginAtZero: true, max: 1, grid: { color: 'rgba(255, 255, 255, 0.1)' } } },
        animation: { duration: 300 }
    };

  // Extract face verification results specifically for display
  const faceVerificationResult = analysisResult?.face_verification;

  // --- JSX Rendering ---
  return (
    // Reuse container class if styles are shared
    <div className="verify-upload-container video-analytics-container">
      <h1>Upload & Verify Video</h1>

      {/* Upload Form */}
      <form onSubmit={handleSubmit} className="upload-form">
          <label htmlFor="verify-upload-input" className="file-label" title={file ? file.name : "Choose Video File"}>
              {file ? file.name : "Choose Video File"}
          </label>
          <input id="verify-upload-input" type="file" accept="video/*" onChange={handleFileChange} disabled={loading} style={{ display: 'none' }} />
          <button type="submit" disabled={loading || !file}>
              {loading ? `Analyzing...` : 'Analyze & Verify'}
          </button>
      </form>

      {/* Loading/Error Indicators */}
      {loading && <div className="progress-section"><p>Processing... Please Wait</p>{/* Consider adding spinner */}</div>}
      {error && <div className="error-message"><strong>Error:</strong> {error}</div>}

      {/* Video Preview */}
      {videoUrl && !analysisResult && !loading && <div className="uploaded-video card"><h3>Video Preview</h3><video src={videoUrl} controls width="720" /></div>}

      {/* --- Combined Results Display --- */}
      {analysisResult && (
        <div className="results-display">
          <h2>Analysis & Verification Results: {file?.name}</h2>

          {/* --- Row for Summary & Face Verification --- */}
          <div className="results-grid"> {/* Use grid or flexbox for layout */}

            {/* Summary Card */}
            {summaryMetrics && (
               <div className="card summary-card"><h3>Summary</h3>
                  <p>Overall Max Deepfake Score: <span className="highlight">{summaryMetrics.finalScore?.toFixed(3) ?? 'N/A'}</span></p>
                  <p className={`alert ${summaryMetrics.alert.includes('High') ? 'alert-high' : 'alert-low'}`}>{summaryMetrics.alert}</p>
                  <p>Average Frame Score: {summaryMetrics.avgScore?.toFixed(3) ?? 'N/A'}</p>
                  <p>Frames Analyzed: {summaryMetrics.frameCount ?? 'N/A'}</p>
               </div>
            )}

            {/* Face Verification Card */}
            {faceVerificationResult && (
                  <div className="card face-verification-card">
                      <h3>Face Verification</h3>
                      <p>Known Face(s) Detected: <span className={`highlight ${faceVerificationResult.verified ? 'verified-yes' : 'verified-no'}`}>
                          {faceVerificationResult.verified ? 'Yes' : 'No'}
                      </span></p>
                      {faceVerificationResult.verified && faceVerificationResult.recognized_names?.length > 0 && (
                          <p>Recognized Names: {faceVerificationResult.recognized_names.join(', ')}</p>
                      )}
                      {!faceVerificationResult.verified && (
                           <p>No known faces from the database were detected in the sampled frames.</p>
                      )}
                  </div>
              )}
          </div> {/* End results-grid */}


          {/* Transcription Card */}
          <div className="card">
              <h3>Transcription</h3>
              <p><strong>Detected Language:</strong> {analysisResult.detected_language || 'N/A'}</p>
              <p><strong>Original Text:</strong></p>
              <textarea readOnly value={analysisResult.original_transcription || '[No transcription]'} rows={5}></textarea>
              {analysisResult.english_transcription && analysisResult.detected_language !== 'en' && (
                  <> <p><strong>English Text:</strong></p> <textarea readOnly value={analysisResult.english_transcription} rows={5}></textarea> </>
              )}
              {analysisResult.original_transcription && (
                  <div className="translate-section">
                      <label htmlFor="language-select">Translate Original to:</label>
                      <select id="language-select" value={language} onChange={(e) => setLanguage(e.target.value)} disabled={loading}>
                           <option value="en">English</option>
                           <option value="hi">Hindi (हिन्दी)</option>
                           {/* Add other languages */}
                      </select>
                      <button onClick={handleTranslate} disabled={loading}> {loading ? 'Translating...' : 'Translate'} </button>
                      {translation && ( <div className="translated-text"><p><strong>Translated Text ({language}):</strong></p><textarea readOnly value={translation} rows={3}></textarea></div> )}
                  </div>
              )}
          </div>

          {/* Deepfake Chart Card */}
          <div className="card">
              <h3>Deepfake Score Over Time</h3>
              {lineChartData ? ( <div className="chart-container"> <Line data={lineChartData} options={chartOptions} /> </div> ) : <p>No frame data available for chart.</p>}
              {analysisResult?.frames_data?.timestamps?.length > 0 && ( <button onClick={handleDownload} className="download-button">Download Frame Data (CSV)</button> )}
          </div>

          {/* Suspicious Frames / Heatmaps Card */}
          {suspiciousFrames.length > 0 && (
            <div className="card suspicious-frames-card">
              <h3>Suspicious Frames (Score {'>'} 0.7)</h3>
              <p>Frames with high deepfake scores. Click button to see attention heatmap.</p>
              <ul className="suspicious-frames-list">
                  {suspiciousFrames.map(frame => ( <li key={frame.index} className="suspicious-frame-item-detailed">
                          <div className="frame-info">
                              <p>Frame: {frame.index}</p> <p>Time: {frame.timestamp?.toFixed(2)}s</p> <p>Score: <span className="highlight">{frame.score?.toFixed(3)}</span></p>
                              <button onClick={() => handleToggleHeatmap(frame.index)} className="toggle-heatmap-button"> {visibleHeatmapFrameIndex === frame.index ? 'Hide Heatmap' : 'Show Heatmap'} </button>
                          </div>
                          {visibleHeatmapFrameIndex === frame.index && ( <div className="heatmap-display-area"> <img src={frame.url} alt={`Heatmap Overlay Frame ${frame.index}`} loading="lazy" className="heatmap-image-large"/> </div> )}
                      </li> ))}
              </ul>
            </div>
          )}

          {/* Text Analysis Card */}
          {analysisResult.text_analysis && (
             <div className="card">
                <h3>Textual Content Analysis</h3>
                {/* Check if political_bias exists before accessing properties */}
                {analysisResult.text_analysis.political_bias ? (
                   <p> Political Bias: {analysisResult.text_analysis.political_bias.label || 'N/A'} (Score: <span className="highlight">{analysisResult.text_analysis.political_bias.score?.toFixed(2) || 'N/A'}</span>)</p>
                ) : <p>Political Bias: N/A</p>}
                <p> Manipulation Score: <span className="highlight">{analysisResult.text_analysis.manipulation_score?.toFixed(2) || 'N/A'}</span></p>
                <p> Emotional Triggers: {analysisResult.text_analysis.emotional_triggers?.join(', ') || 'None'}</p>
                <p> Stereotypes: {analysisResult.text_analysis.stereotypes?.join(', ') || 'None'} </p>
                {analysisResult.text_analysis.entities?.length > 0 && ( <div> <strong>Entities:</strong> <ul> {analysisResult.text_analysis.entities.map((entity, idx) => (<li key={idx}> {entity.text} ({entity.type}) </li> ))} </ul> </div> )}
             </div>
          )}

          {/* Fact Check Card */}
          {/* Ensure fact_check_result exists before trying to access its properties */}
          {analysisResult.text_analysis?.fact_check_result && (
            <div className="card fact-check-section">
              <div className="fact-check-header">
                <h3>Fact Check Details</h3>
                {isSpeaking && ttsSupported && (
                    <button onClick={handleStopSpeaking} className="stop-speak-button global-stop-button"> ⏹️ Stop Speaking </button>
                )}
                <button onClick={() => setIsFactCheckExpanded(!isFactCheckExpanded)} className="toggle-button">
                    {isFactCheckExpanded ? 'Hide Details' : 'Show Details'}
                </button>
              </div>

              {isFactCheckExpanded && (
                <div className="fact-check-content">
                    <h4>Processed Claims</h4>
                    {analysisResult.text_analysis.fact_check_result.processed_claims?.length > 0 ? (
                      analysisResult.text_analysis.fact_check_result.processed_claims.map((claim, idx) => {
                        const textForSpeech = `Verdict for claim ${idx + 1}: ${claim.final_label || 'Not Available'}. Explanation: ${claim.final_explanation || 'None provided.'}`;
                        return (
                            <div key={idx} className="claim-detail with-speak">
                                <div className="claim-text">
                                    <p><strong>Claim {idx + 1}:</strong> "{claim.original_claim || 'N/A'}"</p>
                                    <p>Verdict: <span className="highlight">{claim.final_label || 'N/A'}</span> (Conf: {claim.confidence !== undefined ? (claim.confidence * 100).toFixed(0) + '%' : 'N/A'})</p>
                                    <p>Explanation: {claim.final_explanation || 'N/A'}</p>
                                    <p><em>(Source: {claim.source || 'N/A'})</em></p>
                                </div>
                                {ttsSupported && (
                                    <button onClick={() => speakText(textForSpeech)} disabled={isSpeaking} className="speak-button" title={`Read verdict and explanation for claim ${idx + 1}`}>
                                        <span role="img" aria-label="Speak">🔊</span>
                                    </button>
                                )}
                            </div>
                        );
                       })
                    ) : (<p>No claims processed or found in this text.</p>)}

                    <h4>Processing Summary</h4>
                    <pre className="summary-box">{analysisResult.text_analysis.fact_check_result.summary || 'No summary provided.'}</pre>
                    {!ttsSupported && <p className="tts-warning"><em>(Text-to-Speech is not supported by your browser)</em></p>}
                </div>
              )}
            </div>
          )}

          {/* Map Card (Optional) */}
          {/* {analysisResult.text_analysis?.locations?.length > 0 && ( <div className="card"> ... </div> )} */}

        </div> // End results-display
      )}
    </div> // End container
  );
}

export default VerifyUpload;