// frontend/src/components/VideoAnalytics.js
import React, { useState, useEffect, useRef, useCallback /* Added useCallback */ } from 'react'; // Import useCallback
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
import '../styles/VideoAnalytics.css';

// Chart.js Registration and Leaflet fix (Keep as is)
ChartJS.register( CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler );
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
});

const BACKEND_URL = 'http://127.0.0.1:5001';

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
  const [suspiciousFrames, setSuspiciousFrames] = useState([]);
  const [visibleHeatmapFrameIndex, setVisibleHeatmapFrameIndex] = useState(null);

  // --- NEW TTS State ---
  const [isSpeaking, setIsSpeaking] = useState(false); // Is TTS currently active?
  const [ttsSupported, setTtsSupported] = useState(true); // Assume supported initially
  const [voices, setVoices] = useState([]); // Optional: for voice selection
  const [selectedVoiceURI, setSelectedVoiceURI] = useState(null); // Optional: store selected voice

  // --- Refs ---
  const wsRef = useRef(null);
  // Ref to keep track of the current utterance to manage state correctly on cancel/end
  const currentUtteranceRef = useRef(null);

  // --- Check TTS Support and Load Voices ---
  useEffect(() => {
    const synth = window.speechSynthesis;
    if (!synth) {
      console.warn("Speech Synthesis not supported by this browser.");
      setTtsSupported(false);
      return; // Exit if not supported
    }

    const loadVoices = () => {
        try {
            const availableVoices = synth.getVoices();
            if (availableVoices.length > 0) {
                setVoices(availableVoices);
                console.log("Voices loaded:", availableVoices.length);
                // Optionally set a default voice preference
                // if (!selectedVoiceURI) {
                //     const defaultVoice = availableVoices.find(v => v.default && v.lang.startsWith('en')) || availableVoices.find(v => v.lang.startsWith('en'));
                //     if (defaultVoice) setSelectedVoiceURI(defaultVoice.voiceURI);
                // }
            } else {
                 // Voices might load async, wait for event
                 console.log("getVoices() initial call returned empty list, waiting for event...");
            }
        } catch (e) {
             console.error("Error getting voices:", e);
             setTtsSupported(false); // Mark as unsupported on error
        }
    };

    // Load voices initially
    loadVoices();

    // Listen for changes (essential for some browsers)
    if (synth.onvoiceschanged !== undefined) {
        synth.onvoiceschanged = loadVoices;
    }

    // Cleanup listener on unmount
    return () => {
        if (synth && synth.onvoiceschanged !== undefined) {
            synth.onvoiceschanged = null;
        }
        // Cancel any speech on unmount
        if (synth && synth.speaking) {
             synth.cancel();
        }
    };
  }, []); // Run only on mount

  // --- TTS Handlers ---
  const speakText = useCallback((textToSpeak) => {
    const synth = window.speechSynthesis;
    // Basic checks
    if (!ttsSupported || !synth || !textToSpeak || typeof textToSpeak !== 'string' || textToSpeak.trim() === '') {
        console.warn("TTS: Cannot speak - not supported, synth unavailable, or empty text provided.");
        return;
    }

    // --- Cancel existing speech ---
    // Store ref before cancelling, as cancel might clear synth.speaking prematurely
    const wasSpeaking = synth.speaking;
    if (wasSpeaking) {
        console.log("TTS: Cancelling previous utterance...");
        synth.cancel();
    }

    // Function to actually start speaking (used directly or after timeout)
    const startSpeaking = () => {
         // Double check synth isn't speaking after potential cancel/timeout
         if (synth.speaking) {
             console.warn("TTS: Synth still speaking after cancel attempt, aborting new speech.");
             setIsSpeaking(false); // Ensure state reflects reality
             return;
         }

        const utterance = new SpeechSynthesisUtterance(textToSpeak.trim());
        currentUtteranceRef.current = utterance; // Store ref to this utterance

        // --- Optional: Set Voice ---
        if (selectedVoiceURI) {
            const selectedVoice = voices.find(v => v.voiceURI === selectedVoiceURI);
            if (selectedVoice) {
                utterance.voice = selectedVoice;
                 console.log("TTS: Using voice:", selectedVoice.name);
            } else {
                 console.warn("TTS: Selected voice not found, using default.");
            }
        } else {
             console.log("TTS: Using default voice.");
        }

        // --- Set Rate/Pitch (Optional) ---
        // utterance.rate = 1; // 0.1 to 10
        // utterance.pitch = 1; // 0 to 2

        // --- Event Listeners for State Management ---
        utterance.onstart = () => {
            // Only set state if this is still the active utterance
            if (currentUtteranceRef.current === utterance) {
                console.log("TTS: Speech started for:", textToSpeak.substring(0, 30) + "...");
                setIsSpeaking(true);
            } else {
                 console.log("TTS: onstart fired for stale utterance, ignoring state update.");
            }
        };

        utterance.onend = () => {
             // Only set state if this utterance was the one that finished
            if (currentUtteranceRef.current === utterance) {
                console.log("TTS: Speech finished naturally.");
                setIsSpeaking(false);
                currentUtteranceRef.current = null; // Clear ref
            } else {
                 console.log("TTS: onend fired for stale utterance, ignoring state update.");
            }
        };

        utterance.onerror = (event) => {
            console.error("TTS: Speech Synthesis Error:", event.error);
            setError(`Speech error: ${event.error}`);
             // Only update state if this utterance caused the error
            if (currentUtteranceRef.current === utterance) {
                setIsSpeaking(false);
                currentUtteranceRef.current = null; // Clear ref
            }
        };

        // --- Speak ---
        try {
            synth.speak(utterance);
        } catch(e) {
             console.error("TTS: Error calling synth.speak:", e);
             setError("Failed to initiate speech.");
             setIsSpeaking(false); // Reset state on error
             currentUtteranceRef.current = null;
        }
    };

    // Use a timeout if we just cancelled, to ensure the cancel command processes
    if (wasSpeaking) {
        setTimeout(startSpeaking, 150); // Small delay
    } else {
        startSpeaking(); // Start immediately if nothing was playing
    }

  }, [ttsSupported, voices, selectedVoiceURI, setError]); // Dependencies

  const handleStopSpeaking = useCallback(() => {
    const synth = window.speechSynthesis;
    if (ttsSupported && synth && synth.speaking) {
        synth.cancel();
        // Cancel might not trigger onend immediately/reliably, so force state update
        setIsSpeaking(false);
        currentUtteranceRef.current = null; // Clear ref as well
        console.log("TTS: Speech cancelled by user.");
    }
  }, [ttsSupported]);

  // --- Handlers (handleFileChange, handleSubmit, handleTranslate, handleDownload, handleToggleHeatmap - keep as is) ---
  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);
    setAnalysisResult(null); setError(null); setProgress(0);
    setTranslation(null); setIsFactCheckExpanded(false);
    setSuspiciousFrames([]); setVisibleHeatmapFrameIndex(null);
    handleStopSpeaking(); // Stop any speaking when file changes
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoUrl(selectedFile ? URL.createObjectURL(selectedFile) : null);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file || loading) return;

    // Reset state including TTS
    setLoading(true); setError(null); setProgress(0);
    setAnalysisResult(null); setTranslation(null); setIsFactCheckExpanded(false);
    setSuspiciousFrames([]); setVisibleHeatmapFrameIndex(null);
    handleStopSpeaking(); // Stop any speaking

    const formData = new FormData();
    formData.append('file', file);

    try {
      const apiUrl = `${BACKEND_URL}/api/video/analyze`;
      const response = await axios.post(apiUrl, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 480000,
      });
      console.log("Analysis Response Received:", response.data);
      setAnalysisResult(response.data);
      setProgress(1);

      if (response.data?.frames_data?.overlay_urls) {
          const overlays = response.data.frames_data.overlay_urls;
          const frames = response.data.frames_data;
          const foundSuspicious = Object.keys(overlays).map(frameIndexStr => {
                const frameIndex = parseInt(frameIndexStr, 10);
                const dataIndex = frames.frame_indices?.findIndex(idx => idx === frameIndex);
                if (dataIndex !== -1 && frames.timestamps?.[dataIndex] !== undefined && frames.scores?.[dataIndex] !== undefined) {
                    const relativeUrl = overlays[frameIndexStr];
                    return {
                        index: frameIndex,
                        timestamp: frames.timestamps[dataIndex],
                        score: frames.scores[dataIndex],
                        url: relativeUrl.startsWith('/') ? `${BACKEND_URL}${relativeUrl}` : relativeUrl
                    };
                }
                console.warn(`Could not find matching data for overlay frame index: ${frameIndex}`);
                return null; // Return null for invalid entries
          }).filter(item => item !== null); // Filter out nulls

          foundSuspicious.sort((a, b) => b.score - a.score);
          setSuspiciousFrames(foundSuspicious);
          console.log("Processed suspicious frames with overlays:", foundSuspicious);
      } else {
          console.log("No overlay URLs found.");
      }

    } catch (err) {
        console.error("Analysis Submission Error:", err);
        let errMsg = 'An error occurred during analysis.';
        if (err.code === 'ECONNABORTED') {
            errMsg = `Analysis timed out after ${err.config.timeout / 1000} seconds.`;
        } else {
            errMsg = err.response?.data?.detail || err.message || errMsg;
        }
        setError(errMsg);
        setAnalysisResult(null);
        setProgress(0);
    } finally {
        setLoading(false);
    }
  };

  const handleTranslate = async () => {
      // (Keep existing logic)
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
    // (Keep existing logic)
    const frameData = analysisResult?.frames_data;
    if (!frameData?.timestamps || !frameData?.scores || !frameData?.frame_indices) {
        alert("Frame data incomplete."); return;
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
        const safeFilename = file?.name.replace(/[^a-z0-9._-]/gi, '_').toLowerCase() || 'video_analysis';
        link.download = `${safeFilename}_frames.csv`;
        link.click();
        URL.revokeObjectURL(link.href);
    } catch (e) { setError("Failed to generate CSV."); }
  };

  const handleToggleHeatmap = (frameIndex) => {
      setVisibleHeatmapFrameIndex(prev => prev === frameIndex ? null : frameIndex);
  };

   // WebSocket useEffect (Keep as is)
   useEffect(() => {
    const wsUrl = `ws://${BACKEND_URL.split('//')[1]}/api/video/progress`;
    wsRef.current = new WebSocket(wsUrl);
    wsRef.current.onopen = () => console.log('Progress WS connected');
    wsRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data && typeof data.progress === 'number') {
           setProgress(Math.min(1, Math.max(0, data.progress)));
           if (data.progress < 0) {
              setError("Analysis failed during processing.");
              setLoading(false);
              handleStopSpeaking(); // Stop speech on backend error
           }
        }
      } catch (e) { console.error("Failed to parse progress:", e); }
    };
    wsRef.current.onerror = (error) => console.error('Progress WS error:', error);
    wsRef.current.onclose = (event) => console.log(`Progress WS closed (Code: ${event.code})`);
    return () => { if (wsRef.current) wsRef.current.close(1000); };
  }, []);


  // --- Derived Data for Display (Chart data, metrics, locations - keep as is) ---
  const calculateSummaryMetrics = () => {
    // (Keep existing logic)
    const scores = analysisResult?.frames_data?.scores;
    const finalScore = analysisResult?.final_score;
    if (!scores?.length || finalScore === undefined || finalScore === null) return null;
    const validScores = scores.filter(s => typeof s === 'number');
    if (!validScores.length) return { avgScore: 0, peakScore: 0, frameCount: scores.length, alert: 'No valid scores', finalScore };
    const avgScore = validScores.reduce((a, b) => a + b, 0) / validScores.length;
    return { avgScore, peakScore: finalScore, frameCount: scores.length, alert: finalScore > 0.7 ? 'High Deepfake Probability Detected!' : 'Low Deepfake Probability', finalScore };
  };
  const summaryMetrics = calculateSummaryMetrics();

  const lineChartData = analysisResult?.frames_data?.timestamps?.length > 0 && analysisResult?.frames_data?.scores?.length > 0
    ? { labels: analysisResult.frames_data.timestamps.map((t) => t?.toFixed(2) ?? '?'),
        datasets: [{ label: 'Deepfake Score', data: analysisResult.frames_data.scores, borderColor: 'rgb(0, 123, 255)', backgroundColor: 'rgba(0, 123, 255, 0.1)', fill: true, tension: 0.1, pointRadius: 1, pointHoverRadius: 5, borderWidth: 1.5 }]
      } : null;

  const chartOptions = { /* (Keep existing options) */
        responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
        plugins: { legend: { position: 'top', labels: { color: '#e0e0e0' } }, title: { display: true, text: 'Deepfake Score Over Time', color: '#ffffff' }, tooltip: { bodyColor: '#e0e0e0', titleColor: '#ffffff', backgroundColor: 'rgba(0, 0, 0, 0.8)', callbacks: { label: function(context) { let label = context.dataset.label || ''; if (label) { label += ': '; } if (context.parsed.y !== null) { label += context.parsed.y.toFixed(3); } const frameIndex = analysisResult?.frames_data?.frame_indices?.[context.dataIndex]; if (frameIndex !== undefined) { label += ` (Frame ${frameIndex})`; } return label; } } } },
        scales: { x: { title: { display: true, text: 'Time (s)', color: '#ccc' }, ticks: { color: '#ccc', maxRotation: 0, autoSkip: true, maxTicksLimit: 15 } }, y: { title: { display: true, text: 'Score', color: '#ccc' }, ticks: { color: '#ccc' }, beginAtZero: true, max: 1, grid: { color: 'rgba(255, 255, 255, 0.1)' } } },
        animation: { duration: 300 }
    };

  const locations = analysisResult?.text_analysis?.locations?.map((loc, index) => ({ name: loc.text, latitude: 51.505 + (Math.random() - 0.5) * 0.2, longitude: -0.09 + (Math.random() - 0.5) * 0.2 })) || [];


  // --- JSX Rendering ---
  return (
    <div className="video-analytics-container">
      <h1>Video File Analysis</h1>

      {/* Upload Form (Keep as is) */}
      <form onSubmit={handleSubmit} className="upload-form">
          <label htmlFor="video-upload-input" className="file-label" title={file ? file.name : "Choose Video File"}>
              {file ? file.name : "Choose Video File"}
          </label>
          <input id="video-upload-input" type="file" accept="video/*" onChange={handleFileChange} disabled={loading} style={{ display: 'none' }} />
          <button type="submit" disabled={loading || !file}>
            {loading ? `Analyzing... (${(progress * 100).toFixed(0)}%)` : 'Analyze Video'}
          </button>
      </form>

      {/* Loading/Progress/Error (Keep as is) */}
      {loading && <div className="progress-section"><progress value={progress} max="1" /><span>{(progress * 100).toFixed(0)}%</span></div>}
      {error && <div className="error-message"><strong>Error:</strong> {error}</div>}

      {/* Video Preview (Keep as is) */}
      {videoUrl && !analysisResult && !loading && <div className="uploaded-video card"><h3>Video Preview</h3><video src={videoUrl} controls width="720" /></div>}

      {/* Results Display Area */}
      {analysisResult && (
        <div className="results-display">
          <h2>Analysis Results: {file?.name}</h2>

          {/* Summary Card (Keep as is) */}
          {summaryMetrics && (
             <div className="card summary-card"><h3>Summary</h3>
                <p>Overall Max Deepfake Score: <span className="highlight">{summaryMetrics.finalScore?.toFixed(3) ?? 'N/A'}</span></p>
                <p className={`alert ${summaryMetrics.alert.includes('High') ? 'alert-high' : 'alert-low'}`}>{summaryMetrics.alert}</p>
                <p>Average Frame Score: {summaryMetrics.avgScore?.toFixed(3) ?? 'N/A'}</p>
                <p>Frames Analyzed: {summaryMetrics.frameCount ?? 'N/A'}</p>
             </div>
          )}

          {/* Transcription Card (Keep as is) */}
           <div className="card">
               {/* ... (transcription and translation logic remains the same) ... */}
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
                            {/* ... other language options ... */}
                        </select>
                        <button onClick={handleTranslate} disabled={loading}> {loading ? 'Translating...' : 'Translate'} </button>
                        {translation && ( <div className="translated-text"><p><strong>Translated Text ({language}):</strong></p><textarea readOnly value={translation} rows={3}></textarea></div> )}
                    </div>
                )}
           </div>

          {/* Deepfake Chart Card (Keep as is) */}
            <div className="card">
                <h3>Deepfake Score Over Time</h3>
                {lineChartData ? ( <div className="chart-container"> <Line data={lineChartData} options={chartOptions} /> </div> ) : <p>No frame data available.</p>}
                {analysisResult?.frames_data?.timestamps?.length > 0 && ( <button onClick={handleDownload} className="download-button">Download Frame Data (CSV)</button> )}
            </div>

          {/* Suspicious Frames / Heatmaps Card (Keep as is) */}
           {suspiciousFrames.length > 0 && (
            <div className="card suspicious-frames-card">
              <h3>Suspicious Frames (Score {'>'} 0.7)</h3>
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

          {/* Text Analysis Card (Keep as is) */}
          {analysisResult.text_analysis && ( /* ... existing card content ... */
             <div className="card">
                <h3>Textual Content Analysis</h3>
                <p> Political Bias: {analysisResult.text_analysis.political_bias?.label || 'N/A'} (Score: <span className="highlight">{analysisResult.text_analysis.political_bias?.score?.toFixed(2) || 'N/A'}</span>)</p>
                <p> Manipulation Score: <span className="highlight">{analysisResult.text_analysis.manipulation_score?.toFixed(2) || 'N/A'}</span></p>
                <p> Emotional Triggers: {analysisResult.text_analysis.emotional_triggers?.join(', ') || 'None'}</p>
                <p> Stereotypes: {analysisResult.text_analysis.stereotypes?.join(', ') || 'None'} </p>
                {!!analysisResult.text_analysis.entities?.length && ( <div> <strong>Entities:</strong> <ul> {analysisResult.text_analysis.entities.map((entity, idx) => (<li key={idx}> {entity.text} ({entity.type}) </li> ))} </ul> </div> )}
             </div>
          )}

          {/* Fact Check Card - MODIFIED to include Speak/Stop buttons */}
          {analysisResult.text_analysis?.fact_check_result && (
            <div className="card fact-check-section">
              <div className="fact-check-header">
                <h3>Fact Check Details</h3>
                {/* Global Stop Button for this section */}
                {isSpeaking && ttsSupported && (
                    <button onClick={handleStopSpeaking} className="stop-speak-button global-stop-button">
                         ⏹️ Stop Speaking
                    </button>
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
                        // Prepare text for this specific claim
                        const textForSpeech = `Verdict for claim ${idx + 1}: ${claim.final_label || 'Not Available'}. Explanation: ${claim.final_explanation || 'None provided.'}`;
                        return (
                            <div key={idx} className="claim-detail with-speak"> {/* Add class for styling */}
                                <div className="claim-text">
                                    <p><strong>Claim {idx + 1}:</strong> "{claim.original_claim || 'N/A'}"</p>
                                    <p>Verdict: <span className="highlight">{claim.final_label || 'N/A'}</span> (Conf: {(claim.confidence * 100).toFixed(0)}%)</p>
                                    <p>Explanation: {claim.final_explanation || 'N/A'}</p>
                                    <p><em>(Source: {claim.source || 'N/A'})</em></p>
                                </div>
                                {/* Speak Button per Claim */}
                                {ttsSupported && (
                                    <button
                                        onClick={() => speakText(textForSpeech)}
                                        disabled={isSpeaking} // Disable button if *any* speech is happening
                                        className="speak-button"
                                        title={`Read verdict and explanation for claim ${idx + 1}`}
                                    >
                                        <span role="img" aria-label="Speak">🔊</span> {/* Icon */}
                                    </button>
                                )}
                            </div>
                        );
                       }) // End map
                    ) : (<p>No claims processed or found in this text.</p>)}

                    <h4>Processing Summary</h4>
                    <pre className="summary-box">{analysisResult.text_analysis.fact_check_result.summary || 'No summary provided.'}</pre>
                    {!ttsSupported && <p className="tts-warning"><em>(Text-to-Speech is not supported by your browser)</em></p>}
                </div>
              )}
            </div>
          )} {/* End Fact Check Card */}

          {/* Map Card (Keep as is) */}
          {locations.length > 0 && ( /* ... existing card content ... */
            <div className="card">
                <h3>Mentioned Locations (Map - Placeholder Coords)</h3>
                <MapContainer center={[20, 0]} zoom={2} scrollWheelZoom={false} className="map-container">
                    <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" attribution='© OpenStreetMap contributors' />
                    {locations.map((loc, idx) => ( <Marker key={idx} position={[loc.latitude, loc.longitude]}><Popup>{loc.name}</Popup></Marker> ))}
                </MapContainer>
                <p><em>Note: Map locations use placeholder coordinates.</em></p>
            </div>
           )}

        </div> // End results-display
      )}
    </div> // End video-analytics-container
  );
};

export default VideoAnalytics;