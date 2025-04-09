// frontend/src/components/RealTimeAnalysis.js
import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import axios from 'axios';
import '../styles/RealTimeAnalysis.css'; // Ensure CSS is imported

// --- Define Constants ---
const HISTORY_LIMIT = 15; // Show the last 15 analyzed chunks in history
const BACKEND_URL = 'http://127.0.0.1:5001'; // Ensure this matches your backend

// --- Helper Component for Rendering Analysis Details ---
// (Assuming this component is correct from previous versions - No changes needed here)
const AnalysisDetailsDisplay = React.memo(({ analysisData }) => {
    const [isFcExpanded, setIsFcExpanded] = useState(false);

    if (!analysisData || analysisData.chunk_index === undefined) {
        return null;
    }

    const {
        chunk_index, analysis_timestamp, deepfake_analysis,
        transcription, fact_check_results, fact_check_summary,
        fact_check_context_current
    } = analysisData;

    const relativeHeatmapUrl = deepfake_analysis?.heatmap_url;
    const heatmapDisplayUrl = relativeHeatmapUrl?.startsWith('http')
        ? relativeHeatmapUrl
        : (relativeHeatmapUrl?.startsWith('/') ? `${BACKEND_URL}${relativeHeatmapUrl}` : null);

    const hasFactCheckData = fact_check_results?.length > 0 || fact_check_summary;

    return (
        <div className="analysis-details-content">
            {/* Deepfake */}
            <div className="analysis-card-subsection">
                <h5>Deepfake Score</h5>
                {deepfake_analysis && deepfake_analysis.timestamp >= 0 ? (
                    <>
                        <p>Score: <span className="highlight">{deepfake_analysis.score.toFixed(3)}</span></p>
                        {heatmapDisplayUrl && (
                            <div className="heatmap-container">
                                <p><strong>Attention Heatmap:</strong></p>
                                <img src={heatmapDisplayUrl} alt={`Heatmap for chunk ${chunk_index}`} className="heatmap-image" loading="lazy" />
                            </div>
                        )}
                    </>
                ) : <p>N/A</p>}
            </div>

            {/* Transcription */}
            <div className="analysis-card-subsection">
                <h5>Transcription</h5>
                {transcription ? (
                    <>
                        <p><strong>Lang:</strong> {transcription.detected_language || 'N/A'}</p>
                        <p><strong>Text:</strong> {transcription.original || '[No Text]'}</p>
                        {transcription.detected_language && !transcription.detected_language.startsWith('en') && transcription.english && (
                            <p><strong>English:</strong> {transcription.english}</p>
                        )}
                    </>
                ) : <p>N/A</p>}
            </div>

            {/* Fact Check */}
            {(hasFactCheckData || fact_check_context_current) && (
                 <div className="analysis-card-subsection fact-check-subsection">
                     <h5>
                         Fact Check
                         {fact_check_context_current && <span className="fresh-indicator">(Context Updated Here)</span>}
                     </h5>
                     {hasFactCheckData && (
                         <button onClick={() => setIsFcExpanded(prev => !prev)} className="toggle-button">
                             {isFcExpanded ? 'Hide' : 'Show'} Details
                         </button>
                     )}
                     {isFcExpanded && hasFactCheckData && (
                         <div className="fact-check-details">
                             {fact_check_results?.length > 0 ? (
                                 fact_check_results.map((claim, idx) => (
                                     claim.error ?
                                     <div key={idx} className="claim-detail error-detail">
                                         <p><strong>Fact Check Error:</strong> {claim.error}</p>
                                     </div>
                                     :
                                     <div key={idx} className="claim-detail">
                                         <p><strong>Claim {idx + 1}:</strong> "{claim.original_claim || 'N/A'}"</p>
                                         <p>Verdict: <span className="highlight">{claim.final_verdict || 'N/A'}</span></p>
                                     </div>
                                 ))
                             ) : (
                                 <p>No specific claims processed in the last fact check.</p>
                             )}
                              {fact_check_summary && (
                                  <div className="summary-box">
                                      <strong>Summary:</strong>
                                      <pre>{fact_check_summary}</pre>
                                  </div>
                              )}
                         </div>
                     )}
                     {fact_check_context_current && !hasFactCheckData && !isFcExpanded && (
                         <p><em>(No claims or summary found in last context check)</em></p>
                     )}
                 </div>
             )}
        </div>
    );
});


// --- Main RealTimeAnalysis Component (Simplified) ---
const RealTimeAnalysis = () => {
    // --- State ---
    const [streamUrl, setStreamUrl] = useState('');
    const [analysisDataStore, setAnalysisDataStore] = useState({}); // Stores { chunkIndex: analysisData }
    const [isLoading, setIsLoading] = useState(false); // For Start/Stop API calls
    const [isConnecting, setIsConnecting] = useState(false); // For initial WS connection/data wait
    const [errorState, setErrorState] = useState(null); // For displaying errors
    const [expandedHistoryIndices, setExpandedHistoryIndices] = useState({}); // For history UI
    const [isStreamActive, setIsStreamActive] = useState(false); // Track if analysis process is active

    // --- Refs ---
    const wsRef = useRef(null); // Ref to the WebSocket object
    const reconnectTimeoutRef = useRef(null); // Ref for WS reconnect timeout

    // --- Derived State using useMemo ---
    // Memoize the list of historical indices for display (No 'current' chunk to exclude)
    const historicalIndices = useMemo(() => {
         return Object.keys(analysisDataStore)
             .map(Number) // Convert keys to numbers
             .sort((a, b) => b - a) // Sort descending (newest first)
             .slice(0, HISTORY_LIMIT); // Limit history size
    }, [analysisDataStore]); // Only depends on the store now


    // --- WebSocket Message Handler ---
    // (Removed playbackQueue logic)
    const handleWebSocketMessage = useCallback((event) => {
        setErrorState(null); // Clear error on new data
        try {
            const data = JSON.parse(event.data);
            // We only care about chunk_index and the analysis data itself now
            if (data?.chunk_index !== undefined) {
                const chunkIndex = data.chunk_index;
                console.log(`WS Received: Analysis data for Chunk ${chunkIndex}.`);

                // Store analysis data - triggers useMemo update for historicalIndices
                setAnalysisDataStore(prevStore => ({ ...prevStore, [chunkIndex]: data }));

                // Mark connection as established if it was the first message
                if (isConnecting) { setIsConnecting(false); }

            } else { console.warn("Received invalid WS data structure:", data); }
        } catch (e) { console.error("WS message processing error:", e); setErrorState("Error processing stream data."); }
    }, [isConnecting]); // Dependency on isConnecting state


    // --- WebSocket Connection & Management ---
    // (No changes needed here, handles connection/reconnection/errors)
    const connectWebSocket = useCallback(() => {
        if (wsRef.current && wsRef.current.readyState < 2) {
            console.log("connectWebSocket: WS already open or connecting.");
            return;
        }
        const wsUrl = `ws://${BACKEND_URL.split('//')[1]}/api/stream/results`;
        wsRef.current = new WebSocket(wsUrl);
        console.log('Attempting WebSocket connection...');
        setErrorState(null);
        wsRef.current.onopen = () => {
            console.log('WebSocket connection established.');
            if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
        };
        wsRef.current.onerror = (error) => {
            console.error('WebSocket error:', error);
            setErrorState('WebSocket connection error.');
        };
        wsRef.current.onclose = (event) => {
            console.log(`WebSocket closed (Code: ${event.code}, Reason: ${event.reason || 'No reason given'})`);
            wsRef.current = null;
            if (event.code !== 1000 && !reconnectTimeoutRef.current && isStreamActive) { // Only reconnect if stream was meant to be active
                console.log('Attempting WebSocket reconnect in 5 seconds...');
                reconnectTimeoutRef.current = setTimeout(() => {
                    reconnectTimeoutRef.current = null;
                    connectWebSocket();
                }, 5000);
            }
        };
        wsRef.current.onmessage = handleWebSocketMessage;
    }, [handleWebSocketMessage, isStreamActive]); // Added isStreamActive dependency

    // --- Effect for component lifecycle cleanup ---
    // (No changes needed here)
     useEffect(() => {
         return () => {
             console.log("RealTimeAnalysis unmounting: Cleaning up WS and Timer.");
             if (reconnectTimeoutRef.current) {
                 clearTimeout(reconnectTimeoutRef.current);
                 reconnectTimeoutRef.current = null;
             }
             if (wsRef.current) {
                 wsRef.current.onclose = null;
                 wsRef.current.close(1000, "Component unmounting");
                 wsRef.current = null;
             }
         };
     }, []); // Empty dependency array


    // --- API Handlers (Start/Stop) ---
    // (Removed video player reset logic)
    const handleStart = async () => {
        if (!streamUrl) { setErrorState("Please enter a stream URL."); return; }
        if (isLoading || isConnecting || isStreamActive) return; // Prevent starting if already active/loading

        console.log("Requesting stream start...");
        setIsLoading(true); setIsConnecting(true); setErrorState(null);
        // Reset previous results
        setAnalysisDataStore({});
        setExpandedHistoryIndices({});
        setIsStreamActive(true); // Mark stream as active *before* API call

        connectWebSocket(); // Initiate WebSocket connection
        try {
            await axios.post(`${BACKEND_URL}/api/stream/analyze`, { url: streamUrl });
            console.log("Start request sent to backend.");
            // isConnecting will be set false by the first WS message
        } catch (error) {
            console.error("Error starting stream analysis:", error);
            setErrorState(error.response?.data?.detail || 'Failed to start analysis backend.');
            setIsConnecting(false); // Stop waiting state on start error
            setIsStreamActive(false); // Reset active state on start error
        } finally {
            setIsLoading(false); // API call finished
        }
    };

    const handleStop = async () => {
        // Can stop even if isLoading (e.g., if start API failed but state wasn't fully reset)
        // if (isLoading) { console.warn("Stop requested while another operation in progress."); }

        console.log("Requesting stream stop...");
        setIsLoading(true); setErrorState(null); setIsConnecting(false);
        setIsStreamActive(false); // Mark stream as inactive immediately

        try {
            // --- Frontend Cleanup First ---
            // No video player cleanup needed

            // --- Close WebSocket ---
            if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
            if (wsRef.current) {
                wsRef.current.onclose = null; // Prevent reconnect attempts
                wsRef.current.close(1000, "User stopped stream");
                wsRef.current = null;
            }

            // --- Send Stop Request to Backend ---
            // Add a timeout to the stop request
            await axios.post(`${BACKEND_URL}/api/stream/stop`, {}, { timeout: 10000 }); // 10 second timeout
            console.log("Stop request sent successfully to backend.");

        } catch (error) {
            // Differentiate between timeout and other errors
            if (error.code === 'ECONNABORTED') {
                 console.warn("Backend stop request timed out. Backend might still be running, but frontend is stopped.");
                 setErrorState('Backend stop confirmation timed out. Frontend disconnected.');
            } else {
                 console.error("Error stopping stream analysis:", error);
                 setErrorState(error.response?.data?.detail || 'Failed to confirm stop with backend cleanly.');
            }
        } finally {
            setIsLoading(false); // Reset loading state regardless of outcome
            // Ensure stream state is inactive
            setIsStreamActive(false);
            setIsConnecting(false);
        }
    };

    // --- Toggle History Item Expansion ---
    // (No changes needed here)
    const handleToggleHistoryExpand = (indexToToggle) => {
        setExpandedHistoryIndices(prev => ({ ...prev, [indexToToggle]: !prev[indexToToggle] }));
    };

    // --- JSX ---
    // (Removed the video player and current analysis sections)
    return (
        <div className="realtime-analysis-container">
            <h2>Real-Time Stream Analysis</h2>
            {/* Input Section */}
            <div className="stream-input-section">
                 <input
                    type="text"
                    placeholder="Enter Stream URL (e.g., Twitch, YouTube)"
                    value={streamUrl}
                    onChange={(e) => setStreamUrl(e.target.value)}
                    // Disable if loading, connecting, or stream is actively running
                    disabled={isLoading || isConnecting || isStreamActive}
                />
                 {/* Logic for Start/Stop Button */}
                 {!isStreamActive ? (
                     <button
                         onClick={handleStart}
                         // Disable if no URL OR if loading/connecting
                         disabled={!streamUrl || isLoading || isConnecting}
                         className="start-button"
                     >
                         {isLoading ? 'Processing...' : (isConnecting ? 'Connecting...' : 'Start Analysis')}
                     </button>
                 ) : (
                     <button
                        onClick={handleStop}
                        disabled={isLoading} // Only disable stop if currently processing the stop request itself
                        className="stop-button"
                    >
                         {isLoading ? 'Stopping...' : 'Stop Analysis'}
                     </button>
                 )}
            </div>

            {/* Status/Error Display */}
            {errorState && <p className="status-message error">{errorState}</p>}
            {isConnecting && <p className="status-message info">Connecting to backend, waiting for first analysis...</p>}
            {isStreamActive && !isConnecting && !errorState && historicalIndices.length === 0 && <p className="status-message info">Stream active, waiting for first analysis results...</p>}


            {/* REMOVED: Main Content Area (Video Player & Current Analysis) */}
            {/* <div className="content-area"> ... </div> */}


            {/* Analysis History Section */}
            {/* Always render the history container, show message if empty */}
            <div className="history-section card">
                <h3>Analysis History (Last {HISTORY_LIMIT} Chunks)</h3>
                {historicalIndices.length > 0 ? (
                    <div className="history-list">
                        {historicalIndices.map(index => {
                            const historyData = analysisDataStore[index];
                            const isExpanded = !!expandedHistoryIndices[index];
                            // Simplified collapsed info
                            let collapsedInfo = `Chunk ${index}`;
                            if (historyData?.analysis_timestamp) collapsedInfo += ` @ ${new Date(historyData.analysis_timestamp * 1000).toLocaleTimeString()}`;
                            if (historyData?.deepfake_analysis?.score !== undefined) collapsedInfo += ` | Score: ${historyData.deepfake_analysis.score.toFixed(3)}`;
                            if (historyData?.fact_check_context_current) collapsedInfo += ` [FC Update]`;

                            return (
                                <div key={index} className={`history-item ${isExpanded ? 'expanded' : ''}`}>
                                    <div className="history-item-header" onClick={() => handleToggleHistoryExpand(index)}>
                                        <span>{collapsedInfo}</span>
                                        <button className="toggle-button"> {isExpanded ? 'Collapse' : 'Expand'} </button>
                                    </div>
                                    {/* Render details only if expanded and data exists */}
                                    {isExpanded && historyData && (
                                        <div className="history-item-details">
                                            <AnalysisDetailsDisplay analysisData={historyData} />
                                        </div>
                                    )}
                                </div>
                            );
                        })}
                    </div>
                 ) : (
                    // Show placeholder message when history is empty
                    <p className="history-placeholder">
                        {isStreamActive ? 'Waiting for analysis results...' : 'Analysis results will appear here once a stream is started.'}
                    </p>
                 )}
            </div>
        </div> // End container
    );
};

export default RealTimeAnalysis;