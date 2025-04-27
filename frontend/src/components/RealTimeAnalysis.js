import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import axios from 'axios';
import '../styles/RealTimeAnalysis.css'; // Ensure CSS is imported

// --- Define Constants ---
const HISTORY_LIMIT = 15; // Show the last 15 analyzed chunks in history list
const BACKEND_URL = 'http://127.0.0.1:5001'; // Ensure this matches your backend

// --- Helper Component for Rendering Analysis Details in History ---
// (Assuming this component exists and is correct - displays analysis details)
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
                                         <p>Verdict: <span className="highlight">{claim.final_label || 'N/A'}</span></p>
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

// --- Main RealTimeAnalysis Component ---
const RealTimeAnalysis = () => {
    // --- State ---
    const [streamUrl, setStreamUrl] = useState('');
    const [analysisDataStore, setAnalysisDataStore] = useState({}); // Stores { chunkIndex: analysisData }
    const [isLoading, setIsLoading] = useState(false);
    const [isConnecting, setIsConnecting] = useState(false);
    const [errorState, setErrorState] = useState(null);
    const [expandedHistoryIndices, setExpandedHistoryIndices] = useState({});
    const [isStreamActive, setIsStreamActive] = useState(false);
    // New state for sequential playback
    const [playbackQueue, setPlaybackQueue] = useState([]); // Stores { index: number, url: string }[]
    const [currentPlayingIndex, setCurrentPlayingIndex] = useState(null); // chunkIndex of video in player

    // --- Refs ---
    const wsRef = useRef(null);
    const reconnectTimeoutRef = useRef(null);
    const videoPlayerRef = useRef(null); // Ref for the video player element

    // --- Derived State: Historical Indices for List ---
    const historicalIndices = useMemo(() => {
         return Object.keys(analysisDataStore)
             .map(Number)
             .sort((a, b) => b - a) // Sort descending (newest first)
             .slice(0, HISTORY_LIMIT);
    }, [analysisDataStore]);

    // --- WebSocket Message Handler ---
    const handleWebSocketMessage = useCallback((event) => {
        setErrorState(null); // Clear error on new data
        try {
            const data = JSON.parse(event.data);

            if (data?.chunk_index !== undefined) {
                const chunkIndex = data.chunk_index;
                console.log(`WS Received: Analysis data for Chunk ${chunkIndex}.`);

                // Store analysis data for history list
                setAnalysisDataStore(prevStore => ({ ...prevStore, [chunkIndex]: data }));

                // Add video chunk URL to playback queue if available
                if (data.video_chunk_url) {
                    const relativeUrl = data.video_chunk_url;
                    const fullUrl = `${BACKEND_URL}${relativeUrl}`;
                    const newItem = { index: chunkIndex, url: fullUrl };

                    setPlaybackQueue(prevQueue => {
                        // Prevent adding if already playing or queued
                        if (prevQueue.some(item => item.index === chunkIndex) || currentPlayingIndex === chunkIndex) {
                            return prevQueue;
                        }
                        // Insert in sorted order
                        const insertIndex = prevQueue.findIndex(item => item.index > chunkIndex);
                        const updatedQueue = [...prevQueue];
                        if (insertIndex === -1) { // Append if largest index so far
                            updatedQueue.push(newItem);
                        } else { // Insert at correct position
                            updatedQueue.splice(insertIndex, 0, newItem);
                        }
                         console.log(`Queue updated. Size: ${updatedQueue.length}. Added: ${chunkIndex}`);
                        return updatedQueue;
                    });
                }

                if (isConnecting) { setIsConnecting(false); } // Mark connection as established
            } else {
                console.warn("Received invalid WS data structure:", data);
            }
        } catch (e) {
            console.error("WS message processing error:", e);
            setErrorState("Error processing stream data.");
        }
    }, [isConnecting, currentPlayingIndex]); // Added currentPlayingIndex dependency

    // --- WebSocket Connection & Management ---
    const connectWebSocket = useCallback(() => {
        if (wsRef.current && wsRef.current.readyState < 2) return; // Already open or connecting
        const wsUrl = `ws://${BACKEND_URL.split('//')[1]}/api/stream/results`;
        wsRef.current = new WebSocket(wsUrl);
        console.log('Attempting WebSocket connection...');
        setErrorState(null);
        wsRef.current.onopen = () => {
            console.log('WebSocket connection established.');
            if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
        };
        wsRef.current.onerror = (error) => { console.error('WebSocket error:', error); setErrorState('WebSocket connection error.'); };
        wsRef.current.onclose = (event) => {
            console.log(`WebSocket closed (Code: ${event.code})`);
            wsRef.current = null;
            // Modify reconnect logic: Only try if stream *should* be active but WS closed unexpectedly
            if (isStreamActive && event.code !== 1000 && !reconnectTimeoutRef.current) {
                console.log('Attempting WebSocket reconnect in 5 seconds...');
                reconnectTimeoutRef.current = setTimeout(() => { reconnectTimeoutRef.current = null; connectWebSocket(); }, 5000);
            }
        };
        wsRef.current.onmessage = handleWebSocketMessage;
    }, [handleWebSocketMessage, isStreamActive]); // isStreamActive dependency is important for reconnect logic

    // --- Effect for component lifecycle cleanup ---
     useEffect(() => {
         return () => { // Cleanup function
             console.log("RealTimeAnalysis unmounting: Cleaning up WS and Timer.");
             if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
             if (wsRef.current) { wsRef.current.onclose = null; wsRef.current.close(1000, "Component unmounting"); }
         };
     }, []); // Empty dependency array: run only on mount/unmount

    // --- API Handlers (Start/Stop) ---
    const handleStart = async () => {
        if (!streamUrl || isLoading || isConnecting || isStreamActive) return;

        console.log("Requesting stream start...");
        setIsLoading(true); setIsConnecting(true); setErrorState(null);
        setAnalysisDataStore({}); // Reset results
        setExpandedHistoryIndices({});
        setPlaybackQueue([]); // Reset playback queue
        setCurrentPlayingIndex(null); // Reset playing index
        setIsStreamActive(true); // Mark stream as active

        connectWebSocket(); // Initiate WebSocket connection
        try {
            await axios.post(`${BACKEND_URL}/api/stream/analyze`, { url: streamUrl });
            console.log("Start request sent to backend.");
        } catch (error) {
            console.error("Error starting stream analysis:", error);
            setErrorState(error.response?.data?.detail || 'Failed to start analysis backend.');
            setIsConnecting(false);
            setIsStreamActive(false); // Reset state on start error
        } finally {
            setIsLoading(false);
        }
    };

    const handleStop = async () => {
        console.log("Requesting stream stop...");
        setIsLoading(true); setErrorState(null); setIsConnecting(false);
        setIsStreamActive(false); // Mark stream as inactive immediately

        try {
            // Frontend Cleanup
            setPlaybackQueue([]);
            setCurrentPlayingIndex(null);
            if (videoPlayerRef.current) {
                videoPlayerRef.current.pause();
                videoPlayerRef.current.src = ''; // Clear source
            }
            if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
            if (wsRef.current) {
                wsRef.current.onclose = null; // Prevent reconnect attempts
                wsRef.current.close(1000, "User stopped stream");
                wsRef.current = null;
            }

            // Backend Stop Request
            await axios.post(`${BACKEND_URL}/api/stream/stop`, {}, { timeout: 10000 });
            console.log("Stop request sent successfully to backend.");

        } catch (error) {
            if (error.code === 'ECONNABORTED') {
                 console.warn("Backend stop request timed out.");
                 setErrorState('Backend stop confirmation timed out. Frontend disconnected.');
            } else {
                 console.error("Error stopping stream analysis:", error);
                 setErrorState(error.response?.data?.detail || 'Failed to confirm stop with backend cleanly.');
            }
        } finally {
            setIsLoading(false);
            setIsStreamActive(false); // Ensure state is inactive
            setIsConnecting(false);
        }
    };

    // --- Video Player Event Handlers ---
    const handleVideoEnded = useCallback(() => {
        console.log(`Video for chunk ${currentPlayingIndex} ended.`);
        setPlaybackQueue(prevQueue => {
            const updatedQueue = prevQueue.filter(item => item.index !== currentPlayingIndex);
            if (updatedQueue.length > 0) {
                // Set the next chunk index to play (queue is already sorted)
                setCurrentPlayingIndex(updatedQueue[0].index);
                console.log(`Playing next chunk: ${updatedQueue[0].index}`);
            } else {
                setCurrentPlayingIndex(null); // No more chunks
                console.log("Playback queue empty.");
            }
            return updatedQueue;
        });
    }, [currentPlayingIndex]); // Dependency

    const handleVideoError = useCallback((e) => {
        console.error(`Error loading video for chunk ${currentPlayingIndex}:`, e.target.error);
        setErrorState(`Error loading video chunk ${currentPlayingIndex}. Skipping.`);
        // Skip to the next video in the queue
        setPlaybackQueue(prevQueue => {
            const updatedQueue = prevQueue.filter(item => item.index !== currentPlayingIndex);
            if (updatedQueue.length > 0) {
                setCurrentPlayingIndex(updatedQueue[0].index);
                console.log(`Skipping failed chunk ${currentPlayingIndex}, playing next: ${updatedQueue[0].index}`);
            } else {
                setCurrentPlayingIndex(null);
                console.log("Playback queue empty after error.");
            }
            return updatedQueue;
        });
    }, [currentPlayingIndex]); // Dependency

    // --- Effect to Trigger Initial Playback ---
    useEffect(() => {
        if (isStreamActive && currentPlayingIndex === null && playbackQueue.length > 0) {
            // Start playing the first item in the sorted queue
            const firstItemIndex = playbackQueue[0].index;
            console.log(`Initial playback trigger: Starting chunk ${firstItemIndex}`);
            setCurrentPlayingIndex(firstItemIndex);
        }
        // If stream becomes inactive, ensure we stop trying to play
        else if (!isStreamActive && currentPlayingIndex !== null) {
             setCurrentPlayingIndex(null);
        }
    }, [playbackQueue, currentPlayingIndex, isStreamActive]); // Monitor these states

    // --- Toggle History Item Expansion ---
    const handleToggleHistoryExpand = (indexToToggle) => {
        setExpandedHistoryIndices(prev => ({ ...prev, [indexToToggle]: !prev[indexToToggle] }));
    };

    // --- JSX ---
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
                    disabled={isLoading || isConnecting || isStreamActive} // Disable input when active
                />
                 {!isStreamActive ? (
                     <button onClick={handleStart} disabled={!streamUrl || isLoading || isConnecting} className="start-button">
                         {isLoading ? 'Processing...' : (isConnecting ? 'Connecting...' : 'Start Analysis')}
                     </button>
                 ) : (
                     <button onClick={handleStop} disabled={isLoading} className="stop-button">
                         {isLoading ? 'Stopping...' : 'Stop Analysis'}
                     </button>
                 )}
            </div>

            {/* Status/Error Display */}
            {errorState && <p className="status-message error">{errorState}</p>}
            {isConnecting && <p className="status-message info">Connecting to backend, waiting for first analysis...</p>}
            {isStreamActive && !isConnecting && !errorState && currentPlayingIndex === null && playbackQueue.length === 0 && <p className="status-message info">Stream active, waiting for analysis results and video chunks...</p>}

            {/* Sequential Video Player Section */}
            {isStreamActive && ( // Only show player when stream is active
                 <div className="sequential-player card">
                     <h3>Processed Stream Playback (Delayed)</h3>
                     <video
                         ref={videoPlayerRef}
                         key={currentPlayingIndex} // Force re-render on index change
                         src={playbackQueue.find(item => item.index === currentPlayingIndex)?.url || ''}
                         controls
                         autoPlay
                         muted // Mute often required for autoplay
                         preload="auto" // Preload upcoming chunk if possible
                         width="100%"
                         style={{ maxWidth: '720px', backgroundColor: '#111', marginBottom: '10px' }} // Style adjustments
                         onEnded={handleVideoEnded}
                         onError={handleVideoError}
                     >
                         Your browser does not support the video tag.
                     </video>
                     <div className="player-status">
                        <span>Playing Chunk: {currentPlayingIndex ?? 'Waiting...'}</span>
                        <span>Queued Chunks: {playbackQueue.length}</span>
                     </div>
                     <p className="delay-notice"><em>Note: This playback is delayed from the live stream due to analysis time.</em></p>
                 </div>
             )}

            {/* Analysis History Section (List View) */}
            <div className="history-section card">
                <h3>Analysis History (Last {HISTORY_LIMIT} Chunks)</h3>
                {historicalIndices.length > 0 ? (
                    <div className="history-list">
                        {historicalIndices.map(index => {
                            const historyData = analysisDataStore[index];
                            const isExpanded = !!expandedHistoryIndices[index];
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
                                    {isExpanded && historyData && (
                                        <div className="history-item-details">
                                            {/* Display analysis details using helper component */}
                                            <AnalysisDetailsDisplay analysisData={historyData} />
                                        </div>
                                    )}
                                </div>
                            );
                        })}
                    </div>
                 ) : (
                    <p className="history-placeholder">
                        {isStreamActive ? 'Waiting for analysis results...' : 'Analysis results will appear here once a stream is started.'}
                    </p>
                 )}
            </div>
        </div> // End container
    );
};

export default RealTimeAnalysis;