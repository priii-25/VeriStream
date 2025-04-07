// frontend/src/components/RealTimeAnalysis.js
import React, { useEffect, useRef, useState, useCallback } from 'react';
import axios from 'axios';
import '../styles/RealTimeAnalysis.css'; // Ensure CSS is imported

// --- Define Constants ---
const HISTORY_LIMIT = 15; // Show the last 15 analyzed chunks in history
const BACKEND_URL = 'http://127.0.0.1:5001'; // Ensure this matches your backend

// --- Helper Component for Rendering Analysis Details ---
const AnalysisDetailsDisplay = React.memo(({ analysisData }) => {
    // --- HOOK CALLED AT TOP LEVEL ---
    const [isFcExpanded, setIsFcExpanded] = useState(false); // State for fact-check expansion

    // Now perform checks and early return if necessary
    if (!analysisData || analysisData.chunk_index === undefined) {
        return analysisData === undefined ? null : <p>Loading details...</p>;
    }

    // Destructure data *after* the check
    const {
        chunk_index, analysis_timestamp, deepfake_analysis,
        transcription, fact_check_results, fact_check_context_current
    } = analysisData;

    // Construct full heatmap URL if present
    const relativeHeatmapUrl = deepfake_analysis?.heatmap_url;
    const heatmapDisplayUrl = relativeHeatmapUrl?.startsWith('http')
        ? relativeHeatmapUrl
        : (relativeHeatmapUrl?.startsWith('/') ? `${BACKEND_URL}${relativeHeatmapUrl}` : null);

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
                        {transcription.detected_language && transcription.detected_language !== 'en' && transcription.english && (
                            <p><strong>English:</strong> {transcription.english}</p>
                        )}
                    </>
                ) : <p>N/A</p>}
            </div>

            {/* Fact Check */}
            {(fact_check_results?.length > 0 || fact_check_context_current) && (
                <div className="analysis-card-subsection fact-check-subsection">
                    <h5>
                        Fact Check
                        {fact_check_context_current && <span className="fresh-indicator">(Context Updated)</span>}
                    </h5>
                    {/* Only show button if there are results */}
                    {fact_check_results?.length > 0 && (
                        <button onClick={() => setIsFcExpanded(prev => !prev)} className="toggle-button">
                            {isFcExpanded ? 'Hide' : 'Show'} Details
                        </button>
                    )}
                    {/* Conditionally render details */}
                    {isFcExpanded && fact_check_results?.length > 0 && (
                        // --- THIS IS THE /* ... rendering logic ... */ PART ---
                        <div className="fact-check-details">
                            {fact_check_results.map((claim, idx) => (
                                claim.error ? // Display errors differently
                                <div key={idx} className="claim-detail error-detail">
                                    <p><strong>Fact Check Error:</strong> {claim.error}</p>
                                </div>
                                : // Display normal claim result
                                <div key={idx} className="claim-detail">
                                    <p><strong>Claim {idx + 1}:</strong> "{claim.original_claim || 'N/A'}"</p>
                                    <p>Verdict: <span className="highlight">{claim.final_verdict || 'N/A'}</span></p>
                                    {/* Optionally include explanation if needed/available */}
                                    {/* <p>Explanation: {claim.final_explanation || 'N/A'}</p> */}
                                </div>
                            ))}
                        </div>
                        // --- END OF /* ... rendering logic ... */ ---
                    )}
                    {/* Show message if context updated but no claims found */}
                    {fact_check_context_current && !fact_check_results?.length && !isFcExpanded && (
                        <p><em>(No claims found in last context check)</em></p>
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
    const [playbackQueue, setPlaybackQueue] = useState([]);
    const [analysisDataStore, setAnalysisDataStore] = useState({});
    const [currentlyPlayingUrl, setCurrentlyPlayingUrl] = useState(null);
    const [currentlyPlayingIndex, setCurrentlyPlayingIndex] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [isPlaying, setIsPlaying] = useState(false);
    const [errorState, setErrorState] = useState(null);
    const [expandedHistoryIndices, setExpandedHistoryIndices] = useState({});

    // --- Refs ---
    const videoRef = useRef(null);
    const wsRef = useRef(null);
    const reconnectTimeoutRef = useRef(null);
    const isPlayingRef = useRef(false); // Ref to track isPlaying state reliably

    // Keep ref in sync with state
    useEffect(() => {
        isPlayingRef.current = isPlaying;
    }, [isPlaying]);

    // --- Derived State ---
    const currentAnalysis = currentlyPlayingIndex !== null ? analysisDataStore[currentlyPlayingIndex] : undefined;
    const historicalIndices = Object.keys(analysisDataStore)
        .map(Number)
        .filter(idx => idx !== currentlyPlayingIndex)
        .sort((a, b) => b - a)
        .slice(0, HISTORY_LIMIT);

    // --- Play Next Chunk ---
    const playNextInQueue = useCallback(() => {
        console.log("Attempting playNextInQueue...");
        setPlaybackQueue(prevQueue => {
            if (prevQueue.length === 0) {
                console.log("PlayNext: Queue is empty, stopping.");
                setIsPlaying(false);
                return [];
            }
            const [nextUrl, ...remainingQueue] = prevQueue;
            let chunkIndex = -1;
            try {
                const match = nextUrl.match(/stream_chunk_(\d+)\.mp4/);
                if (match?.[1]) chunkIndex = parseInt(match[1], 10);
                else console.warn("Could not parse chunk index from URL:", nextUrl);
            } catch (e) { console.error("Error parsing chunk index:", e); }

            console.log(`PlayNext: Preparing chunk ${chunkIndex}. URL: ...${nextUrl.slice(-30)}`);
            setCurrentlyPlayingUrl(nextUrl); // Set URL for next render
            setCurrentlyPlayingIndex(chunkIndex); // Set Index for next render

            if (videoRef.current) {
                console.log(`PlayNext: Setting src and calling load/play for chunk ${chunkIndex}`);
                videoRef.current.src = nextUrl;
                videoRef.current.load();
                const playPromise = videoRef.current.play();
                if (playPromise !== undefined) {
                    playPromise.catch(error => { // Only catch errors here
                        console.error(`PlayNext: Playback failed for chunk ${chunkIndex}:`, error.name, error.message);
                        setIsPlaying(false);
                        if (error.name !== 'AbortError') {
                            setErrorState(`Video playback error: ${error.name}. Autoplay might be blocked.`);
                        }
                    });
                } else { console.warn(`Play() did not return a promise for chunk ${chunkIndex}`); }
            } else { console.error("PlayNext: videoRef.current is null!"); }

            return remainingQueue;
        });
    }, []); // No dependencies needed for setter form

    // --- WebSocket Message Handler ---
    const handleWebSocketMessage = useCallback((event) => {
        setErrorState(null);
        try {
            const data = JSON.parse(event.data);
            if (data?.chunk_index !== undefined && data?.video_chunk_url) {
                const chunkIndex = data.chunk_index;
                const relativeUrl = data.video_chunk_url;
                const fullUrl = relativeUrl.startsWith('/') ? `${BACKEND_URL}${relativeUrl}` : relativeUrl;
                const relativeHeatmapUrl = data.deepfake_analysis?.heatmap_url;
                const fullHeatmapUrl = relativeHeatmapUrl?.startsWith('http') ? relativeHeatmapUrl : (relativeHeatmapUrl?.startsWith('/') ? `${BACKEND_URL}${relativeHeatmapUrl}` : null);
                const processedData = { ...data, deepfake_analysis: { ...data.deepfake_analysis, heatmap_url: fullHeatmapUrl } };

                setAnalysisDataStore(prevStore => ({ ...prevStore, [chunkIndex]: processedData }));
                setPlaybackQueue(prevQueue => {
                    if (!prevQueue.includes(fullUrl)) {
                        const newQueue = [...prevQueue, fullUrl];
                        if (prevQueue.length === 0 && !isPlayingRef.current) {
                            console.log("WS: Queue was empty and not playing, triggering playNextInQueue.");
                            playNextInQueue();
                        }
                        return newQueue;
                    }
                    return prevQueue;
                });
                if (isLoading) { setIsLoading(false); } // Turn off initial load flag
            } else { console.warn("Received invalid WS data:", data); }
        } catch (e) { console.error("WS message processing error:", e); setErrorState("Error processing stream data."); }
    }, [playNextInQueue, isLoading]); // isLoading needed here

    // --- WebSocket Connection & Management ---
    const connectWebSocket = useCallback(() => {
        if (wsRef.current && wsRef.current.readyState < 2) return;
        const wsUrl = `ws://${BACKEND_URL.split('//')[1]}/api/stream/results`;
        wsRef.current = new WebSocket(wsUrl);
        console.log('Attempting WebSocket connection...');
        setErrorState(null);
        wsRef.current.onopen = () => { console.log('WebSocket connection established.'); if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current); reconnectTimeoutRef.current = null; };
        wsRef.current.onerror = (error) => { console.error('WebSocket error:', error); setErrorState('WebSocket connection error.'); };
        wsRef.current.onclose = (event) => {
            console.log(`WebSocket closed (Code: ${event.code}, Reason: ${event.reason})`);
            if (event.code !== 1000 && !reconnectTimeoutRef.current) { // Abnormal closure
                console.log('Attempting WebSocket reconnect in 5 seconds...');
                reconnectTimeoutRef.current = setTimeout(() => { reconnectTimeoutRef.current = null; connectWebSocket(); }, 5000);
            }
        };
        wsRef.current.onmessage = handleWebSocketMessage;
    }, [handleWebSocketMessage]);

    useEffect(() => { connectWebSocket(); return () => { /* Cleanup WS and Timer */ if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current); if (wsRef.current) wsRef.current.close(1000); }; }, [connectWebSocket]);


    // --- Video Event Handlers ---
    const handleVideoEnded = useCallback(() => {
        console.log(`Video ended: Chunk ${currentlyPlayingIndex}`);
        setIsPlaying(false);
        playNextInQueue();
    }, [currentlyPlayingIndex, playNextInQueue]);

    const handleVideoPlay = useCallback(() => {
        // Check if the source has actually loaded, avoid setting true if src is invalid
        if (videoRef.current && videoRef.current.currentSrc) {
             console.log(`Video play event: Chunk ${currentlyPlayingIndex}`);
             setIsPlaying(true);
             setErrorState(null); // Clear errors on successful play
        } else {
            console.warn(`Video play event fired but currentSrc is invalid for chunk ${currentlyPlayingIndex}.`);
            setIsPlaying(false); // Ensure false if src not loaded
        }
    }, [currentlyPlayingIndex]);

    const handleVideoPause = useCallback(() => {
        if (videoRef.current && !videoRef.current.ended) { // Check if pause wasn't due to ending
            console.log(`Video paused event: Chunk ${currentlyPlayingIndex}`);
            setIsPlaying(false);
        }
    }, [currentlyPlayingIndex]);

    const handleVideoError = useCallback((e) => {
        console.error(`Video Element Error: Chunk ${currentlyPlayingIndex}`, e, e.target.error);
        setIsPlaying(false);
        let errorMsg = 'Video playback failed.';
        if (e.target.error) {
            switch (e.target.error.code) {
                case MediaError.MEDIA_ERR_ABORTED: errorMsg = 'Video playback aborted by user or script.'; break;
                case MediaError.MEDIA_ERR_NETWORK: errorMsg = 'Network error caused video download to fail.'; break;
                case MediaError.MEDIA_ERR_DECODE: errorMsg = 'Video decoding error occurred.'; break;
                case MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED: errorMsg = 'Video source URL/format is not supported or invalid.'; break;
                default: errorMsg = `An unknown video error occurred (Code ${e.target.error.code}).`; break;
            }
        }
        setErrorState(errorMsg);
        // Consider automatically trying the next chunk after a short delay on error?
        // setTimeout(playNextInQueue, 1000); // Add delay before trying next
    }, [currentlyPlayingIndex]); // Removed playNextInQueue from deps to avoid loops


    // --- API Handlers (Start/Stop) ---
    const handleStart = async () => {
        if (!streamUrl) { setErrorState("Please enter a stream URL."); return; }
        if (isLoading) return;
        console.log("Requesting stream start...");
        setIsLoading(true); setErrorState(null); setPlaybackQueue([]);
        setAnalysisDataStore({}); setCurrentlyPlayingUrl(null); setCurrentlyPlayingIndex(null);
        setIsPlaying(false); setExpandedHistoryIndices({});
        if (videoRef.current) { videoRef.current.src = ''; videoRef.current.load(); }
        connectWebSocket(); // Ensure WS is connecting/connected
        try {
            await axios.post(`${BACKEND_URL}/api/stream/analyze`, { url: streamUrl });
            console.log("Start request sent.");
            // isLoading set false by WS handler
        } catch (error) {
            console.error("Error starting stream analysis:", error);
            setErrorState(error.response?.data?.detail || 'Failed to start analysis.');
            setIsLoading(false);
        }
    };

    const handleStop = async () => {
        if (isLoading && !isPlaying && !currentlyPlayingUrl) return; // Prevent stop during initial load
        console.log("Requesting stream stop...");
        setIsLoading(true); setErrorState(null); // Use isLoading to disable buttons during stop
        try {
            if (videoRef.current) videoRef.current.pause(); // Immediate pause
            setIsPlaying(false);

            await axios.post(`${BACKEND_URL}/api/stream/stop`);
            console.log("Stop request sent successfully.");

            if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current); reconnectTimeoutRef.current = null;
            if (wsRef.current) { wsRef.current.close(1000, "User stopped stream"); wsRef.current = null; }

            setPlaybackQueue([]); setAnalysisDataStore({}); setCurrentlyPlayingUrl(null);
            setCurrentlyPlayingIndex(null); setIsPlaying(false); setExpandedHistoryIndices({});
            if (videoRef.current) { videoRef.current.src = ''; videoRef.current.load(); }
        } catch (error) {
            console.error("Error stopping stream analysis:", error);
            setErrorState(error.response?.data?.detail || 'Failed to stop analysis cleanly.');
        } finally {
            setIsLoading(false);
        }
    };

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
                <input type="text" placeholder="Enter Stream URL (e.g., Twitch, YouTube)" value={streamUrl} onChange={(e) => setStreamUrl(e.target.value)} disabled={isLoading || isPlaying || !!currentlyPlayingUrl} />
                {(!isPlaying && !currentlyPlayingUrl && !isLoading) ? (
                    <button onClick={handleStart} disabled={!streamUrl} className="start-button"> Start Analysis </button>
                ) : (
                    <button onClick={handleStop} disabled={isLoading && !isPlaying && !currentlyPlayingUrl} className="stop-button">
                       {isLoading ? 'Processing...' : 'Stop Analysis'}
                    </button>
                )}
            </div>

            {/* Status/Error Display */}
            {errorState && <p className="status-message error">{errorState}</p>}
            {isLoading && !currentlyPlayingUrl && <p className="status-message info">Starting stream analysis, waiting for first chunk...</p>}

            {/* Main Content Area */}
            <div className="content-area">
                {/* Video Player Area */}
                <div className="stream-video card">
                    <h3>Stream Playback {currentlyPlayingIndex !== null ? `(Chunk ${currentlyPlayingIndex})` : ''}</h3>
                    <video ref={videoRef} controls width="100%" onEnded={handleVideoEnded} onError={handleVideoError} onPlay={handleVideoPlay} onPause={handleVideoPause} muted={false} playsInline />
                    {!currentlyPlayingUrl && !isLoading && <div className="video-placeholder">Video stream will appear here</div>}
                </div>

                {/* Current Analysis Display Area */}
                <div className="results-section card">
                    <h3>Current Analysis {currentlyPlayingIndex !== null ? `(Chunk ${currentlyPlayingIndex})` : ''}</h3>
                    {currentlyPlayingIndex !== null && currentAnalysis !== undefined ? (
                        <AnalysisDetailsDisplay analysisData={currentAnalysis} />
                    ) : (
                        isLoading ? <p>Starting analysis...</p> :
                        (isPlaying ? <p>Waiting for analysis of current chunk ({currentlyPlayingIndex})...</p> :
                         <p>Analysis results will appear here once the stream starts.</p>)
                    )}
                </div>
            </div>

            {/* Analysis History Section */}
            {historicalIndices.length > 0 && (
                <div className="history-section card">
                    <h3>Analysis History (Last {HISTORY_LIMIT} Chunks)</h3>
                    <div className="history-list">
                        {historicalIndices.map(index => {
                            const historyData = analysisDataStore[index];
                            const isExpanded = !!expandedHistoryIndices[index];
                            const collapsedInfo = `Chunk ${index} @ ${historyData?.analysis_timestamp ? new Date(historyData.analysis_timestamp * 1000).toLocaleTimeString() : '?'}` +
                                                `${historyData?.deepfake_analysis?.score !== undefined ? ` | Score: ${historyData.deepfake_analysis.score.toFixed(3)}` : ''}`;

                            return (
                                <div key={index} className={`history-item ${isExpanded ? 'expanded' : ''}`}>
                                    <div className="history-item-header" onClick={() => handleToggleHistoryExpand(index)}>
                                        <span>{collapsedInfo}</span>
                                        <button className="toggle-button"> {isExpanded ? 'Collapse' : 'Expand'} </button>
                                    </div>
                                    {isExpanded && historyData && (
                                        <div className="history-item-details">
                                            <AnalysisDetailsDisplay analysisData={historyData} />
                                        </div>
                                    )}
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}
            {/* End Analysis History Section */}
        </div> // End container
    );
};

export default RealTimeAnalysis;