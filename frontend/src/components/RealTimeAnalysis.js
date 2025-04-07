// frontend/src/components/RealTimeAnalysis.js
import React, { useEffect, useRef, useState, useCallback } from 'react';
import axios from 'axios';
import '../styles/RealTimeAnalysis.css'; // Ensure this CSS file exists

const RealTimeAnalysis = () => {
    // --- State ---
    const [streamUrl, setStreamUrl] = useState('');
    const [playbackQueue, setPlaybackQueue] = useState([]); // URLs only
    const [analysisDataStore, setAnalysisDataStore] = useState({}); // { chunkIndex: data }
    const [currentlyPlayingUrl, setCurrentlyPlayingUrl] = useState(null);
    const [currentlyPlayingIndex, setCurrentlyPlayingIndex] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [isPlaying, setIsPlaying] = useState(false);
    const [isFactCheckExpanded, setIsFactCheckExpanded] = useState(false);
    const [errorState, setErrorState] = useState(null);

    // --- Refs ---
    const videoRef = useRef(null);
    const wsRef = useRef(null);
    const reconnectTimeoutRef = useRef(null);
    const processingQueueRef = useRef(false);

    // --- Derived State: Get current analysis data ---
    const currentAnalysis = currentlyPlayingIndex !== null ? analysisDataStore[currentlyPlayingIndex] : null;

    // --- Play Next Chunk ---
    const playNextInQueue = useCallback(() => {
        // Simplified version - actual implementation likely needs more robustness
        setPlaybackQueue(prevQueue => {
            if (prevQueue.length === 0) {
                setIsPlaying(false);
                return [];
            }
            const [nextUrl, ...remainingQueue] = prevQueue;
            let chunkIndex = -1;
            try {
                const match = nextUrl.match(/stream_chunk_(\d+)\.mp4/);
                if (match && match[1]) chunkIndex = parseInt(match[1], 10);
            } catch { /* ignore */ }

            setCurrentlyPlayingUrl(nextUrl);
            setCurrentlyPlayingIndex(chunkIndex);
            setIsFactCheckExpanded(false); // Collapse details on new chunk

            // Defer video operations slightly
            setTimeout(() => {
                if (videoRef.current) {
                    videoRef.current.load();
                    videoRef.current.play().then(() => setIsPlaying(true)).catch(err => {
                        console.warn(`Autoplay failed for chunk ${chunkIndex}:`, err.name);
                        setIsPlaying(false);
                    });
                }
            }, 50); // Short delay

            return remainingQueue;
        });
    }, []); // No dependencies needed if it only modifies state based on prev state

    // --- WebSocket Message Handler ---
    const handleWebSocketMessage = useCallback((event) => {
        setErrorState(null);
        try {
            const data = JSON.parse(event.data);
            // console.log("WS Message:", data); // DEBUG

            if (data && typeof data.chunk_index === 'number' && typeof data.video_chunk_url === 'string') {
                const chunkIndex = data.chunk_index;
                const relativeUrl = data.video_chunk_url;
                // *** Ensure this matches your backend host/port ***
                const backendHost = 'http://127.0.0.1:5001';
                const fullUrl = relativeUrl.startsWith('/') ? `${backendHost}${relativeUrl}` : relativeUrl;
                // *** Also construct full heatmap URL if present ***
                const relativeHeatmapUrl = data.deepfake_analysis?.heatmap_url;
                const fullHeatmapUrl = relativeHeatmapUrl?.startsWith('/') ? `${backendHost}${relativeHeatmapUrl}` : relativeHeatmapUrl;

                // Store analysis data (including the potentially updated heatmap URL)
                setAnalysisDataStore(prevStore => ({
                    ...prevStore,
                    [chunkIndex]: { // Store the *full* data object received
                       ...data,
                       // Overwrite heatmap_url with the fully qualified one if it exists
                       deepfake_analysis: {
                          ...data.deepfake_analysis,
                          heatmap_url: fullHeatmapUrl // Store the full URL or undefined
                       }
                    }
                }));

                // Add video URL to queue if new
                setPlaybackQueue(prevQueue => {
                    if (!prevQueue.includes(fullUrl)) {
                        console.log(`Enqueuing video chunk ${chunkIndex}`);
                        const newQueue = [...prevQueue, fullUrl];
                        // Trigger play if queue WAS empty and video isn't already playing
                        if (prevQueue.length === 0 && !isPlaying) {
                            playNextInQueue();
                        }
                        return newQueue;
                    }
                    return prevQueue;
                });
                setIsLoading(false); // Stop loading indicator once first chunk arrives
            }
        } catch (e) {
            console.error("WS message processing error:", e);
            setErrorState("Error processing stream data.");
        }
    }, [playNextInQueue, isPlaying]); // Dependencies

    // --- WebSocket Connection ---
    const connectWebSocket = useCallback(() => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            console.log('WebSocket already open.');
            return;
        }
        // *** Ensure this matches backend WebSocket endpoint ***
        wsRef.current = new WebSocket('ws://127.0.0.1:5001/api/stream/results');
        console.log('Attempting WebSocket connection...');
        setErrorState(null); // Clear previous errors

        wsRef.current.onopen = () => {
            console.log('WebSocket connection established.');
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current);
                reconnectTimeoutRef.current = null;
            }
        };
        wsRef.current.onerror = (error) => {
            console.error('WebSocket error:', error);
            setErrorState('WebSocket connection error. Check console.');
            // Don't attempt reconnect immediately on error, wait for close
        };
        wsRef.current.onclose = (event) => {
            console.log(`WebSocket closed (Code: ${event.code}, Reason: ${event.reason})`);
            if (!event.wasClean && !reconnectTimeoutRef.current) { // Attempt reconnect only if closed unexpectedly
                console.log('Attempting WebSocket reconnect in 5 seconds...');
                reconnectTimeoutRef.current = setTimeout(connectWebSocket, 5000);
            }
        };
        wsRef.current.onmessage = handleWebSocketMessage;
    }, [handleWebSocketMessage]); // connectWebSocket depends on handleWebSocketMessage

    // Effect for initial connection and cleanup
    useEffect(() => {
        connectWebSocket();
        return () => {
            if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
            if (wsRef.current) {
                wsRef.current.close(1000, "Component unmounting");
                wsRef.current = null;
            }
        };
    }, [connectWebSocket]); // Run on mount and if connectWebSocket changes

    // --- Video Event Handler ---
    const handleVideoEnded = useCallback(() => {
        console.log(`Video ended: ${currentlyPlayingUrl ? currentlyPlayingUrl.slice(-30) : 'N/A'}`);
        setIsPlaying(false);
        playNextInQueue(); // Automatically play next
    }, [currentlyPlayingUrl, playNextInQueue]);

    // --- API Handlers (Start/Stop) ---
    const handleStart = async () => {
        if (!streamUrl) { setErrorState("Please enter a stream URL."); return; }
        if (isLoading) return; // Prevent double clicks

        console.log("Requesting stream start...");
        setIsLoading(true);
        setErrorState(null);
        setPlaybackQueue([]);
        setAnalysisDataStore({});
        setCurrentlyPlayingUrl(null);
        setCurrentlyPlayingIndex(null);
        setIsPlaying(false);
        setIsFactCheckExpanded(false);
        if (videoRef.current) { videoRef.current.src = ''; videoRef.current.load(); }

        // Ensure WebSocket is ready or connecting
        if (!wsRef.current || wsRef.current.readyState > 1) { // CLOSING or CLOSED
            connectWebSocket();
            // Optionally wait a very short time for connection attempt? Risky.
            // Rely on WS message handler to turn off isLoading.
        }

        try {
             // *** Ensure this matches backend API endpoint ***
            await axios.post('http://127.0.0.1:5001/api/stream/analyze', { url: streamUrl });
            console.log("Start request sent successfully.");
            // Don't set isLoading false here, wait for first WS message
        } catch (error) {
            console.error("Error starting stream analysis:", error);
            setErrorState(error.response?.data?.detail || 'Failed to start analysis. Check backend.');
            setIsLoading(false); // Stop loading on error
        }
    };

    const handleStop = async () => {
        if (isLoading) return; // Prevent double clicks

        console.log("Requesting stream stop...");
        setIsLoading(true); // Indicate stopping process
        setErrorState(null);

        try {
            // *** Ensure this matches backend API endpoint ***
            await axios.post('http://127.0.0.1:5001/api/stream/stop');
            console.log("Stop request sent successfully.");

            // Close WebSocket cleanly from client-side after stop confirmed
            if (wsRef.current) {
                wsRef.current.close(1000, "User stopped stream");
                wsRef.current = null; // Nullify ref after closing
            }
            if (reconnectTimeoutRef.current) { // Clear any pending reconnect
                clearTimeout(reconnectTimeoutRef.current);
                reconnectTimeoutRef.current = null;
            }

            // Reset UI state fully
            setPlaybackQueue([]);
            setAnalysisDataStore({});
            setCurrentlyPlayingUrl(null);
            setCurrentlyPlayingIndex(null);
            setIsPlaying(false);
            setIsFactCheckExpanded(false);
            if (videoRef.current) { videoRef.current.src = ''; videoRef.current.load(); }

        } catch (error) {
            console.error("Error stopping stream analysis:", error);
            setErrorState(error.response?.data?.detail || 'Failed to stop analysis cleanly.');
            // Still reset UI partially on error? Yes.
            setIsPlaying(false);
        } finally {
            setIsLoading(false); // Stop loading indicator
        }
    };

    // --- Toggle Fact Check ---
    const toggleFactCheckExpand = () => setIsFactCheckExpanded(prev => !prev);

    // --- Render Analysis Details ---
    const renderAnalysisDetails = (analysisData) => {
        if (!analysisData || analysisData.chunk_index === undefined) {
             return isPlaying ? <p>Loading analysis data...</p> : null;
        }

        const {
            chunk_index, analysis_timestamp, deepfake_analysis,
            transcription, fact_check_results, fact_check_context_current
        } = analysisData;

        // Get the full heatmap URL (already constructed in handleWebSocketMessage)
        const heatmapDisplayUrl = deepfake_analysis?.heatmap_url;

        return (
            <>
                <h4>Analysis for Chunk {chunk_index}</h4>
                <p><em>(Processed: {new Date(analysis_timestamp * 1000).toLocaleTimeString()})</em></p>

                {/* Deepfake */}
                <div className="analysis-card">
                    <h5>Deepfake</h5>
                    {deepfake_analysis && deepfake_analysis.timestamp >= 0 ? (
                        <>
                            <p>Score: <span className="highlight">{deepfake_analysis.score.toFixed(3)}</span></p>
                            {/* Conditionally render heatmap image */}
                            {heatmapDisplayUrl && (
                                <div className="heatmap-container">
                                    <p><strong>Attention Heatmap (if score {'>'} threshold):</strong></p>
                                    <img
                                        src={heatmapDisplayUrl}
                                        alt={`Deepfake Attention Heatmap for chunk ${chunk_index}`}
                                        className="heatmap-image"
                                    />
                                </div>
                            )}
                        </>
                    ) : <p>N/A</p>}
                </div>

                {/* Transcription */}
                <div className="analysis-card">
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
                {(fact_check_results?.length > 0 || fact_check_context_current) && ( // Show card if results exist OR context is current (even if empty)
                    <div className="analysis-card fact-check-section">
                        <h5>
                            Fact Check
                            {fact_check_context_current && <span className="fresh-indicator">(Updated)</span>}
                        </h5>
                        {fact_check_results?.length > 0 && ( // Only show button if there are results to show/hide
                            <button onClick={toggleFactCheckExpand} className="toggle-button">
                                {isFactCheckExpanded ? 'Hide' : 'Show'} Details
                            </button>
                        )}
                        {isFactCheckExpanded && fact_check_results?.length > 0 && (
                            <div className="fact-check-details">
                                {fact_check_results.map((claim, idx) => (
                                     claim.error ? // Display errors differently
                                     <div key={idx} className="claim-detail error-detail">
                                         <p><strong>Fact Check Error:</strong> {claim.error}</p>
                                     </div>
                                     :
                                    <div key={idx} className="claim-detail">
                                        <p><strong>Claim {idx + 1}:</strong> "{claim.original_claim || 'N/A'}"</p>
                                        <p>Verdict: <span className="highlight">{claim.final_verdict || 'N/A'}</span></p>
                                        {/* Optionally show explanation */}
                                        {/* <p>Explanation: {claim.final_explanation || 'N/A'}</p> */}
                                    </div>
                                ))}
                            </div>
                        )}
                         {fact_check_context_current && fact_check_results?.length === 0 && (
                            <p>No claims found in the latest fact check context.</p>
                         )}
                    </div>
                )}
            </>
        );
    };


    // --- JSX ---
    return (
        <div className="realtime-analysis-container">
            <h2>Real-Time Stream Analysis</h2>
            {/* Input Section */}
            <div className="stream-input-section">
                <input
                    type="text" placeholder="Enter Stream URL (e.g., Twitch, YouTube)" value={streamUrl}
                    onChange={(e) => setStreamUrl(e.target.value)}
                    disabled={isLoading || isPlaying || !!currentlyPlayingUrl}
                />
                {/* Buttons - Conditional Rendering based on state */}
                {(!isPlaying && !currentlyPlayingUrl) ? ( // Show Start only when truly stopped
                    <button onClick={handleStart} disabled={isLoading || !streamUrl} className="start-button">
                        {isLoading ? 'Starting...' : 'Start Analysis'}
                    </button>
                ) : ( // Show Stop if playing or has played
                    <button onClick={handleStop} disabled={isLoading} className="stop-button">
                        {isLoading ? 'Stopping...' : 'Stop Analysis'}
                    </button>
                )}
            </div>

            {/* Status/Error Display */}
            {errorState && <p className="status-message error">{errorState}</p>}
            {isLoading && !currentlyPlayingUrl && <p className="status-message info">Starting stream analysis...</p>}

            {/* Main Content Area */}
            <div className="content-area">
                {/* Video Player Area */}
                <div className="stream-video card"> {/* Added card class */}
                    <h3>Stream Playback</h3>
                    <video
                        ref={videoRef} controls width="100%"
                        src={currentlyPlayingUrl || ''}
                        onEnded={handleVideoEnded}
                        onError={(e) => { console.error("Video Playback Error:", e); setErrorState(`Video Error: ${e.target.error?.message || 'Cannot play video chunk.'}`); setIsPlaying(false); }}
                        onPlay={() => { setIsPlaying(true); setErrorState(null); }} // Clear error on successful play
                        onPause={() => setIsPlaying(false)}
                        muted={false} // Ensure not muted by default, add controls if needed
                        autoPlay // Attempt autoplay when src changes
                    />
                    {/* Placeholder */}
                    {!currentlyPlayingUrl && !isLoading && <div className="video-placeholder">Video chunks will appear here</div>}
                </div>

                {/* Analysis Display Area */}
                <div className="results-section card"> {/* Added card class */}
                    <h3>Analysis Details</h3>
                    {renderAnalysisDetails(currentAnalysis)}
                    {/* Show message if playing but analysis hasn't arrived yet */}
                    {isPlaying && !currentAnalysis && <p>Waiting for analysis of current chunk...</p>}
                    {/* Show message if stopped and no analysis was ever shown */}
                    {!isPlaying && !currentlyPlayingIndex && !isLoading && !errorState && <p>Analysis results will appear here.</p>}
                </div>
            </div>
        </div>
    );
};

export default RealTimeAnalysis;