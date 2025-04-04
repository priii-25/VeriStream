// frontend/src/components/RealTimeAnalysis.js
import React, { useEffect, useRef, useState, useCallback } from 'react';
import axios from 'axios';
import '../styles/RealTimeAnalysis.css';

const RealTimeAnalysis = () => {
    // --- State ---
    const [streamUrl, setStreamUrl] = useState('');
    const [playbackQueue, setPlaybackQueue] = useState([]); // URLs only
    const [analysisDataStore, setAnalysisDataStore] = useState({}); // { chunkIndex: data }
    const [currentlyPlayingUrl, setCurrentlyPlayingUrl] = useState(null); // For <video> src
    const [currentlyPlayingIndex, setCurrentlyPlayingIndex] = useState(null); // Index of the playing chunk
    const [isLoading, setIsLoading] = useState(false);
    const [isPlaying, setIsPlaying] = useState(false);
    const [isFactCheckExpanded, setIsFactCheckExpanded] = useState(false);
    const [errorState, setErrorState] = useState(null);

    // --- Refs ---
    const videoRef = useRef(null);
    const wsRef = useRef(null);
    const reconnectTimeoutRef = useRef(null);
    const processingQueueRef = useRef(false); // Prevent playNext race conditions

    // --- Get current analysis data based on index ---
    // This derived state avoids needing a separate currentAnalysis state variable
    const currentAnalysis = currentlyPlayingIndex !== null ? analysisDataStore[currentlyPlayingIndex] : null;

    // --- Play Next Chunk ---
    const playNextInQueue = useCallback(() => {
        if (processingQueueRef.current) return;
        processingQueueRef.current = true;

        setPlaybackQueue(prevQueue => {
            if (prevQueue.length === 0) {
                console.log("PlayNext: Queue empty.");
                setIsPlaying(false);
                // We don't clear currentlyPlayingIndex here, so the last analysis stays visible
                processingQueueRef.current = false;
                return [];
            }

            const [nextUrl, ...remainingQueue] = prevQueue;
            let chunkIndex = -1;
            try {
                const match = nextUrl.match(/stream_chunk_(\d+)\.mp4/);
                if (match && match[1]) chunkIndex = parseInt(match[1], 10);
            } catch { /* ignore */ }

            console.log(`PlayNext: Dequeuing chunk ${chunkIndex}, URL: ${nextUrl.slice(-30)}`);
            setCurrentlyPlayingUrl(nextUrl); // Update src
            setCurrentlyPlayingIndex(chunkIndex); // *** Update index SIMULTANEOUSLY ***
            setIsFactCheckExpanded(false); // Collapse details

            // Defer video operations
            setTimeout(() => {
                if (videoRef.current) {
                    console.log(`PlayNext: Loading/Playing chunk ${chunkIndex}`);
                    videoRef.current.load();
                    videoRef.current.play()
                        .then(() => setIsPlaying(true))
                        .catch(err => {
                            console.warn(`PlayNext: Autoplay failed chunk ${chunkIndex}:`, err.name);
                            setIsPlaying(false);
                            if (err.name !== 'AbortError') {
                                setErrorState(`Video Error: ${err.name}.`);
                            }
                        });
                }
                processingQueueRef.current = false; // Unlock
            }, 100);

            return remainingQueue;
        });
    }, [analysisDataStore]); // AnalysisDataStore is needed to update display via derived state

    // --- WebSocket Message Handler ---
    const handleWebSocketMessage = useCallback((event) => {
        setErrorState(null);
        try {
            const data = JSON.parse(event.data);
            // console.log("WS Message:", data);

            if (data && typeof data.chunk_index === 'number' && typeof data.video_chunk_url === 'string') {
                const chunkIndex = data.chunk_index;
                const relativeUrl = data.video_chunk_url;
                const fullUrl = relativeUrl.startsWith('/')
                    ? `http://127.0.0.1:5001${relativeUrl}` // ** VERIFY PORT **
                    : relativeUrl;

                // 1. Store analysis data immediately
                setAnalysisDataStore(prevStore => ({ ...prevStore, [chunkIndex]: data }));

                // 2. Add URL to queue if new
                setPlaybackQueue(prevQueue => {
                    if (!prevQueue.includes(fullUrl)) {
                        console.log(`Enqueuing chunk ${chunkIndex}`);
                        const newQueue = [...prevQueue, fullUrl];
                        // 3. Trigger play if queue WAS empty and we are NOT currently playing
                        // Check isPlaying state directly here, not ref
                        if (prevQueue.length === 0 && !isPlaying) {
                            console.log("Triggering playNextInQueue (WS).");
                            // No timeout needed, playNextInQueue handles timing
                            playNextInQueue();
                        }
                        return newQueue;
                    }
                    return prevQueue;
                });
                setIsLoading(false);

            } else { /* Handle non-chunk messages */ }
        } catch (e) {
            console.error("WS message error:", e);
            setErrorState("Error processing WS data.");
        }
    }, [playNextInQueue, isPlaying]); // Added isPlaying dependency

    // --- WebSocket Connection ---
    const connectWebSocket = useCallback(() => {
        // ... (Connection logic identical to previous version) ...
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;
        wsRef.current = new WebSocket('ws://127.0.0.1:5001/api/stream/results');
        console.log('Attempting WebSocket connection...');
        setErrorState(null);

        wsRef.current.onopen = () => { /* ... */ };
        wsRef.current.onerror = (error) => { /* ... setErrorState ... */ };
        wsRef.current.onclose = (event) => { /* ... reconnect logic ... */ };
        wsRef.current.onmessage = handleWebSocketMessage;
    }, [handleWebSocketMessage]); // Dependency

    // Effect for mount/unmount
    useEffect(() => {
        connectWebSocket();
        return () => { /* ... cleanup ... */ };
    }, [connectWebSocket]);

    // --- Video Event Handler ---
    const handleVideoEnded = useCallback(() => {
        console.log(`Video ended: ${currentlyPlayingUrl ? currentlyPlayingUrl.slice(-30) : 'N/A'}`);
        setIsPlaying(false);
        playNextInQueue();
    }, [currentlyPlayingUrl, playNextInQueue]);

    // --- API Handlers (Reset state correctly) ---
    const handleStart = async () => {
        if (!streamUrl) return;
        setIsLoading(true);
        setErrorState(null);
        setPlaybackQueue([]);
        setAnalysisDataStore({});
        // Reset playing state *before* starting
        setCurrentlyPlayingUrl(null);
        setCurrentlyPlayingIndex(null);
        // setCurrentAnalysis(null); // Not needed - derived state
        setIsPlaying(false);
        setIsFactCheckExpanded(false);
        if (videoRef.current) { /* ... reset player ... */ }

        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            connectWebSocket();
        }

        try {
            await axios.post('http://127.0.0.1:5001/api/stream/analyze', { url: streamUrl });
            console.log("Start request sent.");
        } catch (error) { /* ... error handling ... */ }
        // isLoading will be turned off by WS message handler
    };

    const handleStop = async () => {
        setIsLoading(true);
        setErrorState(null);
        try {
            await axios.post('http://127.0.0.1:5001/api/stream/stop');
            console.log("Stop request sent.");
            if (wsRef.current) {
                wsRef.current.close(1000, "User stopped stream");
                wsRef.current = null;
            }
            if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);

            // Reset state
            setPlaybackQueue([]);
            setAnalysisDataStore({});
            // setCurrentAnalysis(null); // Derived state clears automatically
            setCurrentlyPlayingUrl(null);
            setCurrentlyPlayingIndex(null);
            setIsPlaying(false);
            setIsFactCheckExpanded(false);
            if (videoRef.current) { /* ... reset player ... */ }
        } catch (error) { /* ... error handling ... */ }
        finally { setIsLoading(false); }
    };

    // --- Toggle Fact Check ---
    const toggleFactCheckExpand = () => setIsFactCheckExpanded(prev => !prev);

    // --- Render Analysis Details ---
    const renderAnalysisDetails = (analysisData) => {
        // **Guard Clause First**
        if (!analysisData || analysisData.chunk_index === undefined) {
             // Display something informative if we are supposed to be playing
             if (currentlyPlayingUrl && isPlaying) {
                return <p>Loading analysis data for current chunk...</p>;
             }
             return null; // Render nothing otherwise
        }

        // Destructure *after* the check
        const {
            chunk_index, analysis_timestamp, deepfake_analysis,
            transcription, fact_check_results, fact_check_context_current
        } = analysisData;

        // Log that we are actually rendering
        console.log(`Rendering analysis for chunk ${chunk_index}`);

        return (
            <>
                <h4>Analysis for Chunk {chunk_index}</h4>
                <p><em>(Processed: {new Date(analysis_timestamp * 1000).toLocaleTimeString()})</em></p>

                {/* Deepfake */}
                <div className="analysis-card">
                    <h5>Deepfake</h5>
                    {deepfake_analysis && deepfake_analysis.timestamp >= 0 ? (
                        <p>Score: <span className="highlight">{deepfake_analysis.score.toFixed(3)}</span></p>
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
                {(fact_check_results?.length > 0) && (
                    <div className="analysis-card fact-check-section">
                        <h5>
                            Fact Check
                            {fact_check_context_current && <span className="fresh-indicator">(Updated)</span>}
                        </h5>
                        <button onClick={toggleFactCheckExpand}>
                            {isFactCheckExpanded ? 'Hide' : 'Show'}
                        </button>
                        {isFactCheckExpanded && (
                            <div className="fact-check-details">
                                {fact_check_results.map((claim, idx) => (
                                    <div key={idx} className="claim-detail">
                                        <p><strong>Claim {idx + 1}:</strong> "{claim.original_claim || 'N/A'}"</p>
                                        <p>Verdict: <span className="highlight">{claim.final_verdict || 'N/A'}</span></p>
                                    </div>
                                ))}
                            </div>
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
                    type="text" placeholder="Enter Stream URL" value={streamUrl}
                    onChange={(e) => setStreamUrl(e.target.value)}
                    disabled={isLoading || isPlaying || !!currentlyPlayingUrl}
                />
                {/* Buttons */}
                {(!currentlyPlayingUrl && !isPlaying) ? (
                    <button onClick={handleStart} disabled={isLoading || !streamUrl} className="start-button">
                        {isLoading ? 'Starting...' : 'Start Analysis'}
                    </button>
                ) : (
                    <button onClick={handleStop} disabled={isLoading} className="stop-button">
                        {isLoading ? 'Stopping...' : 'Stop Analysis'}
                    </button>
                )}
            </div>

            {/* Display Errors */}
            {errorState && <p className="status-message error">{errorState}</p>}

            {/* Main Content Area */}
            <div className="content-area">
                {/* Video Player Area */}
                <div className="stream-video">
                    <h3>Stream Playback</h3>
                    <video
                        ref={videoRef} controls width="100%"
                        src={currentlyPlayingUrl || ''} // Controlled by state
                        onEnded={handleVideoEnded}
                        onError={(e) => setErrorState(`Video Error: ${e.target.error?.message || 'Cannot play.'}`)}
                        onPlay={() => { setIsPlaying(true); setErrorState(null); }} // Update state/clear error
                        onPause={() => setIsPlaying(false)} // Reflect user pause
                        // key={currentlyPlayingUrl} // Optional: Uncomment if load() is unreliable
                    />
                    {/* Placeholder */}
                    {!currentlyPlayingUrl && !isLoading && <div className="video-placeholder">Video appears here</div>}
                    {isLoading && !currentlyPlayingUrl && <div className="video-placeholder">Starting...</div>}
                </div>

                {/* Analysis Display Area */}
                <div className="results-section">
                    <h3>Analysis for Current Video</h3>
                    {/* Use the derived 'currentAnalysis' state */}
                    {renderAnalysisDetails(currentAnalysis)}
                </div>
            </div>
        </div>
    );
};

export default RealTimeAnalysis;