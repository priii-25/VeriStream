import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';

const RealTimeAnalysis = () => {
  const [streamUrl, setStreamUrl] = useState('');
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const videoRef = useRef(null);
  const wsRef = useRef(null);

  // WebSocket with reconnection logic
  useEffect(() => {
    const connectWebSocket = () => {
      wsRef.current = new WebSocket('ws://127.0.0.1:5000/api/stream/results');
      
      wsRef.current.onopen = () => console.log('WebSocket connected');
      
      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.message !== "Processing...") {
          setResults((prev) => [...prev.slice(-5), data]); // Keep last 5 chunks
          setIsLoading(false); // Stop loading once we get results
          if (videoRef.current) {
            videoRef.current.src = data.video_chunk;
            videoRef.current.play().catch(err => console.error('Video play error:', err));
          }
        }
      };
      
      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
      
      wsRef.current.onclose = () => {
        console.log('WebSocket closed, attempting to reconnect...');
        setTimeout(connectWebSocket, 1000);
      };
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const handleStart = async () => {
    try {
      setIsLoading(true);
      await axios.post('http://127.0.0.1:5000/api/stream/analyze', {
        url: streamUrl || 'https://www.twitch.tv/iskall85',
      });
    } catch (error) {
      console.error('Error starting stream:', error);
      setIsLoading(false);
    }
  };

  const handleStop = async () => {
    try {
      await axios.post('http://127.0.0.1:5000/api/stream/stop');
      setResults([]);
      setIsLoading(false);
      if (videoRef.current) videoRef.current.src = '';
    } catch (error) {
      console.error('Error stopping stream:', error);
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <h2>Real-Time Stream Analysis</h2>
      <input
        type="text"
        placeholder="Enter Stream URL (e.g., Twitch)"
        value={streamUrl}
        onChange={(e) => setStreamUrl(e.target.value)}
        style={{ width: '300px', marginRight: '10px' }}
      />
      <button onClick={handleStart} style={{ marginRight: '10px' }}>
        Start Stream
      </button>
      <button onClick={handleStop}>Stop Stream</button>

      <div style={{ marginTop: '20px' }}>
        {isLoading ? (
          <p>Loading stream (skipping initial Twitch loading screen, please wait up to 60 seconds)...</p>
        ) : (
          <video
            ref={videoRef}
            controls
            autoPlay
            style={{ width: '640px', height: '360px' }}
          />
        )}
      </div>

      <div style={{ marginTop: '20px' }}>
        <h3>Live Results</h3>
        {results.length > 0 ? (
          results.map((result, idx) => (
            <div key={idx} style={{ marginBottom: '10px' }}>
              <p>Timestamp: {new Date(result.timestamp * 1000).toLocaleTimeString()}</p>
              <p>Deepfake Scores:</p>
              <ul>
                <li>First 25 seconds: {result.deepfake_scores.first_half.toFixed(2)}</li>
                <li>Second 25 seconds: {result.deepfake_scores.second_half.toFixed(2)}</li>
              </ul>
              <p>Faces Detected: {result.faces_detected.filter(Boolean).length} out of {result.faces_detected.length} frames</p>
              <p>Transcriptions:</p>
              <ul>
                <li>First 25 seconds: {result.transcriptions.first_half}</li>
                <li>Second 25 seconds: {result.transcriptions.second_half}</li>
              </ul>
              <p>Fact Checks:</p>
              <ul>
                {result.fact_checks.length > 0 ? (
                  result.fact_checks.map((fc, i) => (
                    <li key={i}>{fc.verdict} - {fc.evidence}</li>
                  ))
                ) : (
                  <li>Pending or No Claims</li>
                )}
              </ul>
            </div>
          ))
        ) : (
          <p>No results yet...</p>
        )}
      </div>
    </div>
  );
};

export default RealTimeAnalysis;