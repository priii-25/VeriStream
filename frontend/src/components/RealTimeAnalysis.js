// frontend/src/components/RealTimeAnalysis.js
import React, { useEffect, useRef, useState } from 'react';
import Hls from 'hls.js';
import axios from 'axios';

const RealTimeAnalysis = () => {
  const [streamUrl, setStreamUrl] = useState('');
  const [hlsUrl, setHlsUrl] = useState('');
  const [results, setResults] = useState(null);
  const videoRef = useRef(null);
  const wsRef = useRef(null);

  // Play the stream
  useEffect(() => {
    if (!hlsUrl || !videoRef.current) return;

    const hls = new Hls();
    if (Hls.isSupported()) {
      hls.loadSource(hlsUrl);
      hls.attachMedia(videoRef.current);
      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        videoRef.current.play();
      });
      hls.on(Hls.Events.ERROR, (event, data) => {
        console.error('HLS Error:', data);
      });
    } else if (videoRef.current.canPlayType('application/vnd.apple.mpegurl')) {
      videoRef.current.src = hlsUrl;
      videoRef.current.play();
    }

    return () => {
      hls.destroy();
    };
  }, [hlsUrl]);

  // WebSocket for results
  useEffect(() => {
    wsRef.current = new WebSocket('ws://127.0.0.1:5000/api/stream/results');
    wsRef.current.onopen = () => console.log('WebSocket connected');
    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setResults(data);
    };
    wsRef.current.onerror = (error) => console.error('WebSocket error:', error);
    wsRef.current.onclose = () => console.log('WebSocket closed');

    return () => {
      wsRef.current.close();
    };
  }, []);

  const handleStart = async () => {
    try {
      // Fetch HLS URL
      const urlResponse = await axios.post('http://127.0.0.1:5000/api/stream/get-url', {
        url: streamUrl || 'https://www.twitch.tv/iskall85',
      });
      setHlsUrl(urlResponse.data.hls_url);

      // Start analysis
      await axios.post('http://127.0.0.1:5000/api/stream/analyze', {
        url: streamUrl || 'https://www.twitch.tv/iskall85',
      });
    } catch (error) {
      console.error('Error starting stream:', error);
    }
  };

  const handleStop = async () => {
    try {
      await axios.post('http://127.0.0.1:5000/api/stream/stop');
      setHlsUrl('');
      setResults(null);
    } catch (error) {
      console.error('Error stopping stream:', error);
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <h2>Real-Time Stream Analysis</h2>
      <input
        type="text"
        placeholder="Enter Twitch URL"
        value={streamUrl}
        onChange={(e) => setStreamUrl(e.target.value)}
        style={{ width: '300px', marginRight: '10px' }}
      />
      <button onClick={handleStart} style={{ marginRight: '10px' }}>
        Start Stream
      </button>
      <button onClick={handleStop}>Stop Stream</button>

      <div style={{ marginTop: '20px' }}>
        <video
          ref={videoRef}
          controls
          autoPlay
          style={{ width: '640px', height: '360px' }}
        />
      </div>

      <div style={{ marginTop: '20px' }}>
        <h3>Live Results</h3>
        {results ? (
          <pre style={{ whiteSpace: 'pre-wrap' }}>
            {JSON.stringify(results, null, 2)}
          </pre>
        ) : (
          <p>No results yet...</p>
        )}
      </div>
    </div>
  );
};

export default RealTimeAnalysis;