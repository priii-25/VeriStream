import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import '../styles/RealTimeAnalysis.css';

const RealTimeAnalysis = () => {
  const [streamUrl, setStreamUrl] = useState('');
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [expanded, setExpanded] = useState({}); // For collapsible sections
  const videoRef = useRef(null);
  const wsRef = useRef(null);

  // WebSocket with reconnection logic
  useEffect(() => {
    const connectWebSocket = () => {
      wsRef.current = new WebSocket('ws://127.0.0.1:5001/api/stream/results');
      
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
      await axios.post('http://127.0.0.1:5001/api/stream/analyze', {
        url: streamUrl || 'https://www.twitch.tv/iskall85',
      });
    } catch (error) {
      console.error('Error starting stream:', error);
      setIsLoading(false);
    }
  };

  const handleStop = async () => {
    try {
      await axios.post('http://127.0.0.1:5001/api/stream/stop');
      setResults([]);
      setIsLoading(false);
      if (videoRef.current) videoRef.current.src = '';
    } catch (error) {
      console.error('Error stopping stream:', error);
    }
  };

  const toggleExpand = (index) => {
    setExpanded((prev) => ({ ...prev, [index]: !prev[index] }));
  };

  return (
    <div className="realtime-analysis-container">
      <h2>Real-Time Stream Analysis</h2>
      <div className="stream-input-section">
        <input
          type="text"
          placeholder="Enter Stream URL (e.g., Twitch)"
          value={streamUrl}
          onChange={(e) => setStreamUrl(e.target.value)}
        />
        <button onClick={handleStart} className="start-button">
          Start Stream
        </button>
        <button onClick={handleStop} className="stop-button">
          Stop Stream
        </button>
      </div>

      <div className="stream-video">
        {isLoading ? (
          <p className="loading-message">Loading stream (skipping initial Twitch loading screen, please wait up to 60 seconds)...</p>
        ) : (
          <video ref={videoRef} controls autoPlay />
        )}
      </div>

      <div className="results-section">
        <h3>Live Results</h3>
        {results.length > 0 ? (
          results.map((result, idx) => (
            <div key={idx} className="result-card">
              <p><strong>Timestamp:</strong> {new Date(result.timestamp * 1000).toLocaleTimeString()}</p>
              <p><strong>Deepfake Scores:</strong></p>
              <ul>
                <li>First 25 seconds: <span className="highlight">{result.deepfake_scores.first_half.toFixed(2)}</span></li>
                <li>Second 25 seconds: <span className="highlight">{result.deepfake_scores.second_half.toFixed(2)}</span></li>
              </ul>
              <p><strong>Faces Detected:</strong> {result.faces_detected.filter(Boolean).length} out of {result.faces_detected.length} frames</p>
              <p><strong>Transcriptions:</strong></p>
              <ul>
                <li>First 25 seconds: {result.transcriptions.first_half}</li>
                <li>Second 25 seconds: {result.transcriptions.second_half}</li>
              </ul>

              {result.fact_check_results?.length > 0 && (
                <div className="fact-check-section">
                  <button onClick={() => toggleExpand(idx)}>
                    {expanded[idx] ? 'Hide Fact Check Details' : 'Show Fact Check Details'}
                  </button>
                  {expanded[idx] && (
                    <div>
                      {result.fact_check_results.map((fcResult, fcIdx) => (
                        <div key={fcIdx}>
                          <h5>Raw Google Fact Check API Results</h5>
                          {fcResult.raw_fact_checks && Object.keys(fcResult.raw_fact_checks).length > 0 ? (
                            Object.entries(fcResult.raw_fact_checks).map(([claim, results], i) => (
                              <div key={i}>
                                <p>Claim: "{claim}"</p>
                                <ul>
                                  {results.map((res, j) => (
                                    <li key={j}>Verdict: {res.verdict} | Evidence: {res.evidence}</li>
                                  ))}
                                </ul>
                              </div>
                            ))
                          ) : (
                            <p>No raw fact check data available.</p>
                          )}

                          <h5>Filtered Non-Checkable Sentences</h5>
                          {fcResult.non_checkable_claims?.length > 0 ? (
                            <ul>
                              {fcResult.non_checkable_claims.map((sentence, i) => (
                                <li key={i}>"{sentence}"</li>
                              ))}
                            </ul>
                          ) : (
                            <p>No sentences were filtered out.</p>
                          )}

                          <h5>Processed Claim Details</h5>
                          {fcResult.processed_claims?.length > 0 ? (
                            fcResult.processed_claims.map((claim, i) => (
                              <div key={i}>
                                <p><strong>Claim {i + 1}:</strong> "{claim.original_claim}" [Source: {claim.source}]</p>
                                <p>Preprocessed: "{claim.preprocessed_claim}"</p>
                                {claim.source === "Knowledge Graph" ? (
                                  <>
                                    <p>Verdict: <span className="highlight">{claim.final_verdict}</span></p>
                                    <p>Explanation: {claim.final_explanation}</p>
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
                                          {claim.top_rag_snippets.map((snippet, j) => (
                                            <li key={j}>{snippet}</li>
                                          ))}
                                        </ul>
                                      </div>
                                    )}
                                    <p>Verdict: <span className="highlight">{claim.final_verdict}</span></p>
                                    <p>Explanation: {claim.final_explanation}</p>
                                  </>
                                ) : (
                                  <p>Error: {claim.final_explanation}</p>
                                )}
                              </div>
                            ))
                          ) : (
                            <p>No processed claims available.</p>
                          )}

                          <h5>SHAP Explanations</h5>
                          {fcResult.shap_explanations?.length > 0 ? (
                            <ul>
                              {fcResult.shap_explanations.map((ex, i) => (
                                <li key={i}>"{ex.claim}": {typeof ex.shap_values === 'string' ? ex.shap_values : '[SHAP Values Available]'}</li>
                              ))}
                            </ul>
                          ) : (
                            <p>SHAP analysis skipped or no results.</p>
                          )}

                          <h5>Chain of Thought Summary</h5>
                          <pre>{fcResult.summary || 'No summary available'}</pre>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
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