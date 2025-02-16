#utils.py
import streamlit as st
import folium
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
from typing import Dict, Any
from config import LANGUAGE_MAPPING
from deep_translator import GoogleTranslator
from mongodb_manager import MongoDBManager
import logging
import json
from kafka.admin import KafkaAdminClient
from pyvis.network import Network
logger = logging.getLogger('veristream')

mongo_manager = MongoDBManager()

def create_gis_map():
    """Create a Folium map with markers for the detected locations."""
    geolocator = Nominatim(user_agent="geoapiExercises")
    map_center = [20.5937, 78.9629]  
    m = folium.Map(location=map_center, zoom_start=5)

    locations = mongo_manager.get_all_locations()
    
    for location in locations:
        try:
            geo_location = geolocator.geocode(location)
            if geo_location:
                folium.Marker(
                    location=[geo_location.latitude, geo_location.longitude],
                    popup=location,
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)
        except Exception as e:
            logger.error(f"Error geocoding location {location}: {e}")
    
    return m


def create_monitoring_dashboard():
    st.subheader("System Monitoring")
    
    col1, col2, col3 = st.columns(3)
    
    try:
        from monitoring import MetricsCollector
        metrics = MetricsCollector()
        system_metrics = metrics.get_system_metrics()
        
        with col1:
            cpu_usage = system_metrics['cpu_usage']
            st.metric("CPU Usage", f"{cpu_usage:.1f}%",
                     delta=f"{cpu_usage - 50:.1f}%" if cpu_usage > 50 else None,
                     delta_color="inverse")
            
        with col2:
            memory_usage = system_metrics['memory_usage']
            st.metric("Memory Usage", f"{memory_usage:.1f}%",
                     delta=f"{memory_usage - 70:.1f}%" if memory_usage > 70 else None,
                     delta_color="inverse")
            
        with col3:
            system_healthy = system_metrics['system_healthy']
            health_status = "Healthy" if system_healthy > 0.5 else "Unhealthy"
            health_delta = "OK" if system_healthy > 0.5 else "Check Logs"
            st.metric("System Health", health_status,
                     delta=health_delta,
                     delta_color="normal" if health_status == "Healthy" else "inverse")
        
    except Exception as e:
        logger.error(f"Error creating monitoring dashboard: {e}", exc_info=True)
        st.error("Unable to display monitoring dashboard")

def display_analysis_results(final_score: float, frames_data: Dict):
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=frames_data['timestamps'],
            y=frames_data['scores'],
            mode='lines',
            name='Average Score',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=frames_data['timestamps'],
            y=frames_data['max_scores'],
            mode='lines',
            name='Max Score',
            line=dict(color='red', width=2)
        ))
        fig.update_layout(
            title='Deepfake Detection Confidence Over Time',
            xaxis_title='Time (seconds)',
            yaxis_title='Confidence Score',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=frames_data['max_scores'],
            nbinsx=30,
            name='Score Distribution',
            marker_color='rgb(55, 83, 109)'
        ))
        hist_fig.update_layout(
            title='Detection Score Distribution',
            xaxis_title='Confidence Score',
            yaxis_title='Frequency',
            bargap=0.1
        )
        st.plotly_chart(hist_fig, use_container_width=True)
    
    st.subheader("Analysis Summary")
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        avg_score = float(np.mean(frames_data['max_scores']))
        st.metric(
            "Average Confidence",
            f"{avg_score:.2%}",
            delta=f"{avg_score - 0.5:.2%}",
            delta_color="inverse"
        )
    
    with summary_col2:
        max_score = float(max(frames_data['max_scores']))
        st.metric(
            "Peak Detection Score",
            f"{max_score:.2%}",
            delta=f"{max_score - avg_score:.2%}",
            delta_color="inverse"
        )
    
    with summary_col3:
        total_frames = len(frames_data['timestamps'])
        st.metric(
            "Total Frames Analyzed",
            f"{total_frames:,}",
            f"{total_frames/30:.1f} seconds"
        )
    
    if final_score > 0.7:
        st.error(f"🚨 High probability of deepfake detected (Confidence: {final_score:.2%})")
        st.markdown("""
            **Detection Details:**
            - Multiple frames show signs of manipulation
            - High confidence scores sustained over time
            - Consistent detection patterns across video segments
            
            **Recommended Actions:**
            - Conduct manual review
            - Check video metadata
            - Verify source authenticity
        """)
    elif final_score > 0.4:
        st.warning(f"⚠️ Potential manipulation detected (Confidence: {final_score:.2%})")
        st.markdown("""
            **Analysis Notes:**
            - Some suspicious frames detected
            - Moderate confidence in manipulation
            - Further investigation recommended
        """)
    else:
        st.success(f"✅ Video appears authentic (Confidence: {1-final_score:.2%})")
        st.markdown("""
            **Analysis Notes:**
            - No significant manipulation patterns detected
            - Low confidence scores across frames
            - Normal video characteristics observed
        """)
    
    with st.expander("Detailed Metrics"):
        metrics_df = pd.DataFrame({
            'Time (s)': frames_data['timestamps'],
            'Average Score': frames_data['scores'],
            'Max Score': frames_data['max_scores'],
            'Faces Detected': frames_data['faces_detected']
        })
        st.dataframe(
            metrics_df.style.background_gradient(subset=['Max Score'], cmap='RdYlGn_r'),
            use_container_width=True
        )
        
        csv = metrics_df.to_csv(index=False)
        st.download_button(
            label="Download Detailed Results",
            data=csv,
            file_name="deepfake_analysis_results.csv",
            mime="text/csv"
        )

def visualize_knowledge_graph_interactive(graph_data):
    """
    Visualize the knowledge graph interactively using pyvis.
    
    Args:
        graph_data (dict): The knowledge graph data containing nodes and edges.
    """
    net = Network(notebook=True, directed=True, height="750px", width="100%", bgcolor="#ffffff", font_color="#000000")
    
    for node_id, node_data in graph_data["nodes"]:
        label = node_data.get("text", node_id)
        title = f"Type: {node_data.get('type', 'N/A')}\n"
        if node_data["type"] == "fact":
            title += f"Text: {node_data.get('text', 'N/A')}\nSentiment: {node_data.get('sentiment', 'N/A')}"
        elif node_data["type"] == "verification":
            title += f"Status: {node_data.get('status', 'N/A')}\nDetails: {node_data.get('details', 'N/A')}"
        elif node_data["type"] == "entity":
            title += f"Text: {node_data.get('text', 'N/A')}\nEntity Type: {node_data.get('entity_type', 'N/A')}"
        net.add_node(node_id, label=label, title=title, color=get_node_color(node_data["type"]))
    
    for edge in graph_data["edges"]:
        source, target, edge_data = edge
        net.add_edge(source, target, title=edge_data.get("relation", ""), color="gray")
    
    net.set_options("""
    {
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -2000,
                "centralGravity": 0.3,
                "springLength": 200,
                "springConstant": 0.04,
                "damping": 0.09,
                "avoidOverlap": 0.1
            },
            "minVelocity": 0.75
        }
    }
    """)
    
    net.save_graph("knowledge_graph.html")
    st.components.v1.html(open("knowledge_graph.html", "r").read(), height=800)

def get_node_color(node_type):
    """Return a color based on the node type."""
    colors = {
        "fact": "#ff7f7f",  
        "verification": "#7f7fff",  
        "entity": "#7fff7f"  
    }
    return colors.get(node_type, "#808080")
