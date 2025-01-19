import streamlit as st
import os
from config import LANGUAGE_MAPPING
from logging_config import configure_logging
from analyzer import OptimizedAnalyzer
from video_analyzer import VideoAnalyzer
from utils import create_gis_map, create_monitoring_dashboard, display_analysis_results, visualize_knowledge_graph_interactive
from deep_translator import GoogleTranslator
from kafka.admin import KafkaAdminClient
from streamlit_folium import folium_static

logger = configure_logging()

def main():
    st.title("VERISTREAM")
    st.markdown("### Real-time Deepfake Detection & Transcription")
    
    create_monitoring_dashboard()
    
    with st.sidebar:
        st.title("System Status")
        if st.button("Check Kafka Topics"):
            try:
                admin_client = KafkaAdminClient(bootstrap_servers=['localhost:29092'])
                topics = admin_client.list_topics()
                st.json({"Available Topics": topics})
            except Exception as e:
                st.error(f"Error checking topics: {e}")
                logger.error(f"Kafka error: {e}")
    
    uploaded_file = st.file_uploader("Upload Video for Analysis", type=['mp4', 'avi', 'mov'])

    if uploaded_file:
        temp_path = f"temp_{uploaded_file.name}"
        try:
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            video_file = open(temp_path, "rb")
            st.video(video_file)
            
            progress_bar = st.progress(0)
            st.write("Analyzing video...")
            
            try:
                analyzer = VideoAnalyzer()
                transcription, final_score, frames_data = analyzer.analyze_video(temp_path, progress_bar)
                
                text_analyzer = OptimizedAnalyzer(use_gpu=True)
                analysis_result = text_analyzer.analyze_text(transcription)
                
                display_analysis_results(final_score, frames_data)
                
                with st.expander("Video Transcription"):
                    st.write(transcription)
                    target_language = st.selectbox("Translate to", ["English", "Assamese","Bengali", "Gujarati","Hindi", "Kannada","Malayalam","Marathi","Odia (Oriya)","Urdu"])
    
                    if target_language != "English":
                        target_code = LANGUAGE_MAPPING.get(target_language, "en")
                        translated_text = GoogleTranslator(source='auto', target=target_code).translate(transcription)
                        st.write(f"Translated to {target_language}:")
                        st.write(translated_text)
                
                with st.expander("Text Analysis Results"):
                    st.write("### Sentiment Analysis")
                    if analysis_result.sentiment:
                        sentiment_label = analysis_result.sentiment["label"]
                        st.write(f"**Sentiment:** {sentiment_label}")
                        st.write(f"**Confidence Score:** {analysis_result.sentiment['score']:.4f}")
                    else:
                        st.write("Sentiment analysis not available")
                    
                    st.write("### Fact Checks")
                    st.write(analysis_result.fact_checks)
                    
                    st.write("### Emotional Triggers")
                    st.write(analysis_result.emotional_triggers)
                    
                    st.write("### Stereotypes")
                    st.write(analysis_result.stereotypes)
                    
                    st.write("### Manipulation Score")
                    st.write(analysis_result.manipulation_score)
                    
                    st.write("### Entities")
                    st.write(analysis_result.entities)
                    
                    st.write("### Knowledge Graph")
                    visualize_knowledge_graph_interactive(analysis_result.knowledge_graph)
                    
                    st.write("### Generative Analysis")
                    st.write(analysis_result.generative_analysis)

                    st.subheader("Geospatial Visualization of Detected Locations")

                    gis_map = create_gis_map()
                    folium_static(gis_map)
                
                
                progress_bar.progress(1.0)
                
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
                logger.error(f"Processing error: {str(e)}", exc_info=True)
                if "NoSuchMethodError" in str(e):
                    st.warning("Spark version compatibility issue detected. Please check system configurations.")
                return
            
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            logger.error(f"Processing error: {str(e)}")
            from monitoring import MetricsCollector
            metrics = MetricsCollector()
            metrics.system_healthy.set(0)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    main()