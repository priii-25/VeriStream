import streamlit as st
import os
from datetime import timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from config import LANGUAGE_MAPPING
from logging_config import configure_logging
from analyzer import OptimizedAnalyzer
from video_analyzer import VideoAnalyzer
from utils import create_gis_map, create_monitoring_dashboard, display_analysis_results, visualize_knowledge_graph_interactive
from deep_translator import GoogleTranslator
from kafka.admin import KafkaAdminClient
from streamlit_folium import folium_static
import six
import sys
if sys.version_info >= (3, 12, 0):
    sys.modules['kafka.vendor.six.moves'] = six.moves
logger = configure_logging()

class OutbreakAnalyzer:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
    
    def visualize_trends(self, include_predictions=True, prediction_days=30):
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Outbreaks Over Time',
                    'Outbreaks by Region',
                    'Outbreaks by Reason',
                    'Region-Reason Heatmap'
                )
            )
            
            df_time = self.df.groupby('date').size().reset_index(name='count')
            fig.add_trace(
                go.Scatter(x=df_time['date'], y=df_time['count'], name='Historical Outbreaks'),
                row=1, col=1
            )
            
            if include_predictions:
                end_date = self.df['date'].max() + timedelta(days=prediction_days)
                predictions = predict_range(
                    self.df['date'].max() + timedelta(days=1),
                    end_date
                )
                if predictions is not None and len(predictions) > 0:
                    pred_time = predictions.groupby('date').size().reset_index(name='count')
                    fig.add_trace(
                        go.Scatter(x=pred_time['date'], y=pred_time['count'],
                                 name='Predicted Outbreaks', line=dict(dash='dash')),
                        row=1, col=1
                    )
            
            region_counts = self.df['region'].value_counts()
            fig.add_trace(
                go.Bar(x=region_counts.index, y=region_counts.values, name='Outbreaks by Region'),
                row=1, col=2
            )
            
            reason_counts = self.df['reason'].value_counts()
            fig.add_trace(
                go.Bar(x=reason_counts.index, y=reason_counts.values, name='Outbreaks by Reason'),
                row=2, col=1
            )
            
            heatmap_data = pd.crosstab(self.df['region'], self.df['reason'])
            fig.add_trace(
                go.Heatmap(z=heatmap_data.values, x=heatmap_data.columns,
                          y=heatmap_data.index, name='Region-Reason Distribution'),
                row=2, col=2
            )
            
            fig.update_layout(
                height=800,
                width=1200,
                title_text="Outbreak Analysis Dashboard",
                showlegend=True
            )
            
            return fig
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            return None

def predict_range(start_date, end_date, model_path='outbreak_model.joblib'):
    try:
        model_components = joblib.load(model_path)
        model = model_components['model']
        le_region = model_components['le_region']
        le_reason = model_components['le_reason']
        feature_columns = model_components['feature_columns']
        regions = model_components['regions']
        reasons = model_components['reasons']
        
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        dates = pd.date_range(start=start_date, end=end_date)
        
        prediction_data = []
        for date in dates:
            for region in regions:
                for reason in reasons:
                    prediction_data.append({
                        'date': date,
                        'region': region,
                        'reason': reason,
                        'year': date.year,
                        'month': date.month,
                        'day': date.day,
                        'day_of_week': date.dayofweek,
                        'region_encoded': le_region.transform([region])[0],
                        'reason_encoded': le_reason.transform([reason])[0],
                        'region_frequency': 1,
                        'reason_frequency': 1,
                        'region_7d_count': 0,
                        'region_30d_count': 0,
                        'region_90d_count': 0,
                        'reason_7d_count': 0,
                        'reason_30d_count': 0,
                        'reason_90d_count': 0
                    })
        
        prediction_df = pd.DataFrame(prediction_data)
        predictions = model.predict_proba(prediction_df[feature_columns])
        prediction_df['outbreak_probability'] = predictions.max(axis=1)
        high_risk_outbreaks = prediction_df[prediction_df['outbreak_probability'] > 0.8]
        
        return high_risk_outbreaks[['date', 'region', 'reason', 'outbreak_probability']]
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

def video_analysis_page():
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
            
            analyzer = VideoAnalyzer()
            transcription, final_score, frames_data = analyzer.analyze_video(temp_path, progress_bar)
            
            text_analyzer = OptimizedAnalyzer(use_gpu=True)
            analysis_result = text_analyzer.analyze_text(transcription)
            
            display_analysis_results(final_score, frames_data)
            
            with st.expander("Video Transcription"):
                st.write(transcription)
                target_language = st.selectbox(
                    "Translate to",
                    ["English", "Assamese", "Bengali", "Gujarati", "Hindi", "Kannada",
                     "Malayalam", "Marathi", "Odia (Oriya)", "Urdu"]
                )
                
                if target_language != "English":
                    target_code = LANGUAGE_MAPPING.get(target_language, "en")
                    translated_text = GoogleTranslator(source='auto', target=target_code).translate(transcription)
                    st.write(f"Translated to {target_language}:")
                    st.write(translated_text)
            
            with st.expander("Text Analysis Results"):
                st.write("### Sentiment Analysis")
                if analysis_result.sentiment:
                    st.write(f"**Sentiment:** {analysis_result.sentiment['label']}")
                    st.write(f"**Confidence Score:** {analysis_result.sentiment['score']:.4f}")
                else:
                    st.write("Sentiment analysis not available")
                
                for section in ["Fact Checks", "Emotional Triggers", "Stereotypes",
                              "Manipulation Score", "Entities", "Generative Analysis"]:
                    st.write(f"### {section}")
                    st.write(getattr(analysis_result, section.lower().replace(" ", "_")))
                
                st.write("### Knowledge Graph")
                visualize_knowledge_graph_interactive(analysis_result.knowledge_graph)
                
                st.subheader("Geospatial Visualization of Detected Locations")
                gis_map = create_gis_map()
                folium_static(gis_map)
            
            progress_bar.progress(1.0)
            
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            logger.error(f"Processing error: {str(e)}", exc_info=True)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

def analytics_prediction_page():
    st.title("Analytics & Predictions")
    
    try:
        analyzer = OutbreakAnalyzer('misinformation_dataset.csv')
        
        st.subheader("Prediction Settings")
        prediction_days = st.slider("Number of days to predict", 7, 90, 30)
        include_predictions = st.checkbox("Include predictions", value=True)
        
        fig = analyzer.visualize_trends(
            include_predictions=include_predictions,
            prediction_days=prediction_days
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            if include_predictions:
                st.subheader("Detailed Predictions")
                predictions = predict_range(
                    start_date=pd.Timestamp.now(),
                    end_date=pd.Timestamp.now() + timedelta(days=prediction_days)
                )
                
                if predictions is not None:
                    st.write("High-risk outbreaks predicted:")
                    st.dataframe(predictions.sort_values('outbreak_probability', ascending=False))
                    st.write(f"Total high-risk outbreaks predicted: {len(predictions)}")
        else:
            st.error("Error generating visualizations")
            
    except Exception as e:
        st.error(f"Error in analytics page: {str(e)}")

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Video Analysis", "Analytics & Predictions"])
    
    if page == "Video Analysis":
        video_analysis_page()
    else:
        analytics_prediction_page()

if __name__ == "__main__":
    main()