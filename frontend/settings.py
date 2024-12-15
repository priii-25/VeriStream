import streamlit as st

st.header("Settings")
st.subheader("Filter Settings")
col1, col2, col3 = st.columns(3)

with col1:
    lang = st.selectbox("Language", ["English", "Hindi", "Other"])

with col2:
    region = st.selectbox("Region", ["North", "South", "East", "West"])

with col3:
    time_range = st.selectbox("Time Range", ["Last Hour", "Today", "This Week"])

content_type = st.multiselect("Content Type", ["Video", "Image", "Text"])

confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

st.subheader("Data Sources")

col4, col5, col6 = st.columns(3)

with col4:
    social_media = st.checkbox("Social Media")

with col5:
    news_broadcast = st.checkbox("News Broadcast")

with col6:
    govt_db = st.checkbox("Govt DB")

if st.button("Apply Settings"):
    st.write("Settings applied!")

