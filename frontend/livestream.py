import streamlit as st

col1, col2 = st.columns([7, 3])

with col1:
    st.markdown("## Live Streaming")
    st.video('https://www.youtube.com/watch?v=Io-G_aiF8HA')  
    st.write("[Video content will appear here]")

with col2:
    st.markdown("## Flagged Content")
    st.write("[Flagged content will appear here]")

st.markdown(
    """
    <style>
    .css-1d391kg {border-right: 2px solid #cccccc; padding-right: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)