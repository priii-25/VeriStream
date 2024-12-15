import streamlit as st
import plotly.express as px
import pandas as pd

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    .navbar {
        background-color: #f8f9fa;
        padding: 10px;
        border-bottom: 2px solid #cccccc;
        display: flex;
        justify-content: space-around;
    }
    .navbar a {
        text-decoration: none;
        font-size: 18px;
        font-weight: bold;
        color: #333333;
        padding: 5px 15px;
    }
    .navbar a:hover {
        background-color: #dddddd;
        border-radius: 5px;
    }
    </style>
    <div class="navbar">
        <a href="?page=live_feed">Live Feed</a>
        <a href="?page=gis">GIS</a>
        <a href="?page=knowledge_graph">Knowledge Graph</a>
    </div>
    """,
    unsafe_allow_html=True,
)

query_params = st.query_params

if query_params.get("page") == "live_feed":
    import livestream
elif query_params.get("page") == "gis":
    st.markdown("### GIS Section")
    st.write("[GIS content will appear here]")
elif query_params.get("page") == "knowledge_graph":
    st.markdown("### Knowledge Graph Section")
    st.write("[Knowledge Graph content will appear here]")
else:
    st.markdown("### Search")
    st.text_input("", placeholder="Search...")

    st.markdown("### Dashboard")
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    data = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [10, 20, 15, 30, 25],
    })

    fig1 = px.line(data, x="x", y="y", title="Line Chart")
    fig2 = px.bar(data, x="x", y="y", title="Bar Chart")
    fig3 = px.scatter(data, x="x", y="y", title="Scatter Plot")
    fig4 = px.pie(data, values="y", names="x", title="Pie Chart")

    col1.plotly_chart(fig1, use_container_width=True)
    col2.plotly_chart(fig2, use_container_width=True)
    col3.plotly_chart(fig3, use_container_width=True)
    col4.plotly_chart(fig4, use_container_width=True)
