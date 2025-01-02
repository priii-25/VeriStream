# **VeriStream: Misinformation Detection for Media Streams**
## **Overview**
VeriStream is a cutting-edge tool designed to detect and correct misinformation in live broadcasts and media streams. Leveraging AI, machine learning, and distributed systems, VeriStream empowers broadcasters to identify false or misleading information in real-time, enhancing media integrity and transparency.

## **Key Features**
- **Real-Time Knowledge Graphs**: Builds and maintains dynamic entity connections for claim verification.
- **Chained NLP Pipelines**: Modular workflows for tasks like entity recognition, fact-checking, and sentiment analysis using LangChain and PyTorch.
- **Deepfake Detection**: Identifies tampered audio, frame inconsistencies, and lip-sync issues.
- **Multilingual Support**: Processes misinformation in multiple languages using Hugging Face transformers.
- **Geospatial Misinformation Mapping**: Visualizes misinformation spread using GIS tools like GeoPandas and Folium.
- **Explainable AI**: Ensures transparency by providing justifications for flagged content.

## **Tech Stack**

| **Category**             | **Technologies**                       |
|---------------------------|-----------------------------------------------|
| **Programming Language**  | Python                                        |
| **Data Streaming**        | Apache Kafka & ZooKeeper                      |
| **Distributed Processing**| Apache Spark                                  |
| **AI/ML Frameworks**      | PyTorch, LangChain, Hugging Face, OpenCV      |
| **Databases**             | Neo4j, MongoDB                                |
| **Visualization**         | Streamlit, Plotly                             |
| **GIS Tools**             | GeoPandas, Folium                             |
| **Containerization**      | Docker                                        |


## **Installation Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/priii-25/VeriStream
cd VeriStream
```
### **2. Set Up Docker and Apache Kafka**
Install Docker & Start the Docker containers:
```bash
docker-compose up -d
```
### **3. Create and Activate a Virtual Environment**
Create a virtual environment:
```bash
python -m venv env
```
Activate the virtual environment:
On Windows:
```bash
env\Scripts\activate
```
On macOS/Linux:
```bash
source env/bin/activate
```
### **4. Install Project Dependencies**
```bash
pip install -r requirements.txt
```
5. Set Up Apache Spark
```bash
python setup_spark.py
```
6. Configure API Keys
Obtain API keys for: Google Fact Check API & Google Generative AI API.
Create a .env file in the project root directory
```bash
FACT_CHECK_API_KEY=your_fact_check_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```
7. Run the Application
Start the program using Streamlit:
```bash
streamlit run app.py
```
