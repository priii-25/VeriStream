# **VeriStream: Misinformation Detection for Media Streams**
## **Overview**
VeriStream is a tool designed to detect and verify misinformation in live broadcasts and media streams. Leveraging AI, machine learning, and distributed systems, VeriStream empowers broadcasters to identify false or misleading information in real time, enhancing media integrity and transparency.

## **Project Structure**
The project is organized into two main directories:

### **Frontend**
The frontend is built with React and provides an intuitive user interface for:
- Uploading and analyzing video files
- Real-time stream analysis
- Visualizing analysis results including deepfake detection, fact-checking, and knowledge graphs

**Key Technologies:**
- React 19
- React Router for navigation
- Chart.js and Plotly.js for data visualization
- Leaflet for geospatial mapping
- Axios for API communication

### **Backend**
The backend is built with FastAPI and provides robust analysis capabilities:
- Real-time video and audio processing
- Deepfake detection using computer vision techniques
- Knowledge graph construction for contextual analysis
- Fact-checking against verified sources
- Stream processing with FFmpeg

**Key Technologies:**
- FastAPI for API endpoints
- PyTorch and Transformers for AI/ML models
- OpenCV for image processing
- Whisper for audio transcription
- Streamlink for stream handling

## **Key Features**
- **Real-Time Knowledge Graphs**: Builds and maintains dynamic entity connections for claim verification.
- **Chained NLP Pipelines**: Modular workflows for tasks like entity recognition, fact-checking, and sentiment analysis.
- **Deepfake Detection**: Identifies tampered audio, frame inconsistencies, and lip-sync issues. using finetuned DinoV2
- **Multilingual Support**: Processes misinformation in multiple languages using Hugging Face transformers.
- **Geospatial Misinformation Mapping**: Visualizes misinformation spread using GIS tools.
- **Explainable AI**: Ensures transparency by providing justifications for flagged content.

## **Tech Stack**

| **Category**                    | **Technologies**                            |
|---------------------------------|---------------------------------------------|
| **Frontend**                    | React, Chart.js, Plotly.js, Leaflet         |
| **Backend**                     | FastAPI, PyTorch, OpenCV, Whisper           |
| **Distributed Processing**      | Apache Spark                                |
| **Data Streaming**              | Apache Kafka & ZooKeeper                    |
| **AI/ML Frameworks**            | PyTorch, Transformers, EfficientNet         |
| **Knowledge Graph**             | Neo4j                                       |

## **Installation Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/priii-25/VeriStream
cd VeriStream
```
### **2. Activate the virtual environment and download the requirements**
```bash
pip install -r requirements.txt
```

### **3. Frontend Setup**
Install frontend dependencies:
```bash
cd frontend
npm install
```

### **4. Environment Configuration**
Create a `.env` file in the project root directory with your API keys:
```
GROQ_API_KEY=your_groq_api_key_here
FACT_CHECK_API_KEY = YOUR_FACT_CHECK_API_KEY
```

### **5. Running the Application**
Start the backend:
```bash
cd backend
uvicorn main:app --reload
```

Start the frontend:
```bash
cd frontend
npm start
```

The application will be available at http://localhost:3000

## **Usage**
1. **File Upload Analysis**: Upload a video file to detect deepfakes and check facts.
2. **Real-Time Stream Analysis**: Enter a stream URL to analyze live broadcasts for misinformation.

## **Demo Video**

[![Demo Video](https://img.youtube.com/vi/EfvakYHF7-M/0.jpg)](https://youtu.be/EfvakYHF7-M)

