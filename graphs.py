import matplotlib.pyplot as plt
import numpy as np
import os

# --- Placeholder Data (Replace with actual measured data!) ---

# Component Accuracy (%)
accuracy_data = {
    # From your prompt
    'DinoV2 (Deepfake)': 88.0,
    # Estimated based on common Whisper base model performance (WER ~15%)
    'Whisper (Transcription)': 85.0,
    # Estimated for NER/Text tasks (highly dependent on specific task)
    'Text Analyzer (NER F1)': 90.0,
    # Estimated fact-check accuracy (very dependent on claim complexity & sources)
    'Fact Check (Correct Verdict)': 75.0,
}

# Component Latency (Seconds - time for one typical unit of work)
# These are highly speculative and depend heavily on hardware/input size
latency_data = {
    # Time to score one frame/small batch (CPU estimate)
    'DinoV2 (Frame Score)': 0.15,
    # Time to transcribe a 10s audio chunk (CPU estimate)
    'Whisper (10s Chunk)': 2.5,
    # Time for text analysis models on a ~10s transcription
    'Text Analyze (Chunk Text)': 0.6,
    # Time for full fact-check pipeline (API+RAG+LLM) per claim (HIGHLY VARIABLE)
    'Fact Check (Claim Pipeline)': 6.0,
    # Target cycle time for stream processing (download + analyze buffer)
    'Stream Cycle Target': 10.0, # e.g., chunk duration
}

# Processing Time (Seconds - time for a larger task)
# Again, highly speculative estimates for a hypothetical ~1-minute video upload
processing_time_data = {
    # Scoring all frames
    'DinoV2 (Upload Video)': 15.0,
     # Transcribing full audio
    'Whisper (Upload Video)': 45.0,
    # Analyzing full transcription
    'Text Analyze (Upload Video)': 25.0,
     # Fact-checking all claims
    'Fact Check (Upload Video)': 90.0,
    # Estimated total time for the /api/video/analyze endpoint
    'Total Upload Analysis': 180.0,
}

# --- Plotting Functions ---

output_dir = "performance_graphs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def plot_bar_chart(data, title, xlabel, ylabel, filename, y_limit=None, color='skyblue'):
    """Helper function to create and save a bar chart."""
    labels = list(data.keys())
    values = list(data.values())

    plt.figure(figsize=(12, 7))
    bars = plt.bar(labels, values, color=color)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=30, ha='right', fontsize=10) # Rotate labels slightly
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if y_limit:
        plt.ylim(0, y_limit)

    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}', va='bottom', ha='center', fontsize=10) # Adjust position and format

    plt.tight_layout() # Adjust layout to prevent labels overlapping
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print(f"Saved graph: {filepath}")
    plt.close() # Close the figure to free memory

# --- Generate Graphs ---

print("Generating performance graphs (using placeholder data)...")

# 1. Accuracy Graph
plot_bar_chart(
    data=accuracy_data,
    title='Component Accuracy',
    xlabel='System Component / Task',
    ylabel='Accuracy (%)',
    filename='component_accuracy.png',
    y_limit=100,
    color='mediumseagreen'
)

# 2. Latency Graph
plot_bar_chart(
    data=latency_data,
    title='Component Latency',
    xlabel='Component / Task Unit',
    ylabel='Latency (seconds)',
    filename='component_latency.png',
    # No y-limit needed unless values are very close
    color='lightcoral'
)

# 3. Processing Time Graph
plot_bar_chart(
    data=processing_time_data,
    title='Task Processing Time (~1min Video Upload)',
    xlabel='Task / Endpoint',
    ylabel='Processing Time (seconds)',
    filename='processing_time.png',
    color='cornflowerblue'
)

print(f"Graphs saved in directory: {output_dir}")