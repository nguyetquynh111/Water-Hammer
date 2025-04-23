# Vessel Analysis Application

A Streamlit application for analyzing vessel images to detect patterns and trends.

## Features

- Upload ZIP files containing vessel images
- Process and analyze the images
- Display results in an interactive dashboard
- Visualize window heatmaps, overall trends, and selected windows

## Installation

### Using Docker

1. Build the Docker image:
   ```
   docker build -t vessel-analysis .
   ```

2. Run the container:
   ```
   docker run -p 8501:8501 vessel-analysis
   ```

3. Open your browser and navigate to `http://localhost:8501`

### Local Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd vessel-analysis
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

4. Open your browser and navigate to the URL shown in the terminal (typically `http://localhost:8501`)

## Usage

1. Prepare a ZIP file containing your vessel images
2. Upload the ZIP file using the file uploader
3. Click 'Process Images' to analyze the data
4. View the results displayed below

## Outputs

- **Window Heatmap**: Shows the distribution of pixel values across windows
- **Overall Trend**: Displays the average pixel value over time
- **Selected Windows**: Highlights the windows selected for analysis

## Requirements

- Python 3.9+
- See `requirements.txt` for Python dependencies # Water-Hammer
# Water-Hammer
