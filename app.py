import streamlit as st
import os
import zipfile
import shutil
import logging
from inference import process
import tempfile
import cv2
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure directories
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

# Ensure directories exist with proper permissions
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.chmod(UPLOAD_FOLDER, 0o777)
os.chmod(OUTPUT_FOLDER, 0o777)

# Set page config
st.set_page_config(
    page_title="Vessel Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
    }
    .stImage {
        margin: 1rem 0;
        width: 100%;
    }
    .uploadedFile {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .image-container {
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("Vessel Analysis")
st.markdown("Upload a ZIP file containing vessel images or a video file for analysis.")

# Create tabs for different upload methods
tab1, tab2 = st.tabs(["ZIP File Upload", "Video Upload"])

# Function to extract frames from video
def extract_frames_from_video(video_path, output_folder, video_name):
    """Extract frames from video and save as PNG files."""
    # Create a subfolder for the video frames
    video_folder = os.path.join(output_folder, video_name)
    os.makedirs(video_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error(f"Error opening video file: {video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    st.info(f"Video has {frame_count} frames at {fps} FPS")
    
    # Extract frames
    frame_number = 0
    with st.spinner("Extracting frames from video..."):
        progress_bar = st.progress(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Save frame as PNG
            frame_path = os.path.join(video_folder, f"{video_name}_{frame_number+1}.png")
            cv2.imwrite(frame_path, frame)
            
            frame_number += 1
            progress_bar.progress(min(frame_number / frame_count, 1.0))
    
    cap.release()
    progress_bar.progress(1.0)
    
    st.success(f"Extracted {frame_number} frames to {video_folder}")
    return video_folder

# Tab 1: ZIP File Upload
with tab1:
    st.header("ZIP File Upload")
    st.markdown("Upload a ZIP file containing vessel images for analysis.")
    
    # File uploader for ZIP
    uploaded_zip = st.file_uploader("Choose a ZIP file", type="zip", key="zip_uploader")
    
    if uploaded_zip is not None:
        # Show file details
        file_details = {"Filename": uploaded_zip.name, "FileType": uploaded_zip.type, "FileSize": f"{uploaded_zip.size / 1024 / 1024:.2f} MB"}
        st.write("### File Details")
        for key, value in file_details.items():
            st.write(f"**{key}:** {value}")
        
        # Process button
        if st.button("Process ZIP Images", key="process_zip"):
            with st.spinner("Processing images..."):
                try:
                    # Clear previous uploads and outputs
                    if os.path.exists(UPLOAD_FOLDER):
                        if os.path.isfile(UPLOAD_FOLDER):
                            os.remove(UPLOAD_FOLDER)
                        else:
                            shutil.rmtree(UPLOAD_FOLDER)
                    if os.path.exists(OUTPUT_FOLDER):
                        if os.path.isfile(OUTPUT_FOLDER):
                            os.remove(OUTPUT_FOLDER)
                        else:
                            shutil.rmtree(OUTPUT_FOLDER)
                    
                    # Create fresh directories
                    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
                    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
                    os.chmod(UPLOAD_FOLDER, 0o777)
                    os.chmod(OUTPUT_FOLDER, 0o777)
                    
                    # Save and extract zip file
                    zip_path = os.path.join(UPLOAD_FOLDER, uploaded_zip.name)
                    with open(zip_path, "wb") as buffer:
                        content = uploaded_zip.read()
                        if len(content) > MAX_FILE_SIZE:
                            st.error("File too large. Maximum size is 16MB.")
                            st.stop()
                        buffer.write(content)
                    
                    # Extract zip file
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(UPLOAD_FOLDER)
                    
                    # Remove the zip file after extraction
                    os.remove(zip_path)
                    
                    # Process the extracted files
                    extracted_files = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isdir(os.path.join(UPLOAD_FOLDER, f))]
                    if not extracted_files:
                        st.error("No valid directories found in zip file")
                        st.stop()
                    
                    patient_folder = os.path.join(UPLOAD_FOLDER, extracted_files[0])
                    process(patient_folder)
                    
                    # Verify that output files were generated
                    expected_files = ["window_heatmap.png", "overall_trend.png", "selected_windows.png"]
                    missing_files = [f for f in expected_files if not os.path.exists(os.path.join(OUTPUT_FOLDER, f))]
                    if missing_files:
                        st.error(f"Failed to generate output files: {', '.join(missing_files)}")
                        st.stop()
                    
                    st.success("Processing complete!")
                    
                    # Display results
                    st.write("### Results")
                    
                    # Display images in rows instead of columns
                    st.write("#### Window Heatmap")
                    st.image(os.path.join(OUTPUT_FOLDER, "window_heatmap.png"), use_container_width=True)
                    
                    st.write("#### Overall Trend")
                    st.image(os.path.join(OUTPUT_FOLDER, "overall_trend.png"), use_container_width=True)
                    
                    st.write("#### Selected Windows")
                    st.image(os.path.join(OUTPUT_FOLDER, "selected_windows.png"), use_container_width=True)
                    
                except Exception as e:
                    logger.error(f"Error processing upload: {str(e)}")
                    st.error(f"Error: {str(e)}")

# Tab 2: Video Upload
with tab2:
    st.header("Video Upload")
    st.markdown("Upload a video file to extract frames and analyze them.")
    
    # File uploader for video
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"], key="video_uploader")
    
    if uploaded_video is not None:
        # Show file details
        file_details = {"Filename": uploaded_video.name, "FileType": uploaded_video.type, "FileSize": f"{uploaded_video.size / 1024 / 1024:.2f} MB"}
        st.write("### File Details")
        for key, value in file_details.items():
            st.write(f"**{key}:** {value}")
        
        # Process button
        if st.button("Process Video", key="process_video"):
            with st.spinner("Processing video..."):
                try:
                    # Clear previous uploads and outputs
                    if os.path.exists(UPLOAD_FOLDER):
                        if os.path.isfile(UPLOAD_FOLDER):
                            os.remove(UPLOAD_FOLDER)
                        else:
                            shutil.rmtree(UPLOAD_FOLDER)
                    if os.path.exists(OUTPUT_FOLDER):
                        if os.path.isfile(OUTPUT_FOLDER):
                            os.remove(OUTPUT_FOLDER)
                        else:
                            shutil.rmtree(OUTPUT_FOLDER)
                    
                    # Create fresh directories
                    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
                    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
                    os.chmod(UPLOAD_FOLDER, 0o777)
                    os.chmod(OUTPUT_FOLDER, 0o777)
                    
                    # Save video file
                    video_name = os.path.splitext(uploaded_video.name)[0]
                    video_path = os.path.join(UPLOAD_FOLDER, uploaded_video.name)
                    with open(video_path, "wb") as buffer:
                        content = uploaded_video.read()
                        if len(content) > MAX_FILE_SIZE:
                            st.error("File too large. Maximum size is 16MB.")
                            st.stop()
                        buffer.write(content)
                    
                    # Extract frames from video
                    frames_folder = extract_frames_from_video(video_path, UPLOAD_FOLDER, video_name)
                    
                    if frames_folder:
                        # Process the extracted frames
                        process(frames_folder)
                        
                        # Verify that output files were generated
                        expected_files = ["window_heatmap.png", "overall_trend.png", "selected_windows.png"]
                        missing_files = [f for f in expected_files if not os.path.exists(os.path.join(OUTPUT_FOLDER, f))]
                        if missing_files:
                            st.error(f"Failed to generate output files: {', '.join(missing_files)}")
                            st.stop()
                        
                        st.success("Processing complete!")
                        
                        # Display results
                        st.write("### Results")
                        
                        # Display images in rows instead of columns
                        st.write("#### Window Heatmap")
                        st.image(os.path.join(OUTPUT_FOLDER, "window_heatmap.png"), use_container_width=True)
                        
                        st.write("#### Overall Trend")
                        st.image(os.path.join(OUTPUT_FOLDER, "overall_trend.png"), use_container_width=True)
                        
                        st.write("#### Selected Windows")
                        st.image(os.path.join(OUTPUT_FOLDER, "selected_windows.png"), use_container_width=True)
                    
                except Exception as e:
                    logger.error(f"Error processing video: {str(e)}")
                    st.error(f"Error: {str(e)}")

# Add a sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This application analyzes vessel images to detect patterns and trends.
    
    ### How to use:
    1. Choose between ZIP file or video upload
    2. Upload your file using the file uploader
    3. Click 'Process' to analyze the data
    4. View the results displayed below
    
    ### Outputs:
    - **Window Heatmap**: Shows the distribution of pixel values across windows
    - **Overall Trend**: Displays the average pixel value over time
    - **Selected Windows**: Highlights the windows selected for analysis
    """)
    
    st.header("Contact")
    st.markdown("For questions or support, please contact the development team.") 