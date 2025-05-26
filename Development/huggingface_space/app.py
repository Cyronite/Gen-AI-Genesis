import os
import sys
import base64
import tempfile
import shutil
import zipfile
import io
import subprocess
import gradio as gr
from pathlib import Path

# Ensure we have manim installed
try:
    import manim
    print(f"Manim version: {manim.__version__}")
except ImportError:
    print("Installing manim...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "manim"])
    import manim
    print(f"Manim version: {manim.__version__}")

def extract_zip(zip_data, extract_dir):
    """Extract a base64-encoded zip file to the specified directory."""
    try:
        zip_bytes = base64.b64decode(zip_data)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zip_ref:
            zip_ref.extractall(extract_dir)
        return True, f"Extracted {len(zip_ref.namelist())} files"
    except Exception as e:
        return False, f"Error extracting zip: {str(e)}"

def render_manim_scene(project_zip, main_file, scene_name, quality):
    """
    Render a Manim scene from the provided project files.
    
    Args:
        project_zip (str): Base64-encoded zip file containing the project
        main_file (str): Path to the main Python file within the project
        scene_name (str): Name of the scene class to render
        quality (str): Quality setting (low_quality, medium_quality, high_quality)
    
    Returns:
        tuple: (status, result)
            - status: 'success' or 'error'
            - result: base64-encoded video on success, error message on failure
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Extract the zip file
            success, message = extract_zip(project_zip, temp_dir)
            if not success:
                return "error", message
            
            # Quality settings
            quality_flags = {
                "low_quality": "-ql",
                "medium_quality": "-qm",
                "high_quality": "-qh",
            }
            
            # Ensure main file path is correct
            main_file_path = os.path.join(temp_dir, main_file)
            if not os.path.exists(main_file_path):
                return "error", f"Main file not found: {main_file}"
            
            # Create output directory
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # Build the command
            quality_flag = quality_flags.get(quality, "-qm")  # Default to medium quality
            cmd = [
                sys.executable, "-m", "manim",
                quality_flag,
                "-o", os.path.join(output_dir, "video.mp4"),
                main_file_path,
                scene_name
            ]
            
            # Execute the command
            print(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=temp_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Check if rendering was successful
            if result.returncode != 0:
                print(f"Command output: {result.stdout}")
                print(f"Command error: {result.stderr}")
                return "error", f"Error rendering scene: {result.stderr}"
            
            # Find the output video
            video_files = list(Path(output_dir).glob("*.mp4"))
            if not video_files:
                return "error", "No video file was generated"
            
            # Return the video file as base64
            video_path = video_files[0]
            with open(video_path, "rb") as f:
                video_data = f.read()
            
            video_base64 = base64.b64encode(video_data).decode("utf-8")
            return "success", video_base64
            
        except Exception as e:
            return "error", f"Unexpected error: {str(e)}"

# Create Gradio interface
def gradio_render(project_zip, main_file, scene_name, quality):
    if not project_zip:
        return "error", "No project file provided"
    
    status, result = render_manim_scene(project_zip, main_file, scene_name, quality)
    return status, result

demo = gr.Interface(
    fn=gradio_render,
    inputs=[
        gr.Textbox(label="Project ZIP (base64)"),
        gr.Textbox(label="Main File"),
        gr.Textbox(label="Scene Name"),
        gr.Radio(
            choices=["low_quality", "medium_quality", "high_quality"],
            value="medium_quality",
            label="Quality"
        ),
    ],
    outputs=[
        gr.Textbox(label="Status"),
        gr.Textbox(label="Result (base64 video or error message)")
    ],
    title="Manim Renderer",
    description="Render Manim scenes from uploaded project files."
)

if __name__ == "__main__":
    demo.launch()