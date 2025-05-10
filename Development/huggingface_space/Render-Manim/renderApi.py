import os
import sys
import base64
import tempfile
import gradio as gr
from renderVid import runCommand

def gradio_render(project_zip, main_file, scene_name, quality):
    """
    A simplified version that just echoes back the input parameters without rendering.

    Args:
        project_zip (str): Base64-encoded zip file containing the project
        main_file (str): Path to the main Python file within the project
        scene_name (str): Name of the scene class to render
        quality (str): Quality setting (low_quality, medium_quality, high_quality)
    """
    # Log inputs for debugging (excluding the potentially large project_zip)
    print(f"Received request - Main file: {main_file}, Scene: {scene_name}, Quality: {quality}")
    print(f"ZIP data length: {len(project_zip) if project_zip else 0} characters")

    # First extract the project files and set up directories
    dir = decodeFile(project_zip)
    
    # Create vids directory if it doesn't exist
    vids_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vids")
    if not os.path.exists(vids_dir):
        os.makedirs(vids_dir, exist_ok=True)
    
    # Generate a timestamp for the output video filename
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{scene_name}_{timestamp}.mp4"
    output_path = os.path.join(vids_dir, output_filename)
    
    # Define the media directory - where manim will output files
    media_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Command to render MP4 without GUI
    # Different Manim versions have different CLI arguments, so we'll try multiple approaches
    print(f"Rendering {scene_name} to {output_path}")
    
    # Check which version of Manim is installed
    version_cmd = ["manimgl", "--version"]
    version_success, version_output, _ = runCommand(version_cmd, working_dir=dir)
    
    # Default to manimgl style arguments
    cmd = [
        "manimgl", 
        main_file, 
        scene_name, 
        "-o",  # Output to file without GUI
        f"--{quality}",
        "--media_dir", media_dir
    ]
    
    # If we can detect manim (not manimgl), adjust the commands
    if not version_success:
        print("Could not determine Manim version. Trying manim command instead of manimgl...")
        # Try manim (instead of manimgl) command format
        cmd = [
            "manim",
            main_file,
            scene_name,
            "-p",  # Preview file after rendering
            f"-{quality[0]}",  # e.g., -l for low_quality, -m for medium_quality, -h for high_quality
            "--media_dir", media_dir
        ]
    
    print(f"Running command: {' '.join(cmd)}")
    success, stdout, stderr = runCommand(cmd, working_dir=dir)
    
    # If first attempt failed, try alternate command structure
    if not success:
        print("First rendering attempt failed. Trying alternate command...")
        if "manimgl" in cmd[0]:
            # Try manim instead
            alt_cmd = [
                "manim",
                main_file,
                scene_name,
                "-p",  # Preview file after rendering
                f"-{quality[0]}",  # e.g., -l for low_quality
                "--media_dir", media_dir
            ]
        else:
            # Try manimgl instead
            alt_cmd = [
                "manimgl", 
                main_file, 
                scene_name, 
                "--write_to_movie",
                f"--{quality}",
                "--media_dir", media_dir
            ]
            
        print(f"Running alternate command: {' '.join(alt_cmd)}")
        success, stdout, stderr = runCommand(alt_cmd, working_dir=dir)
    print(f"Command output:\n{stdout}\n\nErrors:\n{stderr}")
    
    # After rendering, look for the video file in the media output
    if success:
        status = "success"
        message = (
            f"Video rendered successfully:\n"
            f"- Main file: {main_file}\n"
            f"- Scene name: {scene_name}\n"
            f"- Quality: {quality}\n"
            f"- Output directory: {media_dir}"
        )
        
        # Manim typically outputs to a videos directory inside the media_dir
        # Different versions put videos in different locations, so check multiple paths
        potential_video_paths = [
            # manimgl output paths
            os.path.join(media_dir, "videos", f"{scene_name}.mp4"),
            os.path.join(media_dir, "videos", main_file.replace(".py", ""), quality, f"{scene_name}.mp4"),
            os.path.join(media_dir, "videos", scene_name, quality, f"{scene_name}.mp4"),
            # manim (not manimgl) output paths
            os.path.join(media_dir, "videos", main_file.replace(".py", ""), "480p15", f"{scene_name}.mp4"),
            os.path.join(media_dir, "videos", main_file.replace(".py", ""), "720p30", f"{scene_name}.mp4"),
            os.path.join(media_dir, "videos", main_file.replace(".py", ""), "1080p60", f"{scene_name}.mp4"),
            # Try with file name
            os.path.join(media_dir, "videos", f"{main_file.replace('.py', '')}_{scene_name}.mp4")
        ]
        
        # Also look for .mp4 files that were created in the last minute
        import glob
        import time
        current_time = time.time()
        for mp4_file in glob.glob(os.path.join(media_dir, "**", "*.mp4"), recursive=True):
            if os.path.getmtime(mp4_file) > current_time - 60:  # Created in the last minute
                potential_video_paths.append(mp4_file)
        
        found_video = None
        for path in potential_video_paths:
            if os.path.exists(path):
                found_video = path
                break
        
        if found_video:
            # Copy the file to our vids directory with the timestamped name
            import shutil
            shutil.copy2(found_video, output_path)
            print(f"Video copied to: {output_path}")
            
            # Return an embedded video player
            # Get relative path for display
            rel_path = os.path.relpath(output_path, os.path.dirname(os.path.abspath(__file__)))
            placeholder_html = f"""
            <div style="width: 100%; padding: 10px;">
                <p>Video rendered to: vids/{output_filename}</p>
                <video width="100%" height="auto" controls>
                    <source src="file://{output_path}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <p style="font-size:0.8em; color:#666;">Note: The video file has been saved to the 'vids' directory.</p>
            </div>
            """
        else:
            status = "error"
            message += "\n\nHowever, no output video file was found."
            placeholder_html = """
            <div style="width: 100%; height: 300px; background-color: #fff0e0; display: flex;
                        justify-content: center; align-items: center; border-radius: 5px;">
                <p style="text-align: center; color: #cc7700;">Rendering succeeded but video file not found.<br>
                Check the logs for more information.</p>
            </div>
            """
    else:
        status = "error"
        message = (
            f"Video rendering failed:\n"
            f"- Main file: {main_file}\n"
            f"- Scene name: {scene_name}\n"
            f"- Quality: {quality}\n"
            f"- Error: {stderr}"
        )
        
        # Return an error placeholder
        placeholder_html = """
        <div style="width: 100%; height: 300px; background-color: #ffe0e0; display: flex;
                    justify-content: center; align-items: center; border-radius: 5px;">
            <p style="text-align: center; color: #cc0000;">Video rendering failed.<br>
            Check the logs for more information.</p>
        </div>
        """

    # Note: All rendering logic has been moved above

    return status, message, placeholder_html


def decodeFile(projectZip):
    """
    Decodes a base64-encoded ZIP file, extracts it, and saves its contents to a 'tmp' directory
    next to the script file.

    Args:
        projectZip (str): Base64-encoded ZIP file

    Returns:
        str: Path to the 'tmp' directory containing the extracted files
    """
    try:
        # Create a 'tmp' directory next to the script file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tmp_dir = os.path.join(script_dir, "tmp")

        # Create the directory if it doesn't exist, clear it if it does
        if os.path.exists(tmp_dir):
            import shutil
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        print(f"Created/cleared directory: {tmp_dir}")

        # Create a temporary file for the decoded ZIP
        temp_zip_path = os.path.join(tmp_dir, "temp_project.zip")

        # Decode the base64 string and write it to the ZIP file
        decoded_zip = base64.b64decode(projectZip)
        with open(temp_zip_path, 'wb') as temp_zip:
            temp_zip.write(decoded_zip)
        print(f"Decoded ZIP file saved to: {temp_zip_path}")

        # Extract the ZIP file to the tmp directory
        import zipfile
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)
            print(f"ZIP file extracted to: {tmp_dir}")

            # List extracted files for debugging
            files = os.listdir(tmp_dir)
            print(f"Extracted files: {files}")

        # Clean up the temporary ZIP file
        os.unlink(temp_zip_path)

        return tmp_dir

    except Exception as e:
        print(f"Error decoding/extracting ZIP file: {str(e)}")
        return None



# Define the interface
with gr.Blocks(title="Manim Renderer") as demo:
    gr.Markdown("""
    # Manim Scene Renderer

    This application renders Manim scenes from uploaded project files.

    ## Instructions:
    1. Convert your project files to a base64-encoded ZIP file
    2. Enter the relative path to the main Python file within the ZIP
    3. Enter the name of the scene class to render
    4. Select the quality setting
    5. Click Submit to render your scene

    ## API Usage Documentation
    This application exposes an API for programmatic access. Check the documentation below.
    """)

    with gr.Row():
        with gr.Column():
            project_zip_input = gr.Textbox(
                label="Project ZIP (base64)",
                lines=5,
                placeholder="Paste your base64-encoded ZIP file here"
            )
            main_file_input = gr.Textbox(
                label="Main File Path (relative to ZIP root)",
                placeholder="e.g., main.py",
                value="test.py"
            )
            scene_name_input = gr.Textbox(
                label="Scene Class Name",
                placeholder="e.g., SquareToCircle",
                value="OpeningManimExample"
            )
            quality_input = gr.Radio(
                choices=["low_quality", "medium_quality", "high_quality"],
                value="medium_quality",
                label="Rendering Quality"
            )
            submit_button = gr.Button("Submit")

        with gr.Column():
            status_output = gr.Textbox(label="Status")
            message_output = gr.Textbox(label="Message", lines=5)
            video_output = gr.HTML(label="Rendered Video")

    # Connect the button to the render function
    submit_button.click(
        fn=gradio_render,
        inputs=[project_zip_input, main_file_input, scene_name_input, quality_input],
        outputs=[status_output, message_output, video_output]
    )

    # Add API documentation
    gr.Markdown("""
    ## API Documentation

    ### Using the Gradio Client

    The easiest way to use this API is with the Gradio Client:

    ```python
    from gradio_client import Client

    client = Client("https://your-app-url")
    result = client.predict(
        "BASE64_ZIP_DATA",  # Project ZIP (base64)
        "test.py",          # Main File Path
        "OpeningManimExample", # Scene Class Name
        "medium_quality",   # Rendering Quality
        api_name="/predict"
    )
    status, message, video_html = result
    ```

    ### Using Direct API Calls

    For raw HTTP requests:

    ```
    POST /run/predict
    Content-Type: application/json

    {
        "data": [
            "BASE64_ZIP_DATA",
            "main_file.py",
            "SceneName",
            "quality_setting"
        ]
    }
    ```

    For long-running jobs:

    ```
    POST /submit/predict
    Content-Type: application/json

    {
        "data": [
            "BASE64_ZIP_DATA",
            "main_file.py",
            "SceneName",
            "quality_setting"
        ]
    }
    ```

    This will return a job_id. Then:

    ```
    GET /status/{job_id}
    GET /result/{job_id}   # When status is complete
    ```
    """)

if __name__ == "__main__":
    # Print environment information
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Gradio version: {gr.__version__}")

    demo.queue()  # Enable queuing for better handling of concurrent requests

    # Launch the Gradio app with proper API configuration
    demo.launch(
        server_name="0.0.0.0",  # Listen on all network interfaces
        server_port=7860,       # Use port 7860
        share=False             # Don't create a public link
    )
