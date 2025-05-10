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

    # Simply return the parameters for confirmation
    status = "received"
    message = (
        f"Parameters received successfully:\n"
        f"- Main file: {main_file}\n"
        f"- Scene name: {scene_name}\n"
        f"- Quality: {quality}\n"
        f"- ZIP data length: {len(project_zip) if project_zip else 0} characters"
    )

    # Return a placeholder for the video
    placeholder_html = """
    <div style="width: 100%; height: 300px; background-color: #f0f0f0; display: flex;
                justify-content: center; align-items: center; border-radius: 5px;">
        <p style="text-align: center;">Video rendering placeholder.<br>
        Implement your own rendering logic to replace this.</p>
    </div>
    """

    dir = decodeFile(project_zip)
    print(dir)
    print(f"manimgl {main_file} {scene_name}")
    print(runCommand(["manimgl", main_file, scene_name], working_dir="/Users/sidak/Development/huggingface_space/Render-Manim/tmp"))


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
