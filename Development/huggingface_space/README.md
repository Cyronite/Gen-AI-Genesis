# Manim Renderer Space

This Hugging Face Space provides an API for rendering [Manim](https://github.com/ManimCommunity/manim) animations without requiring a local installation.

## How It Works

This Space exposes an API endpoint that accepts:
- A base64-encoded ZIP file containing your Manim project
- The main Python file name
- The Scene class name to render
- Quality settings (low, medium, high)

It returns:
- A base64-encoded MP4 file of the rendered animation
- Status information

## API Usage

You can call this Space's API endpoint programmatically:

```python
import requests
import base64
import json

# Encode your project as ZIP file in base64
with open("project.zip", "rb") as f:
    project_zip_base64 = base64.b64encode(f.read()).decode("utf-8")

# Call the API
response = requests.post(
    "https://your-username-manim-renderer.hf.space/api/predict",
    json={
        "data": [
            project_zip_base64,
            "main.py",
            "MyScene",
            "medium_quality"
        ]
    }
)

result = response.json()
status = result["data"][0]
video_base64 = result["data"][1]

# Save the video
if status == "success":
    with open("output.mp4", "wb") as f:
        f.write(base64.b64decode(video_base64))
```

## Quick Start

1. ZIP your Manim project
2. Convert the ZIP file to base64
3. Make an API request
4. Decode the returned video

## Limitations

- Maximum project size: 100MB
- Maximum rendering time: 5 minutes
- Some advanced Manim features may not work

## Troubleshooting

If you encounter errors:
1. Check that your Manim code works locally
2. Ensure all required files are included in the ZIP
3. Check for syntax errors in your code

## Credits

Built with Gradio and Manim Community Edition.