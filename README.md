# Face Detection and Image Generation Pipeline

This project implements a pipeline for detecting faces in images using a YOLO-based detector and processing them with an InstantID pipeline powered by Stable Diffusion XL. The pipeline extracts faces, resizes them, and uses embeddings to generate high-quality images based on user-defined prompts.

---

## Features
- **YOLO-based Face Detection**: Uses a YOLO model to detect faces in images.
- **Face Cropping and Resizing**: Extracts and resizes faces with adjustable enlargement settings.
- **Face Embedding Extraction**: Uses InsightFace to generate facial embeddings for improved identity preservation.
- **Stable Diffusion XL Image Generation**: Generates images based on facial embeddings and a user-defined prompt.
- **Customizable Image Generation Settings**: Supports control over face detection, cropping, resizing, and generation settings.

---

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- PyTorch (with CUDA if using GPU)
- OpenCV
- Diffusers
- InsightFace
- Ultralytics YOLO

### Install Dependencies
Run the following command to install the necessary dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage
### Running the Script
To process a single image or a directory of images, run:
```bash
python script.py <input_path> --output <output_directory> --prompt "Your prompt here"
```

### Arguments
| Argument | Description |
|----------|-------------|
| `input` | Path to an image file or directory containing images. |
| `--output`, `-o` | Output directory for generated images. Default: current directory. |
| `--prompt`, `-p` | Prompt for image generation. Default: "girl, blonde hair, blue eyes, cinematic, highly detailed, 4k, high resolution, color photo". |
| `--num-images`, `-n` | Number of images to generate per input image. Default: 1. |
| `--enlargement`, `-e` | Enlargement factor for face cropping. Default: 100 pixels. |
| `--skip-yolo` | Skip YOLO face detection and use the full original image. |

---

## Implementation Details
### 1. **Face Detection**
- Uses a YOLO-based face detector to find the largest face in the image.
- Extracts bounding box coordinates and enlarges the crop to maintain a square aspect ratio.
- Converts the cropped face region to a PIL image.

### 2. **Face Processing with InsightFace**
- Extracts facial embeddings using InsightFace.
- Draws key points on the detected face for better alignment.

### 3. **Image Generation using Stable Diffusion XL**
- Uses a ControlNet model for conditioning on facial features.
- Generates images with the specified prompt and negative prompt.
- Uses a DPMSolverMultistepScheduler for efficient inference.

---

## Example Usage
### Processing a Single Image
```bash
python script.py /path/to/image.jpg --output ./results --prompt "A futuristic cyberpunk character"
```

### Processing a Directory of Images
```bash
python script.py /path/to/images/ --output ./results --num-images 3
```

### Skipping Face Detection
```bash
python script.py /path/to/image.jpg --output ./results --skip-yolo
```

---

## Output Files
Generated images are saved in the specified output directory with progressive numbering to prevent overwriting. The format follows:
```
output_directory/
    input_image_001.png
    input_image_002.png
    input_image_003.png
```

---

## Model Checkpoints and Paths
Ensure you have the following model checkpoints downloaded and placed correctly:
- **YOLO Model**: `yolov11n-face.pt`
- **Stable Diffusion XL Checkpoint**: `.checkpoints/realvisxlV50_v50Bakedvae.safetensors`
- **ControlNet Model**: `./checkpoints/ControlNetModel`
- **IP Adapter Checkpoint**: `./checkpoints/ip-adapter.bin`

---

## Performance Considerations
- For **GPU acceleration**, ensure you have a compatible CUDA setup and PyTorch installed with GPU support.
- Using a higher number of inference steps (`num_inference_steps`) will improve image quality but increase generation time.
- Face enlargement (`--enlargement`) can be adjusted to capture more facial context.

---

## Troubleshooting
### 1. **No Faces Detected**
- Ensure the input images contain clear, front-facing faces.
- Try increasing the enlargement factor (`--enlargement`).

### 2. **CUDA Out of Memory Error**
- Reduce the batch size or `num-images` parameter.
- Use model CPU offloading (`pipe.enable_model_cpu_offload()`).

### 3. **Generated Images Do Not Resemble Input Face**
- Increase the IP adapter scale (`ip_adapter_scale`).
- Ensure facial embeddings are correctly extracted from InsightFace.

---

## Acknowledgments
This project leverages:
- **Ultralytics YOLO** for face detection.
- **InsightFace** for facial embedding extraction.
- **Stable Diffusion XL** for high-quality image generation.
- **Diffusers Library** for efficient deep learning inference.

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.

---

## Contact
For questions or contributions, please create an issue or pull request on GitHub.

