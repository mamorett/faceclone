import cv2
import torch
import numpy as np
from PIL import Image
import os
import argparse
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers import StableDiffusionXLPipeline

from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps
from diffusers.schedulers import DDPMScheduler, HeunDiscreteScheduler, KarrasVeScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler


class FaceDetector:
    def __init__(self, yolo_model_path, target_enlargement=100):
        self.model = YOLO(yolo_model_path)
        self.target_enlargement = target_enlargement
    
    def predict(self, image_path):
        # Run inference on the source
        results = self.model(image_path)
        
        # Load the original image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return None

        # Process results
        if len(results) > 0 and len(results[0].boxes) > 0:
            # Get the first detected face
            boxes = results[0].boxes
            if boxes.xyxy.shape[0] == 0:
                print("No faces detected in the image")
                return None
                
            # Get the box with highest confidence
            conf_vals = boxes.conf.cpu().numpy()
            best_idx = np.argmax(conf_vals)
            box = boxes.xyxy[best_idx].cpu().numpy()
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box)
            
            # Calculate original width and height
            width = x2 - x1
            height = y2 - y1
            
            # Find the larger dimension
            max_dim = max(width, height)
            
            # Calculate the enlargement needed (we'll use 200px as a base and adjust)
            target_enlargement = self.target_enlargement
            total_size = max_dim + 2 * target_enlargement  # Total size of the square crop
            
            # Calculate center of the bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Calculate new coordinates for square crop
            half_size = total_size // 2
            x1_new = max(0, center_x - half_size)
            y1_new = max(0, center_y - half_size)
            x2_new = min(img.shape[1], center_x + half_size)
            y2_new = min(img.shape[0], center_y + half_size)
            
            # Adjust if we hit image boundaries
            actual_width = x2_new - x1_new
            actual_height = y2_new - y1_new
            final_size = max(actual_width, actual_height)
            
            # Recalculate to ensure square crop
            if actual_width < final_size:
                x1_new = max(0, x2_new - final_size)
            elif actual_height < final_size:
                y1_new = max(0, y2_new - final_size)
            
            x2_new = min(img.shape[1], x1_new + final_size)
            y2_new = min(img.shape[0], y1_new + final_size)
            
            # Crop the face region
            face_crop = img[y1_new:y2_new, x1_new:x2_new]
            
            # Convert to PIL Image
            face_crop_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            
            print(f"Face detected")
            print(f"Original box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            print(f"Square crop: x1={x1_new}, y1={y1_new}, x2={x2_new}, y2={y2_new}")
            print(f"Crop size: {final_size}x{final_size}")
            
            return face_crop_pil
        else:
            print("No faces detected in the image")
            return None


def resize_img(input_image, max_side=1024, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image

def get_output_filename(output_dir, input_path, index=None):
    """Generate output filename with progressive numbering to prevent overwriting"""
    input_name = Path(input_path).stem
    output_base = os.path.join(output_dir, input_name)
    
    if index is not None:
        # For multiple images in one run
        base_name = f"{output_base}_{index:03d}"
    else:
        # For single image case
        base_name = output_base
    
    # Check if the base filename already exists and find the next available number
    output_file = f"{base_name}.png"
    if not os.path.exists(output_file):
        return output_file
    
    # If file exists, find the next available number
    counter = 1
    while True:
        if index is not None:
            # For multiple images, include both index and counter
            output_file = f"{output_base}_{index:03d}_{counter:03d}.png"
        else:
            # For single image, just use counter
            output_file = f"{output_base}_{counter:03d}.png"
        if not os.path.exists(output_file):
            return output_file
        counter += 1


def process_image(input_path, output_dir, pipe, app, prompt, face_detector=None, 
                 skip_yolo=False, face_image=None, generator=None, num_images=1, 
                 progress_bar=None):
    """Process a single image and save multiple results"""
    if face_image is None:
        face_image = load_image(input_path)
    
    # Use YOLO detection unless skipped
    if not skip_yolo and face_detector is not None:
        face_image = face_detector.predict(input_path)
        if face_image is None:
            if progress_bar:
                progress_bar.write(f"Skipping {Path(input_path).name} - no face detection")
            else:
                print(f"Skipping {input_path} due to no face detection")
            return
    else:
        if progress_bar:
            progress_bar.write(f"Using full original image for {Path(input_path).name} (YOLO skipped)")
        else:
            print(f"Using full original image for {input_path} (YOLO skipped)")

    face_image = resize_img(face_image)
    
    face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    if not face_info:
        if progress_bar:
            progress_bar.write(f"No face detected by InsightFace in: {Path(input_path).name}")
        else:
            print(f"No face detected by InsightFace in: {input_path}")
        return
    
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
    face_emb = face_info['embedding']
    face_kps = draw_kps(face_image, face_info['kps'])

    n_prompt = "(hands:1.2), (arms:1.2), upper arm, hand, (veil:1.2), cap, headset, (lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured)"

    # Create a nested progress bar for image generation if generating multiple images
    if num_images > 1:
        generation_desc = f"Generating {num_images} images for {Path(input_path).name}"
        if progress_bar:
            progress_bar.write(generation_desc)
    
    images = pipe(
        prompt=prompt,
        negative_prompt=n_prompt,
        image_embeds=face_emb,
        image=face_kps,
        controlnet_conditioning_scale=0.8,
        ip_adapter_scale=0.8,
        num_inference_steps=40,
        guidance_scale=2,
        generator=generator,
        num_images_per_prompt=num_images,
    ).images

    # Save images with progress tracking
    save_progress = tqdm(images, desc=f"Saving images for {Path(input_path).name}", 
                        leave=False, disable=(num_images == 1))
    
    for i, image in enumerate(save_progress):
        output_file = get_output_filename(output_dir, input_path, i if num_images > 1 else None)
        image.save(output_file, format="PNG")
        
        if progress_bar:
            progress_bar.write(f"Saved: {Path(output_file).name}")
        else:
            print(f"Saved result to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process image(s) with InstantID pipeline")
    parser.add_argument("input", help="Path to input image or directory")
    parser.add_argument("--output", "-o", default=".", help="Output directory (default: current directory)")
    parser.add_argument("--prompt", "-p", 
                       default="girl, blonde hairs, blue eyes, cinematic, highly detailed, 4k, high resolution, color photo",
                       help="Prompt for image generation")
    parser.add_argument("--num-images", "-n", type=int, default=1,
                       help="Number of images to generate for each input (default: 1)")
    parser.add_argument("--enlargement", "-e", type=int, default=100,
                       help="Face crop enlargement value in pixels (default: 100)")
    parser.add_argument("--skip-yolo", action="store_true",
                       help="Skip YOLO face detection and use full original image")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    print("Loading models...")
    
    # Initialize models with progress indication
    with tqdm(total=5, desc="Loading models") as model_pbar:
        app = FaceAnalysis(name='antelopev2', root='./', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        model_pbar.update(1)
        model_pbar.set_description("Loaded FaceAnalysis")

        face_adapter = './checkpoints/ip-adapter.bin'
        controlnet_path = './checkpoints/ControlNetModel'
        checkpoint_path = './realvisxlV50_v50Bakedvae.safetensors'

        controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.bfloat16)
        model_pbar.update(1)
        model_pbar.set_description("Loaded ControlNet")
        
        base_pipe = StableDiffusionXLPipeline.from_single_file(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        )
        model_pbar.update(1)
        model_pbar.set_description("Loaded base pipeline")

        pipe = StableDiffusionXLInstantIDPipeline(
            vae=base_pipe.vae,
            text_encoder=base_pipe.text_encoder,
            text_encoder_2=base_pipe.text_encoder_2,
            unet=base_pipe.unet,
            scheduler=base_pipe.scheduler,
            controlnet=controlnet,
            tokenizer=base_pipe.tokenizer,
            tokenizer_2=base_pipe.tokenizer_2,
        )

        scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++"
        )
        pipe.scheduler = scheduler

        pipe.cuda()
        pipe.load_ip_adapter_instantid(face_adapter)
        pipe.vae.enable_tiling()
        pipe.enable_model_cpu_offload()
        model_pbar.update(1)
        model_pbar.set_description("Configured pipeline")

        generator = torch.Generator(device="cuda").manual_seed(torch.randint(0, 2**32, (1,)).item())

        # Initialize YOLO face detector only if not skipping
        face_detector = None if args.skip_yolo else FaceDetector('yolov11n-face.pt', target_enlargement=args.enlargement)
        model_pbar.update(1)
        model_pbar.set_description("Models loaded successfully")

    print("Starting image processing...")

    input_path = os.path.abspath(args.input)
    if os.path.isdir(input_path):
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        image_files = [f for f in os.listdir(input_path) if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print("No image files found in the directory!")
            return
        
        # Main progress bar for processing all images
        with tqdm(image_files, desc="Processing images", unit="image") as pbar:
            for image_file in pbar:
                pbar.set_description(f"Processing {image_file}")
                full_path = os.path.join(input_path, image_file)
                process_image(full_path, output_dir, pipe, app, args.prompt, 
                             face_detector=face_detector, skip_yolo=args.skip_yolo,
                             generator=generator, num_images=args.num_images,
                             progress_bar=pbar)
    else:
        # Single image processing
        print(f"Processing single image: {Path(input_path).name}")
        process_image(input_path, output_dir, pipe, app, args.prompt, 
                     face_detector=face_detector, skip_yolo=args.skip_yolo,
                     generator=generator, num_images=args.num_images)

    print("Processing complete!")

if __name__ == "__main__":
    main()
