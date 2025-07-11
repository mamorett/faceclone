from huggingface_hub import hf_hub_download
import gdown
import os

# download models
hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="ControlNetModel/config.json",
    local_dir="./checkpoints",
)
hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="ControlNetModel/diffusion_pytorch_model.safetensors",
    local_dir="./checkpoints",
)
hf_hub_download(
    repo_id="InstantX/InstantID", filename="ip-adapter.bin", local_dir="./checkpoints"
)
hf_hub_download(
    repo_id="xxxpo13/RealVisXL_5",
    filename="realvisxlV50_v50Bakedvae.safetensors",
    local_dir="./checkpoints",
)



# Download the file
hf_hub_download(
    repo_id="AdamCodd/YOLOv11n-face-detection",
    filename="model.pt",
    local_dir=".",
)

# Rename the file
os.rename("model.pt", "yolov11n-face.pt")


hf_hub_download(
    repo_id="latent-consistency/lcm-lora-sdxl",
    filename="pytorch_lora_weights.safetensors",
    local_dir="./checkpoints",
)
# # download antelopev2
# gdown.download(url="https://drive.google.com/file/d/18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8/view?usp=sharing", output="./models/", quiet=False, fuzzy=True)
# # unzip antelopev2.zip
# os.system("unzip ./models/antelopev2.zip -d ./models/")