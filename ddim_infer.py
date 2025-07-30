# infer_apple2banana_ddim.py

import os
import torch
from PIL import Image
from torchvision import transforms
from diffusers import DDIMScheduler, UNet2DConditionModel

def load_image(path: str, image_size: int) -> torch.Tensor:
    """Load a PIL image, resize, normalize to [-1,1], and return a (1,3,H,W) tensor."""
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),                    # [0,1]
        transforms.Normalize([0.5]*3, [0.5]*3),   # to [-1,1]
    ])
    img = Image.open(path).convert("RGB")
    return preprocess(img).unsqueeze(0)          # shape (1,3,H,W)

def save_image(tensor: torch.Tensor, path: str):
    """Clamp tensor to [-1,1], map to [0,1], convert to PIL and save."""
    img = (tensor.clamp(-1, 1) + 1) / 2          # [0,1]
    arr = (img.cpu().permute(0,2,3,1).numpy() * 255).round().astype("uint8")[0]
    Image.fromarray(arr).save(path)

def main():
    device = "cuda"                       # or "cpu"
    image_size = 256
    num_inference_steps = 50
    eta = 0.0                          # 0.0 = deterministic DDIM
    seed = 42
    model_dir = "checkpoints/apple2banana-final"
    projector_path = "checkpoints/projector.pt"   # saved during training
    input_path = "examples/apple.jpg"
    output_path = "results/apple2banana_ddim.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Load the trained UNet and the auxiliary projector
    unet = UNet2DConditionModel.from_pretrained(model_dir).to(device)
    # This projector was defined during training as: nn.Linear(3, 512)
    projector = torch.nn.Linear(3, unet.config.cross_attention_dim).to(device)
    projector.load_state_dict(torch.load(projector_path, map_location=device))
    projector.eval()

    # 2. Create & configure the DDIM scheduler
    ddim = DDIMScheduler(
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        num_train_timesteps=1000,  # must match training
        clip_sample=False,
        set_alpha_to_one=False,
    )
    ddim.set_timesteps(num_inference_steps, device=device)

    # 3. Load and embed the conditioning image
    cond_img = load_image(input_path, image_size).to(device)  # (1,3,H,W)
    B, C, H, W = cond_img.shape
    # Flatten spatial dims: (1,3,H,W) → (1,3,H*W) → (1,H*W,3)
    seq = cond_img.view(B, C, H*W).permute(0, 2, 1)
    # Project to cross_attention_dim: (1,H*W,3) → (1,H*W,512)
    cond_emb = projector(seq)

    # 4. Initialize random noise to start diffusion
    img = torch.randn((1, 3, image_size, image_size), device=device)
    gen = torch.Generator(device=device).manual_seed(seed)

    # 5. DDIM sampling loop
    for t in ddim.timesteps:
        # Predict the noise residual with your conditional UNet
        noise_pred = unet(img, t, encoder_hidden_states=cond_emb).sample
        # Take one DDIM step
        step = ddim.step(noise_pred, t, img, eta=eta, generator=gen)
        img = step.prev_sample

    # 6. Post‑process and save the output
    save_image(img, output_path)
    print(f"✅ Saved translated image to {output_path}")

if __name__ == "__main__":
    main()
