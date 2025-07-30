import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from diffusers import DDPMScheduler, UNet2DConditionModel
from tqdm import tqdm

class PairedImageDataset(Dataset):
    """Loads paired images: for each apple_xxx.jpg there is banana_xxx.jpg"""
    def __init__(self, rootA_dir, rootB_dir, split="train", image_size=256):
        self.rootA = os.path.join(rootA_dir, f'{split}A')
        self.rootB = os.path.join(rootB_dir, f'{split}B')
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),               # [0,1]
            transforms.Normalize([0.5]*3, [0.5]*3)  # to [-1,1]
        ])
        # assume files named apple_0001.jpg, banana_0001.jpg, …
        self.idsA = sorted({fn.split('_')[1].split('.')[0]
                           for fn in os.listdir(self.rootA)})
        self.idsB = sorted({fn.split('_')[1].split('.')[0]
                           for fn in os.listdir(self.rootB)})
    def __len__(self):
        return len(self.idsA)
    
    def __getitem__(self, idx):
        idx_strA = self.idsA[idx]
        idx_strB = self.idsB[idx]
        apple = Image.open(os.path.join(self.rootA, f"n07740461_{idx_strA}.jpg")).convert("RGB")
        banana = Image.open(os.path.join(self.rootB, f"n07749192_{idx_strB}.jpg")).convert("RGB")
        return self.transform(apple), self.transform(banana)

def main():
    rootA_dir = "/data1/home/baehyunsung/hynix/pytorch-CycleGAN-and-pix2pix/datasets/apple2orange"
    rootB_dir = "/data1/home/baehyunsung/hynix/pytorch-CycleGAN-and-pix2pix/datasets/apple2orange"
    device = "cuda"
    image_size = 128
    batch_size = 1
    epochs = 5
    lr = 1e-4
    num_train_steps = 1000

    # 1. Dataset & Dataloader
    ds = PairedImageDataset(rootA_dir, rootB_dir, split="train", image_size=image_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    # 2. Model: conditional U‑Net
    model = UNet2DConditionModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        cross_attention_dim=512  # size of your “condition” embedding
    ).to(device)

    projector = torch.nn.Linear(3, 512).to(device)

    # 3. Noise scheduler (for training)
    noise_scheduler = DDPMScheduler(
        beta_start=0.0001, beta_end=0.02,
        beta_schedule="linear",
        num_train_timesteps=num_train_steps
    )

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(projector.parameters()),
        lr=lr
    )

    # 4. Training loop
    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,} ({total_params/1e6:.1f} M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f} M)")

    for epoch in range(epochs):
        for step, (apple_imgs, banana_imgs) in tqdm(enumerate(dl), desc="Training Progress.."):
            apple_imgs   = apple_imgs.to(device)    # condition
            banana_imgs  = banana_imgs.to(device)   # target

            B, C, H, W = apple_imgs.shape
            # (B,3,H,W) → (B,3,H*W) → (B,H*W,3)
            apple_seq = apple_imgs.view(B, C, H*W).permute(0, 2, 1)
            # (B,H*W,3) → (B,H*W,512)
            cond_emb  = projector(apple_seq)

            # sample random timesteps for each image
            t = torch.randint(0, noise_scheduler.num_train_timesteps,
                              (batch_size,), device=device).long()
            # sample noise
            noise = torch.randn_like(banana_imgs)
            # add noise to target banana_imgs
            noisy = noise_scheduler.add_noise(banana_imgs, noise, t)

            # forward: predict the noise given the noisy image + *embedded* apple
            noise_pred = model(
                noisy, t,
                encoder_hidden_states=cond_emb
            ).sample

            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 100 == 0:
                print(f"[Epoch {epoch}] step {step} loss: {loss.item():.4f}")

        # save checkpoint each epoch
        model.save_pretrained(f"checkpoints/uni_{epoch:02d}")
    # final save
    model.save_pretrained("checkpoints/apple2banana-unet")
    print("✨ Training complete!")
    
if __name__ == "__main__":
    main()
