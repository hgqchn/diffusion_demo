import matplotlib.pyplot as plt
from diffusers import DiffusionPipeline,EulerDiscreteScheduler,DDPMScheduler,UNet2DModel
from PIL import Image
import numpy as np
import torch
import tqdm

def sample_process(sample):
    image_processed=sample.cpu().permute(0,2,3,1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)
    image_processed = image_processed.squeeze(0)
    return image_processed


torch.manual_seed(3407)

repo_id = "google/ddpm-cat-256"

scheduler = DDPMScheduler.from_pretrained(repo_id)
model = UNet2DModel.from_pretrained(repo_id, use_safetensors=True)
model=model.to("cuda")

noisy_sample = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
noisy_sample=noisy_sample.to("cuda")


sample = noisy_sample
imgs=[]
imgs_t=[]

for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
    # 1. predict noise residual
    with torch.no_grad():
        residual = model(sample, t).sample

    # 2. compute less noisy image and set x_t -> x_t-1
    sample = scheduler.step(residual, t, sample).prev_sample

    # 3. optionally look at image
    if (i+1) % 50 == 0:
        img = sample_process(sample)
        imgs.append(img)
        imgs_t.append(t)
fig,axes=plt.subplots(5,4)
for i,ax in enumerate(axes.flatten()):
    ax.imshow(imgs[i])
    ax.set_title(f'T: {imgs_t[i+1]}')
    ax.axis('off')
plt.tight_layout()
plt.savefig('ddpm_cat_256.png')
plt.show()

