from dataclasses import dataclass
from XarrayDataset import XarrayDataset
from PredictionPipeline import PredictionPipeline
from datasets import load_dataset
from torchvision import transforms
import torch.nn.functional as F

import numpy as np
from diffusers import UNet2DModel
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid

from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
from PIL import Image
import os
import torch


@dataclass
class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 6
    eval_batch_size = 6  # how many images to sample during evaluation
    num_epochs = 100
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 5
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "hrrr-2tm"  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 42
    past_frames = 3
    predict_frames = 2


config = TrainingConfig()

model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=config.past_frames + config.predict_frames,  # the number of input channels, 3 for RGB images
    out_channels=config.predict_frames,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    # block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    # block_out_channels=(32,32,64,64,128,128),
    block_out_channels=(64,64,128,128,256,256),
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

dataset = load_dataset("hrrr_disk", split="train", trust_remote_code=True)
dataset.cleanup_cache_files()

t2m_mean = 278.6412
t2m_std = 21.236853

preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size),antialias=None),
        # transforms.RandomHorizontalFlip(),
        transforms.Normalize([t2m_mean], [t2m_std]),
    ]
)

def transform(examples):
    # for data in examples["data"]:
    #     # print(data)
    #     # print (data)
    #     print (np.array(data).shape)
    dataset = [preprocess(torch.from_numpy(np.array(data,dtype=np.float32))) for data in examples["past"]]
    predicted =  [preprocess(torch.from_numpy(np.array(data,dtype=np.float32))) for data in examples["predict"]]
    return {"past": dataset, "predict": predicted }

dataset.set_transform(transform)
train_dataloader = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=config.train_batch_size)#, shuffle=True)

past_image = dataset[0]["past"].unsqueeze(0)
pred_image = dataset[0]["predict"].unsqueeze(0)
sample_set = torch.concat([past_image, pred_image], axis=1)


print("Input shape:", sample_set.shape)
print("Output shape:", model(sample_set, timestep=0).sample.shape)


pred_img = Image.fromarray(((pred_image[0,-1,...] + 0.5) *127.5).type(torch.uint8).numpy())
past_img = Image.fromarray(((past_image[0,0,...] + 0.5) *127.5).type(torch.uint8).numpy())

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)


optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

def evaluate(config, epoch, pipeline):
    try:

        test_dir = os.path.join(config.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
    
        data = pipeline(past_image,
                 batch_size=2,
                 predict_frames=config.predict_frames,
                 generator=torch.Generator().manual_seed(config.seed),
                 output_type="np.array",
                 return_dict=False)[0]

        images=[]
        img_cnt = 1
        images.append(past_img)
        
        for i in range(0,config.predict_frames):
            img = Image.fromarray(((data[0,i,...] + 0.5) * 127.5).type(torch.uint8).numpy())
            images.append(img)
            img_cnt += 1

        images.append(past_img)
        for i in range(0,config.predict_frames):
            img = Image.fromarray(((pred_image[0,i,...] + 0.5) * 127.5).type(torch.uint8).numpy())
            images.append(img)

        # Make a grid out of the images, rows cols must match image count
        image_grid = make_image_grid(images, rows=2, cols=img_cnt)
        image_grid.save(f"{test_dir}/{epoch:04d}.png")

    except Exception as e:
        print (e)


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            past_images = batch["past"]
            pred_images = batch["predict"]

            # Sample noise to add to the images
            noise = torch.randn(pred_images.shape, device=pred_images.device)
            bs = past_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=pred_images.device,
                dtype=torch.int64
            )

            # Add noise to the predicted images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(pred_images, noise, timesteps)

            # unet trained with clean past images and new noisy images
            noisy_set = torch.concat([past_images, noisy_images], axis=1)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_set, timesteps, return_dict=False)[0]
                # compare predicted noise to noise generated
                loss = F.mse_loss(noise_pred, noise)
                # 
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = PredictionPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                print ("saving model")
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    print ("saving to {}".format(config.output_dir))
                    pipeline.save_pretrained(config.output_dir)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                print ("saving test image")
                evaluate(config, epoch, pipeline)
                print ("done saving")

           


#from accelerate import notebook_launcher
#
#args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
#train_loop(args)
train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
#
#notebook_launcher(train_loop, args, num_processes=1)

