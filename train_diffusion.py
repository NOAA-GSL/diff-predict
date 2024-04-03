# -*- coding: utf-8 -*-
#
# @Author: Jebb Q. Stewart
# @Date:   2023-12-16
# @Email: jebb.q.stewart@noaa.gov 
#
# @Last modified by:   Jebb Q. Stewart
# @Last Modified time: 2024-04-03 18:34:39

from dataclasses import dataclass, asdict
from XarrayDataset import XarrayDataset
from PredictionPipeline import PredictionPipeline
from datasets import load_dataset
from torchvision import transforms
import torch.nn.functional as F
from torch import nn

import numpy as np
import math
import fastcore.all as fc

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
    # resume information
    resume_from_checkpoint: bool = False
    starting_epoch: int = 0
    resume_step: int = 0

    # training information
    image_size: int = 64
    train_batch_size: int = 25
    eval_batch_size: int = 25         # how many images to sample during evaluation
    num_timesteps: int = 1000         # how many timesteps to generate noise
    num_workers: int = 12             # for data loading, how many parallel workers
    num_epochs: int = 75 
    gradient_accumulation_steps: int = 3
    learning_rate: float = 5e-4
    lr_warmup_epochs: int = 3
    save_image_epochs: int = 5        # how often to save model weights
    save_model_epochs: int = 5        # how often to save model weights
    mixed_precision: str = "fp16"     # `no` for float32, `fp16` for automatic mixed precision
    output_dir   = "hrrr-4x2"         # the model name locally and on the HF Hub
    seed: int = 42
    num_features: int = 5
    past_frames: int = 4
    predict_frames: int = 2
    start_time: str = "2018-01-20T00:00:00"
    end_time: str = "2023-07-31T12:59:00"
    val_start_time: str = "2023-08-01T12:00:00"
    val_end_time: str = "2023-09-30T23:59:00"

    # hugging face information
    push_to_hub: bool = False         # whether to upload the saved model to the HF Hub
    hub_model_id: str = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    hub_private_repo: bool = False
    overwrite_output_dir: bool = True  # overwrite the old model when re-running the notebook

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}

config = TrainingConfig()

# borrowed from https://github.com/tcapelle/cloud_diffusion.git
def init_unet(model):
    "From Jeremy's bag of tricks on fastai V2 2023"
    for o in model.down_blocks:
        for p in o.resnets:
            p.conv2.weight.data.zero_()
            for p in fc.L(o.downsamplers): nn.init.orthogonal_(p.conv.weight)

    for o in model.up_blocks:
        for p in o.resnets: p.conv2.weight.data.zero_()

    model.conv_out.weight.data.zero_()

model = UNet2DModel(
    #https://huggingface.co/docs/diffusers/en/api/models/unet2d
    sample_size=config.image_size,  # the target image resolution
    in_channels=(config.past_frames + config.predict_frames) * config.num_features,  # the number of input channels, 3 for RGB images
    out_channels=config.predict_frames * config.num_features,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    # block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    # block_out_channels=(32,32,64,64,128,128),
    block_out_channels=(64,64,128,128,256,256),
    # down_block_types defaults ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    # up_block_types defaults to ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

# reset weights before training
init_unet(model)

dataset = XarrayDataset(name="hrrr_v4_more_analysis", 
                        data_start=config.start_time, 
                        data_end=config.end_time,
                        batch_size=config.train_batch_size,
                        image_size=config.image_size,
                        future_frames=config.predict_frames,
                        previous_frames=config.past_frames,
                        ).shuffle(seed=42)

val_dataset = XarrayDataset(name="hrrr_v4_more_analysis",
                        data_start=config.val_start_time,
                        data_end=config.val_end_time,
                        batch_size=config.train_batch_size,
                        future_frames=config.predict_frames,
                        previous_frames=config.past_frames,
                        image_size=config.image_size)


train_dataloader = torch.utils.data.DataLoader(dataset,
                                               num_workers=config.num_workers, 
                                               batch_size=config.train_batch_size) 

val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                               num_workers=config.num_workers, 
                                               batch_size=config.eval_batch_size) 

print (f"dataloader length {len(train_dataloader)}")

past_image,pred_image = next(iter(dataset))
past_image = past_image[np.newaxis,...]
pred_image = pred_image[np.newaxis,...]
sample_set = torch.concat([past_image, pred_image], axis=1).to(model.device)

print("Input shape:", sample_set.shape)
print("Output shape:", model(sample_set, timestep=0).sample.shape)


pred_img = Image.fromarray(((pred_image[0,-1,...] + 0.5) *127.5).type(torch.uint8).cpu().numpy())
past_img = Image.fromarray(((past_image[0,0,...] + 0.5) *127.5).type(torch.uint8).cpu().numpy())

steps_per_epoch = math.ceil(len(train_dataloader)/config.num_workers)*config.num_workers
update_steps = math.ceil(len(train_dataloader)/config.gradient_accumulation_steps)
max_steps = update_steps* config.num_epochs

print ("steps per epoch: ", steps_per_epoch)
print ("update_steps: ", update_steps)
print ("max_steps: ", max_steps)

noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_timesteps)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, eps=1e-5)
# lr_scheduler = get_cosine_schedule_with_warmup(
#     optimizer=optimizer,
#     num_warmup_steps=config.lr_warmup_epochs*steps_per_epoch,
#     num_training_steps=steps_per_epoch * config.num_epochs)

lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                         max_lr=config.learning_rate,
                         #total_steps=max_steps)
                         epochs=config.num_epochs,
                         steps_per_epoch=update_steps)# steps_per_epoch)
                         #steps_per_epoch=round(steps_per_epoch / config.gradient_accumulation_steps))
                         #total_steps=steps_per_epoch * config.num_epochs)

def evaluate(config, epoch, pipeline, device, accelerator, step):
    try:

        test_dir = os.path.join(config.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
    
        data = pipeline(past_image,
                 batch_size=2,
                 predict_frames=config.predict_frames,
                 num_features=config.num_features,
                 num_inference_steps=333,
                 generator=torch.Generator(device).manual_seed(config.seed),
                 output_type="np.array",
                 return_dict=False)[0]

        images=[]
        img_cnt = 1

        # Make a row for ground truth data
        images.append(past_img)
        for i in range(0,config.predict_frames):
            img = Image.fromarray(((pred_image[0,i*config.num_features,...] + 0.5) * 127.5).type(torch.uint8).cpu().numpy())
            images.append(img)

        # Make a row for predicted values
        images.append(past_img)
        for i in range(0,config.predict_frames):
            img = Image.fromarray(((data[0,i*config.num_features,...] + 0.5) * 127.5).type(torch.uint8).cpu().numpy())
            images.append(img)
            img_cnt += 1

        # Make a row for comparing differences
        diff_img = Image.fromarray((((past_image[0,0,...]-past_image[0,0,...]) + 0.5) * 127.5).type(torch.uint8).cpu().numpy())
        images.append(diff_img)
        # bring back to cpu
        data = data.cpu()
        for i in range(0,config.predict_frames):
            diff_img = Image.fromarray((((data[0,i*config.num_features,...]-pred_image[0,i*config.num_features,...]) + 0.5) * 127.5).type(torch.uint8).cpu().numpy())
            images.append(diff_img)

        # Make a grid out of the images, rows cols must match image count
        image_grid = make_image_grid(images, rows=3, cols=img_cnt)
        image_grid.save(f"{test_dir}/{epoch:04d}.png")

    except Exception as e:
        print (e)


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, val_dataloader, lr_scheduler):

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        step_scheduler_with_optimizer=True,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=["tensorboard"],
        project_dir=os.path.join(config.output_dir, "logs"),
    )


    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("hrrr-t2m", config=asdict(config))

    scaler = torch.cuda.amp.GradScaler()

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler, scaler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler, scaler
    )

    global_step = 0
    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        total_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            if config.resume_from_checkpoint and epoch == config.starting_epoch:
                if config.resume_step is not None and step < config.resume_step:
                    progress_bar.update(1)
                    global_step += 1
                    continue

            past_frames,predict_frames = batch

            # Sample noise to add to the images
            noise = torch.randn(predict_frames.shape, device=predict_frames.device)
            bs = past_frames.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=predict_frames.device,
                dtype=torch.int64
            )

            # Add noise to the predicted frames according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_frames = noise_scheduler.add_noise(predict_frames, noise, timesteps)

            # unet trained with clean past images and new noisy images
            noisy_set = torch.concat([past_frames, noisy_frames], axis=1).to(model.device)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_set, timesteps, return_dict=False)[0]
                # compare predicted noise to noise generated
                loss = F.mse_loss(noise, noise_pred)
                total_loss += loss.detach().float()

                optimizer.zero_grad()
                accelerator.backward(scaler.scale(loss))
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()

            progress_bar.update(1)
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # validation
        model.eval()
        losses = []
        completed_eval_steps = 0
        #progress_bar.set_description(f"Eval Epoch {epoch}")

        for eval_step, batch in enumerate(val_dataloader):
            past_frames,predict_frames = batch

            bs = past_frames.shape[0]

            noise = torch.randn(predict_frames.shape, device=predict_frames.device)

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=predict_frames.device,
                dtype=torch.int64
            )

            noisy_frames = noise_scheduler.add_noise(predict_frames, noise, timesteps)

            # unet trained with clean past images and new noisy images
            noisy_set = torch.concat([past_frames, noisy_frames], axis=1).to(model.device)

            with torch.no_grad():
                predicted_noise = model(noisy_set, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise, predicted_noise)

            completed_eval_steps += 1
            losses.append(accelerator.gather(loss.repeat(config.eval_batch_size)))        
            
        # Get average loss for epoch and log some information
        losses = torch.cat(losses)
        eval_loss = torch.mean(losses)
    
        accelerator.log(
                {
                    "epoch_eval_loss": eval_loss,
                    "epoch_train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": global_step,
                },
                step=global_step,
            )

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = PredictionPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                progress_bar.write("saving model")
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    progress_bar.write("saving to {}".format(config.output_dir))
                    pipeline.save_pretrained(config.output_dir)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                progress_bar.write("saving test image")
                evaluate(config, epoch, pipeline, model.device, accelerator, global_step)

           

train_loop(config, model, noise_scheduler, optimizer, train_dataloader, val_dataloader, lr_scheduler)
      
