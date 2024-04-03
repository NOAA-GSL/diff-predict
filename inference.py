
from dataclasses import dataclass
from XarrayDataset import XarrayDataset
from PredictionPipeline import PredictionPipeline
from transformers import pipeline
from diffusers import DiffusionPipeline
from diffusers.utils import make_image_grid
from PIL import Image

import numpy as np
import torch
import os


@dataclass
class TrainingConfig:
    # must match training configuration
    image_size = 64              # the generated image resolution
    num_features = 5
    past_frames = 4
    predict_frames = 2
    sampler_steps=333            # number of sampler steps on the diffusion process
    num_random_experiments = 4   # we will perform inference multiple times on the same inputs
    future_steps = 2           # how many steps to cycle forward forecast

    batch_size = 5
    num_workers = 1
    output_dir = "infer_t2m_4x2_lb_cd"  # location to store output

    # model information
    model_name = "/home/jstewart/git/diff-predict/hrrr-4x2-largebatch-cd"  # location of model weights
    hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    hub_private_repo = False

    seed = 42
    # Data
    start_time = "2023-10-25T18:00:00"
    end_time = "2023-10-26T23:59:00"

    
config = TrainingConfig()

dataset = XarrayDataset(name="hrrr_v4_more_analysis", 
                        data_start=config.start_time, 
                        data_end=config.end_time,
                        batch_size=config.batch_size,
                        image_size=config.image_size,
                        future_frames=config.predict_frames,
                        previous_frames=config.past_frames)
print (len(dataset))

it_dataset = iter(dataset)
past_image,pred_image = next(it_dataset)
past_image = past_image[np.newaxis,...]

# Gather ground truth data
pred_image = pred_image[np.newaxis,...]
pred_image = np.concatenate((past_image, pred_image), axis=1)

# loop forward to get future ground truth images
for i in range(1,config.future_steps):
   empty, pred_image_tmp = next(it_dataset)
   pred_image = np.concatenate((pred_image, pred_image_tmp[np.newaxis,...]),axis=1)

print (pred_image.shape)
print (past_image.shape)

# assumes huggingface directory structure "{repo-id}/unet/diffusion_pytorch_model.safetensors"
pipeline = PredictionPipeline.from_pretrained(config.model_name, use_safetensors=True)

def evaluate(config, pipeline, experiment, future_steps, seed=None):

    try:

        test_dir = os.path.join(config.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)

        generator=torch.Generator()
        if seed is not None:
            print ("setting seed")
            generator=torch.Generator().manual_seed(seed)

        frames = past_image
        gt = pred_image

        # roll forward forecast
        for s in range(0,future_steps):
    
            data = pipeline(frames[-1:,-(config.num_features*config.past_frames):,...],
                     batch_size=config.batch_size,
                     predict_frames=config.predict_frames,
                     num_features=config.num_features,
                     generator=None,
                     num_inference_steps=config.sampler_steps,
                     output_type="np.array",
                     return_dict=False)[0]

            frames = torch.cat([frames, data.to(frames.device)], dim=1)
            data = data.to(frames.device)
            print (frames.shape)

            images=[]
            img_cnt = 0

            offset = config.num_features*(config.predict_frames)

            pred = frames[-1:,-15:,...]
            past_offset = ((config.past_frames-1)*config.num_features)

            # gather ground truth images
            gt_images = []
            for i in range(0,config.predict_frames+1):
                img_data = gt[-1,past_offset+(s*offset+(i*config.num_features)),...]
                gt_images.append(img_data)
                img = Image.fromarray(((img_data + 1.0) * 127.5).astype(np.uint8))
                images.append(img)

            # Make a row for predicted values
            pd_images = []
            for i in range(0,config.predict_frames+1):
                img_data = pred[0,i*config.num_features,...] 
                pd_images.append(img_data)
                img = Image.fromarray(((img_data + 1.0) * 127.5).type(torch.uint8).numpy())
                images.append(img)
                img_cnt += 1

            # Make a row for comparing differences
            for i in range(0,config.predict_frames+1):
                # print(-(s*offset+(i*config.num_features)))
                img_data = pd_images[i] - gt_images[i]
                diff_img = Image.fromarray(((img_data + 1.0) * 127.5).type(torch.uint8).numpy())
                images.append(diff_img)

            # Make a grid out of the images, rows cols must match image count
            image_grid = make_image_grid(images, rows=3, cols=img_cnt)
            image_grid.save(f"{test_dir}/exp_{experiment:04d}_{s:04d}.png")


    except Exception as e:
        print (e)

    return frames.numpy()


ensemble = []
print (pred_image.shape)
ensemble.append(np.squeeze(pred_image))

# for each ensemble based on noise generate a forecast sequence
for e in range(0,3):
    ens_data = np.array(evaluate(config, pipeline, e, config.future_steps))
    print (ens_data.shape)
    ensemble.append(np.squeeze(ens_data))

ens = np.array(ensemble)
print (ens.shape)

# save data for other analysis
test_dir = os.path.join(config.output_dir, "data")
os.makedirs(test_dir, exist_ok=True)
np.save(f"{test_dir}/ens_forecast.npy", ens)


