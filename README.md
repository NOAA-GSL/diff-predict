# AI Diffusion Prediction Project

This is an **experimental** project that implements AI Diffusion Techniques for predicting future frames.  Based off work outlined at: https://huggingface.co/docs/diffusers/main/en/tutorials/basic_training

This assumes you have preprocessed zarr data somewhere in advance. 

## Training 
```
python3 -u train_diffusion.py
```

This will create a hrrr-2tm directory with logs, sample images as various epochs, and trained weights

## Inference

Modify TrainingConfig information at top to match your trained model then:

```
python3 -u inference.py
```

## To Do

- [x] Add easy inference script
- [x] Investigate streaming dataset (IterableDataset) for better storage and memory management
- [x] Add capability for multiple variables
- [ ] Expand to vertical levels 

## Disclaimer

This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an “as is” basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercia products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.
