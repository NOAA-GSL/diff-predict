from typing import List, Optional, Tuple, Union

import torch

from diffusers import DiffusionPipeline, ImagePipelineOutput


class PredictionPipeline(DiffusionPipeline):
    r"""
    Pipeline for prediction.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        past_frames,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            past_frames (`np.array`):
                The frames of data used for prediction
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # Sample gaussian noise to begin loop
        # if isinstance(self.unet.config.sample_size, int):
        #     image_shape = (
        #         batch_size,
        #         self.unet.config.in_channels,
        #         self.unet.config.sample_size,
        #         self.unet.config.sample_size,
        #     )
        # else:
        #     image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        # if self.device.type == "mps":
        #     # randn does not work reproducibly on mps
        #     image = randn_tensor(image_shape, generator=generator)
        #     image = image.to(self.device)
        # else:
        #     image = randn_tensor(image_shape, generator=generator, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        new_frames = torch.randn_like(past_frames, dtype=past_frames.dtype, device=self.device)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            noise = self.unet(torch.cat([past_frames, new_frames], dim=1), t).sample

            # 2. compute previous image: x_t -> x_t-1
            new_frames = self.scheduler.step(noise, t, new_frames, generator=generator).prev_sample

        # convert from Tensor to numpy
        new_frames.float().cpu().numpy()
        
        if not return_dict:
            return (new_frames,)

        return ImagePipelineOutput(predictions=new_frames)