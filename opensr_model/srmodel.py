import pathlib
from typing import Union
import requests
import torch
from opensr_model.diffusion.latentdiffusion import LatentDiffusion
from skimage.exposure import match_histograms
import torch.utils.checkpoint as checkpoint
from opensr_model.diffusion.utils import DDIMSampler
from tqdm import tqdm
import numpy as np
from typing import Literal
import shutil
import random
import numpy as np
from einops import rearrange
from opensr_model.utils import suppress_stdout
from opensr_model.utils import assert_tensor_validity
from opensr_model.utils import revert_padding
from opensr_model.utils import create_no_data_mask
from opensr_model.utils import apply_no_data_mask
from opensr_model.utils import normalize_s1, normalize_s2, normalize_aerial



class SRLatentDiffusion(torch.nn.Module):
    def __init__(self,config, device: Union[str, torch.device] = "cpu"):
        super().__init__()

        # Set up the model
        self.config = config
        self.model = LatentDiffusion(
            config.first_stage_config,
            config.cond_stage_config,
            timesteps=config.denoiser_settings.timesteps,
            unet_config=config.cond_stage_config,
            linear_start=config.denoiser_settings.linear_start,
            linear_end=config.denoiser_settings.linear_end,
            concat_mode=config.other.concat_mode,
            cond_stage_trainable=config.other.cond_stage_trainable,
            first_stage_key=config.other.first_stage_key,
            cond_stage_key=config.other.cond_stage_key,
            scale_factor=0.18215, # 512/2800, adjust if you change the target HR size or VAE downscale
        )


        # Set up the model for inference
        self.set_normalization() # decide wether to use norm
        self.device = device # set self device
        self.model.device = device # set model device as selected
        self.model = self.model.to(device) # move model to device
        self.model.eval() # set model state
        self = self.eval() # set main model state
        self._X_s2 = None # placeholder for LR S2 image (used in histogram matching)
        self.encode_conditioning = config.encode_conditioning # encode LR images before dif?

        # Scale factor: LR→HR spatial ratio (original=4, S1+S2 fusion=10)
        self.scale_factor = getattr(config, 'scale_factor', 4)
        # VAE spatial downscale factor: 2^(num_downsample_steps)
        ch_mult = list(config.first_stage_config.ch_mult)
        self.vae_downscale = 2 ** (len(ch_mult) - 1)
        # Number of VAE latent channels
        self.z_channels = config.first_stage_config.z_channels

    def set_normalization(self):
        if self.config.apply_normalization==True:
            from opensr_model.utils import linear_transform_4b
            self.linear_transform = linear_transform_4b
        else:
            from opensr_model.utils import linear_transform_placeholder
            self.linear_transform = linear_transform_placeholder
            print("Normalization disabled.")

    def _tensor_encode(self, X_s2: torch.Tensor, X_s1: torch.Tensor):
        """Encode S2 and S1 inputs into a fused conditioning tensor in latent space.

        S2 (4ch) is upsampled to HR size and encoded through the VAE to get a
        4-channel latent representation.  S1 (2ch) is upsampled directly to the
        same latent spatial size (it cannot pass through the 4-ch VAE).
        The two are concatenated to form a 6-channel conditioning tensor.

        Args:
            X_s2: (B, 4, H, W) Sentinel-2 RGBNIR at LR resolution.
            X_s1: (B, 2, H, W) Sentinel-1 VV/VH at LR resolution.

        Returns:
            conditioning: (B, 6, latent_H, latent_W) fused conditioning tensor.
        """
        # Store S2 copy for histogram matching during decode
        self._X_s2 = X_s2.clone()

        # Compute target sizes
        lr_size = X_s2.shape[-1]                          # e.g. 128
        hr_size = lr_size * self.scale_factor              # e.g. 128 * 10 = 1280
        latent_size = hr_size // self.vae_downscale        # e.g. 1280 / 4 = 320

        # --- S2 conditioning (4ch) ---
        # Normalize S2 reflectance DN → [-1, 1]
        X_s2_norm = normalize_s2(X_s2, stage="norm")
        if self.encode_conditioning:
            # Upsample to HR size, then encode through VAE → 4ch latent
            X_s2_up = torch.nn.functional.interpolate(
                X_s2_norm,
                size=(hr_size, hr_size),
                mode='bilinear',
                align_corners=False
            )
            cond_s2 = self.model.first_stage_model.encode(X_s2_up).mode()
        else:
            # Skip VAE, just resize to latent spatial size
            cond_s2 = torch.nn.functional.interpolate(
                X_s2_norm,
                size=(latent_size, latent_size),
                mode='bilinear',
                align_corners=False
            )

        # --- S1 conditioning (2ch) ---
        # Normalize S1 dB → [-1, 1]
        X_s1_norm = normalize_s1(X_s1, stage="norm")
        # Upsample directly to latent spatial size
        cond_s1 = torch.nn.functional.interpolate(
            X_s1_norm,
            size=(latent_size, latent_size),
            mode='bilinear',
            align_corners=False
        )

        # --- Fuse: concat S2 latent (4ch) + S1 (2ch) → 6ch conditioning ---
        conditioning = torch.cat([cond_s2, cond_s1], dim=1)
        conditioning = conditioning.to(self.device)
        return conditioning



    def _tensor_decode(self, latent: torch.Tensor, spe_cor: bool = False) -> torch.Tensor:
        """Decode a latent tensor back to aerial image space.

        Args:
            latent: (B, 4, latent_H, latent_W) latent tensor from DDIM sampling.
            spe_cor: If True, apply histogram matching against the stored S2 input.

        Returns:
            (B, 4, HR_H, HR_W) aerial image in [0, 255] range.
        """
        # Decode latent → [-1, 1] image space via VAE decoder
        decoded = self.model.decode_first_stage(latent)

        # Denormalize [-1, 1] → [0, 255]
        sr = normalize_aerial(decoded, stage="denorm")

        # Spectral correction: histogram match SR to S2 input per sample
        if spe_cor and self._X_s2 is not None:
            # Upsample S2 to SR spatial size for reference
            s2_up = torch.nn.functional.interpolate(
                self._X_s2.to(sr.device),
                size=sr.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            corrected = []
            for i in range(sr.shape[0]):
                corrected.append(self.hq_histogram_matching(sr[i], s2_up[i]))
            sr = torch.stack(corrected)

        return sr

    def _prepare_model(
        self,
        conditioning: torch.Tensor,
        eta: float = 1.0,
        custom_steps: int = 100,
        verbose: bool = False
    ):
        # Create the DDIM sampler
        ddim = DDIMSampler(self.model)

        # make schedule to compute alphas and sigmas
        ddim.make_schedule(ddim_num_steps=custom_steps, ddim_eta=eta, verbose=verbose)

        # Create the HR latent noise — 4ch (z_channels), same spatial as conditioning
        latent_shape = (
            conditioning.shape[0],      # batch size
            self.z_channels,            # 4 (VAE latent channels)
            conditioning.shape[2],      # latent H (e.g. 320)
            conditioning.shape[3],      # latent W (e.g. 320)
        )
        latent = torch.randn(latent_shape, device=conditioning.device)

        # Create the vector with the timesteps
        timesteps = ddim.ddim_timesteps
        time_range = np.flip(timesteps)

        return ddim, latent, time_range

    def _ddim_step(self, latent, e_t, index, ddim, temperature):
        """Single DDIM update step given a pre-computed noise prediction e_t."""
        B, device = latent.shape[0], latent.device
        a_t     = torch.full((B, 1, 1, 1), ddim.ddim_alphas[index],              device=device)
        a_prev  = torch.full((B, 1, 1, 1), ddim.ddim_alphas_prev[index],         device=device)
        sigma_t = torch.full((B, 1, 1, 1), ddim.ddim_sigmas[index],              device=device)
        sqrt_one_minus_at = torch.full((B, 1, 1, 1), ddim.ddim_sqrt_one_minus_alphas[index], device=device)
        pred_x0 = (latent - sqrt_one_minus_at * e_t) / a_t.sqrt()
        dir_xt  = (1.0 - a_prev - sigma_t ** 2).sqrt() * e_t
        noise   = sigma_t * torch.randn_like(latent) * temperature
        return a_prev.sqrt() * pred_x0 + dir_xt + noise

    @torch.no_grad()
    def forward(
        self,
        X_s2: torch.Tensor,
        X_s1: torch.Tensor,
        sampling_eta: float = None,
        sampling_steps: int = None,
        sampling_temperature: float = None,
        histogram_matching: bool = True,
        save_iterations: bool = False,
        verbose: bool = False,
        guidance_scale: float = 1.0,
    ):
        """Obtain the super resolution from fused S2+S1 input.

        Args:
            X_s2 (torch.Tensor): Sentinel-2 RGBNIR (B, 4, H, W) at LR resolution.
            X_s1 (torch.Tensor): Sentinel-1 VV/VH (B, 2, H, W) at LR resolution.
            sampling_steps (int, optional): Number of DDIM steps. Defaults to config.
            sampling_eta (float, optional): DDIM eta (0=deterministic, 1=stochastic).
            sampling_temperature (float, optional): Noise temperature. Defaults to config.
            histogram_matching (bool, optional): Apply spectral correction. Defaults to True.
            save_iterations (bool, optional): Return all intermediate SR images.
            verbose (bool, optional): Show progress bar.

        Returns:
            torch.Tensor: Super resolved aerial image (B, 4, HR_H, HR_W).
        """
        # fall back on config if args are None
        if sampling_eta is None:
            sampling_eta = self.config.denoiser_settings.sampling_eta
        if sampling_temperature is None:
            sampling_temperature = self.config.denoiser_settings.sampling_temperature
        if sampling_steps is None:
            sampling_steps = self.config.denoiser_settings.sampling_steps

        # Assert shape, size, dimensionality. Add padding if necessary
        X_s2, padding = assert_tensor_validity(X_s2)
        X_s1, _ = assert_tensor_validity(X_s1)

        # create no_data_mask from S2 (primary optical input)
        no_data_mask = create_no_data_mask(X_s2, target_size=X_s2.shape[-1] * self.scale_factor)

        # Encode S2+S1 into fused conditioning in latent space
        conditioning = self._tensor_encode(X_s2.clone(), X_s1.clone())
        null_conditioning = torch.zeros_like(conditioning)  # used for CFG

        # ddim, latent (4ch noise) and time_range
        ddim, latent, time_range = self._prepare_model(
            conditioning=conditioning, eta=sampling_eta,
            custom_steps=sampling_steps, verbose=verbose
        )
        iterator = tqdm(time_range, desc="DDIM Sampler", total=sampling_steps, disable=True)

        # Iterate over the timesteps
        if save_iterations:
            save_iters = []

        for i, step in enumerate(iterator):
            index = sampling_steps - i - 1
            t = torch.full((latent.shape[0],), step, device=conditioning.device, dtype=torch.long)

            if guidance_scale > 1.0:
                # CFG: two UNet passes, amplify conditioned direction
                e_t_uncond = self.model.apply_model(latent, t, cond=null_conditioning)
                e_t_cond   = self.model.apply_model(latent, t, cond=conditioning)
                e_t = e_t_uncond + guidance_scale * (e_t_cond - e_t_uncond)
                latent = self._ddim_step(latent, e_t, index, ddim, sampling_temperature)
            else:
                outs = ddim.p_sample_ddim(
                    x=latent, c=conditioning, t=step, index=index,
                    use_original_steps=False, temperature=sampling_temperature,
                )
                latent, _ = outs

            if save_iterations:
                save_iters.append(
                    self._tensor_decode(latent, spe_cor=histogram_matching)
                )

        if save_iterations:
            return save_iters

        sr = self._tensor_decode(latent, spe_cor=histogram_matching) # decode the latent image

        # Post-processing
        sr = apply_no_data_mask(sr, no_data_mask) # apply no data mask as in LR image
        sr = revert_padding(sr, padding) # remove padding from the SR image if there was any
        return sr


    def hq_histogram_matching(
        self, image1: torch.Tensor, image2: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies histogram matching to align the color distribution of image1 to image2.

        This function adjusts the pixel intensity distribution of `image1` (typically the
        low-resolution or degraded image) to match that of `image2` (typically the
        high-resolution or reference image). The operation is done per channel and
        assumes both images are in (C, H, W) format.

        Args:
            image1 (torch.Tensor): The source image whose histogram will be modified (C, H, W).
            image2 (torch.Tensor): The reference image whose histogram will be matched (C, H, W).

        Returns:
            torch.Tensor: A new tensor with the same shape as `image1`, but with pixel
                        intensities adjusted to match the histogram of `image2`.

        Raises:
            ValueError: If input tensors are not 2D or 3D.
        """

        # Go to numpy
        np_image1 = image1.detach().cpu().numpy()
        np_image2 = image2.detach().cpu().numpy()

        if np_image1.ndim == 3:
            np_image1_hat = match_histograms(np_image1, np_image2, channel_axis=0)
        elif np_image1.ndim == 2:
            np_image1_hat = match_histograms(np_image1, np_image2, channel_axis=None)
        else:
            raise ValueError("The input image must have 2 or 3 dimensions.")

        # Go back to torch
        image1_hat = torch.from_numpy(np_image1_hat).to(image1.device)

        return image1_hat

    def load_pretrained(self, weights_file: str):
        """
        Loads pretrained model weights from a local file or downloads them from Hugging Face if not present.

        If the specified `weights_file` does not exist locally, it is automatically downloaded from the
        Hugging Face model hub under `simon-donike/RS-SR-LTDF`. A progress bar is shown during download.

        After loading, the method removes any perceptual loss-related weights from the state dict and
        loads the remaining weights into the model.

        Args:
            weights_file (str): Path to the local weights file. If the file is not found, it will be downloaded
                                using this name from the Hugging Face repository.

        Raises:
            RuntimeError: If the weights cannot be loaded or parsed correctly.

        Example:
            self.load_pretrained("model_weights.ckpt")
        """

        # download pretrained model
        # create download link based on input
        hf_model = str("https://huggingface.co/simon-donike/RS-SR-LTDF/resolve/main/"+str(weights_file))

        # Total size in bytes.
        if not pathlib.Path(weights_file).exists():
            print("Downloading pretrained weights from: ", hf_model)
            response = requests.get(hf_model, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte

            # Open the file to write as binary - write bytes to a file
            with open(weights_file, "wb") as f:
                # Setup the progress bar
                with tqdm(total=total_size, unit='iB', unit_scale=True, desc=weights_file) as bar:
                    for data in response.iter_content(block_size):
                        bar.update(len(data))
                        f.write(data)

        weights = torch.load(weights_file, map_location=self.device)["state_dict"]

        # Remote perceptual tensors from weights
        for key in list(weights.keys()):
            if "loss" in key:
                del weights[key]

        self.model.load_state_dict(weights, strict=True)
        print("Loaded pretrained weights from: ", weights_file)


    def uncertainty_map(self, x, n_variations=15, sampling_steps=100):
        """
        Estimates uncertainty maps for each sample in the input batch using repeated stochastic forward passes.

        For each input sample, the method generates multiple super-resolved outputs by varying the random seed.
        It then computes the per-pixel standard deviation across these outputs as a proxy for uncertainty.
        The returned uncertainty map represents the average width of the confidence interval per pixel.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W), where B is batch size.
            n_variations (int): Number of stochastic forward passes per input sample.
            custom_steps (int): Custom inference steps passed to the forward method.

        Returns:
            torch.Tensor: Uncertainty maps of shape (B, 1, H, W), where each value indicates pixel-wise uncertainty.
        """
        assert n_variations>3, "n_variations must be greater than 3 to compute uncertainty."


        batch_size = x.shape[0]
        rand_seed_list = random.sample(range(1, 9999), n_variations)

        all_variations = []
        for b in range(batch_size):
            variations = []
            x_b = x[b].unsqueeze(0)  # shape (1, 4, 512, 512)
            for seed in rand_seed_list:
                with suppress_stdout():
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    random.seed(seed)
                    #pytorch_lightning.utilities.seed.seed_everything(seed=seed, workers=True)

                sr = self.forward(x_b, sampling_steps=sampling_steps,sampling_eta=0.0)  # shape (1, C, H, W)
                variations.append(sr.detach().cpu())

            variations = torch.stack(variations)  # (n_variations, 1, C, H, W)
            srs_mean = variations.mean(dim=0)
            srs_stdev = variations.std(dim=0)
            interval_size = (srs_stdev * 2).mean(dim=1)  # mean over channels

            all_variations.append(interval_size)  # each is (1, H, W)

        result = torch.stack(all_variations)  # (B, 1, H, W)
        return result



    def _attribution_methods(
        self,
        X: torch.Tensor,
        grads: torch.Tensor,
        attribution_method: Literal[
            "grad_x_input", "max_grad", "mean_grad", "min_grad"
                        ],
                    ):
        """
        DEPRECIATED; SUBJECT TO REMOVAL
        """
        if attribution_method == "grad_x_input":
            return torch.norm(grads * X, dim=(0, 1))
        elif attribution_method == "max_grad":
            return grads.abs().max(dim=0).max(dim=0)
        elif attribution_method == "mean_grad":
            return grads.abs().mean(dim=0).mean(dim=0)
        elif attribution_method == "min_grad":
            return grads.abs().min(dim=0).min(dim=0)
        else:
            raise ValueError(
                "The attribution method must be one of: grad_x_input, max_grad, mean_grad, min_grad"
            )

    def explainer(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        eta: float = 0.0,
        temperature: float = 1.0,
        custom_steps: int = 100,
        steps_to_consider_for_attributions: list = list(range(100)),
        attribution_method: Literal[
            "grad_x_input", "max_grad", "mean_grad", "min_grad"
        ] = "grad_x_input",
        verbose: bool = False,
        enable_checkpoint = True,
        histogram_matching=True
            ):
        """
        DEPRECIATED; SUBJECT TO REMOVAL
        """
        # Normalize and encode the LR image
        X = X.clone()
        Xnorm = self._tensor_encode(X)

        # ddim, latent and time_range
        ddim, latent, time_range = self._prepare_model(
            X=Xnorm, eta=eta, custom_steps=custom_steps, verbose=verbose
        )

        # Iterate over the timesteps
        container = []
        iterator = tqdm(time_range, desc="DDIM Sampler", total=custom_steps,disable=True)
        for i, step in enumerate(iterator):

            # Activate or deactivate gradient tracking
            if i in steps_to_consider_for_attributions:
                torch.set_grad_enabled(True)
            else:
                torch.set_grad_enabled(False)

            # Compute the latent image
            if enable_checkpoint:
                outs = checkpoint.checkpoint(
                    ddim.p_sample_ddim,
                    latent,
                    Xnorm,
                    step,
                    custom_steps - i - 1,
                    temperature,
                    use_reentrant=False,
                )
            else:
                outs = ddim.p_sample_ddim(
                    x=latent,
                    c=Xnorm,
                    t=step,
                    index=custom_steps - i - 1,
                    temperature=temperature
                )
            latent, _ = outs


            if i not in steps_to_consider_for_attributions:
                continue

            # Apply the mask
            output_graph = (latent*mask).mean()

            # Compute the gradients
            grads = torch.autograd.grad(output_graph, Xnorm, retain_graph=True)[0]

            # Compute the attribution and save it
            with torch.no_grad():
                to_save = {
                    "latent": self._tensor_decode(latent, spe_cor=histogram_matching),
                    "attribution": self._attribution_methods(
                        Xnorm, grads, attribution_method
                    )
                }
            container.append(to_save)

        return container



# -----------------------------------------------------------------------------
# Logic to create PyTorch Lightning Model from dif model
# Logic to handle outputs from PL model and save them
# -----------------------------------------------------------------------------

import torch
import pytorch_lightning
from pytorch_lightning import LightningModule

class SRLatentDiffusionLightning(LightningModule):
    """
    This Pytorch Lightning Class wraps around the torch model to
    aid in distrubuted GPU processing and optimized dataloaders
    provided by PL. ToDo: implement demo showcase
    """
    def __init__(self,config, device: Union[str, torch.device] = "cpu"):
        super().__init__()
        self.model = SRLatentDiffusion(config,device=device)
        self.model = self.model.eval()

    @torch.no_grad()
    def forward(self, x,**kwargs):
        #print("Dont call 'forward' on the PL model, instead use 'predict'")
        return self.model(x)

    def load_pretrained(self, weights_file: str):
        self.model.load_pretrained(weights_file)
        print("PL Model: Model loaded from ", weights_file)

    @torch.no_grad()
    def predict_step(self, x, idx: int = 0,**kwargs):
        # perform SR
        assert self.model.training == False, "Model in Training mode. Abort." # make sure we're in eval
        p = self.model.forward(x)
        return(p)

    @torch.no_grad()
    def uncertainty_map(self, x,n_variations=15,custom_steps=100):
        uncertainty_map = self.model.uncertainty_map(x,n_variations,custom_steps)
        return(uncertainty_map)
