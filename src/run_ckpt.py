import os
import torch
from torch import autocast
from src import SimpleStableDiffusionPipeline
from omegaconf import OmegaConf
from diffusers import (
    AutoencoderKL,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from .scripts.convert_original_stable_diffusion_to_diffusers import (
        create_unet_diffusers_config,
        convert_ldm_unet_checkpoint,
        create_vae_diffusers_config,
        convert_ldm_vae_checkpoint,
        convert_ldm_clip_checkpoint,
        convert_open_clip_checkpoint,
        safe_open
)
from transformers import AutoFeatureExtractor, CLIPTextModel, CLIPTokenizer

def get_huggingface_cache_path():
    return os.path.join(os.path.expanduser('~'), ".cache", "huggingface", "diffusers")

def run_and_cache_custom_model(custom_model_path, name, folder_path, config_file = None, vae_file = None, cache=True):
    _, extension = os.path.splitext(custom_model_path)
    if extension == ".safetensors":
        checkpoint = {}
        with safe_open(custom_model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                checkpoint[key] = f.get_tensor(key)
    else:
        checkpoint = torch.load(custom_model_path, map_location="cuda")

    global_step = checkpoint["global_step"] if "global_step" in checkpoint else None
    checkpoint = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    upcast_attention = False
    prediction_type = None
    image_size = None

    if config_file is None:
        key_name = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
        if key_name in checkpoint and checkpoint[key_name].shape[-1] == 1024:
            config_file = "src/scripts/v2-inference-v.yaml"
            if global_step == 110000:
                # v2.1 needs to upcast attention
                upcast_attention = True
        else:
            config_file = "src/scripts/v1-inference.yaml"

    original_config = OmegaConf.load(config_file)

    if (
            "parameterization" in original_config["model"]["params"]
            and original_config["model"]["params"]["parameterization"] == "v"
        ):
            if prediction_type is None:
                # NOTE: For stable diffusion 2 base it is recommended to pass `prediction_type=="epsilon"`
                # as it relies on a brittle global step parameter here
                prediction_type = "epsilon" if global_step == 875000 else "v_prediction"
            if image_size is None:
                # NOTE: For stable diffusion 2 base one has to pass `image_size==512`
                # as it relies on a brittle global step parameter here
                image_size = 512 if global_step == 875000 else 768
    else:
        if prediction_type is None:
            prediction_type = "epsilon"
        if image_size is None:
            image_size = 512

    num_train_timesteps = original_config.model.params.timesteps
    beta_start = original_config.model.params.linear_start
    beta_end = original_config.model.params.linear_end

    scheduler = DDIMScheduler(
        beta_end=beta_end,
        beta_schedule="scaled_linear",
        beta_start=beta_start,
        num_train_timesteps=num_train_timesteps,
        steps_offset=1,
        clip_sample=False,
        set_alpha_to_one=False,
        prediction_type=prediction_type,
    )

    unet_config = create_unet_diffusers_config(original_config, image_size=image_size)
    unet_config["upcast_attention"] = upcast_attention
    unet = UNet2DConditionModel(**unet_config)

    converted_unet_checkpoint = convert_ldm_unet_checkpoint(
        checkpoint, unet_config, path=custom_model_path, extract_ema=False
    )

    unet.load_state_dict(converted_unet_checkpoint)

    vae_config = create_vae_diffusers_config(original_config, image_size=image_size)
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(converted_vae_checkpoint)

    model_type = original_config.model.params.cond_stage_config.target.split(".")[-1]

    if model_type == "FrozenOpenCLIPEmbedder":
        text_model = convert_open_clip_checkpoint(checkpoint)
        tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2", subfolder="tokenizer")
        safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
        pipe = SimpleStableDiffusionPipeline.SimpleStableDiffusionPipeline(
            vae=vae,
            text_encoder=text_model,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=None,
            safety_checker=safety_checker,
            requires_safety_checker=False
        )
    elif model_type == "FrozenCLIPEmbedder":
        text_model = convert_ldm_clip_checkpoint(checkpoint)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        feature_extractor = AutoFeatureExtractor.from_pretrained("CompVis/stable-diffusion-safety-checker")
        safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
        pipe =  SimpleStableDiffusionPipeline.SimpleStableDiffusionPipeline(
            vae=vae,
            text_encoder=text_model,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            safety_checker=safety_checker,
            requires_safety_checker=False
        )

    if cache:
        pipe.save_pretrained(os.path.join(folder_path, name))

    #remember to pipe to cuda!!
    return pipe, prediction_type
