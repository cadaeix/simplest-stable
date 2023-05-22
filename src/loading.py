

import os
import subprocess
import requests
import torch
from tqdm import tqdm
from typing import Dict, Optional, Tuple
from diffusers import AutoencoderKL

from src.SimpleStableDiffusionPipeline import SimpleStableDiffusionPipeline
from src.utils import (
    find_modules_and_assign_padding_mode,
    get_huggingface_cache_path,
    login_to_huggingface,
    process_embeddings_folder
)
from src.scripts.convert_from_ckpt import (
    convert_ldm_vae_checkpoint_from_file,
)


def prepare_pipe_from_filepath(filepath: str, enable_attention_slicing: bool = False, enable_xformers: bool = False, to_cuda: bool = True) -> SimpleStableDiffusionPipeline:
    # TODO: check integrity of filepath
    pipe = load_ckpt(filepath)
    pipe = prepare_pipe_options(
        pipe, enable_attention_slicing, enable_xformers, to_cuda)

    pipe_info = {
        "prediction_type": pipe.scheduler.prediction_type,
        "image_size": pipe.vae.config.sample_size
    }

    return pipe, pipe_info


def prepare_pipe_from_diffusers_repo_id(repo_id: str, enable_attention_slicing: bool = False, enable_xformers: bool = False, to_cuda: bool = True) -> SimpleStableDiffusionPipeline:
    pipe = load_diffusers_from_repo_id(repo_id)
    pipe = prepare_pipe_options(
        pipe, enable_attention_slicing, enable_xformers, to_cuda)

    pipe_info = {
        "prediction_type": pipe.scheduler.prediction_type,
        "image_size": pipe.vae.config.sample_size
    }

    return pipe, pipe_info


def prepare_pipe_from_presets(model_choice: dict, enable_attention_slicing: bool = False, enable_xformers: bool = False, to_cuda: bool = True) -> SimpleStableDiffusionPipeline:
    model_download_folder = "non_hf_downloads/"

    if model_choice["type"] == "diffusers":
        pipe = load_diffusers_from_repo_id(model_choice["repo_id"])
    elif model_choice["type"] == "hf_file":
        filepath = download_file_from_hf(
            model_choice["repo_id"], model_choice["filename"], f'{model_download_folder}{model_choice.get("save_as", model_choice["filename"])}')
        pipe = load_ckpt(filepath)
    elif model_choice["type"] == "civitai-model":
        filepath = get_model_file_from_civitai_with_model_id(
            model_choice["model_id"], f"{model_download_folder}{model_choice['filename']}")
        pipe = load_ckpt(filepath)

    if "vae" in model_choice:
        if model_choice["vae"]["type"] == "hf-file":
            vae_filepath = download_file_from_hf()
        elif model_choice["vae"]["type"] == "civitai":
            vae_filepath = get_vae_file_from_civitai_with_model_id(
                model_choice["model_id"], f"{model_download_folder}{model_choice['filename']}")
        pipe = load_vae_file_to_current_pipe(pipe, vae_filepath)

    pipe = prepare_pipe_options(
        pipe, enable_attention_slicing, enable_xformers, to_cuda)

    pipe_info = {
        "keyword": model_choice.get("keyword"),
        "prediction_type": model_choice["prediction"],
        "negative_keyword": model_choice.get("negative_keyword"),
        "image_size": model_choice["image_size"]
    }

    return pipe, pipe_info


def prepare_pipe_options(pipe: SimpleStableDiffusionPipeline, enable_attention_slicing: bool = False, enable_xformers: bool = False, to_cuda: bool = True) -> SimpleStableDiffusionPipeline:
    if enable_xformers:
        pipe.enable_xformers_memory_efficient_attention()
    elif enable_attention_slicing:
        pipe.enable_attention_slicing()

    find_modules_and_assign_padding_mode(pipe, "setup")
    if to_cuda:
        pipe = pipe.to("cuda")

    return pipe


def load_diffusers_from_repo_id(repo_id: str) -> SimpleStableDiffusionPipeline:
    # takes a diffusers repo id
    return SimpleStableDiffusionPipeline.from_pretrained(
        repo_id, safety_checker=None, requires_safety_checker=False, torch_dtype=torch.float16)


def load_ckpt(ckpt_link: str) -> SimpleStableDiffusionPipeline:
    # takes a huggingface link or a filepath to a ckpt/safetensors
    return SimpleStableDiffusionPipeline.from_ckpt(ckpt_link, torch_dtype=torch.float16, load_safety_checker=False)


def load_vae_file_to_current_pipe(pipe: SimpleStableDiffusionPipeline, vae_file_path: str) -> SimpleStableDiffusionPipeline:
    vae_config = dict(
        sample_size=pipe.vae.sample_size,
        in_channels=pipe.vae.in_channels,
        out_channels=pipe.vae.out_channels,
        down_block_types=pipe.vae.down_block_types,
        up_block_types=pipe.vae.up_block_types,
        block_out_channels=pipe.vae.block_out_channels,
        latent_channels=pipe.vae.latent_channels,
        layers_per_block=pipe.vae.layers_per_block,
    )

    vae_ckpt = torch.load(vae_file_path, map_location="cuda")
    vae_dict_1 = {k: v for k, v in vae_ckpt["state_dict"].items(
    ) if k[0:4] != "loss" and k not in {"model_ema.decay", "model_ema.num_updates"}}
    converted_vae_checkpoint = convert_ldm_vae_checkpoint_from_file(
        vae_dict_1, vae_config)

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(converted_vae_checkpoint)
    pipe.vae = vae
    find_modules_and_assign_padding_mode(pipe, "setup")
    pipe = pipe.to("cuda")
    pipe = pipe.to(torch.float16)
    return pipe

# functions for downloading files


def download_file_with_requests(url: str, filename: str) -> str:
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(filename, 'wb') as file, tqdm(
        desc="Downloading model",
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=8192):
            size = file.write(data)
            bar.update(size)
    return filename


def download_file_from_hf(repo_id: str, filename_to_download: str, filename_to_save_as: str) -> str:
    # returns the filepath to the downloaded file for load_ckpt
    # was initially set up for hf_hub_download, but that doesn't let you rename files, and from_ckpt doesn't like underscores, and i was too lazy to fix the model file
    return download_file_with_requests(f"https://huggingface.co/{repo_id}/resolve/main/{filename_to_download}", filename_to_save_as)

# civitai functions


def get_model_file_from_civitai_with_model_id(model_id: str, filename: str):
    if not os.path.exists(filename):
        url = f"https://civitai.com/api/download/models/{model_id}"
        file = download_file_with_requests(url, filename)
        return file
    else:
        return filename


def get_config_file_from_civitai_with_model_id(model_id: str, filename: str):
    if not os.path.exists(filename):
        url = f"https://civitai.com/api/download/models/{model_id}?type=Config"
        file = download_file_with_requests(url, filename)
        return file
    else:
        return filename


def get_vae_file_from_civitai_with_model_id(model_id: str, filename: str):
    if not os.path.exists(filename):
        url = f"https://civitai.com/api/download/models/{model_id}?type=VAE"
        file = download_file_with_requests(url, filename)
        return file
    else:
        return filename

###


def download_a_list_of_embeddings(embeddings_folder: str, embeddings_list):
    for emb in embeddings_list:
        if not os.path.exists(os.path.join(embeddings_folder, emb["filename"])):
            if emb.get("type") == "civitai_embedding":
                get_model_file_from_civitai_with_model_id(
                    emb["model_id"], os.path.join(embeddings_folder, emb["filename"]))
            else:
                subprocess.run(
                    ["wget", "-O", os.path.join(embeddings_folder, emb["filename"]), emb["download_url"]])


def load_embeddings(embeddings_folder: str, pipe: SimpleStableDiffusionPipeline) -> SimpleStableDiffusionPipeline:
    if embeddings_folder and os.path.exists(embeddings_folder):
        emb_list = process_embeddings_folder(embeddings_folder)

        for emb_path in emb_list:
            token = os.path.basename(emb_path.split('.')[0])
            pipe.load_textual_inversion(emb_path, token=token)

    return pipe
