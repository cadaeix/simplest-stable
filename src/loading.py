from typing import Optional, Tuple
from src.run_ckpt import run_and_cache_custom_model
from src.utils import find_modules_and_assign_padding_mode, get_huggingface_cache_path, login_to_huggingface, process_embeddings_folder
from src.SimpleStableDiffusionPipeline import SimpleStableDiffusionPipeline
from diffusers import AutoencoderKL
import torch
import json
from huggingface_hub import hf_hub_download
import os


def prepare_pipe(model_name: str, model_type: str, downloadable_model_dict: dict, custom_model_dict: Optional[dict], cached_model_dict: Optional[dict], attention_slicing: bool = False) -> Tuple[SimpleStableDiffusionPipeline, dict]:
    pipe_info = None

    if cached_model_dict and ((model_name in cached_model_dict) or (model_type == "Installed Models")):
        pipe_info = cached_model_dict[model_name]
        pipe = load_installed_model_from_hf_cache(pipe_info["path"])
    elif model_type == "Downloadable Models":
        model_choice = downloadable_model_dict[model_name]
        if model_choice["type"] == "diffusers":
            pipe = load_diffusers_model(model_choice)
        elif model_choice["type"] == "hf-file":
            pipe = download_and_load_non_diffusers_model(
                model_choice["repo_id"], model_name, model_choice["filename"], model_choice["config_file"], model_choice["vae"])
        pipe_info = {
            "keyword": model_choice["keyword"],
            "prediction_type": model_choice["prediction"]
        }
    elif custom_model_dict and model_type == "Custom Models":
        custom_model = custom_model_dict[model_name]
        pipe, prediction_type = load_custom_model_from_local_file(
            custom_model["path"], model_name, custom_model["config"], custom_model["vae"])

        pipe_info = {
            "keyword": custom_model["keywords"],
            "prediction_type": prediction_type
        }
    else:
        raise ValueError(f"Tried to load {model_name} and failed.")

    if attention_slicing:
        pipe.enable_attention_slicing()

    find_modules_and_assign_padding_mode(pipe, "setup")
    pipe = pipe.to("cuda")
    return pipe, pipe_info


def load_custom_model_from_local_file(model_path: str, model_name: str, model_config: Optional[str], vae_file: Optional[str]) -> Tuple[SimpleStableDiffusionPipeline, str]:
    hf_cache_folder = get_huggingface_cache_path()

    return run_and_cache_custom_model(model_path, model_name, hf_cache_folder, model_config, vae_file, True)


def load_installed_model_from_hf_cache(model_path: str) -> SimpleStableDiffusionPipeline:
    return SimpleStableDiffusionPipeline.from_pretrained(
        model_path, safety_checker=None, requires_safety_checker=False, local_files_only=True)


def download_and_load_non_diffusers_model(repo_id: str, model_name: str, filename: str, config_file: Optional[str], vae_file: Optional[str]) -> SimpleStableDiffusionPipeline:
    hf_cache_folder = get_huggingface_cache_path()
    ckpt = hf_hub_download(repo_id=repo_id, filename=filename)
    if config_file:
        config = hf_hub_download(repo_id=repo_id, filename=config_file)
    else:
        config = None
    if vae_file:
        vae = hf_hub_download(repo_id=repo_id, filename=vae)
    else:
        vae = None
    pipe, _ = run_and_cache_custom_model(
        ckpt, model_name, hf_cache_folder, config, vae, True)
    return pipe


def load_diffusers_model(model_choice):
    if model_choice["vae"] != "":
        if model_choice["requires_hf_login"] or model_choice["vae"]["requires_hf_login"]:
            login_to_huggingface()
        vae = AutoencoderKL.from_pretrained(model_choice["vae"]["repo_id"])
        pipe = SimpleStableDiffusionPipeline.from_pretrained(
            model_choice["repo_id"], vae=vae, safety_checker=None, requires_safety_checker=False)
    else:
        if model_choice["requires_hf_login"]:
            login_to_huggingface()
        pipe = SimpleStableDiffusionPipeline.from_pretrained(
            model_choice["repo_id"], safety_checker=None, requires_safety_checker=False)
    return pipe


def load_embeddings(embeddings_folder, pipe):
    if embeddings_folder and os.path.exists(embeddings_folder):
        emb_list = process_embeddings_folder(embeddings_folder)

        for emb_path in emb_list:
            pipe.embedding_database.add_embedding_path(emb_path)
        pipe.load_embeddings()
    return pipe
