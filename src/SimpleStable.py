from datetime import datetime
import random
import os
from IPython.display import display, clear_output
import torch
import json
from huggingface_hub import hf_hub_download
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, LMSDiscreteScheduler, DPMSolverSinglestepScheduler, DPMSolverMultistepScheduler
from src import utils, SimpleStableDiffusionPipeline, run_ckpt
from PIL import Image, ImageFilter, PngImagePlugin

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
with open('src/models.json') as modelfile:
    model_dict = json.load(modelfile)

sampler_dict = {
        "Euler a": {
            "type": "diffusers",
            "sampler": EulerAncestralDiscreteScheduler
        },
        "Euler": {
            "type": "diffusers",
            "sampler": EulerDiscreteScheduler
        },
        "KLMS": {
            "type": "diffusers",
            "sampler": LMSDiscreteScheduler
        },
        "DPMSolver++ (2S) (has issues with img2img)": {
            "type": "diffusers_DPMSolver",
            "sampler": DPMSolverSinglestepScheduler
        },
        "DPMSolver++ (2M)": {
            "type": "diffusers_DPMSolver",
            "sampler": DPMSolverMultistepScheduler
        }
}

sampler_list = list(sampler_dict.keys())

res_dict = {"Custom (Select this and put width and height below)": "",
            "Square 512x512 (default, good for most models)": [512,512],
            "Landscape 768x512": [768,512],
            "Portrait 512x768": [512,768],
            "Square 768x768 (good for 768 models)": [768,768],
            "Landscape 1152x768 (does not work on free colab)": [1152,768],
            "Portrait 768x1152 (does not work on free colab)":[768,1152]}


def load_cached_model(model_name, custom_model_dict, cached_model_dict):
    pipe_info = cached_model_dict[model_name]
    pipe = SimpleStableDiffusionPipeline.SimpleStableDiffusionPipeline.from_pretrained(
            pipe_info["path"], safety_checker=None, requires_safety_checker=False, local_files_only=True).to("cuda")
    utils.find_modules_and_assign_padding_mode(pipe, "setup")
    #pipe.enable_attention_slicing()
    return pipe, pipe_info


def load_custom_model(model_name, custom_model_dict, cached_model_dict = None):
    custom_model = custom_model_dict[model_name]
    hf_cache_folder = utils.get_huggingface_cache_path()

    pipe, prediction_type = run_ckpt.run_and_cache_custom_model(custom_model["path"], model_name, hf_cache_folder, custom_model["yaml"], custom_model["vae"], True)

    pipe_info = {
        "keyword": custom_model["keywords"],
        "prediction_type": prediction_type
    }
    utils.find_modules_and_assign_padding_mode(pipe, "setup")

    return pipe, pipe_info

def load_diffusers_model(model_choice):
    if model_choice["vae"] != "":
        if model_choice["requires_hf_login"] or model_choice["vae"]["requires_hf_login"]:
            utils.login_to_huggingface()
        vae = AutoencoderKL.from_pretrained(model_choice["vae"]["repo_id"])
        pipe = SimpleStableDiffusionPipeline.SimpleStableDiffusionPipeline.from_pretrained(
            model_choice["repo_id"], vae=vae, safety_checker=None, requires_safety_checker=False).to("cuda")
    else:
        if model_choice["requires_hf_login"]:
            utils.login_to_huggingface()
        pipe = SimpleStableDiffusionPipeline.SimpleStableDiffusionPipeline.from_pretrained(
            model_choice["repo_id"], safety_checker=None, requires_safety_checker=False).to("cuda")
    return pipe

def load_downloadable_model(model_name, custom_model_dict=None, cached_model_dict=None):
    model_choice = model_dict[model_name]

    if model_choice["type"] == "diffusers":
        pipe = load_diffusers_model(model_choice)
    elif model_choice["type"] == "hf-file":
        if model_name not in cached_model_dict:
            print("Loading from HF")
            hf_cache_folder = utils.get_huggingface_cache_path()
            ckpt = hf_hub_download(repo_id=model_choice["repo_id"], filename=model_choice["filename"])
            if model_choice["config"] != "":
                config = hf_hub_download(repo_id=model_choice["repo_id"], filename=model_choice["config"])
            else:
                config = None
            if model_choice["vae"] != "":
                vae = hf_hub_download(repo_id=model_choice["repo_id"], filename=model_choice["vae"])
            else:
                vae = None
            pipe, _ = run_ckpt.run_and_cache_custom_model(ckpt, model_name, hf_cache_folder, config, vae, True)
        else:
            print("loading from cache")
            return load_cached_model(model_name, custom_model_dict, cached_model_dict)

    utils.find_modules_and_assign_padding_mode(pipe, "setup")

    pipe_info = {
        "keyword": model_choice["keyword"],
        "prediction_type": model_choice["prediction"]
    }

    return pipe, pipe_info

def load_embeddings(embeddings_folder, pipe):
    if embeddings_folder and os.path.exists(embeddings_folder):
        emb_list = utils.process_embeddings_folder(embeddings_folder)

        for emb_path in emb_list:
            pipe.embedding_database.add_embedding_path(emb_path)
        pipe.load_embeddings()

def gradio_main(opt, pipe):
    if sampler_dict[opt["sampler"]]["type"] == "diffusers":
        pipe.scheduler = sampler_dict[opt["sampler"]]["sampler"](
            beta_end = 0.012,
            beta_schedule = "scaled_linear",
            beta_start = 0.00085,
            num_train_timesteps = 1000,
            prediction_type = opt["prediction_type"],
            trained_betas = None,
        )
    elif sampler_dict[opt["sampler"]]["type"] == "diffusers_DPMSolver":
        pipe.scheduler = sampler_dict[opt["sampler"]]["sampler"](
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            trained_betas=None,
            prediction_type=opt["prediction_type"],
            thresholding=False,
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            lower_order_final=True,
            solver_order=2
        )

    tiling_type = "tiling" if opt["tiling"] else "original"

    if opt["init_img"] != None:
        if opt["mask_image"] != None: #inpainting

            mask = opt["mask_image"].resize([opt["W"], opt["H"]])
            mask_image = mask.filter(ImageFilter.GaussianBlur(radius=4))

            prompt_options = {
                "prompt": opt["prompt"],
                "negative_prompt": None if opt["negative"] == "" else opt["negative"],
                "image": opt["init_img"].resize([opt["W"], opt["H"]]),
                "mask_image": mask_image,
                "strength": opt["strength"],
                "height": opt["H"],
                "width": opt["W"],
                "num_inference_steps": opt["steps"],
                "guidance_scale": opt["scale"],
                "num_images_per_prompt": 1,
                "eta": opt["eta"]
            }
        else: #img2img
            prompt_options = {
                "prompt": opt["prompt"],
                "negative_prompt": None if opt["negative"] == "" else opt["negative"],
                "image": opt["init_img"].resize([opt["W"], opt["H"]]),
                "strength": opt["strength"],
                "height": opt["H"],
                "width": opt["W"],
                "num_inference_steps": opt["steps"],
                "guidance_scale": opt["scale"],
                "num_images_per_prompt": 1,
                "eta": opt["eta"]
            }
    else: #txt2img
        prompt_options = {
            "prompt": opt["prompt"],
            "negative_prompt": None if opt["negative"] == "" else opt["negative"],
            "height": opt["H"],
            "width": opt["W"],
            "num_inference_steps": opt["steps"],
            "guidance_scale": opt["scale"],
            "num_images_per_prompt": 1,
            "eta": opt["eta"]
        }

    images = []
    images_details = []
    batch_name = datetime.now().strftime("%H_%M_%S")
    seed = random.randint(0, 2**32) if opt["seed"] < 0 else opt["seed"]
    for _b in range(opt["number_of_images"]):
        utils.set_seed(seed)
        utils.find_modules_and_assign_padding_mode(pipe, tiling_type)
        prompt_options["prompt"] = utils.process_prompt_and_add_keyword(
            opt["prompt"], opt["keyword"] if opt["add_keyword"] else "")
        if prompt_options["negative_prompt"]:
            prompt_options["negative_prompt"] = utils.process_prompt_and_add_keyword(
                opt["negative"], "")

        image = pipe(**prompt_options).images[0]
        image_name = f"{batch_name}_{seed}_{_b}"

        settings_info = utils.save_image(image, image_name, prompt_options, opt, seed, opt["outputs_folder"])

        saved_image = image

        if opt["upscale"]:
            utils.find_modules_and_assign_padding_mode(pipe, "original")
            saved_image = utils.sd_upscale_gradio(image, image_name, opt, pipe, seed)

        images.append(saved_image)
        images_details.append(settings_info)
        seed += 1

    return images, images_details

def main(opt, pipe, recreate, embeddings_list):
    model_choice = model_dict[opt["model_name"]]

    if pipe == None or recreate:
        del pipe
        all_cached_hf_models = utils.get_all_cached_hf_models([])
        print("Loading the model... If this is the first time downloading this model this session, this may take a while...")
        pipe, pipe_info = load_downloadable_model(opt["model_name"], custom_model_dict={}, cached_model_dict=all_cached_hf_models)
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
        clear_output(wait=False)
        name = opt["model_name"]
        print(f"{name} has been loaded!")
        for emb_path in embeddings_list:
            pipe.embedding_database.add_embedding_path(emb_path)
        pipe.load_embeddings()

    if sampler_dict[opt["sampler"]]["type"] == "diffusers":
        pipe.scheduler = sampler_dict[opt["sampler"]]["sampler"].from_pretrained(
            model_choice["repo_id"], subfolder="scheduler")
    elif sampler_dict[opt["sampler"]]["type"] == "diffusers_DPMSolver":
        pipe.scheduler = sampler_dict[opt["sampler"]]["sampler"](
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            trained_betas=None,
            prediction_type=model_choice["prediction"],
            thresholding=False,
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            lower_order_final=True,
            solver_order=2
        )

    tiling_type = "tiling" if opt["tiling"] else "original"

    if opt["init_img"] != None:
        prompt_options = {
            "prompt": opt["prompt"],
            "negative_prompt": None if opt["negative"] == "" else opt["negative"],
            "image": utils.load_img(opt["init_img"], shape=(opt["W"], opt["H"])),
            "strength": opt["strength"],
            "height": opt["H"],
            "width": opt["W"],
            "num_inference_steps": opt["steps"],
            "guidance_scale": opt["scale"],
            "num_images_per_prompt": 1,
            "eta": opt["eta"]
        }
    else:
        prompt_options = {
            "prompt": opt["prompt"],
            "negative_prompt": None if opt["negative"] == "" else opt["negative"],
            "height": opt["H"],
            "width": opt["W"],
            "num_inference_steps": opt["steps"],
            "guidance_scale": opt["scale"],
            "num_images_per_prompt": 1,
            "eta": opt["eta"]
        }

    seed = random.randint(0, 2**32) if opt["seed"] < 0 else opt["seed"]
    batch_name = datetime.now().strftime("%H_%M_%S")
    for _b in range(opt["batches"]):
        utils.set_seed(seed)
        utils.find_modules_and_assign_padding_mode(pipe, tiling_type)
        prompt_options["prompt"] = utils.process_prompt_and_add_keyword(
            opt["prompt"], model_choice["keyword"] if opt["add_keyword"] else "")
        if prompt_options["negative_prompt"]:
            prompt_options["negative_prompt"] = utils.process_prompt_and_add_keyword(
                opt["negative"], "")

        print(prompt_options["prompt"])
        image = pipe(**prompt_options).images[0]
        image_name = f"{batch_name}_{seed}_{_b}"
        utils.save_image(image, image_name, prompt_options, opt, seed, opt["outputs_folder"])
        display(image)
        seed += 1

        if opt["upscale"]:
            utils.find_modules_and_assign_padding_mode(pipe, "original")
            utils.sd_upscale(image, image_name, opt, pipe)

    return pipe
