from datetime import datetime
from IPython.display import display, clear_output
import torch
import json
from huggingface_hub import login
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, LMSDiscreteScheduler, DPMSolverSinglestepScheduler, DPMSolverMultistepScheduler
from src import utils, SimpleStableDiffusionPipeline
from PIL import Image, ImageFilter

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
m = open('src/models.json')
model_dict = json.load(m)
m.close()

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


def setup_pipe(model_opt):
    model_choice = model_dict[model_opt]
    if model_choice["vae"] != "":
        if model_choice["requires_hf_login"] or model_choice["vae"]["requires_hf_login"]:
            utils.login_to_huggingface()
        vae = AutoencoderKL.from_pretrained(model_choice["vae"]["url"])
        pipe = SimpleStableDiffusionPipeline.SimpleStableDiffusionPipeline.from_pretrained(
            model_choice["url"], vae=vae, safety_checker=None, requires_safety_checker=False).to("cuda")
    else:
        if model_choice["requires_hf_login"]:
            utils.login_to_huggingface()
        pipe = SimpleStableDiffusionPipeline.SimpleStableDiffusionPipeline.from_pretrained(
            model_choice["url"], safety_checker=None, requires_safety_checker=False).to("cuda")
    utils.find_modules_and_assign_padding_mode(pipe, "setup")
    # name = opt["model_name"]
    # print(f"{name} has been loaded!")
    # for emb_path in embeddings_list:
    #     pipe.embedding_database.add_embedding_path(emb_path)
    # pipe.load_embeddings()
    pipe.enable_attention_slicing()
    return pipe

def gradio_main(opt, pipe):
    model_choice = model_dict[opt["model_name"]]

    if sampler_dict[opt["sampler"]]["type"] == "diffusers":
        pipe.scheduler = sampler_dict[opt["sampler"]]["sampler"].from_pretrained(
            model_choice["url"], subfolder="scheduler")
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
    batch_name = datetime.now().strftime("%H_%M_%S")
    for _b in range(opt["number_of_images"]):
        utils.find_modules_and_assign_padding_mode(pipe, tiling_type)
        utils.set_seed(opt["seed"])
        prompt_options["prompt"] = utils.process_prompt_and_add_keyword(
            opt["prompt"], model_choice["keyword"] if opt["add_keyword"] else "")
        if prompt_options["negative_prompt"]:
            prompt_options["negative_prompt"] = utils.process_prompt_and_add_keyword(
                opt["negative"], "")

        image = pipe(**prompt_options).images[0]
        image_name = f"{batch_name}_{_b}"
        image.save(f"{image_name}.png")
        saved_image = image

        if opt["upscale"]:
            utils.find_modules_and_assign_padding_mode(pipe, "original")
            saved_image = utils.sd_upscale_gradio(image, image_name, opt, pipe)

        images.append(saved_image)
        opt["seed"] += 1

    return images

def main(opt, pipe, recreate, embeddings_list):
    model_choice = model_dict[opt["model_name"]]

    if pipe == None or recreate:
        print("Loading the model... If this is the first time downloading this model this session, this may take a while...")
        if model_choice["vae"] != "":
            if model_choice["requires_hf_login"] or model_choice["vae"]["requires_hf_login"]:
                utils.login_to_huggingface()
            vae = AutoencoderKL.from_pretrained(model_choice["vae"]["url"])
            pipe = SimpleStableDiffusionPipeline.SimpleStableDiffusionPipeline.from_pretrained(
                model_choice["url"], vae=vae, safety_checker=None, requires_safety_checker=False).to("cuda")
        else:
            if model_choice["requires_hf_login"]:
                utils.login_to_huggingface()
            pipe = SimpleStableDiffusionPipeline.SimpleStableDiffusionPipeline.from_pretrained(
                model_choice["url"], safety_checker=None, requires_safety_checker=False).to("cuda")
        utils.find_modules_and_assign_padding_mode(pipe, "setup")
        clear_output(wait=False)
        name = opt["model_name"]
        print(f"{name} has been loaded!")
        for emb_path in embeddings_list:
            pipe.embedding_database.add_embedding_path(emb_path)
        pipe.load_embeddings()

    if sampler_dict[opt["sampler"]]["type"] == "diffusers":
        pipe.scheduler = sampler_dict[opt["sampler"]]["sampler"].from_pretrained(
            model_choice["url"], subfolder="scheduler")
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

    pipe.enable_attention_slicing()
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

    batch_name = datetime.now().strftime("%H_%M_%S")
    for _b in range(opt["batches"]):
        utils.find_modules_and_assign_padding_mode(pipe, tiling_type)
        utils.set_seed(opt["seed"])
        prompt_options["prompt"] = utils.process_prompt_and_add_keyword(
            opt["prompt"], model_choice["keyword"] if opt["add_keyword"] else "")
        if prompt_options["negative_prompt"]:
            prompt_options["negative_prompt"] = utils.process_prompt_and_add_keyword(
                opt["negative"], "")

        print(prompt_options["prompt"])
        image = pipe(**prompt_options).images[0]
        image_name = f"{batch_name}_{_b}"
        image.save(f"{image_name}.png")
        display(image)
        opt["seed"] += 1

        if opt["upscale"]:
            utils.find_modules_and_assign_padding_mode(pipe, "original")
            utils.sd_upscale(image, image_name, opt, pipe)

    return pipe
