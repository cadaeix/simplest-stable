from datetime import datetime
from IPython.display import display, clear_output
import torch
import json
from huggingface_hub import login
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, LMSDiscreteScheduler, DPMSolverSinglestepScheduler, DPMSolverMultistepScheduler
from src import utils, SimpleStableDiffusionPipeline

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
    "DPMSolver++ (2S)": {
        "type": "diffusers_DPMSolver",
        "sampler": DPMSolverSinglestepScheduler
    },
    "DPMSolver++ (2M)": {
        "type": "diffusers_DPMSolver",
        "sampler": DPMSolverMultistepScheduler
    }
}


def main(opt, pipe, recreate):
    model_choice = model_dict[opt["model_name"]]
    opt["prompt"] = utils.process_prompt(
        opt["prompt"], model_choice["keyword"])

    if pipe == None or recreate:
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
    utils.find_modules_and_assign_padding_mode(pipe, tiling_type)

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
        utils.set_seed(opt["seed"])
        image = pipe(**prompt_options).images[0]
        image_name = f"{batch_name}_{_b}"
        image.save(f"{image_name}.png")
        display(image)
        opt["seed"] += 1

        if opt["upscale"]:
            utils.find_modules_and_assign_padding_mode(pipe, "original")
            utils.sd_upscale(image, image_name, opt, pipe)

    return pipe
