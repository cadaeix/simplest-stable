import importlib
import random
import json
from datetime import datetime
from typing import List, Optional, Tuple
import torch
from src.SimpleStableDiffusionPipeline import SimpleStableDiffusionPipeline
from src.utils import (
    combine_grid,
    load_img,
    load_img_for_upscale,
    resize_image,
    set_seed,
    find_modules_and_assign_padding_mode,
    process_prompt_and_add_keyword,
    save_image,
    split_grid,
    free_ram
)
from PIL import Image, ImageFilter


with open('src/resources/schedulers.json') as schedulerfile:
    scheduler_dict = json.load(schedulerfile)
del schedulerfile

res_dict = {"Custom (Select this and put width and height below)": "",
            "Square 512x512 (default, good for most models)": [512, 512],
            "Landscape 768x512": [768, 512],
            "Portrait 512x768": [512, 768],
            "Square 768x768 (good for 768 models)": [768, 768],
            "Landscape 1152x768 (does not work on free colab)": [1152, 768],
            "Portrait 768x1152 (does not work on free colab)": [768, 1152]}


def load_sampler(sampler_name: str, model_prediction_type: str, pipe: SimpleStableDiffusionPipeline) -> SimpleStableDiffusionPipeline:
    # TODO: replace with k-diffusion samplers, probably in the pipeline itself
    scheduler_info = scheduler_dict[sampler_name]

    library = importlib.import_module("diffusers")
    scheduler = getattr(library, scheduler_info["sampler"])

    pipe.scheduler = scheduler(
        **scheduler_info["params"],
        prediction_type=model_prediction_type
    )

    return pipe

# opt params:
    # "model_name": model_name,
    # "prompt": prompt,
    # "negative": negative
    # "init_img": init_img,
    # "mask_image": mask_image,
    # "strength": strength,
    # "number_of_images": number_of_images,
    # "H" : height - height % 64,
    # "W" : width - width % 64,
    # "steps": steps,
    # "sampler": sampler,
    # "scale": scale,
    # "eta" : 0.0,
    # "tiling" : "Tiling" in additional_options,
    # "upscale": "SD Upscale" in additional_options,
    # "upscale_strength": upscale_strength
    # "detail_scale" : 10,
    # "seed": used_seed,
    # "add_keyword": "Don't insert model keyword" not in additional_options,
    # "keyword": pipe_info["keyword"],
    # "outputs_folder": session_folder,
    # "prediction_type": pipe_info["prediction_type"]
    # "program_version": "Simple Stable 2.0 (Gradio UI)" or "Simple Stable 2.0 (Notebook)"


def process_and_generate(
        opt: dict,
        pipe: SimpleStableDiffusionPipeline,
        # gradio.progress, don't want to import it in this file
        progress: Optional[any],
        randomizer: dict,
        display_and_print: bool = False,
        save_settings_as_text: Optional[bool] = True) -> Tuple[SimpleStableDiffusionPipeline, List, List]:

    # load sampler
    pipe = load_sampler(opt["sampler"], opt["prediction_type"], pipe)

    if "tiling" in opt:
        tiling_type = "tiling" if opt["tiling"] else "original"

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

    if opt["init_img"] != None:  # img2img
        if opt["mask_image"] != None:  # inpainting
            mask = opt["mask_image"].resize([opt["W"], opt["H"]])
            mask_image = mask.filter(ImageFilter.GaussianBlur(radius=4))

            prompt_options["mask_image"] = mask_image

            if opt.get("latent_noise_inpaint"):
                if opt["latent_noise_inpaint"]:
                    prompt_options["latent_noise_inpaint"] = True

        if type(opt["init_img"]) is str:
            prompt_options["image"] = load_img(
                opt["init_img"], shape=(opt["W"], opt["H"]))
        else:
            prompt_options["image"] = opt["init_img"].resize(
                [opt["W"], opt["H"]])
        prompt_options["strength"] = opt["strength"]

    if "controlnet_model" in opt and "controlnet_image" in opt:
        prompt_options["controlnet_model"] = opt["controlnet_model"]
        prompt_options["controlnet_image"] = opt["controlnet_image"]

    # generation
    # if progress:
    #     progress(
    #         0, desc=f'Preparing to generate {opt["number_of_images"]} number of image(s)...')

    saved_settings = {"settings": opt, "prompts": []}

    images = []
    images_details = []
    batch_name = datetime.now().strftime("%H_%M_%S")
    seed = random.randint(0, 2**32) if opt["seed"] < 0 else opt["seed"]
    for index in range(opt["number_of_images"]):
        free_ram()
        set_seed(seed)
        if "tiling" in opt:
            find_modules_and_assign_padding_mode(pipe, tiling_type)
        prompt_options["prompt"] = process_prompt_and_add_keyword(
            opt["prompt"], opt["keyword"] if opt["add_keyword"] else None, randomizer)
        if prompt_options["negative_prompt"]:
            prompt_options["negative_prompt"] = process_prompt_and_add_keyword(
                opt["negative"], opt["negative_keyword"] if opt["add_keyword"] else None, randomizer)
        else:
            prompt_options["negative_prompt"] = process_prompt_and_add_keyword(
                "", opt["negative_keyword"] if opt["add_keyword"] else None, randomizer)

        image = pipe(**prompt_options).images[0]
        image_name = f"{batch_name}_{seed}_{index}"

        settings_info = save_image(
            image, image_name, prompt_options, opt, seed, opt["outputs_folder"], opt["program_version"])

        if display_and_print:
            print(prompt_options["prompt"])
            display(image)

        saved_image = image

        if opt["upscale"]:
            free_ram()
            if "tiling" in opt:
                find_modules_and_assign_padding_mode(pipe, "original")
            saved_image = generate_sd_upscale(
                image, image_name, prompt_options["prompt"], prompt_options["negative_prompt"], opt, pipe, seed, display_and_print)

        images.append(saved_image)
        images_details.append(settings_info)
        saved_settings["prompts"].append(
            [prompt_options["prompt"], prompt_options["negative_prompt"]])
        seed += 1

    if save_settings_as_text:
        with open(f"{opt['outputs_folder']}/{batch_name}_settings.txt", "w+", encoding="utf-8") as f:
            json.dump(saved_settings, f, ensure_ascii=False, indent=4)

    return pipe, images, images_details


def generate_higher_res_upscale(image: any, image_name: str, opt: dict, pipe: SimpleStableDiffusionPipeline, seed: int, display_and_print: bool = False) -> any:
    # automatically go for 2x
    upscale_factor = 2

    image = torch.nn.functional.interpolate(image, size=(int(
        image.size[0] * upscale_factor) // 8, int(image.size[1] * upscale_factor) // 8), mode="bilinear", antialias=False)


def generate_sd_upscale(image: any, image_name: str, prompt: str, negative: str, opt: dict, pipe: SimpleStableDiffusionPipeline, seed: int, display_and_print: bool = False) -> any:
    tile_w = 704
    tile_h = 704

    resized_image = resize_image(image)
    grid = split_grid(resized_image, tile_w=tile_w, tile_h=tile_h, overlap=128)

    work = []

    for y, h, row in grid.tiles:
        for tiledata in row:
            work.append(tiledata[2])

    batch_count = len(work)

    work_results = []

    prompt_options = {
        "prompt": prompt,
        "negative_prompt": None if negative == "" else negative,
        "strength": opt["upscale_strength"],
        "height": tile_h,
        "width": tile_w,
        "num_inference_steps": opt["steps"],
        "guidance_scale": opt["scale"],
        "num_images_per_prompt": 1,
        "eta": opt["eta"]
    }

    for i in range(batch_count):
        work_results.append(pipe(
            **prompt_options, image=load_img_for_upscale(work[i], tile_w, tile_h)).images[0])

    image_index = 0
    for y, h, row in grid.tiles:
        for tiledata in row:
            tiledata[2] = work_results[image_index] if image_index < len(
                work_results) else Image.new("RGB", (tile_w, tile_h))
            image_index += 1

    final_result = combine_grid(grid)

    save_image(final_result, f"{image_name}_upscale",
               prompt_options, opt, seed, opt["outputs_folder"], opt["program_version"])

    if display_and_print:
        display(final_result)

    return final_result
