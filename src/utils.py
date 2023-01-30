import gc
import os
import re
import os
import random
import math
import random
import requests
import torch
import glob
import json
import PIL
import numpy as np
import torch.nn as nn
from PIL import Image, PngImagePlugin
from src import EverythingsPromptRandomizer
from collections import namedtuple
from packaging import version


def mini_model_lookup():  # this is awful, fix this soon
    with open('src/resources/models.json') as modelfile:
        model_dict = json.load(modelfile)

    model_dict_under_urls = {}
    for i in model_dict.items():
        if i[1]["type"] == "hf-file":
            model_name = i[0]
        elif i[1]["type"] == "diffusers":
            model_name = i[1]["repo_id"]

        model_dict_under_urls[model_name] = {
            "keyword": i[1]["keyword"],
            "prediction_type": i[1]["prediction"]
        }

    return model_dict_under_urls


model_dict_under_urls = mini_model_lookup()

try:
    from diffusers.utils import PIL_INTERPOLATION
except ImportError:
    if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
        PIL_INTERPOLATION = {
            "linear": PIL.Image.Resampling.BILINEAR,
            "bilinear": PIL.Image.Resampling.BILINEAR,
            "bicubic": PIL.Image.Resampling.BICUBIC,
            "lanczos": PIL.Image.Resampling.LANCZOS,
            "nearest": PIL.Image.Resampling.NEAREST,
        }
    else:
        PIL_INTERPOLATION = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
            "nearest": PIL.Image.NEAREST,
        }

Grid = namedtuple("Grid", ["tiles", "tile_w",
                  "tile_h", "image_w", "image_h", "overlap"])


def free_ram():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def process_embeddings_folder(embeddings_path):
    return glob.glob(os.path.join(embeddings_path, "*.pt")) + glob.glob(os.path.join(embeddings_path, "*.bin"))


def get_huggingface_cache_path():
    return os.path.join(os.path.expanduser('~'), ".cache", "huggingface", "diffusers")


def process_custom_model_glob(globlist):
    results = {}
    for file in globlist:
        stemname, filename = os.path.split(file)
        basename, _ = os.path.splitext(filename)
        config = os.path.join(stemname, f"{basename}.yaml") if os.path.exists(
            os.path.join(stemname, f"{basename}.yaml")) else None
        vae = os.path.join(stemname, f"{basename}.vae.pt") if os.path.exists(
            os.path.join(stemname, f"{basename}.vae.pt")) else None
        kw = re.search(r"\[(.*?)\]", basename)
        if kw:
            kw = kw.group(0).replace('[', '').replace(']', '')
            kw = kw.split(',')

        results[basename] = {
            "path": file,
            "config": config,
            "vae": vae,
            "keywords": kw
        }

    return results


def find_custom_models(path):
    if not path:
        return {}
    if not os.path.exists(path):
        print("Could not find path!")
        # return statement

    # assume ones without yamls are v1/epsilon
    ckpts = glob.glob(os.path.join(path, "*.ckpt"))
    safetensors = glob.glob(os.path.join(path, "*.safetensors"))

    return {**process_custom_model_glob(ckpts), **process_custom_model_glob(safetensors)}


def get_info(name, folderpath, model_dict, custom_model_dict=None):
    if name in model_dict:
        return model_dict[name]
    elif custom_model_dict and name in custom_model_dict:
        prediction_type = get_prediction_type_from_diffusers_cache(folderpath)
        return {
            'keyword': custom_model_dict[name]["keywords"],
            'prediction_type': prediction_type
        }
    else:
        prediction_type = get_prediction_type_from_diffusers_cache(folderpath)
        return {
            'keyword': "",
            'prediction_type': prediction_type
        }


def get_prediction_type_from_diffusers_cache(folderpath):
    with open(os.path.join(folderpath, "scheduler", "scheduler_config.json")) as jsonfile:
        model_json = json.load(jsonfile)
    if "prediction_type" not in model_json:
        prediction_type = "epsilon"
    else:
        prediction_type = model_json["prediction_type"]

    return prediction_type


def check_saved_models(custom_model_dict=None):
    folderpath = os.path.join(os.path.expanduser(
        '~'), ".cache", "huggingface", "diffusers", "*", "model_index.json")

    result = {}
    for model_folderpath in glob.glob(folderpath):
        folderpath, _ = os.path.split(model_folderpath)
        _, name = os.path.split(folderpath)
        info = get_info(name, folderpath,
                        model_dict_under_urls, custom_model_dict)
        info["path"] = folderpath
        result[name] = info
    return result


def check_cached_models(custom_model_dict=None):
    folderpath = os.path.join(os.path.expanduser(
        '~'), ".cache", "huggingface", "diffusers", "*", "snapshots", "*", "model_index.json")

    result = {}
    for model_folderpath in glob.glob(folderpath):
        with open(model_folderpath) as json_file:
            model_json = json.load(json_file)
        if model_json["_class_name"] == "StableDiffusionPipeline":
            pathlist = str.split(model_folderpath, os.path.sep)
            if "model" in pathlist[-4]:
                name = pathlist[-4][8:].replace("--", "/")
                folderpath, _ = os.path.split(model_folderpath)
                info = get_info(name, folderpath,
                                model_dict_under_urls, custom_model_dict)
                info["path"] = folderpath
                result[name] = info
    return result


def get_all_cached_hf_models(custom_model_dict=None):
    return {**check_saved_models(custom_model_dict), **check_cached_models(custom_model_dict)}


def save_image(image, image_name, prompt_options, opt, seed, outputs_folder, program_version, is_upscale=False):
    pnginfo = PngImagePlugin.PngInfo()

    if opt["sampler"] == "DPMSolver++ (2S) (has issues with img2img)":
        opt["sampler"] = "DPMSolver++ (2S)"

    prompt_info = f'Prompt: {prompt_options["prompt"]}' if prompt_options["prompt"] else ""
    negative_info = f'\nNegative: {prompt_options["negative_prompt"]}' if prompt_options["negative_prompt"] else ""
    upscale_options = f'\t\t\tUpscale Strength: {opt["upscale_strength"]}' if is_upscale else ""
    tiling_options = f'\t\\ttTiling: True' if opt["tiling"] else ""

    settings_info = f'{prompt_info}{negative_info}\nSeed: {seed}\t\t\tSteps: {str(prompt_options["num_inference_steps"])}\t\t\tSampler: {opt["sampler"]}\t\t\tGuidance Scale: {prompt_options["guidance_scale"]}\t\tResolution: {opt["W"]}x{opt["H"]}{upscale_options}{tiling_options}\nModel: {opt["model_name"]}\t\t\tProgram: {program_version}'

    pnginfo.add_text("parameters", settings_info)

    filepath = os.path.join(outputs_folder, f"{image_name}.png")
    image.save(filepath, pnginfo=pnginfo)
    return settings_info


def find_modules_and_assign_padding_mode(pipe, mode):
    module_names, _, _ = pipe.extract_init_dict(dict(pipe.config))
    for module_name in module_names:
        module = getattr(pipe, module_name)
        if isinstance(module, torch.nn.Module):
            for m in module.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    if mode == "setup":
                        m._orig_padding_mode = m.padding_mode
                    elif mode == "original":
                        m.padding_mode = m._orig_padding_mode
                    elif mode == "tiling":
                        m.padding_mode = 'circular'


def login_to_huggingface():
    print("This model requires an authentication token, which you can get by logging in at https://huggingface.co/ and going to https://huggingface.co/settings/tokens.")
    print("You may also have to accept the model's terms of service.")

    token = input("What is your huggingface token?:")
    login(token)


def process_prompt_and_add_keyword(prompt, keyword):
    result = EverythingsPromptRandomizer.random_prompt(prompt)
    result = EverythingsPromptRandomizer.random_prompt(
        prompt)  # run it twice because there's some sublists
    if type(keyword) is list:
        for kw in keyword:
            kw_strip = kw.strip()
            if kw_strip not in prompt:
                result = f"({kw_strip}:1), {result}"
    elif keyword != "" and keyword != None and keyword not in prompt:
        result = f"({keyword}:1), {result}"
    return result


def set_seed(seed):
    seed = random.randint(0, 2**32) if seed < 0 else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Using the seed {seed}")
    return seed


def load_img(path, shape):
    if path.startswith('http://') or path.startswith('https://'):
        image = Image.open(requests.get(path, stream=True).raw).convert('RGB')
    else:
        if os.path.isdir(path):
            files = [file for file in os.listdir(path) if file.endswith(
                '.png') or file .endswith('.jpg')]
            path = os.path.join(path, random.choice(files))
            print(f"Chose random init image {path}")
        image = Image.open(path).convert('RGB')
    image = image.resize(shape, resample=PIL_INTERPOLATION["lanczos"])
    image = np.array(image).astype(np.float16) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def load_img_for_upscale(img, w, h):
    # resize to integer multiple of 32
    w, h = map(lambda x: x - x % 32, (w, h))
    image = img.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def resize_image(image):
    return image.resize((int(image.size[0] * 2), int(image.size[1] * 2)), PIL_INTERPOLATION["lanczos"])


def split_grid(image, tile_w=512, tile_h=512, overlap=64):
    w = image.width
    h = image.height

    non_overlap_width = tile_w - overlap
    non_overlap_height = tile_h - overlap

    cols = math.ceil((w - overlap) / non_overlap_width)
    rows = math.ceil((h - overlap) / non_overlap_height)

    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

    grid = Grid([], tile_w, tile_h, w, h, overlap)
    for row in range(rows):
        row_images = []

        y = int(row * dy)

        if y + tile_h >= h:
            y = h - tile_h

        for col in range(cols):
            x = int(col * dx)

            if x + tile_w >= w:
                x = w - tile_w

            tile = image.crop((x, y, x + tile_w, y + tile_h))

            row_images.append([x, tile_w, tile])

        grid.tiles.append([y, tile_h, row_images])

    return grid


def combine_grid(grid):
    def make_mask_image(r):
        r = r * 255 / grid.overlap
        r = r.astype(np.uint8)
        return Image.fromarray(r, 'L')

    mask_w = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape(
        (1, grid.overlap)).repeat(grid.tile_h, axis=0))
    mask_h = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape(
        (grid.overlap, 1)).repeat(grid.image_w, axis=1))

    combined_image = Image.new("RGB", (grid.image_w, grid.image_h))
    for y, h, row in grid.tiles:
        combined_row = Image.new("RGB", (grid.image_w, h))
        for x, w, tile in row:
            if x == 0:
                combined_row.paste(tile, (0, 0))
                continue

            combined_row.paste(
                tile.crop((0, 0, grid.overlap, h)), (x, 0), mask=mask_w)
            combined_row.paste(
                tile.crop((grid.overlap, 0, w, h)), (x + grid.overlap, 0))

        if y == 0:
            combined_image.paste(combined_row, (0, 0))
            continue

        combined_image.paste(combined_row.crop(
            (0, 0, combined_row.width, grid.overlap)), (0, y), mask=mask_h)
        combined_image.paste(combined_row.crop(
            (0, grid.overlap, combined_row.width, h)), (0, y + grid.overlap))

    return combined_image
