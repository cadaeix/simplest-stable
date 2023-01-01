import os
import random
import math
import random
import requests
import torch
from tqdm import tqdm
import PIL
import numpy as np
from PIL import Image
from src import EverythingsPromptRandomizer
from collections import namedtuple

Grid = namedtuple("Grid", ["tiles", "tile_w", "tile_h", "image_w", "image_h", "overlap"])

def process_prompt(prompt, keyword):
    result = EverythingsPromptRandomizer.random_prompt(prompt)
    if keyword != "" and keyword not in prompt:
        result = f"({keyword}:1), {result}"
    return result


def set_seed(seed):
    seed = random.randint(0, 2**32) if seed < 0 else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #print(f"Using the seed {seed}")
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
    image = image.resize(shape, resample=PIL.Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float16) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def load_img_for_upscale(img, w, h):
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = img.resize((w, h), resample=PIL.Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def resize_image(image):
        return image.resize((int(image.size[0] * 2), int(image.size[1] * 2)), Image.Resampling.LANCZOS)

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

    mask_w = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((1, grid.overlap)).repeat(grid.tile_h, axis=0))
    mask_h = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((grid.overlap, 1)).repeat(grid.image_w, axis=1))

    combined_image = Image.new("RGB", (grid.image_w, grid.image_h))
    for y, h, row in grid.tiles:
        combined_row = Image.new("RGB", (grid.image_w, h))
        for x, w, tile in row:
            if x == 0:
                combined_row.paste(tile, (0, 0))
                continue

            combined_row.paste(tile.crop((0, 0, grid.overlap, h)), (x, 0), mask=mask_w)
            combined_row.paste(tile.crop((grid.overlap, 0, w, h)), (x + grid.overlap, 0))

        if y == 0:
            combined_image.paste(combined_row, (0, 0))
            continue

        combined_image.paste(combined_row.crop((0, 0, combined_row.width, grid.overlap)), (0, y), mask=mask_h)
        combined_image.paste(combined_row.crop((0, grid.overlap, combined_row.width, h)), (0, y + grid.overlap))

    return combined_image

def sd_upscale(image, name, opt, pipe):
    tile_w = 768
    tile_h = 768

    resized_image = resize_image(image)
    grid = split_grid(resized_image, tile_w=tile_w, tile_h=tile_h, overlap = 128)

    work = []

    for y, h, row in grid.tiles:
        for tiledata in row:
            work.append(tiledata[2])

    batch_count = len(work)

    work_results = []

    for i in range(batch_count):
        work_results.append(pipe(
            prompt = opt["prompt"],
            negative_prompt = None if opt["negative"] == "" else opt["negative"],
            image = load_img_for_upscale(work[i], tile_w, tile_h),
            strength = opt["upscale_strength"],
            height = tile_h,
            width = tile_w,
            num_inference_steps = opt["steps"],
            guidance_scale = opt["detail_scale"],
            num_images_per_prompt = 1,
            eta = opt["eta"]
        ).images[0])

    image_index = 0
    for y, h, row in grid.tiles:
        for tiledata in row:
            tiledata[2] = work_results[image_index] if image_index < len(work_results) else Image.new("RGB", (tile_w, tile_h))
            image_index += 1

    final_result = combine_grid(grid)
    final_result.save(f"{name}_upscale.png")
    display(final_result)
