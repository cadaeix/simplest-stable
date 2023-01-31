import os
import json
import random
from typing import Optional
from PIL import Image
from src.SimpleStableDiffusionPipeline import SimpleStableDiffusionPipeline
from src.loading import load_vae_file_to_current_pipe, prepare_pipe, load_embeddings
from src.process import process_and_generate, res_dict, scheduler_dict
from src.utils import find_custom_models, get_all_cached_hf_models, free_ram
import gradio as gr
import gradio.routes


class LoadJavaScript():
    def __init__(self):
        self.original_template = gradio.routes.templates.TemplateResponse

        with open("src/gradio.js", "r", encoding="utf8") as jsfile:
            self.javascript = f'<script>{jsfile.read()}</script>'

        gradio.routes.templates.TemplateResponse = self.template_response

    def template_response(self, *args, **kwargs):
        response = self.original_template(*args, **kwargs)
        response.body = response.body.replace(
            '</head>'.encode(
                'utf-8'), f"{''.join(self.javascript)}\n</head>".encode("utf-8")
        )
        response.init_headers()
        return response


def load_embeddings_from_folders(pipe: SimpleStableDiffusionPipeline, embeddings_path: Optional[str], downloaded_embeddings: Optional[str]) -> SimpleStableDiffusionPipeline:
    if embeddings_path:
        pipe = load_embeddings(embeddings_path, pipe)
    if downloaded_embeddings and os.path.exists(downloaded_embeddings):
        pipe = load_embeddings(downloaded_embeddings, pipe)
    return pipe


def return_selected_image_from_gallery(i: int) -> Optional[any]:
    return i["name"] if i else None

# gradio parts, don't feel like typing these right now


def mode_buttons():
    with gr.Row():
        txt2img_show = gr.Button(value="txt2img", variant="primary")
        img2img_show = gr.Button(value="img2img")
        inpaint_show = gr.Button(value="inpainting")
    return txt2img_show, img2img_show, inpaint_show


def prompt_section():
    prompt = gr.Textbox(placeholder="Describe a prompt here", label="Prompt")
    negative = gr.Textbox(placeholder="Negative prompt", label="Negative")

    return prompt, negative


def img2img_and_inpainting():
    input_image = gr.Image(value=None, source="upload", interactive=True,
                           type="pil", visible=False, elem_id="img2img_input")
    inpaint_image = gr.Image(value=None, source="upload", interactive=True,
                             type="pil", visible=False, tool="sketch", elem_id="inpaint_input")
    img2img_strength = gr.Slider(minimum=0.1, maximum=1, value=0.75, step=0.05,
                                 label="img2img strength", interactive=True, visible=False, elem_id="img2img_strength")
    inpaint_strength = gr.Slider(minimum=0.1, maximum=1, value=0.75, step=0.05,
                                 label="inpaint strength", interactive=True, visible=False, elem_id="inpaint_strength")
    return input_image, inpaint_image, img2img_strength, inpaint_strength


def image_options():
    with gr.Row():
        with gr.Column(scale=1):
            number_of_images = gr.Number(
                value=1, precision=0, label="Number of Images")
        with gr.Column(scale=2):
            resolution = gr.Dropdown(choices=list(res_dict.keys(
            )), label="Image Resolution", value="Square 512x512 (default, good for most models)")
    return number_of_images, resolution


def advanced_settings():
    with gr.Accordion(label="Advanced Settings", elem_id="adv_settings"):
        with gr.Row():
            with gr.Column():
                custom_width = gr.Slider(minimum=512, maximum=1152, value=512, step=64,
                                         label="Width (if Custom is selected)", interactive=True, elem_id="custom_width")
                sampler = gr.Dropdown(choices=list(scheduler_dict.keys(
                )), label="Sampler", value="Euler a", elem_id="sampler_choice")
                with gr.Row():
                    with gr.Column(elem_id="seed_col"):
                        seed = gr.Number(value=-1, precision=0,
                                         label="Seed", interactive=True)
                    with gr.Column(elem_id="seed_button_col"):
                        reuse_seed_button = gr.Button(
                            value="Last Seed", elem_id="reuse_seed")
                        random_seed_button = gr.Button(
                            value="Random Seed", elem_id="random_seed")
                additional_options = gr.CheckboxGroup(["Tiling", "SD Upscale", "Don't insert model keyword",
                                                      "Insert standard Danbooru model quality prompt"], interactive=True, label="Additional Settings")
            with gr.Column():
                custom_height = gr.Slider(minimum=512, maximum=1152, value=512, step=64,
                                          label="Height (if Custom is selected)", interactive=True, elem_id="custom_height")
                steps = gr.Slider(minimum=1, maximum=100, value=20, step=1,
                                  label="Step Count", interactive=True, elem_id="step_count")
                scale = gr.Slider(minimum=1, maximum=20, value=7, step=0.5,
                                  label="Guidance Scale", interactive=True, elem_id="guidance_scale")
                upscale_strength = gr.Slider(minimum=0.1, maximum=1, value=0.2, step=0.05,
                                             label="Upscale Strength", interactive=True, elem_id="upscale_strength")
    return custom_width, custom_height, steps, sampler, seed, scale, additional_options, upscale_strength, reuse_seed_button, random_seed_button


def output_section():
    with gr.Row(elem_id="generate_row"):
        generate_button = gr.Button(
            value="Generate", variant="primary", elem_id="generate_button")
        # interrupt_button = gr.Button(value="Interrupt", variant="secondary", elem_id="generate_button")
    image_output = gr.Gallery(elem_id="output_gallery")
    with gr.Row(elem_id="edit_row"):
        to_img2img_button = gr.Button(
            value="img2img Selected Image", variant="secondary", elem_id="to_img2img_button", visible=False)
        to_inpaint_button = gr.Button(
            value="inpaint Selected Image", variant="secondary", elem_id="to_inpaint_button", visible=False)
    log_output = gr.Textbox(interactive=False, elem_id="log_output",
                            show_label=False, visible=False, lines=7)
    return generate_button, image_output, to_img2img_button, to_inpaint_button, log_output

# main function
# xformers takes precedent over attention_slicing


def main(starting_model_to_load: str, outputs_folder: str, custom_models_path: Optional[str], embeddings_path: Optional[str], downloaded_embeddings: Optional[str], enable_attention_slicing: bool = False, enable_xformers: bool = False):
    global pipe, pipe_info, session_folder, model_dict, all_custom_models, all_cached_hf_models, all_vae_files

    with open('src/resources/models.json') as modelfile:
        model_dict = json.load(modelfile)
    del modelfile

    failed_placeholder = Image.open("src/resources/failedgen.png")

    try:
        pipe, pipe_info = prepare_pipe(
            starting_model_to_load, "Downloadable Models", model_dict, None, None, enable_attention_slicing, enable_xformers)
    except Exception as e:
        print(
            f"Failed to load selected model for some reason, defaulting to loading Stable Diffusion 1.5.\nError: {e}")
        pipe, pipe_info = prepare_pipe(
            "Stable Diffusion 1.5", "Downloadable Models", model_dict, None, None, enable_attention_slicing, enable_xformers)

    pipe = load_embeddings_from_folders(
        pipe, embeddings_path, downloaded_embeddings)

    all_custom_models, all_vae_files = find_custom_models(custom_models_path)
    all_cached_hf_models = get_all_cached_hf_models(all_custom_models)
    model_dropdown_type_choice = ["Installed Models", "Downloadable Models"]
    if all_custom_models != {}:
        model_dropdown_type_choice.append("Custom Models")
        if all_vae_files != {}:
            model_dropdown_type_choice.append(
                "Load Custom Vae To Current Model")

    def model_selections():
        with gr.Row(elem_id="model_row"):
            model_dropdown_type = gr.Dropdown(
                choices=model_dropdown_type_choice, show_label=False, elem_id="model_dropdown_type")
            downloadable_models = gr.Dropdown(choices=list(model_dict.keys(
            )), value="Stable Diffusion 1.5", show_label=False, elem_id="download_model_choice", interactive=True)
            cached_models = gr.Dropdown(choices=list(all_cached_hf_models.keys(
            )), show_label=False, elem_id="cached_model_choice", interactive=True)
            custom_models = gr.Dropdown(choices=list(all_custom_models.keys(
            )), show_label=False, elem_id="custom_model_choice", interactive=True)
            custom_vae = gr.Dropdown(
                choices=list(all_vae_files.keys()), show_label=False, elem_id="custom_vae_choice", interactive=True)
            model_submit = gr.Button(
                value="Load Model", interactive=True, elem_id="model_submit")
        loading_status = gr.Markdown("", elem_id="model_status")

        return model_dropdown_type, downloadable_models, cached_models, custom_models, model_submit, loading_status, custom_vae

    def update_model_lists():
        global all_custom_models, all_cached_hf_models, all_vae_files
        all_custom_models = find_custom_models(custom_models_path)
        all_cached_hf_models = get_all_cached_hf_models(all_custom_models)

    def update_model_lists_in_gradio():
        return {
            cached_models: gr.update(choices=list(all_cached_hf_models.keys())),
            custom_models: gr.update(choices=list(all_custom_models.keys()))
        }

    def has_image(input):
        return {
            to_img2img_button: gr.update(visible=(input is not None)),
            to_inpaint_button: gr.update(visible=(input is not None)),
            log_output: gr.update(visible=(input is not None)),
        }

    def show_state(input: str):
        states = {
            "txt2img": [True, False, False],
            "img2img": [False, True, False],
            "inpainting": [False, False, True]
        }

        txt2img_button = "primary" if states[input][0] else "secondary"
        img2img_button = "primary" if states[input][1] else "secondary"
        inpaint_button = "primary" if states[input][2] else "secondary"

        return {
            current_mode: input,
            input_image: gr.update(visible=states[input][1]),
            img2img_strength: gr.update(visible=states[input][1]),
            inpaint_image: gr.update(visible=states[input][2]),
            inpaint_strength: gr.update(visible=states[input][2]),
            txt2img_show: gr.update(variant=txt2img_button),
            img2img_show: gr.update(variant=img2img_button),
            inpaint_show: gr.update(variant=inpaint_button)
        }

    def show_state_and_clear_inpaint(input: str):
        states = {
            "txt2img": [True, False, False],
            "img2img": [False, True, False],
            "inpainting": [False, False, True]
        }

        txt2img_button = "primary" if states[input][0] else "secondary"
        img2img_button = "primary" if states[input][1] else "secondary"
        inpaint_button = "primary" if states[input][2] else "secondary"

        return {
            current_mode: input,
            input_image: gr.update(visible=states[input][1]),
            img2img_strength: gr.update(visible=states[input][1]),
            inpaint_image: gr.update(value=None, visible=states[input][2]),
            inpaint_strength: gr.update(visible=states[input][2]),
            txt2img_show: gr.update(variant=txt2img_button),
            img2img_show: gr.update(variant=img2img_button),
            inpaint_show: gr.update(variant=inpaint_button)
        }

    def choose_type_and_load_model(dropdown_type: str, loaded_model_name: str, downloadable_model_name: str, cached_model_name: str, custom_model_name: str, custom_vae_name: str, progress=gr.Progress(track_tqdm=True)):
        global pipe, pipe_info, all_custom_models, all_cached_hf_models, all_vae_files, model_dict

        if dropdown_type == "Installed Models" and cached_model_name:
            chosen_model_name = cached_model_name
        elif dropdown_type == "Downloadable Models" and downloadable_model_name:
            chosen_model_name = downloadable_model_name
        elif dropdown_type == "Custom Models" and custom_model_name:
            chosen_model_name = custom_model_name
        elif dropdown_type == "Load Custom Vae To Current Model":
            if loaded_model_name and custom_vae_name not in loaded_model_name:
                pipe = load_vae_file_to_current_pipe(
                    pipe, all_vae_files[custom_vae_name])
                return f"Loaded {custom_vae_name} VAE to current model", f"{loaded_model_name}+{custom_vae_name}"
            else:
                return "VAE already loaded", loaded_model_name
        else:
            return "Choice not valid", loaded_model_name

        if loaded_model_name == chosen_model_name:
            return "Model already loaded", loaded_model_name

        if pipe:
            del pipe
        free_ram()
        try:
            pipe, pipe_info = prepare_pipe(chosen_model_name,
                                           dropdown_type,
                                           model_dict,
                                           all_custom_models,
                                           all_cached_hf_models,  enable_attention_slicing,
                                           enable_xformers)
            return f"{chosen_model_name} loaded", chosen_model_name
        except Exception as e:
            return f"Failed to load model, generation will not work. Please try loading another model.\nError: {e}", "Please load another model!"

    def generate(mode, prompt, negative, number_of_images, resolution, custom_width, custom_height, steps, sampler, seed, scale, additional_options, upscale_strength, input_image, img2img_strength, inpaint_image, inpaint_strength, model_name, progress=gr.Progress(track_tqdm=True)):
        global pipe, pipe_info

        progress(0, desc="Starting generation...")

        if mode == "txt2img":
            init_img = None
            mask_image = None
            strength = None
        elif mode == "img2img":
            init_img = input_image
            mask_image = None
            strength = img2img_strength
        elif mode == "inpainting":
            init_img = inpaint_image["image"]
            mask_image = inpaint_image["mask"]
            strength = inpaint_strength

        negative = negative if negative != None else ""
        used_seed = random.randint(0, 2**32) if seed < 0 else seed
        width, height = [custom_width, custom_height] if (
            resolution == "Custom (Select this and put width and height below)") else res_dict[resolution]

        if "Insert standard Danbooru model quality prompt" in additional_options:
            prompt = "masterpiece, best quality, " + prompt
            standard_negative = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username"
            negative = standard_negative if negative is None else standard_negative + ", " + negative

        try:
            pipe, images, image_detail_list = process_and_generate({
                "model_name": model_name,
                "prompt": prompt,
                "negative": negative if negative != None else "",
                "init_img": init_img,
                "mask_image": mask_image,
                "strength": strength,
                "number_of_images": number_of_images,
                "H": height - height % 64,
                "W": width - width % 64,
                "steps": steps,
                "sampler": sampler,
                "scale": scale,
                "eta": 0.0,
                "tiling": "Tiling" in additional_options,
                "upscale": "SD Upscale" in additional_options,
                "upscale_strength": upscale_strength if "SD Upscale" in additional_options else None,
                "detail_scale": 10,
                "seed": used_seed,
                "add_keyword": "Don't insert model keyword" not in additional_options,
                "keyword": pipe_info["keyword"],
                "outputs_folder": session_folder,
                "prediction_type": pipe_info["prediction_type"],
                "program_version": "Simple Stable 2.0 (Gradio UI, pre-release 20230129)"
            }, pipe, progress, False)

            message = '\n\n'.join(image_detail_list)
            return images, used_seed, message
        except Exception as e:
            return [failed_placeholder], used_seed, f"Error: {e}"

    # control flow
    if not os.path.exists(outputs_folder):
        os.mkdir(outputs_folder)
    session_folder = outputs_folder

    print(f"Saving images to {session_folder}.")
    if custom_models_path:
        print(f"Found custom models folder at {custom_models_path}!")
    if embeddings_path:
        print(f"Found custom embeddings folder at {embeddings_path}!")

    print("Loading models and files...")

    css = ""

    with open("src/gradio.css") as file:
        css += file.read() + "\n"

    load_javascript = LoadJavaScript()
    with gr.Blocks(css=css, title="Simple Stable") as main:
        current_loaded_model_name = gr.Markdown("Stable Diffusion 1.5")
        current_mode = gr.State("txt2img")
        last_used_seed = gr.State(-1)

        model_dropdown_type, downloadable_models, cached_models, custom_models, model_submit, loading_status, custom_vae = model_selections()

        with gr.Row():
            with gr.Column(scale=3):
                txt2img_show, img2img_show, inpaint_show = mode_buttons()

                prompt, negative = prompt_section()

                number_of_images, resolution = image_options()

                input_image, inpaint_image, img2img_strength, inpaint_strength = img2img_and_inpainting()

                custom_width, custom_height, steps, sampler, seed, scale, additional_options, upscale_strength, reuse_seed_button, random_seed_button = advanced_settings()

            with gr.Column(scale=2):
                generate_button, image_output, to_img2img_button, to_inpaint_button, log_output = output_section()

        model_submit.click(choose_type_and_load_model, inputs=[model_dropdown_type, current_loaded_model_name, downloadable_models, cached_models, custom_models, custom_vae], outputs=[
                           loading_status, current_loaded_model_name], preprocess=False, postprocess=False)
        model_submit.click(update_model_lists_in_gradio, inputs=[], outputs=[
                           cached_models, custom_models])

        reuse_seed_button.click(lambda x: x, inputs=[last_used_seed], outputs=[
                                seed])
        random_seed_button.click(
            lambda: -1, inputs=[], outputs=[seed], preprocess=False, postprocess=False)

        txt2img_show.click(show_state, inputs=[txt2img_show], outputs=[current_mode, input_image, img2img_strength,
                           inpaint_image, inpaint_strength, txt2img_show, img2img_show, inpaint_show], preprocess=False, postprocess=False)
        img2img_show.click(show_state, inputs=[img2img_show], outputs=[current_mode, input_image, img2img_strength,
                           inpaint_image, inpaint_strength, txt2img_show, img2img_show, inpaint_show], preprocess=False, postprocess=False)
        inpaint_show.click(show_state, inputs=[inpaint_show], outputs=[current_mode, input_image, img2img_strength,
                           inpaint_image, inpaint_strength, txt2img_show, img2img_show, inpaint_show], preprocess=False, postprocess=False)

        to_img2img_button.click(show_state, inputs=[img2img_show], outputs=[current_mode, input_image, img2img_strength,
                                inpaint_image, inpaint_strength, txt2img_show, img2img_show, inpaint_show], preprocess=False, postprocess=False)
        to_inpaint_button.click(show_state_and_clear_inpaint, inputs=[inpaint_show], outputs=[
                                current_mode, input_image, img2img_strength, inpaint_image, inpaint_strength, txt2img_show, img2img_show, inpaint_show], preprocess=False, postprocess=False)
        to_img2img_button.click(return_selected_image_from_gallery, inputs=[
                                image_output], outputs=[input_image], _js="findSelectedImageFromGallery")
        to_inpaint_button.click(return_selected_image_from_gallery, inputs=[
                                image_output], outputs=[inpaint_image], _js="findSelectedImageFromGallery")

        generate_button.click(generate, inputs=[current_mode, prompt, negative, number_of_images, resolution, custom_width, custom_height, steps, sampler, seed, scale, additional_options,
                              upscale_strength, input_image, img2img_strength, inpaint_image, inpaint_strength, current_loaded_model_name], outputs=[image_output, last_used_seed, log_output])
        generate_button.click(has_image, inputs=[image_output], outputs=[
                              to_img2img_button, to_inpaint_button, log_output])
        model_dropdown_type.change(
            None, _js="handleModelDropdowns", inputs=[], outputs=[])

    main.queue()
    main.launch(debug=True, inline=True, height=1000)
