# Utils are placed here when they don't yet have a better home :)

import requests
import torch
from PIL import Image
from transformers import logging

# Set logging level to suppress info messages
logging.set_verbosity_error()


def load_image_from_url(url: str) -> Image.Image:
    """Load an image from a given URL."""
    return Image.open(requests.get(url, stream=True).raw)


def prepare_inputs(processor, messages, image=None):
    """Prepare inputs for the model."""
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = (
        processor(text=[text], images=image, return_tensors="pt")
        if image
        else processor(text=[text], return_tensors="pt")
    )
    return inputs


def move_inputs_to_device(inputs, model):
    """Move inputs to the appropriate device and dtype."""
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    return {k: v.to(model.device) for k, v in inputs.items()}


def generate_output(model, inputs, processor, max_new_tokens=500, **kwargs):
    """Generate output from the model."""
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            tokenizer=processor.tokenizer,
            **kwargs,
        )
        output_ids = output[0][inputs["input_ids"].shape[1] :]
        result = processor.decode(output_ids, skip_special_tokens=True)
    return result
