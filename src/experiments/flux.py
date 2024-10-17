import io

import PIL.Image
import requests

from src.settings import IMAGE_PATH, replicate_client

MODEL_NAME = "black-forest-labs/flux-1.1-pro"


def flux_image_api(
    prompt: str,
    steps: int = 25,
    guidance: int = 3,
    interval: int = 2,
    aspect_ratio: str = "1:1",
    safety_tolerance: int = 5,
    show: bool = True,
    save: bool = True,
):
    assert 1 <= steps <= 50
    assert 2 <= guidance <= 5
    assert 1 <= interval <= 4
    assert 1 <= safety_tolerance <= 5
    # assert aspect_ratio in ["16:9", "4:3", "1:1"]

    _input = {
        "steps": steps,  # 1-50
        "prompt": prompt,
        "guidance": guidance,  # 2-5
        "interval": interval,  # 1-4
        "aspect_ratio": aspect_ratio,
        "safety_tolerance": safety_tolerance,  # 1-5
    }

    image_url = replicate_client.run(MODEL_NAME, input=_input)
    response = requests.get(image_url)

    image_bytes = io.BytesIO(response.content)
    image = PIL.Image.open(image_bytes)
    if show:
        image.show()
    if save:
        filepath = IMAGE_PATH / (prompt.replace(" ", "_").lower() + ".jpg")
        image.save(filepath)

    return image


if __name__ == "__main__":
    print("Running flux_image_api")
    prompt = "Pink elephant"
    image = flux_image_api(prompt, steps=5)
