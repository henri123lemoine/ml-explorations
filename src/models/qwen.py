from threading import Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    logging,
)

# Set logging level to suppress info messages
logging.set_verbosity_error()


def load_model_and_tokenizer(model_name: str):
    """Load the model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def prepare_input(tokenizer, prompt: str, system_message: str):
    """Prepare the input for the model."""
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer([text], return_tensors="pt")


def generate_stream(model, tokenizer, model_inputs, max_new_tokens: int = 512):
    """Generate text stream from the model."""
    model_inputs = model_inputs.to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=max_new_tokens)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for text in streamer:
        yield text

    thread.join()


def main():
    MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
    SYSTEM_MESSAGE = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    PROMPT = "Give me a short introduction to large language model."

    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    model_inputs = prepare_input(tokenizer, PROMPT, SYSTEM_MESSAGE)

    print("Streaming output:")
    for text in generate_stream(model, tokenizer, model_inputs):
        print(text, end="", flush=True)

    print("\nStreaming finished.")


if __name__ == "__main__":
    main()
