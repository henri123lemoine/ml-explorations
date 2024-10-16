import ell

from src.settings import ANTHROPIC_CLIENT, OPENAI_CLIENT

# Models: e.g. claude-3-5-sonnet-20240620, gpt-4o-mini, o1-mini, o1-preview


@ell.simple(model="gpt-4o-mini", client=OPENAI_CLIENT)
def example(input1: str, input2: str) -> str:
    """This is the system prompt"""
    prompt = input1 + input2  # prompt processing
    return prompt


@ell.simple(model="gpt-4o-mini", client=OPENAI_CLIENT)
def hello_chatty(name: str):
    """You are a helpful assistant"""
    name = name.capitalize()
    return f"Say hello to {name}!"


@ell.simple(model="claude-3-5-sonnet-20240620", client=ANTHROPIC_CLIENT, max_tokens=8192)
def hello_claude(world: str):
    """You are a helpful assistant"""
    name = world.capitalize()
    return f"Say hello to {name}!"


def main():
    name = "henri"

    result = hello_chatty(name)
    print(f"Chatty result: {result}")

    result = hello_claude(name)
    print(f"Claude result: {result}")


if __name__ == "__main__":
    main()
