from src.settings import replicate_client


def main():
    prompt = "Hello"

    output = replicate_client.run(
        "meta/llama-2-70b-chat",
        input={"prompt": prompt},
    )
    for item in output:
        print(item, end="")


if __name__ == "__main__":
    main()
