from src.utils.ell_help import hello_chatty, hello_claude


def main():
    name = "henri"

    result = hello_chatty(name)
    print(f"Chatty result: {result}")

    result = hello_claude(name)
    print(f"Claude result: {result}")


if __name__ == "__main__":
    main()
