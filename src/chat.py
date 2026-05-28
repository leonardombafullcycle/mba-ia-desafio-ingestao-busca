import argparse
from search import search_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["openai", "gemini"], default="openai")
    args = parser.parse_args()

    print("Chat RAG - Digite 'sair' para encerrar\n")

    while True:
        question = input("PERGUNTA: ").strip()
        if question.lower() == "sair":
            break
        if not question:
            continue
        response = search_prompt(question, args.provider)
        print(f"RESPOSTA: {response}")
        print("---\n")


if __name__ == "__main__":
    main()
