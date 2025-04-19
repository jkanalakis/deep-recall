# examples/user_history_example.py

from memory.memory_retriever import MemoryRetriever
from memory.memory_store import MemoryStore
from models.deepseek_r1_integration import DeepSeekR1Model


def main():
    # 1) Initialize Memory + Retriever
    embedding_dim = 384  # e.g., from all-MiniLM-L6-v2
    memory_store = MemoryStore(embedding_dim)
    retriever = MemoryRetriever(memory_store)

    # 2) Initialize LLM
    llm = DeepSeekR1Model(model_path="local_checkpoint/deepseek_r1")

    # 3) Add some historical data
    user_messages = [
        "I love sushi, especially salmon rolls.",
        "I'm planning a trip to Japan next year.",
        "My favorite programming language is Python.",
    ]
    for msg in user_messages:
        retriever.add_to_memory(msg)

    # 4) New user query
    query = "Where should I visit for great sushi in Tokyo?"
    relevant_memories = retriever.get_relevant_memory(query, k=2)

    # Construct final prompt
    context_snippets = "\n".join([f"- {rm['text']}" for rm in relevant_memories])
    prompt = (
        "SYSTEM:\n"
        "You are a helpful AI assistant with access to a user's personal preferences.\n"
        "Here are some relevant memories:\n"
        f"{context_snippets}\n"
        "USER:\n"
        f"{query}\n"
        "ASSISTANT:\n"
    )

    # 5) Generate a reply
    response = llm.generate_reply(prompt)
    print(response)

    # 6) Save the FAISS index to disk
    memory_store.save_index()


if __name__ == "__main__":
    main()
