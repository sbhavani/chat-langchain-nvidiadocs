import pickle
from query_data import get_chain
import json
import os

with open('secrets.json') as f:
    data = json.load(f)

os.environ['OPENAI_API_KEY'] = data['OPENAI_API_KEY']

if __name__ == "__main__":
    with open("vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    qa_chain = get_chain(vectorstore)
    chat_history = []
    print("Chat with your docs!")
    while True:
        print("Human:")
        question = input()
        result = qa_chain({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))
        print("AI:")
        print(result["answer"])
