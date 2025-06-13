from typing import Annotated
from openai import OpenAI
import os
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import os
from langchain.chat_models import init_chat_model


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    user_id: str
    message:str
    messages: Annotated[list, add_messages]


os.environ["OPENAI_API_KEY"] = (
openai_key)

llm = init_chat_model("openai:gpt-4o-mini")

import csv
import os
from datetime import datetime

CSV_FILE = "chat_history.csv"


def save_message(user_id, message, role):
    with open(CSV_FILE, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([user_id, message, role, datetime.now().isoformat()])


def load_history(user_id):
    history = []
    if not os.path.exists(CSV_FILE):
        return history
    with open(CSV_FILE, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == user_id:
                history.append({"role": row[2], "content": row[1]})
            
    return history


# from utils import save_message, load_history

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class InputNode:
    def __call__(self, state):
        return state


class MemoryNode:
    def __call__(self, state):
        state["messages"] = load_history(state["user_id"])
        return state

def convert_to_openai_format(messages):
    converted = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            continue  # Skip unknown types
        converted.append({"role": role, "content": msg.content})
    return converted


class ChatNode:
    def __call__(self, state):
        state["messages"].append({"role": "user", "content": state["message"]})
        save_message(state["user_id"], state["message"], "user")
        history = state["messages"][:-1]
        k = convert_to_openai_format(history)
        k.append(state["messages"][-1])
        # context = client.chat.completions.create(
        # model="gpt-4o-mini",
        # messages=[
        #     {"role": "developer", "content": "you will be given a series of messages between chatbot and user. you task is to generate a summary on, what ae the key things discussed in the chat and give a geneic summary in few words."},
        #     {"role": "user", "content": history}
        # ]
        # )

        response = (
            client.chat.completions.create(
                model="gpt-4o-mini", messages=k
            )
            .choices[0]
            .message.content
        )

        state["messages"].append({"role": "assistant", "content": response})
        save_message(state["user_id"], response, "assistant")
        return {"response": response}


# from nodes import InputNode, MemoryNode, ChatNode

builder = StateGraph(State)
builder.add_node("input", InputNode())
builder.add_node("memory", MemoryNode())
builder.add_node("chat", ChatNode())

builder.set_entry_point("input")
builder.add_edge("input", "memory")
builder.add_edge("memory", "chat")
builder.add_edge("chat", END)

chatbot = builder.compile()

from fastapi import FastAPI
from pydantic import BaseModel

# from graph import chatbot
# from utils import load_history

app = FastAPI()

class ChatRequest(BaseModel):
    user_id: str
    message: str


@app.post("/chat")
def chat(request: ChatRequest):
    print(request)
    state = {"user_id": request.user_id, "message": request.message}
    result = chatbot.invoke(state)
    # result["AI Response"] = result["messages"][-1]["content"]
    return {
        "ai response": result["messages"][-1]["content"],
        "contexxt": result
    }


@app.get("/history/{user_id}")
def get_history(user_id: str):
    return load_history(user_id)
