from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
import uuid

class State(TypedDict):
    val: str

def my_node(state: State):
    print("my_node started")
    v1 = interrupt("First interrupt")
    print(f"Resumed with: {v1}")
    v2 = interrupt("Second interrupt")
    print(f"Resumed again with: {v2}")
    return {"val": "done"}

builder = StateGraph(State)
builder.add_node("my_node", my_node)
builder.add_edge(START, "my_node")
builder.add_edge("my_node", END)
graph = builder.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "test_123"}}

print("--- Initial Run ---")
for event in graph.stream({"val": "init"}, config):
    print("Event:", event)

print("\n--- First Resume ---")
for event in graph.stream(Command(resume="Answer 1"), config):
    print("Event:", event)

print("\n--- Second Resume ---")
for event in graph.stream(Command(resume="Answer 2"), config):
    print("Event:", event)

print("\n--- Final State ---")
state = graph.get_state(config)
print("Next:", state.next)
print("Values:", state.values)
