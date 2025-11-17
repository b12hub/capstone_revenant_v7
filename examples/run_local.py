import os
import json
BASE = os.path.dirname(os.path.abspath(__file__))
TASK_PATH = os.path.join(BASE, "example_task.json")

async def main():
    with open(TASK_PATH) as f:
        task = json.load(f)

