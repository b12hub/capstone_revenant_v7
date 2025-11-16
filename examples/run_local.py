import json, asyncio
from orchestration.d_orchestrator import DOrchestrator

async def main():
    with open("examples/example_task.json") as f:
        task = json.load(f)
    orch = DOrchestrator()
    res = await orch.run(task)
    print(res)

if __name__ == "__main__":
    asyncio.run(main())
