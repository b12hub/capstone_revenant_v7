import pytest
import asyncio
from agents.enterprise.enterprise_agent import EnterpriseOrchestratorAgent as EnterpriseAgent

@pytest.mark.asyncio
async def test_enterprise_agent():
    agent = EnterpriseAgent()
    res = await agent.run({"task": {"task_id": "t1"}})
    assert res["status"] == "accepted"
