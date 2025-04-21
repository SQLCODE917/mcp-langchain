# based on https://www.polarsparc.com Primer on MCP/LangChain/ReAct

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

def load_config():
    load_dotenv(find_dotenv())

    return {
        "llm_temperature": float(os.getenv("LLM_TEMPERATURE", 0.7)),
        "ollama_model": os.getenv("OLLAMA_MODEL"),
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL")
    }

logger = logging.getLogger("interest_mcp_client")
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s - %(message)s"
)

async def explain_result(llm, tool_result: str, prompt: str) -> str:
    """
    Use LLM to convert a raw tool result into a user-friendly explanation.
    """
    explanation_prompt = f"""The user asked: "{prompt}"
The tool returned: "{tool_result}"
Explain this result clearly and helpfully."""
    response = await llm.ainvoke(explanation_prompt)
    return response.content if hasattr(response, "content") else str(response)

async def run_agent(agent, llm):
    prompts = [
        "explain the definition of simple interest ?",
        "compute the simple interest for a principal of 1000 at rate 3.75 ?",
        "compute the compound interest for a principal of 1000 at rate 4.25 ?"
    ]

    for prompt in prompts:
        response = await agent.ainvoke({
            "messages": [{"role": "user", "content": prompt}]
        })
        logger.info("Agent response:\n%s", response["messages"][::-1])
        result_message = response["messages"][-1].content
        logger.info("Raw tool result:\n%s", result_message)

        final_output = await explain_result(llm, result_message, prompt)
        logger.info("Final user-friendly output:\n%s", final_output)

async def main():
    config = load_config()

    server_script = Path(__file__).parent / "server.py"
    server_params = StdioServerParameters(
        command="python3",
        args=[str(server_script)]
    )

    llm = ChatOllama(
        base_url=config["ollama_base_url"],
        model=config["ollama_model"],
        temperature=config["llm_temperature"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await load_mcp_tools(session)
            logger.info("Loaded MCP Tools: %s", tools)

            agent = create_react_agent(llm, tools)
            await run_agent(agent, llm)

if __name__ == "__main__":
    asyncio.run(main())
