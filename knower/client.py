"""MCP Client facilitating codebase queries."""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession
from mcp.client.sse import sse_client

def load_config():
    """Loads .env values"""
    load_dotenv(find_dotenv())

    return {
        "llm_temperature": float(os.getenv("LLM_TEMPERATURE", "0.9")),
        "ollama_model": os.getenv("OLLAMA_MODEL"),
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL")
    }

os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILENAME = f"logs/client_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILENAME, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("knower_mcp_client")

async def explain_result(llm, tool_result: str, prompt: str) -> str:
    """
    Use LLM to convert a raw tool result into a user-friendly explanation.
    """
    explanation_prompt = f"""You are an assistant tasked with providing a user-friendly explanation based on the following tool output for the question:

    QUESTION:
    {prompt}

    TOOL OUTPUT:
    {tool_result}

    INSTRUCTIONS:
    - Analyze the tool output and question.
    - Provide a clear, concise explanation answering the question based solely on the tool output.
    - If the tool output contains relevant information (e.g., code snippets, file locations), summarize it accurately.
    - If the tool output is empty or indicates no results, state that the information could not be found in the codebase and avoid speculating.
    - Do not call additional tools or request more information.
    - Format the response in natural language, suitable for a user.
    """
    response = await llm.ainvoke(explanation_prompt)
    return response.content if hasattr(response, "content") else str(response)

async def run_agent(agent, llm):
    """
    Runs a series of questions through the agent and processes the responses using the LLM.
    Logs intermediate and final outputs for each prompt.
    """
    prompts = [
    	"In the codebase, what does the function `yearly_simple_interest` do?",
    	"Where in the repo is compound interest calculated?",
    	"List all tools defined in the MCP server defined in the codebase?"
    ]

    for question in prompts:

        response = await agent.ainvoke({
            "messages": [{"role": "user", "content": question}]
            }, config={"recursion_limit": 10})
        logger.info("Agent response:\n%s", response["messages"][::-1])

        for message in response["messages"]:
            if hasattr(message, "tool_calls") and message.tool_calls:
                logger.info("Tool calls: %s", message.tool_calls)
            if isinstance(message, ToolMessage):
                logger.info("Tool output: %s", message.content)

        result_message = response["messages"][-1].content
        logger.info("Raw tool result:\n%s", result_message)

        final_output = await explain_result(llm, result_message, question)
        logger.info("Final user-friendly output:\n%s", final_output)

async def main():
    """
    Main entry point for the client. Initializes the LLM, connects to the MCP server,
    loads tools, and runs the agent.
    """
    config = load_config()

    llm = ChatOllama(
        base_url=config["ollama_base_url"],
        model=config["ollama_model"],
        temperature=config["llm_temperature"]
    )

    try:
        async with sse_client("http://127.0.0.1:8082/sse") as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = await load_mcp_tools(session)
                logger.info("Loaded MCP Tools: %s", tools)

                agent = create_react_agent(llm, tools)
                await run_agent(agent, llm)
    except Exception as e:
        logger.exception("Failed to connect to SSE server")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
