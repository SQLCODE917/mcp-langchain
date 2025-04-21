#
# @Author: Bhaskar S
# @Blog:   https://www.polarsparc.com
# @Date:   06 April 2025
#

import asyncio
import logging
import os

from dotenv import load_dotenv, find_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

logging.basicConfig(format='%(levelname)s %(asctime)s - %(message)s', level=logging.INFO)

logger = logging.getLogger('interest_client')

load_dotenv(find_dotenv())

home_dir = os.getenv('HOME')
llm_temperature = float(os.getenv('LLM_TEMPERATURE'))
ollama_model = os.getenv('OLLAMA_MODEL')
ollama_base_url = os.getenv('OLLAMA_BASE_URL')

ollama_chat_llm = ChatOllama(base_url=ollama_base_url, model=ollama_model, temperature=llm_temperature)

@tool
def dummy():
  """This is a dummy tool"""
  return None

async def main():
  tools = [dummy]

  # Initialize a ReACT agent
  agent = create_react_agent(ollama_chat_llm, tools)

  # Case - 1 : Simple interest definition
  agent_response_1 = await agent.ainvoke(
    {'messages': 'what is the simple interest ?'})
  logger.info(agent_response_1['messages'][::-1])

  # Case - 2 : Simple interest calculation
  agent_response_2 = await agent.ainvoke(
    {'messages': 'compute the simple interest for a principal of 1000 at rate 3.75 ?'})
  logger.info(agent_response_2['messages'][::-1])

  # Case - 3 : Compound interest calculation
  agent_response_3 = await agent.ainvoke(
    {'messages': 'compute the compound interest for a principal of 1000 at rate 4.25 ?'})
  logger.info(agent_response_3['messages'][::-1])


if __name__ == '__main__':
  asyncio.run(main())
