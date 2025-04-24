# MCP-Langchain-ReAct

Agentic AI Experiments

## Installation

### Python Environment

```bash
conda env create -f full-environment.yml
conda activate mcp-langchain
```

if you change the dependencies, don't forget to regenerate the yml:

```bash
conda env create -f full-environment.yml
```

### Local Model

I am using DeepSeek-R1 (Abliterated),
on ollama running on my host machine.
By default, that model was not able to use tools,
so I have created a Modelfile that allows it to.
Find it in the `llm` directory.

Make sure to review the `.env` to configure the correct URLs that
match your environment.

Also, in case your Windows host machine's firewall is not allowing
connections from your VM where your Python env is at, you can run
PowerShell as admin and allow that connection:

```powershell
New-NetFirewallRule -DisplayName "Allow Ollama Port 11435" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 11435
```

## This Repo

```bash
.
├── README.md
├── forerunner
│   └── interest_client.py
├── full-environment.yml
├── knower
│   ├── README.md
│   ├── client.py
│   ├── llm
│   │   └── Modelfile
│   └── server.py
├── llm
│   └── Modelfile
└── prototype
    ├── client.py
    └── server.py
```

- forefunner: lightweight but meaningful example of Agentic AI in action.
Demonstrates how a LangChain ReAct agent powered by a local LLM via Ollama can userstand prompts, decide when to use tools and generate human-like responses.
Run it with `python3 interest_client.py`.
- prototype: more advanced and modular implementation of Agentic AI, closer to production-ready architecture that improves on the `forefunner` with:
    - Dynamic tool discovery via MCP
    - Agent reasoning with ReAct + tool delegation + result explanation
    - Modular client-server separation
    - Cross-process tool transport for local development via `stdio`
- llm/Modelfile: allows DeepSeek-R1 to use tools, for `forefunner` and `prototype`
- knower: after `forerunner` and `prototype`, a more mature Agentic AI application for chatting with your Python codebase
    - Using the `nvidia_Llama-3.1-8B-UltraLong-4M-Instruct-Q6_K_L` local model
    - Employing advanced ReAct prompting techniques
    - Developing intuitive understanding of the relationship between semantic embeddings and the LLM
    - Comprehensive logging
    -
