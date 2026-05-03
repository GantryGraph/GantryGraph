"""Example 1 — QA agent that runs tests and fixes failures.

Run:
    pip install 'gantrygraph[dev]'
    ANTHROPIC_API_KEY=sk-ant-... python examples/01_qa_agent.py
"""
from gantrygraph.presets import qa_agent
from langchain_anthropic import ChatAnthropic

agent = qa_agent(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    workspace=".",
    max_steps=20,
)

result = agent.run(
    "Run pytest on the current directory. "
    "If any tests fail, read the source files and fix them. "
    "Re-run until all tests pass, then summarise what you changed."
)
print(result)
