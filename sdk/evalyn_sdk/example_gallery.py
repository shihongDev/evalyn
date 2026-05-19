"""
Example agent gallery for evalyn.

Provides bundled working example agents across multiple frameworks,
with code templates, dataset templates, and scaffolding utilities.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExampleAgent:
    """A bundled example agent with code and dataset templates."""

    framework: str
    name: str
    description: str
    code_template: str
    dataset_template: str
    expected_metrics: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "name": self.name,
            "description": self.description,
            "code_template": self.code_template,
            "dataset_template": self.dataset_template,
            "expected_metrics": list(self.expected_metrics),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExampleAgent:
        return cls(
            framework=data.get("framework", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            code_template=data.get("code_template", ""),
            dataset_template=data.get("dataset_template", ""),
            expected_metrics=data.get("expected_metrics", []),
            tags=data.get("tags", []),
        )


@dataclass
class ExampleGallery:
    """A collection of example agents grouped by framework."""

    agents: list[ExampleAgent] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "agents": [a.as_dict() for a in self.agents],
            "frameworks": list(self.frameworks),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExampleGallery:
        return cls(
            agents=[ExampleAgent.from_dict(a) for a in data.get("agents", [])],
            frameworks=data.get("frameworks", []),
        )


# ---------------------------------------------------------------------------
# Built-in examples
# ---------------------------------------------------------------------------

_OPENAI_CODE = '''\
"""OpenAI chat completion agent."""

import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def run_agent(input_text: str) -> str:
    """Run a simple OpenAI chat completion."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_text},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "Hello"
    print(run_agent(query))
'''

_OPENAI_DATASET = (
    '{"input": "What is the capital of France?", "expected": "Paris"}\n'
    '{"input": "Summarize photosynthesis in one sentence.", "expected": "Plants convert sunlight into energy using chlorophyll."}\n'
    '{"input": "What is 7 * 8?", "expected": "56"}\n'
)

_ANTHROPIC_CODE = '''\
"""Anthropic Claude-based agent."""

import os
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def run_agent(input_text: str) -> str:
    """Run a Claude chat completion."""
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": input_text},
        ],
    )
    return message.content[0].text


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "Hello"
    print(run_agent(query))
'''

_ANTHROPIC_DATASET = (
    '{"input": "Explain gravity to a 5-year-old.", "expected": "Things fall down because the Earth pulls them."}\n'
    '{"input": "Write a haiku about coding.", "expected": "Lines of logic flow / bugs hide in the syntax weeds / tests bring the dawn light"}\n'
    '{"input": "What language is evalyn written in?", "expected": "Python"}\n'
)

_LANGCHAIN_CODE = '''\
"""LangChain ReAct agent with tool use."""

import os
import ast
import operator
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY"),
    temperature=0.0,
)

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def _safe_eval(expr: str) -> float:
    """Evaluate a simple arithmetic expression safely."""
    tree = ast.parse(expr, mode="eval")
    def _walk(node):
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            left = _walk(node.left)
            right = _walk(node.right)
            return _OPS[type(node.op)](left, right)
        raise ValueError(f"Unsupported: {node}")
    return _walk(tree)


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        return str(_safe_eval(expression))
    except Exception as e:
        return f"Error: {e}"


PROMPT = PromptTemplate.from_template(
    "Answer the question using tools if needed.\\n"
    "Tools: {tools}\\nTool names: {tool_names}\\n"
    "{agent_scratchpad}\\nQuestion: {input}"
)

agent = create_react_agent(llm, [calculator], PROMPT)
executor = AgentExecutor(agent=agent, tools=[calculator], verbose=False)


def run_agent(input_text: str) -> str:
    """Run the ReAct agent."""
    result = executor.invoke({"input": input_text})
    return result["output"]


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "What is 2+2?"
    print(run_agent(query))
'''

_LANGCHAIN_DATASET = (
    '{"input": "What is 123 * 456?", "expected": "56088"}\n'
    '{"input": "Calculate 99 + 1", "expected": "100"}\n'
    '{"input": "What is the square root of 144?", "expected": "12"}\n'
    '{"input": "Compute 2 to the power of 10", "expected": "1024"}\n'
)

_CUSTOM_CODE = '''\
"""Plain Python function agent with no external dependencies."""

import os


def run_agent(input_text: str) -> str:
    """A simple rule-based agent."""
    text = input_text.strip().lower()
    if "hello" in text or "hi" in text:
        return "Hello! How can I help you today?"
    if "capital of" in text:
        capitals = {
            "france": "Paris",
            "japan": "Tokyo",
            "brazil": "Brasilia",
            "germany": "Berlin",
        }
        for country, capital in capitals.items():
            if country in text:
                return f"The capital of {country.title()} is {capital}."
        return "I don't know the capital of that country."
    if "?" in text:
        return "That is a good question. Let me think about it."
    return f"You said: {input_text}"


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "Hello"
    print(run_agent(query))
'''

_CUSTOM_DATASET = (
    '{"input": "Hello there!", "expected": "Hello! How can I help you today?"}\n'
    '{"input": "What is the capital of France?", "expected": "The capital of France is Paris."}\n'
    '{"input": "What is the capital of Japan?", "expected": "The capital of Japan is Tokyo."}\n'
)

_MULTI_TURN_CODE = '''\
"""Multi-turn conversation agent."""

import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


class ConversationAgent:
    """Agent that maintains conversation history."""

    def __init__(self) -> None:
        self.history: list[dict] = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]

    def send(self, user_message: str) -> str:
        """Send a message and get a response."""
        self.history.append({"role": "user", "content": user_message})
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.history,
            temperature=0.0,
        )
        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self) -> None:
        """Clear conversation history."""
        self.history = self.history[:1]


def run_agent(turns: list[str]) -> list[str]:
    """Run a multi-turn conversation and return all replies."""
    agent = ConversationAgent()
    return [agent.send(turn) for turn in turns]


if __name__ == "__main__":
    replies = run_agent(["Hello", "What did I just say?", "Thanks!"])
    for r in replies:
        print(r)
'''

_MULTI_TURN_DATASET = (
    '{"input": ["Hi", "What is 2+2?", "Thanks"], "expected": ["greeting", "4", "acknowledgment"]}\n'
    '{"input": ["Tell me a joke", "Explain it"], "expected": ["joke", "explanation"]}\n'
    '{"input": ["What is Python?", "What about JavaScript?", "Compare them"], "expected": ["python_desc", "js_desc", "comparison"]}\n'
)

_BUILTIN_AGENTS = [
    ExampleAgent(
        framework="openai",
        name="OpenAI Chat Agent",
        description="Simple chat completion agent using the OpenAI API with gpt-4o-mini.",
        code_template=_OPENAI_CODE,
        dataset_template=_OPENAI_DATASET,
        expected_metrics=["correctness", "helpfulness", "latency"],
        tags=["openai", "chat", "simple", "beginner"],
    ),
    ExampleAgent(
        framework="anthropic",
        name="Anthropic Claude Agent",
        description="Chat agent using the Anthropic API with Claude.",
        code_template=_ANTHROPIC_CODE,
        dataset_template=_ANTHROPIC_DATASET,
        expected_metrics=["correctness", "helpfulness", "latency"],
        tags=["anthropic", "claude", "chat", "beginner"],
    ),
    ExampleAgent(
        framework="langchain",
        name="LangChain ReAct Agent",
        description="ReAct agent with tool use built on LangChain and OpenAI.",
        code_template=_LANGCHAIN_CODE,
        dataset_template=_LANGCHAIN_DATASET,
        expected_metrics=["correctness", "tool_accuracy", "latency", "cost"],
        tags=["langchain", "react", "tools", "intermediate"],
    ),
    ExampleAgent(
        framework="custom",
        name="Plain Python Agent",
        description="Rule-based agent with no external dependencies for quick testing.",
        code_template=_CUSTOM_CODE,
        dataset_template=_CUSTOM_DATASET,
        expected_metrics=["correctness"],
        tags=["custom", "python", "simple", "beginner", "no-api"],
    ),
    ExampleAgent(
        framework="multi_turn",
        name="Multi-Turn Conversation Agent",
        description="Stateful multi-turn conversation agent using OpenAI with history tracking.",
        code_template=_MULTI_TURN_CODE,
        dataset_template=_MULTI_TURN_DATASET,
        expected_metrics=["correctness", "coherence", "helpfulness", "latency"],
        tags=["openai", "multi-turn", "conversation", "stateful", "intermediate"],
    ),
]

BUILTIN_EXAMPLES = ExampleGallery(
    agents=_BUILTIN_AGENTS,
    frameworks=["openai", "anthropic", "langchain", "custom", "multi_turn"],
)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def list_examples(
    gallery: ExampleGallery,
    framework: str | None = None,
    tag: str | None = None,
) -> list[ExampleAgent]:
    """Filter examples by framework and/or tag."""
    results = gallery.agents
    if framework is not None:
        results = [a for a in results if a.framework == framework]
    if tag is not None:
        results = [a for a in results if tag in a.tags]
    return results


def get_example(gallery: ExampleGallery, framework: str) -> ExampleAgent | None:
    """Get a single example by framework name. Returns None if not found."""
    for agent in gallery.agents:
        if agent.framework == framework:
            return agent
    return None


def scaffold_example(agent: ExampleAgent, output_dir: str) -> list[str]:
    """Write agent.py, dataset.jsonl, and README.txt to output_dir.

    Creates output_dir if it does not exist. Returns a list of created
    file paths (absolute).
    """
    os.makedirs(output_dir, exist_ok=True)
    created: list[str] = []

    agent_path = os.path.join(output_dir, "agent.py")
    with open(agent_path, "w") as f:
        f.write(agent.code_template)
    created.append(os.path.abspath(agent_path))

    dataset_path = os.path.join(output_dir, "dataset.jsonl")
    with open(dataset_path, "w") as f:
        f.write(agent.dataset_template)
    created.append(os.path.abspath(dataset_path))

    readme_lines = [
        f"Example: {agent.name}",
        f"Framework: {agent.framework}",
        "",
        agent.description,
        "",
        "Files:",
        "  agent.py      - Agent implementation",
        "  dataset.jsonl  - Sample evaluation dataset",
        "",
        "Expected metrics: " + ", ".join(agent.expected_metrics),
        "",
        "Tags: " + ", ".join(agent.tags),
        "",
    ]
    readme_path = os.path.join(output_dir, "README.txt")
    with open(readme_path, "w") as f:
        f.write("\n".join(readme_lines))
    created.append(os.path.abspath(readme_path))

    return created


def format_gallery_listing(gallery: ExampleGallery) -> str:
    """Format a human-readable listing of all examples in the gallery."""
    lines = ["Example Agent Gallery", "=" * 21, ""]
    for agent in gallery.agents:
        lines.append(f"  [{agent.framework}] {agent.name}")
        lines.append(f"    {agent.description}")
        lines.append(f"    Metrics: {', '.join(agent.expected_metrics)}")
        lines.append(f"    Tags: {', '.join(agent.tags)}")
        lines.append("")
    return "\n".join(lines)


def search_examples(gallery: ExampleGallery, query: str) -> list[ExampleAgent]:
    """Search examples by matching query against name, description, and tags."""
    q = query.lower()
    results: list[ExampleAgent] = []
    for agent in gallery.agents:
        searchable = (
            agent.name.lower()
            + " "
            + agent.description.lower()
            + " "
            + " ".join(t.lower() for t in agent.tags)
        )
        if q in searchable:
            results.append(agent)
    return results
