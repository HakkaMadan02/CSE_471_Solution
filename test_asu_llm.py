import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv(Path(__file__).resolve().parent / ".env")

for name in ("ASU_LLM_KEY", "ASU_MODEL"):
    if not os.getenv(name):
        raise RuntimeError(f"{name} is missing from .env")

llm = ChatOpenAI(
    model=os.environ["ASU_MODEL"],
    api_key=os.environ["ASU_LLM_KEY"],
    base_url="https://openai.rc.asu.edu/v1",
    temperature=0,
    max_retries=0,
    timeout=120,
    use_responses_api=False,
)

print("Model:", os.environ["ASU_MODEL"])


# Test 1: ordinary text generation.
print("\n=== 1. Basic generation ===")

try:
    response = llm.invoke("Reply with exactly: ASU connection works")
    print(response.text)
except Exception as exc:
    print(f"FAILED: {type(exc).__name__}: {exc}")
    raise SystemExit(1)


# Test 2: ask for a tool call without forcing tool_choice.
print("\n=== 2. Full tool-calling round trip ===")
@tool
def lookup_test_value(key: str) -> str:
    """Retrieve an otherwise unknown test value for a specified key."""
    return "test-value-742"
try:
    tool_model = llm.bind_tools([lookup_test_value])

    messages = [
        HumanMessage(
            content=(
                "Use lookup_test_value with key='demo'. "
                "Then report the exact value returned by the tool. "
                "Do not guess."
            )
        )
    ]

    response = tool_model.invoke(messages)

    if not response.tool_calls:
        print("NOT CONFIRMED: no structured tool call returned")
        print("Text:", response.text)
    else:
        messages.append(response)

        for call in response.tool_calls:
            if call["name"] != lookup_test_value.name:
                raise RuntimeError(
                    f"Unexpected tool requested: {call['name']}"
                )

            tool_result = lookup_test_value.invoke(call["args"])

            print("Requested tool:", call["name"])
            print("Arguments:", call["args"])
            print("Tool result:", tool_result)

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=call["id"],
                )
            )

        final_response = tool_model.invoke(messages)

        print("Final response:", final_response.text)

        if (
            not final_response.tool_calls
            and "test-value-742" in final_response.text
        ):
            print("PASS: full tool-result round trip")
        else:
            print("NOT CONFIRMED: inspect final response")

except Exception as exc:
    print(f"FAILED: {type(exc).__name__}: {exc}")


# Test 3: native JSON-schema structured output.
class TestRecord(BaseModel):
    topic: str = Field(description="The topic mentioned in the input.")
    number: int = Field(description="The integer mentioned in the input.")


print("\n=== 3. Structured output ===")

try:
    structured_model = llm.with_structured_output(
        TestRecord,
        method="json_schema",
    )

    result = structured_model.invoke(
        "Extract the topic and number from: deposit insurance, 250000."
    )

    print("PASS:", result.model_dump())

except Exception as exc:
    print(f"FAILED: {type(exc).__name__}: {exc}")