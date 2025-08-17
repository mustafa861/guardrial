import os
from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
    output_guardrail,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    RunConfig
)

from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel

load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")


external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)


#                """Implementation of Input Guardrail"""

class MathHomeworkOutput(BaseModel):
    is_math_homework: bool
    reasoning: str

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking you to do their math homework.",
    output_type=MathHomeworkOutput,
)


@input_guardrail
async def math_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context, run_config = config)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        # tripwire_triggered=False #result.final_output.is_math_homework,
        tripwire_triggered=result.final_output.is_math_homework,
    )

agent = Agent(
    name="Customer support agent",
    instructions="You are a customer support agent. You help customers with their questions.",
    input_guardrails=[math_guardrail],
)

# This should trip the guardrail

try:
    result = Runner.run_sync(agent, "Hello, can you help me solve for x: 2x + 3 = 11?", run_config = config)
    print("Guardrail didn't trip - this is unexpected")
    print(result.final_output)

except InputGuardrailTripwireTriggered:
    print("Math homework guardrail tripped")

try:
    result = Runner.run_sync(agent, "Hello", run_config = config)
    print(result.final_output)

except InputGuardrailTripwireTriggered:
    print("Math homework guardrail tripped")

try:
    result = Runner.run_sync(agent, "can you solve 2+3 for me", run_config = config)
    print(result.final_output)

except InputGuardrailTripwireTriggered:
    print("Math homework guardrail tripped")



#           """Implementation of Onput Guardrail:"""


class MessageOutput(BaseModel):
    response: str

class MathOutput(BaseModel):
    is_math: bool
    reasoning: str

guardrail_agent2 = Agent(
    name="Guardrail check",
    instructions="Check if the output includes any math.",
    output_type=MathOutput,
)

@output_guardrail
async def math_guardrail2(
    ctx: RunContextWrapper, agent: Agent, output: MessageOutput
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent2, output.response, context=ctx.context, run_config = config)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_math,
    )   

agent2 = Agent(
    name="Customer support agent",
    instructions="You are a customer support agent. You help customers with their questions.",
    output_guardrails=[math_guardrail2],
    output_type=MessageOutput,
)

# This should trip the guardrail
try:
    Runner.run_sync(agent2, "Hello, can you help me solve for x: 2x + 3 = 11?", run_config = config)
    print("Guardrail didn't trip - this is unexpected")

except OutputGuardrailTripwireTriggered:
    print("Math output guardrail tripped")