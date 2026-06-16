import os
import json
from dotenv import load_dotenv
from openai import OpenAI


# =============================================================================
# Lab 2: Compare OpenAI, Gemini, and Groq
# =============================================================================
# Goal:
# Generate one challenging evaluation question, ask the same question to three
# different LLM providers, and store their answers for comparison.
#
# Providers compared in this lab:
# 1. OpenAI
# 2. Google Gemini
# 3. Groq


# =============================================================================
# 1. Load API Keys
# =============================================================================
# API keys are loaded from the local .env file. Keeping keys in .env avoids
# hardcoding secrets directly in the Python file.
load_dotenv(override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GEMINI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")


# =============================================================================
# 2. Check Which Provider Keys Are Available
# =============================================================================
# Print only a short prefix of each key. This confirms that the key was loaded
# without exposing the full secret in terminal output.
def print_key_status(provider_name, api_key, prefix_length):
    """Print whether a provider API key is available."""
    if api_key:
        print(f"{provider_name} API Key exists and begins {api_key[:prefix_length]}")
    else:
        print(f"{provider_name} API Key not set")


print_key_status("OpenAI", openai_api_key, 8)
print_key_status("Gemini", google_api_key, 2)
print_key_status("Groq", groq_api_key, 4)
print()

# =============================================================================
# 3. Create Provider Clients
# =============================================================================
# All three calls use the OpenAI Python client interface.
# - OpenAI uses the default OpenAI endpoint.
# - Gemini uses Google's OpenAI-compatible endpoint.
# - Groq uses Groq's OpenAI-compatible endpoint.
openai = OpenAI(api_key=openai_api_key)

gemini = OpenAI(
    api_key=google_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

groq = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1",
)


# =============================================================================
# 4. Generate One Shared Evaluation Question
# =============================================================================
# OpenAI is used here only to create the test question. The same generated
# question is then sent to OpenAI, Gemini, and Groq so the answers are comparable.
question_request = "Please come up with a challenging, nuanced question that I can ask a number of LLMs to evaluate their intelligence. Questions should not be more than 200 chars"

question_messages = [{"role": "user", "content": question_request}]
print(f"Question generation message: {question_messages}")

response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=question_messages,
)

question = response.choices[0].message.content
print()
print(f"Generated question: {question}")


# =============================================================================
# 5. Prepare Shared Answer Prompt and Comparison Storage
# =============================================================================
# Every provider receives the exact same prompt. The competitors and answers
# lists keep model names aligned with their generated answers.
answer_messages = [
    {"role": "user", "content": f"Answer the question in 1 line: {question}"}
]

competitors = []
answers = []


# =============================================================================
# 6. Helper Function for Provider Comparisons
# =============================================================================
# This keeps the OpenAI, Gemini, and Groq calls consistent. Each provider gets
# the same messages, and each result is stored in the same comparison lists.
def ask_provider(provider_name, client, model_name):
    """Ask one provider to answer the shared evaluation question."""
    response = client.chat.completions.create(
        model=model_name,
        messages=answer_messages,
    )

    answer = response.choices[0].message.content
    competitors.append(f"{provider_name}: {model_name}")
    answers.append(answer)

# =============================================================================
# 7. Provider 1: OpenAI
# =============================================================================
# OpenAI answers the shared question using an OpenAI model.
ask_provider("OpenAI", openai, "gpt-4o-mini")

# =============================================================================
# 8. Provider 2: Gemini
# =============================================================================
# Gemini answers the same question through Google's OpenAI-compatible endpoint.
ask_provider("Gemini", gemini, "gemini-2.5-flash")

# =============================================================================
# 9. Provider 3: Groq
# =============================================================================
# Groq answers the same question through Groq's OpenAI-compatible endpoint.
ask_provider("Groq", groq, "llama-3.3-70b-versatile")


# =============================================================================
# 10. Prepare Responses for the Judge
# =============================================================================
# The judge needs to see each competitor number, model name, and answer.
# Competitor numbers start at 1 so the judge can rank them clearly.
together = "\n\n".join(
    f"Competitor {index}: {competitor}\nResponse: {answer}"
    for index, (competitor, answer) in enumerate(zip(competitors, answers), start=1)
)


# =============================================================================
# 11. Judge the Competition with OpenAI GPT-5 Mini
# =============================================================================
# GPT-5 mini acts as the evaluator. It ranks the competitors from best to worst
# based on clarity and strength of argument.
judge = f"""You are judging a competition between {len(competitors)} competitors.
Each model has been given this question:

{question}

Your job is to evaluate each response for clarity and strength of argument, and rank them in order of best to worst.
Respond with JSON, and only JSON, with the following format:
{{"results": ["best competitor number", "second best competitor number", "third best competitor number", ...]}}

Here are the responses from each competitor:

{together}

Now respond with the JSON with the ranked order of the competitors, nothing else. Do not include markdown formatting or code blocks."""

judge_messages = [{"role": "user", "content": judge}]

judge_response = openai.chat.completions.create(
    model="gpt-5-mini",
    messages=judge_messages,
    response_format={"type": "json_object"},
)

judge_result_text = judge_response.choices[0].message.content
judge_result = json.loads(judge_result_text)


# =============================================================================
# 12. Print Final Model Ranking
# =============================================================================
# Convert the judge's competitor numbers into readable model names and print
# only the final ranking.
ranked_competitors = [
    competitors[int(competitor_number) - 1]
    for competitor_number in judge_result["results"]
]

print()
print("Final ranking:")
for rank, competitor in enumerate(ranked_competitors, start=1):
    print(f"{rank}. {competitor}")
