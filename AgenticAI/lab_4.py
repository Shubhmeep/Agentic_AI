import os
from typing import Optional
import dotenv
import gradio as gr
from openai import OpenAI
from pydantic import BaseModel, Field
from pypdf import PdfReader
import requests, json

# =============================================================================
# Pydantic Schemas
# =============================================================================
# Pydantic defines the exact structure we want from the LLM.
# The LLM may still generate bad JSON, but Pydantic will reject it if it does
# not match these models.
class WorkExperience(BaseModel):
    company: str = Field(description="Company or organization name")
    role: str = Field(description="Job title or role held by the candidate")
    date_range: Optional[str] = Field(
        default=None,
        description="Employment date range, such as 'Jan 2023 - Sep 2023'",
    )
    yoe: Optional[float] = Field(
        default=None,
        description="Years worked at this company, calculated from date_range",
    )

class Achievement(BaseModel):
    name: str = Field(default=None, description='achievement name of the candidate')

class Resume(BaseModel):
    name: str = Field(description="Candidate's full name")
    summary: str = Field(description="Professional summary of the candidate")
    skills: list[str] = Field(description="List of candidate skills")
    work_experience: list[WorkExperience] = Field(description="Candidate's work experience")
    candidateAchievement: list[Achievement]= Field(description="Candidate's all achievements")

class ContactDetails(BaseModel):
    name: Optional[str] = Field(default=None, description="User's real name, only if explicitly provided by the user")
    email: Optional[str] = Field(default=None, description="User's real email address, only if explicitly provided by the user")
    notes: Optional[str] = Field(default=None, description="Short context for why the user wants to get in touch")

# =============================================================================
# Setup
# =============================================================================
dotenv.load_dotenv(override=True)
groq_api_key = os.getenv("GROQ_API_KEY")
pushover_user = os.getenv("USER_KEY")
pushover_token = os.getenv("PUSHOVER_API_KEY")
pushover_url = "https://api.pushover.net/1/messages.json"

groq = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1",
)

# =============================================================================
# Python Actions
# =============================================================================
def push(message):
    print(f"Push: {message}")
    payload = {"user": pushover_user, "token": pushover_token, "message": message}
    requests.post(pushover_url, data=payload)

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording interest from {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question} asked that I couldn't answer")
    return {"recorded": "ok"}

# =============================================================================
# Read LinkedIn PDF
# =============================================================================
reader = PdfReader("SHUBH SEHGAL.pdf")

linkedin = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        linkedin += text + "\n"

# =============================================================================
# Ask LLM To Extract Resume Data
# =============================================================================
structure_prompt = f"""
You are given text extracted from a candidate's LinkedIn profile.

Extract the candidate information and return valid JSON only.
Do not include markdown, explanations, or code blocks.

Important rules:
- Return all keys using the exact schema field names.
- "skills" must be a list of strings.
- "work_experience" must be a list of objects.
- Each work experience object must include "company", "role", "date_range", and "yoe".
- If a value is missing from the LinkedIn text, use null instead of guessing.
- Return yoe as a number, not a string.
- Extract the achievements of the candidate

Candidate text:
{linkedin}
""".strip()

messages = [
    {
        "role": "user",
        "content": structure_prompt,
    }
]

response = groq.responses.parse(
    model="openai/gpt-oss-120b",
    input=messages,
    text_format=Resume
)

result = response.output_parsed

# =============================================================================
# Build Website Chatbot System Prompt
# =============================================================================
name = result.name
resume_context = result.model_dump_json(indent=2)
system_prompt = f"""
You are acting as {name}.

You answer questions on {name}'s personal website.

Answer questions about:
- Career
- Background
- Skills
- Work experience
- Achievements

Use the resume information below.
Do not invent information.
If the answer is not available, say that you do not know.

NOTE:
- If you do not know the answer, say that you do not know. Do not mention tools.
- If the user seems interested in getting in touch, ask them to share both their name and email address.
- Do not claim that you recorded contact details unless the user actually provided both a real name and a real email address.
- Never write function calls, tool calls, XML tags, or JSON tool syntax in your answer.

Resume information:
{resume_context}
""".strip()

# =============================================================================
# Chat function
# =============================================================================

def assistant_does_not_know(answer):
    unknown_phrases = [
        "i do not know",
        "i don't know",
        "i do not have",
        "i don't have",
        "not available",
        "not mentioned",
        "not provided",
        "not in the resume",
        "not in the provided",
    ]
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in unknown_phrases)


def extract_contact_details(message, history):
    conversation = []

    for old_message in history:
        role = old_message["role"]
        content = old_message["content"]

        if role in ["user", "assistant"]:
            conversation.append(f"{role}: {content}")

    conversation.append(f"user: {message}")

    contact_prompt = f"""
Extract contact details from this conversation.

Rules:
- Extract name only if the user explicitly gave their own real name.
- Extract email only if the user explicitly gave their own real email address.
- Do not use the resume owner's name as the user's name.
- Do not invent missing values.
- If the user has not provided a name or email, return null for that field.

Conversation:
{chr(10).join(conversation)}
""".strip()

    response = groq.responses.parse(
        model="openai/gpt-oss-120b",
        input=[{"role": "user", "content": contact_prompt}],
        text_format=ContactDetails,
    )

    return response.output_parsed


def chat(message, history):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    # Take previous Gradio messages,
    # but only copy the fields Groq needs.
    for old_message in history:
        role = old_message["role"]
        content = old_message["content"]

        if role in ["user", "assistant"]:
            messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )

    # Add the user's newest message.
    messages.append(
        {
            "role": "user",
            "content": message,
        }
    )

    response = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
    )

    answer = response.choices[0].message.content or ""
    contact_details = extract_contact_details(message, history)

    if contact_details.email and contact_details.name:
        record_user_details(
            email=contact_details.email,
            name=contact_details.name,
            notes=contact_details.notes or f"User provided contact details in chat: {message}",
        )
        return f"{answer}\n\nThanks, I have recorded your name and email address."

    if assistant_does_not_know(answer):
        record_unknown_question(message)
        return f"{answer}\n\nI have recorded this question so it can be improved in a future version."

    return answer

# =============================================================================
# Launch website
# =============================================================================

demo = gr.ChatInterface(
    fn=chat,
    title=f"Chat with {name}",
    description=(
        f"Ask questions about {name}'s "
        "background, skills, and experience."
    ),
)

demo.launch()
