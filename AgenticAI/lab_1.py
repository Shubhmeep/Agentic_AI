from dotenv import load_dotenv
import os
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown

console = Console()
load_dotenv(override=True)
open_ai_key = os.getenv("OPENAI_API_KEY")

if open_ai_key:
    print(f"OpenAI API Key exists and begins {open_ai_key[:8]}")
else:
    print("OpenAI API Key not set")
   
openai = OpenAI()

# list of dictionaries
question = "Please propose a hard, challenging question to assess someone's IQ. Respond only with the question."
messages = [{"role": "user", "content": question}]

response = openai.chat.completions.create(
    model="gpt-4.1-mini",
    messages=messages
)

question = response.choices[0].message.content
print(f'Question being asked : {question}')

answer = [{'role':'user','content':question}]

final_response = openai.chat.completions.create(
    model="gpt-4.1-mini",
    messages=answer
)

final_text = final_response.choices[0].message.content
console.print()
console.print(f'Answer the the above question is ')
console.print(Markdown(final_text))
