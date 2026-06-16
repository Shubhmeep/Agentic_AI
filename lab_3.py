import os
import json
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


# =============================================================================
# Lab 3: Prompt Chaining with LangChain LCEL
# =============================================================================
# This lab uses LangChain Expression Language (LCEL) to build a two-step chain:
# 1. Extract technical specifications from raw text.
# 2. Transform the extracted specifications into JSON.


# =============================================================================
# 1. Load Environment Variables
# =============================================================================
# ChatGroq reads GROQ_API_KEY from the environment.
load_dotenv(override=True)


# =============================================================================
# 2. Initialize the Groq Chat Model
# =============================================================================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)


# =============================================================================
# 3. Define Prompt Templates
# =============================================================================
# The first prompt extracts only the useful technical specification details.
extract = ChatPromptTemplate.from_template(
    "Extract the technical specifications from the following text:\n\n{text_input}"
)

# JsonOutputParser gives the model clear format instructions and parses the
# final response into a Python dictionary.
json_parser = JsonOutputParser()

# The second prompt converts those extracted details into a strict JSON object.
prompt_transform = ChatPromptTemplate.from_template(
    "Transform the following specifications into a JSON object with "
    "'cpu', 'memory', and 'storage' as keys.\n\n"
    "{format_instructions}\n\n"
    "Specifications:\n{specifications}"
).partial(
    format_instructions=json_parser.get_format_instructions()
)


# =============================================================================
# 4. Build the LangChain LCEL Chains
# =============================================================================
# StrOutputParser converts the chat model's message output into a plain string.
extraction_chain = extract | llm | StrOutputParser()
print(f'this si extraction chain : {extraction_chain}')
# The full chain passes the extraction output into the transform prompt.
full_chain = (
    {"specifications": extraction_chain}
    | prompt_transform
    | llm
    | json_parser
)


# =============================================================================
# 5. Run the Chain
# =============================================================================
# The input is passed as text_input because prompt_extract expects that variable.
input_text = (
    "The new laptop model features a 3.5 GHz octa-core processor, "
    "16GB of RAM, and a 1TB NVMe SSD."
)

final_result = full_chain.invoke({"text_input": input_text})
print("\n--- Final JSON Output ---")
print(json.dumps(final_result, indent=2))
