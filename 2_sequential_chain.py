# --- Educational Note: Imports ---
# ChatOpenAI: The main class for interacting with OpenAI models in LangChain.
# load_dotenv: Loads environment variables (like API keys) from a .env file.
# PromptTemplate: Allows creating generic template strings with slot placeholders for dynamic values.
# StrOutputParser: Extracts just the string content from an LLM response message.
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os 

# --- Educational Note: LangSmith Setup ---
# Setting the LANGCHAIN_PROJECT environment variable logs our trace in LangSmith under a specific project.
# This drastically enhances visibility and debugging!
os.environ['LANGCHAIN_PROJECT'] = 'Sequential LLM app'

# Loads the .env variables natively
load_dotenv()

# --- Educational Note: Prompt Templates ---
# Prompts are instructions given to an LLM. Here we use PromptTemplates to build dynamic prompts.
# prompt1 takes in a 'topic' variable and tells the model to write a detailed report about it.
prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

# prompt2 takes the 'text' (generated from prompt1's model) and asks the next model to summarize it.
prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

# --- Educational Note: Models and Parsers ---
# model1 uses 'gpt-4o-mini' with a higher temperature (0.7) for more creative report generation.
model1 = ChatOpenAI(model='gpt-4o-mini', temperature=0.7)

# model2 uses 'gpt-4o' with a lower temperature (0.4) for a more focused, concise 5 point summary.
model2 = ChatOpenAI(model='gpt-4o', temperature=0.4)

# parser converts the raw AIMessage output from the models into a plain Python string.
parser = StrOutputParser()

# --- Educational Note: LangChain Expression Language (LCEL) ---
# The '|' operator chains these components together sequentially.
# Flow: 
# 1. prompt1 formats the input topic into a full prompt string.
# 2. model1 generates a detailed report based on prompt1.
# 3. parser extracts the string from model1's output.
# 4. prompt2 takes that string (as 'text') and formats the summary prompt.
# 5. model2 generates the 5 point summary.
# 6. parser extracts the final string summary.
chain = prompt1 | model1 | parser | prompt2 | model2 | parser

config = {
    'tags' : ['llm app', 'report generation', 'summarization'],
    'metadata' : {
        'version' : '1.0.0',
        'author' : 'John Doe'
    }
}

result = chain.invoke({'topic': 'Unemployment in India'}, config=config)

print(result)
