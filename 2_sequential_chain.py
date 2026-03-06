# --- Educational Note: Imports and Setup ---
# ChatOpenAI: The main class for interacting with OpenAI models in LangChain.
# load_dotenv: Loads environment variables (like API keys) from a .env file.
# PromptTemplate: Allows creating generic template strings with slot placeholders for dynamic values.
# StrOutputParser: Extracts just the string content from an LLM response message.
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os 

# Setting the LANGCHAIN_PROJECT environment variable logs our trace in LangSmith under a specific project.
# This drastically enhances visibility and debugging!
os.environ['LANGCHAIN_PROJECT'] = 'Sequential LLM app'

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

model1 = ChatOpenAI(model='gpt-4o-mini', temperature=0.7)

model2 = ChatOpenAI(model='gpt-4o', temperature=0.4)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

config = {
    'run_name': 'Sequential Chain',
    'tags' : ['llm app', 'report generation', 'summarization'],
    'metadata' : {
        'version' : '1.0.0',
        'author' : 'John Doe'
    }
}

result = chain.invoke({'topic': 'Unemployment in India'}, config=config)

print(result)
