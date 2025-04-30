import os
import warnings
from typing import Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Environment setup
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
warnings.filterwarnings("ignore")
load_dotenv()

# Imports (removed unused ChatOllama import)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, chain
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_groq import ChatGroq
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

# Initialize FastAPI
app = FastAPI(
    title="Text to SQL API",
    description="API for converting natural language to SQL queries",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
db = SQLDatabase.from_uri(
    "mysql+pymysql://avnadmin:AVNS_fXEOEYr3BoFiYyRAlLN@mysql-25fb3831-aayushkumarhigh-a519.l.aivencloud.com:23558/defaultdb"
)

# Initialize LLM
llm = ChatGroq(
    groq_api_key="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",
    model_name="llama-3.3-70b-versatile",
)


# Define request models
class QueryRequest(BaseModel):
    question: str


class AgentQueryRequest(BaseModel):
    question: str


# Set up SQL chain
sql_chain = create_sql_query_chain(llm, db)

# QnA chain setup
system = SystemMessagePromptTemplate.from_template(
    """You are helpful AI assistant who answer user question based on the provided context."""
)
prompt_template = """Answer user question based on the provided context ONLY! If you do not know the answer, just say "I don't know".
            ### Context:
            {context}

            ### Question:
            {question}

            ### Answer:"""

prompt = HumanMessagePromptTemplate.from_template(prompt_template)
messages = [system, prompt]
template = ChatPromptTemplate(messages)
qna_chain = template | llm | StrOutputParser()


# Helper function
def ask_llm(context, question):
    return qna_chain.invoke({"context": context, "question": question})


@chain
def get_correct_sql_query(input):
    context = input["context"]
    question = input["question"]

    instruction = f"""
        Use above context to fetch the correct SQL query for following question
        {question}

        Do not enclose query in ```
        You MUST return only single SQL query.
    """

    response = ask_llm(context=context, question=instruction)
    return response


# SQL execution tools
execute_query = QuerySQLDataBaseTool(db=db)
sql_query = create_sql_query_chain(llm, db)

# Final chain
final_chain = (
    {"context": sql_query, "question": RunnablePassthrough()}
    | get_correct_sql_query
    | execute_query
    | StrOutputParser()
)

# Agent setup
SQL_PREFIX = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 10 results.
You can order the results by a relevant column to return the most related examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

INCLUDE PRICE IN RUPEES.
INCLUDE THE NAME OF THE EVENTS and Place

DO NOT INCLUDE DATABASE CONNECTION INFORMATION IN YOUR QUERY. GIVE HUMAN READABLE ANSWERS.
DO NOT GIVE INFORMATION ABOUT THE DATABASE LIKE TABLE NAME OR COLUMN NAME.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

To start you should ALWAYS look at the tables in the database to see what you can query.
Do NOT skip this step.
Then you should query the schema of the most relevant tables."""

system_message = SystemMessage(content=SQL_PREFIX)

# Setup agent toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
try:
    tool = TavilySearchResults(max_results=2)
    tools = tools + [tool]
except Exception:
    pass

agent_executor = create_react_agent(
    llm, tools, state_modifier=system_message, debug=False
)


# API endpoints
@app.post("/")
async def root():
    return {"message": "Text to SQL API is running"}


@app.post("/sql")
async def generate_sql(request: QueryRequest) -> Dict:
    try:
        response = final_chain.invoke(request.question)
        return {"result": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent")
async def agent_query(request: AgentQueryRequest) -> Dict:
    try:
        result = agent_executor.invoke(
            {"messages": [HumanMessage(content=request.question)]}
        )
        return {"result": result["messages"][-1].content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
