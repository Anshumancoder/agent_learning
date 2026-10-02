from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from tools import search_tool, wikipedia_tool, save_tool 

load_dotenv()


class ResponseSchema(BaseModel):
    answer: str = Field(description="Direct answer to the query")
    summary: str = Field(description="Summary of facts")
    source: str = Field(description="Sources used")
    tools_used: list[str] = Field(description="List of tool names used")


# Initialize Model
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# Set up Tools
tools = [search_tool, wikipedia_tool, save_tool]

# Define Agent Prompt
agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful research assistant. Use tools when necessary to answer the query."),
        ("human", "{query}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Build Agent and Executor
agent = create_tool_calling_agent(
    llm=llm,
    prompt=agent_prompt,
    tools=tools,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Build Formatting Chain
structured_llm = llm.with_structured_output(ResponseSchema)

formatter_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Structure the raw research notes into the specified format."),
        ("human", "User Query: {query}\n\nResearch Notes:\n{raw_notes}"),
    ]
)

formatting_chain = formatter_prompt | structured_llm

if __name__ == "__main__":
    query = input("Enter your query: ")

    raw_result = agent_executor.invoke({"query": query})
    
    structured_output = formatting_chain.invoke(
        {
            "query": query,
            "raw_notes": raw_result["output"],
        }
    )

    print("\n--- Structured Result ---")
    print(structured_output)