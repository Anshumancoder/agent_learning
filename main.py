from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq  
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser 
from tools import search_tool
from langchain.agents import create_tool_calling_agent

load_dotenv() 

class ResponseSchema(BaseModel): # how u want the output to be displayed
    answer: str 
    summary : str
    source : str
    tools_used : list[str]


llm = ChatGroq(
    model="openai/gpt-oss-120b", # basically the loading of the model 
    temperature=0
)

parser = PydanticOutputParser(pydantic_object=ResponseSchema) # u are take the output and format it the way u want it to display

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
        """
        You are a generative AI and will generate responses to user queries.
        Wrap the output in this format and provide no other text:
        {format_instructions}
        """
        ),
        ("human", "{query}")
    ]
).partial(format_instructions=parser.get_format_instructions())


tools = [search_tool] # list of tools that u want to use in the agent


agent = create_tool_calling_agent( # making the agent 
    llm=llm,
    promt=prompt,
    tools=[search_tool]
)


chain = prompt | llm | parser  # chain the prompt, model, and output parser directly


# Run the chain
result = chain.invoke({"query": "What is the capital of France?"})
print(result)
print("Answer:", result.answer)

try:
    structured_output = parser.parse(result.get("output")[0]["text"]) # structured output is the output of the model in the format we want it to be displayed
    print(structured_output)
except Exception as e:
    print(f"Error parsing structured output: {e}")