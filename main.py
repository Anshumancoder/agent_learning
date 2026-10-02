import tkinter as tk
from tkinter import simpledialog
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from tools import search_tool, wikipedia_tool, save_tool, load_file_history, history

load_dotenv()

#this tells the agent what all the output should contain
class ResponseSchema(BaseModel):                                                
    answer: str = Field(description="Direct answer to the query")
    summary: str = Field(description="Summary of facts")
    source: str = Field(description="Sources used")


# this is our LLM (Groq)
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

tools = [search_tool, wikipedia_tool, save_tool] 

#this is a template tht tells the agent what all it should do
agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful research assistant tht gives short(10 - 20 lines) and precise answers. "
            "Below is the persistent file history of past interactions. "
            "Use it as reference whenever you need past memory:\n\n"
            "{file_history}\n\n"
            "Use tools when necessary to answer the query.",
        ),
        ("human", "{query}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

#this is the agent 
agent = create_tool_calling_agent(
    llm=llm,
    prompt=agent_prompt,
    tools=tools,
)


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#for structuring the output 
structured_llm = llm.with_structured_output(ResponseSchema)

formatter_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Structure the raw research notes into the specified format."),
        ("human", "User Query: {query}\n\nResearch Notes:\n{raw_notes}"),
    ]
)


formatting_chain = formatter_prompt | structured_llm

# this open tinkter
def get_user_query():
    root = tk.Tk()
    root.withdraw()
    query = simpledialog.askstring("Research Assistant", "Enter your query:")
    root.destroy()
    return query

def display_result_window(output_data):
    root = tk.Tk()
    root.title("Research Result Output")
    root.geometry("600x500")

    text_widget = tk.Text(root, wrap=tk.WORD, font=("Arial", 11))
    text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    formatted_text = (
        f"--- STRUCTURED RESULT ---\n\n"
        f"ANSWER:\n{output_data.answer}\n\n"
        f"SUMMARY:\n{output_data.summary}\n\n"
        f"SOURCES:\n{output_data.source}\n\n"
    )

    text_widget.insert(tk.END, formatted_text)
    text_widget.config(state=tk.DISABLED)

    root.mainloop()

# this is the loop which makes the agent remmember the history
if __name__ == "__main__":
    while True:
        query = get_user_query()
        if not query or query.lower() in ["quit", "exit"]:
            break

        past_history = load_file_history()

        raw_result = agent_executor.invoke(
            {
                "query": query,
                "file_history": past_history,
            }
        )
        
        structured_output = formatting_chain.invoke(
            {
                "query": query,
                "raw_notes": raw_result["output"],
            }
        )

        record_entry = (
            f"Query: {query}\n"
            f"Answer: {structured_output.answer}\n"
            f"Summary: {structured_output.summary}\n"
            f"Sources: {structured_output.source}\n"
        )
        history(record_entry)

        print("\n--- Structured Result ---")
        print(structured_output)
        display_result_window(structured_output)