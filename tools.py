import os
import wikipedia
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool
from datetime import datetime

MEMORY_FILE = "research_output.txt"

#this is the function which saves the output to a text file
def history(data: str, filename: str = MEMORY_FILE) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"

save_tool = Tool(
    name="history",
    func=history,
    description="Save research findings to text file.",
)


wikipedia.set_user_agent("ResearchAssistantAgent/1.0 (contact@example.com)")
search_engine = DuckDuckGoSearchRun()

search_tool = Tool(
    name="Google_Search",
    func=search_engine.run,
    description="Search the web for information related to the query.",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_char_limit=300)

def safe_wikipedia_run(query: str) -> str:
    try:
        return api_wrapper.run(query)
    except Exception as e:
        return f"Error querying Wikipedia: {str(e)}"

wikipedia_tool = Tool(
    name="wikipedia",
    func=safe_wikipedia_run,
    description="Search Wikipedia for background facts and encyclopedia details.",
)

#this loads the file history
def load_file_history(filename: str = MEMORY_FILE) -> str:
    if not os.path.exists(filename):
        return "No prior history recorded."
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return content if content else "No prior history recorded."

def append_interaction(query: str, response: str, filename: str = MEMORY_FILE) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"--- Interaction [{timestamp}] ---\nUser Query: {query}\nResponse: {response}\n\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(entry)