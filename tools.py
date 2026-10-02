import wikipedia
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool
from datetime import datetime

def history(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"


save_tool = Tool(
    name="history",
    func=history,
    description="Save the research output to a text file with a timestamp. Useful for keeping a record of findings.",
)


# Set User-Agent properly to avoid 403 Forbidden / JSONDecodeError
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

def save_history(data: str) -> str:
    # Your save_history logic here
    return "History saved successfully."