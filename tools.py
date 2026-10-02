from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool
from datetime import datetime

search = DuckDuckGoSearchRun()

search_tool = Tool(
    name = "Google_Search",
    func = search.run,
    description="Search the web for information related to the query. Useful for when you need to answer questions about current events or find specific information online.",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_char_limit=100)
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)