from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchResults
from langchain_community.utilities import  WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

search = DuckDuckGoSearchResults()

search_tool = Tool(
    name = "Google Search",
    func=search.run,
    description="Search the web for information related to the query. Useful for when you need to answer questions about current events or find specific information online."
)