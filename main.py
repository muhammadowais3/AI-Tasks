from langgraph.graph import StateGraph, END
from langchain.agents import tool
# from langchain_community.tools import ArxivAPIWrapper
from langchain_community.tools.arxiv.tool import ArxivAPIWrapper

from langchain_community.utilities import SerpAPIWrapper

from langchain_core.runnables import RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.output_parsers import PydanticOutputParser
import logging
import json


