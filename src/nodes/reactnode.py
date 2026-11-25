""""
LangGraph ReactNode is a node that uses the React framework to generate responses based on a given question and context. It utilizes a language model (LLM) to process the input and produce an answer. The node can also integrate with external tools, such as Wikipedia, to enhance its capabilities.
"""

from typing import Optional, List
from src.state.rag_state import RAGState
from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# WikiPedia Tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun


class RAGNode:
    """Node for the RAG workflow that uses React framework for generating responses."""

    def __init__(self, retriever, llm):
        """Initialize the RAGNode with retriever and llm instances."""
        self.retriever = retriever
        self.llm = llm
        self._agent = None  # Lazy Initialization of the React agent

    def retriver_docs(self, state: RAGState) -> RAGState:
        """
        Retrieve relevant documents based on the input state.

        :param state: The current RAG state
        :return: The updated RAG state with retrieved documents
        """
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs,
        )

    def _build_tools(self) -> List[Tool]:
        """Build Retriever and Wikipedia tools for the React agent."""

        def retriever_tool_fn(query: str) -> str:
            """Tool to retrieve documents based on the query."""
            docs: List[Document] = self.retriever.invoke(query)

            if not docs:
                return "No relevant documents found."
            merged = []

            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, 'metadata') else {}
                title = meta.get('title') or meta.get('source') or f"Document {i}"
                merged.append(f"Document {i} - {title}:\n{d.page_content}\n")
            return "\n".join(merged)

        retriever_tool = Tool(
            name="retriever",
            func=retriever_tool_fn,
            description="Useful for answering questions based on user provided documents. Input should be a search query."
        )

        wiki = WikipediaAPIWrapper(top_k_results=3, lang="en")

        wiki_tool = WikipediaQueryRun(
            name="Wikipedia",
            func=wiki.run,
            description="Useful for answering questions based on general knowledge. Input should be a search query."
        )

        return [retriever_tool, wiki_tool]


# Build agent
def _build_agent(self):
    """React Agent with Tools"""
    tools = self._build_tools()
    system_prompt = (
        "You are a helpful RAG Agent. "
        "Prefer 'retriever' for user provide docs and 'Wikipedia' for general knowledge. "
        "return only final user-facing answer, no inner thoughts or reasoning."
    )
    self.agent = create_react_agent(self.llm, tools=tools, system_prompt=system_prompt)

def generate_answer(self, state: RAGState) -> RAGState:
    """
    Create answer using React agent with retriever and Wikipedia tools.
    """
    if self.agent is None:
        self._build_agent()

    # Create a HumanMessage for the agent
    human_message = HumanMessage(content=state.question)

    # Invoke the agent to generate an answer
    result = self.agent.invoke([human_message])
    message = result.get("messages", [])
    answer = Optional[str] = None
    if message:
        answer_msg = message[-1]
        answer =getattr(answer_msg, 'content', None)

    return RAGState(
        question=state.question,
        retrieved_docs=state.retrieved_docs,
        answer= answer or "could not generate an answer based on the provided documents and context."
    )
