from langgraph.constants import START, END
from langgraph.graph import StateGraph
from src.state.rag_state import RAGState
from src.nodes.node import RAGNode


class GraphBuilder:
    """Builder for creating LangGraph graphs with RAG state"""

    def __init__(self, retriever, llm):
        """Initialize the GraphBuilder.

        args:
            retriever: RAG state
            llm: llm instance
        """
        # self.retriever = retriever
        # self.llm = llm
        self.nodes = RAGNode(retriever, llm)
        self.graph = None


def build(self):
    """
    Build the RAG Workflow graph

    Return:
        Complied graph Instance
    """
    #create State graph
    builder = StateGraph(RAGState)

    # Add node
    builder.add_node("retriever", self.node.retriever_docs)
    builder.add_node("responder", self.node.generate_answer)

    # Set Entry Point
    builder.set_entry_point("retriever")

    # Add Edges

    builder.add_edge("retriever","responder")
    builder.add_edge("responder",END)

    # Compile the graph
    return builder.compile()

def run (self, question: str)->dict:
    """
    Run the RAG Workflow graph

    args:
        question: Input question for the RAG workflow

    Return:
        Final state after executing the graph
    """

    if self.graph is None:
        self.graph = self.build()
        
    # Create initial state
    initial_state = RAGState(question=question)

    # Execute the graph
    final_state = self.graph.invoke(initial_state)

    return final_state

