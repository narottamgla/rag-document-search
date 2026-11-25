"""
Node implementations for the RAG workflow
"""
from src.state.rag_state import RAGState


class RAGNode(RAGState):

    def __init__(self, retriever, llm):
        """Initialize the RAGNode with retriever and llm instances."""

        self.retriever = retriever
        self.llm = llm

    def retriever_docs(self, state: RAGState) -> RAGState:
        """
        Retrieve relevant documents based on the input state.

        :param state: The current RAG state
        :return: The updated RAG state with retrieved documents
        """
        docs = self.retriever.retrieve(state.question)
        state.retrieved_docs = docs
        return RAGState(
            question=state.question,
            retrieved_docs=docs,
        )

    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate an answer based on the input state.

        :param state: The current RAG state
        :return: The updated RAG state with the generated answer
        """

        # Combined retrieved documents into context for the LLM
        context = "\n\n".join(doc.page_content for doc in state.retrieved_docs)

        # Create prompt for the LLM using the question and context
        prompt = (f"""Generate the answer based on context"
                context: {context}
                question: {state.question}
                 
                 """
                  )

        answer = self.llm.invoke(prompt)
        state.answer = answer
        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer
        )
