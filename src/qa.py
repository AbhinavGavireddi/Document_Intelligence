"""
AnswerGenerator: orchestrates retrieval, re-ranking, and answer generation.

This module contains:
 - Retriever: Hybrid BM25 + dense retrieval over parsed chunks
 - Reranker: Cross-encoder based re-ranking of candidate chunks
 - AnswerGenerator: ties together retrieval, re-ranking, and LLM generation

Each component is modular and can be swapped or extended (e.g., add HyDE retriever).
"""
import os
import random
from typing import List, Dict, Any, Tuple

from src import logger, RetrieverConfig
from src.utils import LLMClient
from src.retriever import Retriever

class AnswerGenerator:
    """
    Generates answers by retrieving documents from a vector store
    and using them to build a context for an LLM.
    This version is optimized for low latency by skipping the reranking step.
    """
    def __init__(self, collection_name: str):
        self.retriever = Retriever(collection_name, RetrieverConfig)
        self.context_chunks_count = 5 # Use top 5 chunks for the final prompt
        self.greetings = [
            "Hello! I'm ready to answer your questions about the document. What would you like to know?",
            "Hi there! How can I help you with your document today?",
            "Hey! I've got the document open and I'm ready for your questions.",
            "Greetings! Ask me anything about the document, and I'll do my best to find the answer for you."
        ]

    def _truncate_to_last_sentence(self, text: str) -> str:
        """Finds the last period or newline and truncates the text to that point."""
        # Find the last period
        last_period = text.rfind('.')
        # Find the last newline
        last_newline = text.rfind('\n')
        # Find the last of the two
        last_marker = max(last_period, last_newline)

        if last_marker != -1:
            return text[:last_marker + 1].strip()
        
        # If no sentence-ending punctuation, return the text as is (or a portion)
        return text

    def answer(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieves documents, builds a context, and generates an answer.
        Handles simple greetings separately to improve user experience.
        """
        # Handle simple greetings to avoid a failed retrieval
        normalized_question = question.lower().strip().rstrip('.,!')
        greeting_triggers = ["hi", "hello", "hey", "hallo", "hola"]
        if normalized_question in greeting_triggers:
            return random.choice(self.greetings), []

        # Retrieve candidate documents from the vector store
        candidates = self.retriever.retrieve(question)
        
        if not candidates:
            logger.warning("No candidates retrieved from vector store.")
            return "The document does not contain information on this topic.", []
        
        # Use the top N chunks for context, without reranking
        top_chunks = candidates[:self.context_chunks_count]
        
        context = "\n\n".join(f"- {c['narration']}" for c in top_chunks)
        
        # A more robust prompt that encourages a natural, conversational tone
        prompt = (
            "You are a helpful and friendly AI assistant for document analysis. "
            "Your user is asking a question about a document. "
            "Based *only* on the context provided below, formulate a clear and conversational answer. "
            "Adopt a helpful and slightly informal tone, as if you were a knowledgeable colleague.\n\n"
            "CONTEXT:\n"
            "---------------------\n"
            f"{context}\n"
            "---------------------\n\n"
            "USER'S QUESTION: "
            f'"{question}"\n\n'
            "YOUR TASK:\n"
            "1. Carefully read the provided context.\n"
            "2. If the context contains the answer, explain it to the user in a natural, conversational way. Do not just repeat the text verbatim.\n"
            "3. If the context does not contain the necessary information, respond with: "
            "'I've checked the document, but I couldn't find any information on that topic.'\n"
            "4. **Crucially, do not use any information outside of the provided context.**\n\n"
            "Answer:"
        )
        
        answer, finish_reason = LLMClient.generate(prompt, max_tokens=256)

        # Handle cases where the response might be cut off
        if finish_reason == 'length':
            logger.warning("LLM response was truncated due to token limit.")
            truncated_answer = self._truncate_to_last_sentence(answer)
            answer = truncated_answer + " ... (response shortened)"

        return answer, top_chunks

