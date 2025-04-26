"""
AnswerGenerator: orchestrates retrieval, re-ranking, and answer generation.

This module contains:
 - Retriever: Hybrid BM25 + dense retrieval over parsed chunks
 - Reranker: Cross-encoder based re-ranking of candidate chunks
 - AnswerGenerator: ties together retrieval, re-ranking, and LLM generation

Each component is modular and can be swapped or extended (e.g., add HyDE retriever).
"""
import os
import json
import numpy as np
import redis
from typing import List, Dict, Any, Tuple

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from src import sanitize_html
from src.utils import LLMClient, logger
from src.retriever import Retriever, RetrieverConfig


class RerankerConfig:
    MODEL_NAME = 'BAAI/bge-reranker-v2-Gemma'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Reranker:
    """
    Cross-encoder re-ranker using a transformer-based sequence classification model.
    """
    def __init__(self, config: RerankerConfig):
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.MODEL_NAME)
        self.model.to(config.DEVICE)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Score each candidate and return top_k sorted by relevance."""
        inputs = self.tokenizer(
            [query] * len(candidates),
            [c['narration'] for c in candidates],
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(RerankerConfig.DEVICE)
        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(-1)
            scores = torch.sigmoid(logits).cpu().numpy()
        # pair and sort
        paired = list(zip(candidates, scores))
        ranked = sorted(paired, key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:top_k]]


class AnswerGenerator:
    """
    Main interface: given parsed chunks and a question, returns answer and supporting chunks.
    """
    def __init__(self):
        self.ret_config = RetrieverConfig()
        self.rerank_config = RerankerConfig()

    def answer(self, chunks: List[Dict[str, Any]], question: str) -> Tuple[str, List[Dict[str, Any]]]:
        logger.info('Answering question', question=question)
        question = sanitize_html(question)
        # 1. Retrieval
        retriever = Retriever(chunks, self.ret_config)
        candidates = retriever.retrieve(question)
        # 2. Re-ranking
        reranker = Reranker(self.rerank_config)
        top_chunks = reranker.rerank(question, candidates, top_k=5)
        # 3. Assemble prompt
        context = "\n\n".join([f"- {c['narration']}" for c in top_chunks])
        prompt = (
            f"You are a knowledgeable assistant. "
            f"Use the following extracted document snippets to answer the question."
            f"\n\nContext:\n{context}"
            f"\n\nQuestion: {question}\nAnswer:"
        )
        # 4. Generate answer
        answer = LLMClient.generate(prompt)
        return answer, top_chunks

# Example usage:
# generator = AnswerGenerator()
# ans, ctx = generator.answer(parsed_chunks, "What was the Q2 revenue?")
