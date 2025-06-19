"""
AnswerGenerator: orchestrates retrieval, re-ranking, and answer generation.

This module contains:
 - Retriever: Hybrid BM25 + dense retrieval over parsed chunks
 - Reranker: Cross-encoder based re-ranking of candidate chunks
 - AnswerGenerator: ties together retrieval, re-ranking, and LLM generation

Each component is modular and can be swapped or extended (e.g., add HyDE retriever).
"""
import os
from typing import List, Dict, Any, Tuple
import streamlit as st

from src import RerankerConfig, logger
from src.utils import LLMClient
from src.retriever import Retriever, RetrieverConfig

class Reranker:
    """
    Cross-encoder re-ranker using a transformer-based sequence classification model.
    """
    @staticmethod
    @st.cache_resource(show_spinner="Loading reranker model...")
    def load_model_and_tokenizer(model_name, device):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(device)
        return tokenizer, model

    def __init__(self, config: RerankerConfig):
        try:
            self.tokenizer, self.model = self.load_model_and_tokenizer(config.MODEL_NAME, config.DEVICE)
        except Exception as e:
            logger.error(f'Failed to load reranker model: {e}')
            raise

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Score each candidate and return top_k sorted by relevance."""
        if not candidates:
            logger.warning('No candidates provided to rerank.')
            return []
        try:
            import torch
            inputs = self.tokenizer(
                [query] * len(candidates),
                [c.get('narration', '') for c in candidates],
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(RerankerConfig.DEVICE)
            with torch.no_grad():
                out = self.model(**inputs)
            
            logits = out.logits
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(-1)  # only squeeze if it's (batch, 1)

            probs = torch.sigmoid(logits).cpu().numpy().flatten()  # flatten always ensures 1D array
            paired = [(c, float(probs[idx])) for idx, c in enumerate(candidates)]

            ranked = sorted(paired, key=lambda x: x[1], reverse=True)
            return [c for c, _ in ranked[:top_k]]
        except Exception as e:
            logger.error(f'Reranking failed: {e}')
            return candidates[:top_k]


class AnswerGenerator:
    """
    Main interface: initializes Retriever + Reranker once, then
    answers multiple questions without re-loading models each time.
    """
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.retriever = Retriever(chunks, RetrieverConfig)
        self.reranker  = Reranker(RerankerConfig)
        self.top_k = RetrieverConfig.TOP_K // 2

    def answer(
        self, question: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        candidates = self.retriever.retrieve(question)
        top_chunks = self.reranker.rerank(question, candidates, self.top_k)
        context = "\n\n".join(f"- {c['narration']}" for c in top_chunks)
        prompt = (
            "You are a knowledgeable assistant. Use the following snippets to answer."
            f"\n\nContext information is below: \n"
            '------------------------------------'
            f"{context}"
            '------------------------------------'
            "Given the context information above I want you \n"
            "to think step by step to answer the query in a crisp \n"
            "manner, incase you don't have enough information, \n"
            "just say I don't know!. \n\n"
            f"\n\nQuestion: {question} \n"
            "Answer:"
        )
        answer = LLMClient.generate(prompt)
        return answer, top_chunks

