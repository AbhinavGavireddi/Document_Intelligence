import os
import json
import pytest
import torch
import numpy as np

from src.gpp import parse_markdown_table, GPP, GPPConfig
from src.qa import Retriever, RetrieverConfig, Reranker, RerankerConfig, AnswerGenerator
from src.utils import LLMClient

# --- Tests for parse_markdown_table ---
def test_parse_markdown_table_valid():
    md = """
    |h1|h2|
    |--|--|
    |a|b|
    |c|d|
    """
    res = parse_markdown_table(md)
    assert res['headers'] == ['h1', 'h2']
    assert res['rows'] == [['a', 'b'], ['c', 'd']]


def test_parse_markdown_table_invalid():
    md = "not a table"
    assert parse_markdown_table(md) is None

# --- Tests for GPP.chunk_blocks ---
class DummyGPPConfig(GPPConfig):
    CHUNK_TOKEN_SIZE = 4  # small threshold for testing

@pytest.fixture
def gpp():
    return GPP(DummyGPPConfig())

@pytest.fixture
def blocks():
    return [
        {'type': 'text', 'text': 'one two three four'},
        {'type': 'table', 'text': '|h|\n|-|\n|v|'},
        {'type': 'text', 'text': 'five six'}
    ]

def test_chunk_blocks_table_isolation(gpp, blocks):
    chunks = gpp.chunk_blocks(blocks)
    # Expect 3 chunks: one text (4 tokens), one table, one text (2 tokens)
    assert len(chunks) == 3
    assert chunks[1]['type'] == 'table'
    assert 'table_structure' in chunks[1]

# --- Tests for Retriever.retrieve combining sparse & dense ---
def test_retriever_combine_unique(monkeypatch):
    chunks = [{'narration': 'a'}, {'narration': 'b'}, {'narration': 'c'}]
    config = RetrieverConfig()
    retr = Retriever(chunks, config)
    # Monkey-patch methods
    monkeypatch.setattr(Retriever, 'retrieve_sparse', lambda self, q, top_k: [chunks[0], chunks[1]])
    monkeypatch.setattr(Retriever, 'retrieve_dense', lambda self, q, top_k: [chunks[1], chunks[2]])
    combined = retr.retrieve('query', top_k=2)
    assert combined == [chunks[0], chunks[1], chunks[2]]

# --- Tests for Reranker.rerank with dummy model and tokenizer ---
class DummyTokenizer:
    def __call__(self, queries, contexts, padding, truncation, return_tensors):
        batch = len(queries)
        return {
            'input_ids': torch.ones((batch, 1), dtype=torch.long),
            'attention_mask': torch.ones((batch, 1), dtype=torch.long)
        }

class DummyModel:
    def __init__(self): pass
    def to(self, device): return self
    def __call__(self, **kwargs):
        # Generate logits: second candidate more relevant
        batch = kwargs['input_ids'].shape[0]
        logits = torch.tensor([[0.1], [0.9]]) if batch == 2 else torch.rand((batch,1))
        return type('Out', (), {'logits': logits})

@pytest.fixture(autouse=True)
def dummy_pretrained(monkeypatch):
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda name: DummyTokenizer())
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, 'from_pretrained', lambda name: DummyModel())
    return

def test_reranker_order():
    config = RerankerConfig()
    rer = Reranker(config)
    candidates = [{'narration': 'A'}, {'narration': 'B'}]
    ranked = rer.rerank('q', candidates, top_k=2)
    # B should be ranked higher than A
    assert ranked[0]['narration'] == 'B'
    assert ranked[1]['narration'] == 'A'

# --- Tests for AnswerGenerator end-to-end logic ---
def test_answer_generator(monkeypatch):
    # Dummy chunks
    chunks = [{'narration': 'hello world'}]
    # Dummy Retriever and Reranker
    class DummyRetriever:
        def __init__(self, chunks, config): pass
        def retrieve(self, q, top_k=10): return chunks
    class DummyReranker:
        def __init__(self, config): pass
        def rerank(self, q, cands, top_k): return chunks

    # Patch in dummy classes
    monkeypatch.setattr('src.qa.Retriever', DummyRetriever)
    monkeypatch.setattr('src.qa.Reranker', DummyReranker)
    # Patch LLMClient.generate
    monkeypatch.setattr(LLMClient, 'generate', staticmethod(lambda prompt: 'TEST_ANSWER'))

    ag = AnswerGenerator()
    ans, sup = ag.answer(chunks, 'What?')
    assert ans == 'TEST_ANSWER'
    assert sup == chunks
