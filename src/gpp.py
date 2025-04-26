"""
Generic Pre-Processing Pipeline (GPP) for Document Intelligence

This module handles:
 1. Parsing PDFs via MinerU Python API (OCR/text modes)
 2. Extracting markdown, images, and content_list JSON
 3. Chunking multimodal content (text, tables, images), ensuring tables/images are in single chunks
 4. Parsing markdown tables into JSON 2D structures for dense tables
 5. Narration of tables/images via LLM
 6. Semantic enhancements (deduplication, coreference, metadata summarization)
 7. Embedding computation and storage in Redis & BM25

Each step is modular to support swapping components (e.g. different parsers or stores).
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
import re

from mineru.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from mineru.data.dataset import PymuDocDataset
from mineru.model.doc_analyze_by_custom_model import doc_analyze
from mineru.config.enums import SupportedPdfParseMethod

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
import redis

# LLM client abstraction
from src.utils import LLMClient

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_markdown_table(md: str) -> Optional[Dict[str, Any]]:
    """
    Parses a markdown table into a JSON-like dict:
    { headers: [...], rows: [[...], ...] }
    Handles multi-level headers by nesting lists if needed.
    """
    lines = [l for l in md.strip().splitlines() if l.strip().startswith('|')]
    if len(lines) < 2:
        return None
    header_line = lines[0]
    sep_line = lines[1]
    # Validate separator line
    if not re.match(r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?", sep_line):
        return None
    def split_row(line):
        parts = [cell.strip() for cell in line.strip().strip('|').split('|')]
        return parts
    headers = split_row(header_line)
    rows = [split_row(r) for r in lines[2:]]
    return {'headers': headers, 'rows': rows}

class GPPConfig:
    """
    Configuration for GPP pipeline.
    """
    CHUNK_TOKEN_SIZE = 256
    DEDUP_SIM_THRESHOLD = 0.9
    EXPANSION_SIM_THRESHOLD = 0.85
    COREF_CONTEXT_SIZE = 3

    # Embedding models
    TEXT_EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
    META_EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

    # Redis settings
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    REDIS_VECTOR_INDEX = 'gpp_vectors'

class GPP:
    def __init__(self, config: GPPConfig):
        self.config = config
        # Embedding models
        self.text_embedder = SentenceTransformer(config.TEXT_EMBED_MODEL)
        self.meta_embedder = SentenceTransformer(config.META_EMBED_MODEL)
        # Redis for vectors + metadata
        self.redis = redis.Redis(host=config.REDIS_HOST,
                                 port=config.REDIS_PORT,
                                 db=config.REDIS_DB)
        self.bm25 = None

    def parse_pdf(self, pdf_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Uses MinerU API to parse PDF in OCR/text mode,
        dumps markdown, images, layout PDF, content_list JSON.
        Returns parsed data plus file paths for UI traceability.
        """
        name = os.path.splitext(os.path.basename(pdf_path))[0]
        img_dir = os.path.join(output_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        writer_imgs = FileBasedDataWriter(img_dir)
        writer_md = FileBasedDataWriter(output_dir)
        reader = FileBasedDataReader("")
        pdf_bytes = reader.read(pdf_path)
        ds = PymuDocDataset(pdf_bytes)
        if ds.classify() == SupportedPdfParseMethod.OCR:
            infer = ds.apply(doc_analyze, ocr=True)
            pipe = infer.pipe_ocr_mode(writer_imgs)
        else:
            infer = ds.apply(doc_analyze, ocr=False)
            pipe = infer.pipe_txt_mode(writer_imgs)
        # Visual layout
        pipe.draw_layout(os.path.join(output_dir, f"{name}_layout.pdf"))
        # Dump markdown & JSON
        pipe.dump_md(writer_md, f"{name}.md", os.path.basename(img_dir))
        pipe.dump_content_list(writer_md, f"{name}_content_list.json", os.path.basename(img_dir))

        content_list_path = os.path.join(output_dir, f"{name}_content_list.json")
        with open(content_list_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # UI traceability paths
        data.update({
            'md_path': os.path.join(output_dir, f"{name}.md"),
            'images_dir': img_dir,
            'layout_pdf': os.path.join(output_dir, f"{name}_layout.pdf")
        })
        return data

    def chunk_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Creates chunks of ~CHUNK_TOKEN_SIZE tokens, but ensures any table/image block
        becomes its own chunk (unsplittable), flushing current text chunk as needed.
        """
        chunks, current, token_count = [], {'text': '', 'type': None, 'blocks': []}, 0
        for blk in blocks:
            btype = blk.get('type')
            text = blk.get('text', '')
            if btype in ('table', 'img_path'):
                # Flush existing text chunk
                if current['blocks']:
                    chunks.append(current)
                    current = {'text': '', 'type': None, 'blocks': []}
                    token_count = 0
                # Create isolated chunk for the table/image
                tbl_chunk = {'text': text, 'type': btype, 'blocks': [blk]}
                # Parse markdown table into JSON structure if applicable
                if btype == 'table':
                    tbl_struct = parse_markdown_table(text)
                    tbl_chunk['table_structure'] = tbl_struct
                chunks.append(tbl_chunk)
                continue
            # Standard text accumulation
            count = len(text.split())
            if token_count + count > self.config.CHUNK_TOKEN_SIZE and current['blocks']:
                chunks.append(current)
                current = {'text': '', 'type': None, 'blocks': []}
                token_count = 0
            current['text'] += text + '\n'
            current['type'] = current['type'] or btype
            current['blocks'].append(blk)
            token_count += count
        # Flush remaining
        if current['blocks']:
            chunks.append(current)
        logger.info(f"Chunked into {len(chunks)} pieces (with tables/images isolated).")
        return chunks

    def narrate_multimodal(self, chunks: List[Dict[str, Any]]) -> None:
        """
        For table/image chunks, generate LLM narration. Preserve table_structure in metadata.
        """
        for c in chunks:
            if c['type'] in ('table', 'img_path'):
                prompt = f"Describe this {c['type']} concisely:\n{c['text']}"
                c['narration'] = LLMClient.generate(prompt)
            else:
                c['narration'] = c['text']

    def deduplicate(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Drops near-duplicate narrations via cosine sim > threshold.
        """
        embs = self.text_embedder.encode([c['narration'] for c in chunks], convert_to_tensor=True)
        keep = []
        for i, emb in enumerate(embs):
            if not any((emb @ embs[j]).item() / (np.linalg.norm(emb) * np.linalg.norm(embs[j]))
                       > self.config.DEDUP_SIM_THRESHOLD for j in keep):
                keep.append(i)
        deduped = [chunks[i] for i in keep]
        logger.info(f"Deduplicated: {len(chunks)}→{len(deduped)}")
        return deduped

    def coref_resolution(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Resolve pronouns using preceding context via LLM.
        """
        for idx, c in enumerate(chunks):
            start = max(0, idx-self.config.COREF_CONTEXT_SIZE)
            ctx = "\n".join(chunks[i]['narration'] for i in range(start, idx))
            prompt = f"Context:\n{ctx}\nRewrite pronouns in:\n{c['narration']}"
            c['narration'] = LLMClient.generate(prompt)

    def metadata_summarization(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Summarize sections and attach to metadata for self-contained context.
        """
        sections: Dict[str, List[Dict[str, Any]]] = {}
        for c in chunks:
            sec = c.get('section', 'default')
            sections.setdefault(sec, []).append(c)
        for sec, items in sections.items():
            blob = "\n".join(i['narration'] for i in items)
            summ = LLMClient.generate(f"Summarize this section:\n{blob}")
            for i in items:
                i.setdefault('metadata', {})['section_summary'] = summ

    def build_bm25(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Build BM25 index on token lists for sparse retrieval.
        """
        tokenized = [c['narration'].split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def compute_and_store(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Encode narrations & metadata, store vectors and chunk metadata in Redis.
        """
        txts = [c['narration'] for c in chunks]
        metas = [c.get('metadata', {}).get('section_summary', '') for c in chunks]
        txt_embs = self.text_embedder.encode(txts)
        meta_embs = self.meta_embedder.encode(metas)

        pipe = self.redis.pipeline()
        for i, (c, te) in enumerate(zip(chunks, txt_embs)):
            key = f"chunk:{i}"
            # store metadata
            store = {'narration': c['narration'], 'type': c['type']}
            if 'table_structure' in c:
                store['table_structure'] = json.dumps(c['table_structure'])
            pipe.hset(key, mapping=store)
            # store dense vector
            pipe.hset(self.config.REDIS_VECTOR_INDEX, key, te.tobytes())
        pipe.execute()
        logger.info("Stored embeddings and metadata in Redis.")

    def run(self, pdf_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Executes full GPP: parse → chunk → narrate → enhance → index.
        Returns parse output dict augmented with `chunks` for downstream processes.
        """
        parsed = self.parse_pdf(pdf_path, output_dir)
        blocks = parsed.get('blocks', [])
        chunks = self.chunk_blocks(blocks)
        self.narrate_multimodal(chunks)
        chunks = self.deduplicate(chunks)
        self.coref_resolution(chunks)
        self.metadata_summarization(chunks)
        self.build_bm25(chunks)
        self.compute_and_store(chunks)
        parsed['chunks'] = chunks
        logger.info("GPP pipeline complete.")
        return parsed

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pdf')
    parser.add_argument('outdir')
    args = parser.parse_args()
    gpp = GPP(GPPConfig())
    gpp.run(args.pdf, args.outdir)
