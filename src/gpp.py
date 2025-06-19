"""
Generic Pre-Processing Pipeline (GPP) for Document Intelligence

This module handles:
 1. Parsing PDFs via MinerU Python API (OCR/text modes)
 2. Extracting markdown, images, and content_list JSON
 3. Chunking multimodal content (text, tables, images), ensuring tables/images are in single chunks
 4. Parsing markdown tables into JSON 2D structures for dense tables
 5. Narration of tables/images via LLM
 6. Semantic enhancements (deduplication, coreference, metadata summarization)
 7. Embedding computation for in-memory use

Each step is modular to support swapping components (e.g. different parsers or stores).
"""

import os
import json
from typing import List, Dict, Any, Optional
import re

from src import EmbeddingConfig, GPPConfig
from src.utils import OpenAIEmbedder, LLMClient
from src import logger

def parse_markdown_table(md: str) -> Optional[Dict[str, Any]]:
    """
    Parses a markdown table into a JSON-like dict:
    { headers: [...], rows: [[...], ...] }
    Handles multi-level headers by nesting lists if needed.
    """
    lines = [l for l in md.strip().splitlines() if l.strip().startswith("|")]
    if len(lines) < 2:
        return None
    header_line = lines[0]
    sep_line = lines[1]
    # Validate separator line
    if not re.match(r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?", sep_line):
        return None

    def split_row(line):
        parts = [cell.strip() for cell in line.strip().strip("|").split("|")]
        return parts

    headers = split_row(header_line)
    rows = [split_row(r) for r in lines[2:]]
    return {"headers": headers, "rows": rows}


class GPP:
    def __init__(self, config: GPPConfig):
        self.config = config
        # Lazy import heavy libraries
        from sentence_transformers import SentenceTransformer
        # Embedding models
        if EmbeddingConfig.PROVIDER == "openai":
            self.text_embedder = OpenAIEmbedder(EmbeddingConfig.TEXT_MODEL)
            self.meta_embedder = OpenAIEmbedder(EmbeddingConfig.META_MODEL)
        else:
            self.text_embedder = SentenceTransformer(
                EmbeddingConfig.TEXT_MODEL, use_auth_token=True
            )
            self.meta_embedder = SentenceTransformer(
                EmbeddingConfig.META_MODEL, use_auth_token=True
            )

        self.bm25 = None

    def parse_pdf(self, pdf_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Uses MinerU API to parse PDF in OCR/text mode,
        dumps markdown, images, layout PDF, content_list JSON.
        Returns parsed data plus file paths for UI traceability.
        """
        # Lazy import heavy libraries
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
        from magic_pdf.data.dataset import PymuDocDataset
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
        from magic_pdf.config.enums import SupportedPdfParseMethod
        
        name = os.path.splitext(os.path.basename(pdf_path))[0]
        img_dir = os.path.join(output_dir, "images")
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
        pipe.dump_content_list(
            writer_md, f"{name}_content_list.json", os.path.basename(img_dir)
        )

        content_list_path = os.path.join(output_dir, f"{name}_content_list.json")
        with open(content_list_path, "r", encoding="utf-8") as f:
            blocks = json.load(f)
        # UI traceability paths
        return {
            "blocks": blocks,
            "md_path": os.path.join(output_dir, f"{name}.md"),
            "images_dir": img_dir,
            "layout_pdf": os.path.join(output_dir, f"{name}_layout.pdf"),
            "spans_pdf": os.path.join(output_dir, f"{name}_spans.pdf"),
        }

    def chunk_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Creates chunks of ~CHUNK_TOKEN_SIZE tokens, but ensures any table/image block
        becomes its own chunk (unsplittable), flushing current text chunk as needed.
        """
        # Lazy import heavy libraries
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        chunks, current, token_count = [], {"text": "", "type": None, "blocks": []}, 0
        for blk in blocks:
            btype = blk.get("type")
            text = blk.get("text", "")
            if btype in ("table", "img_path"):
                # Flush existing text chunk
                if current["blocks"]:
                    chunks.append(current)
                    current = {"text": "", "type": None, "blocks": []}
                    token_count = 0
                # Create isolated chunk for the table/image
                tbl_chunk = {"text": text, "type": btype, "blocks": [blk]}
                # Parse markdown table into JSON structure if applicable
                if btype == "table":
                    tbl_struct = parse_markdown_table(text)
                    tbl_chunk["table_structure"] = tbl_struct
                chunks.append(tbl_chunk)
                continue
            # Standard text accumulation
            count = len(text.split())
            if token_count + count > self.config.CHUNK_TOKEN_SIZE and current["blocks"]:
                chunks.append(current)
                current = {"text": "", "type": None, "blocks": []}
                token_count = 0
            current["text"] += text + "\n"
            current["type"] = current["type"] or btype
            current["blocks"].append(blk)
            token_count += count
        # Flush remaining
        if current["blocks"]:
            chunks.append(current)
        logger.info(f"Chunked into {len(chunks)} pieces (with tables/images isolated).")
        return chunks

    def narrate_multimodal(self, chunks: List[Dict[str, Any]]) -> None:
        """
        For table/image chunks, generate LLM narration. Preserve table_structure in metadata.
        """
        for c in chunks:
            if c["type"] in ("table", "img_path"):
                prompt = f"Describe this {c['type']} concisely:\n{c['text']}"
                c["narration"] = LLMClient.generate(prompt)
            else:
                c["narration"] = c["text"]

    def deduplicate(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            # Lazy import heavy libraries
            import numpy as np
            from sentence_transformers import SentenceTransformer
            
            narrations = [c.get("narration", "") for c in chunks]
            if EmbeddingConfig.PROVIDER == "openai":
                embs = self.text_embedder.embed(narrations)
            else:
                embs = self.text_embedder.encode(narrations)

            keep = []
            for i, emb in enumerate(embs):
                if not any(
                    (emb @ embs[j]).item()
                    / (np.linalg.norm(emb) * np.linalg.norm(embs[j]) + 1e-8)
                    > self.config.DEDUP_SIM_THRESHOLD
                    for j in keep
                ):
                    keep.append(i)
            deduped = [chunks[i] for i in keep]
            logger.info(f"Deduplicated: {len(chunks)}→{len(deduped)}")
            return deduped
        except Exception as e:
            logger.error(f"Deduplication failed: {e}")
            return chunks

    def coref_resolution(self, chunks: List[Dict[str, Any]]) -> None:
        for idx, c in enumerate(chunks):
            start = max(0, idx - self.config.COREF_CONTEXT_SIZE)
            ctx = "\n".join(chunks[i].get("narration", "") for i in range(start, idx))
            prompt = f"Context:\n{ctx}\nRewrite pronouns in:\n{c.get('narration', '')}"
            try:
                c["narration"] = LLMClient.generate(prompt)
            except Exception as e:
                logger.error(f"Coref resolution failed for chunk {idx}: {e}")

    def metadata_summarization(self, chunks: List[Dict[str, Any]]) -> None:
        sections: Dict[str, List[Dict[str, Any]]] = {}
        for c in chunks:
            sec = c.get("section", "default")
            sections.setdefault(sec, []).append(c)
        for sec, items in sections.items():
            blob = "\n".join(i.get("narration", "") for i in items)
            try:
                summ = LLMClient.generate(f"Summarize this section:\n{blob}")
                for i in items:
                    i.setdefault("metadata", {})["section_summary"] = summ
            except Exception as e:
                logger.error(f"Metadata summarization failed for section {sec}: {e}")

    def build_bm25(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Build BM25 index on token lists for sparse retrieval.
        """
        # Lazy import heavy libraries
        from rank_bm25 import BM25Okapi
        
        tokenized = [c["narration"].split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def compute_and_store(self, chunks: List[Dict[str, Any]], output_dir: str) -> None:
        """
        1. Compute embeddings for each chunk's narration (text_vec)
           and section_summary (meta_vec).
        2. Build two HNSWlib indices (one for text_vecs, one for meta_vecs).
        3. Save both indices to disk.
        4. Dump human-readable chunk metadata (incl. section_summary)
           for traceability in the UI.
        """
        # Lazy import heavy libraries
        import numpy as np
        import hnswlib
        from sentence_transformers import SentenceTransformer

        # --- 1. Prepare embedder ---
        if EmbeddingConfig.PROVIDER.lower() == "openai":
            embedder = OpenAIEmbedder(EmbeddingConfig.TEXT_MODEL)
            embed_fn = embedder.embed
        else:
            st_model = SentenceTransformer(
                EmbeddingConfig.TEXT_MODEL, use_auth_token=True
            )
            embed_fn = lambda texts: st_model.encode(
                texts, show_progress_bar=False
            ).tolist()

        # Batch compute text & meta embeddings ---
        narrations = [c["narration"] for c in chunks]
        meta_texts = [c.get("section_summary", "") for c in chunks]
        logger.info(
            "computing_embeddings",
            provider=EmbeddingConfig.PROVIDER,
            num_chunks=len(chunks),
        )

        text_vecs = embed_fn(narrations)
        meta_vecs = embed_fn(meta_texts)

        if len(text_vecs) != len(chunks) or len(meta_vecs) != len(chunks):
            raise RuntimeError(
                f"Embedding count mismatch: text_vecs={len(text_vecs)}, meta_vecs={len(meta_vecs)}, chunks={len(chunks)}"
            )

        # Convert to numpy arrays
        text_matrix = np.vstack(text_vecs).astype(np.float32)
        meta_matrix = np.vstack(meta_vecs).astype(np.float32)

        # Build HNSW indices ---
        dim = text_matrix.shape[1]
        text_index = hnswlib.Index(space="cosine", dim=dim)
        text_index.init_index(
            max_elements=len(chunks),
            ef_construction=GPPConfig.HNSW_EF_CONSTRUCTION,
            M=GPPConfig.HNSW_M,
        )
        ids = [c["id"] for c in chunks]
        text_index.add_items(text_matrix, ids)
        text_index.set_ef(GPPConfig.HNSW_EF_SEARCH)
        logger.info("text_hnsw_built", elements=len(chunks))

        # Meta index (same dim)
        meta_index = hnswlib.Index(space="cosine", dim=dim)
        meta_index.init_index(
            max_elements=len(chunks),
            ef_construction=GPPConfig.HNSW_EF_CONSTRUCTION,
            M=GPPConfig.HNSW_M,
        )
        meta_index.add_items(meta_matrix, ids)
        meta_index.set_ef(GPPConfig.HNSW_EF_SEARCH)
        logger.info("meta_hnsw_built", elements=len(chunks))

        # Persist indices to disk ---
        text_idx_path = os.path.join(output_dir, "hnsw_text_index.bin")
        meta_idx_path = os.path.join(output_dir, "hnsw_meta_index.bin")
        text_index.save_index(text_idx_path)
        meta_index.save_index(meta_idx_path)
        logger.info(
            "hnsw_indices_saved", text_index=text_idx_path, meta_index=meta_idx_path
        )

        # Dump chunk metadata for UI traceability ---
        meta_path = os.path.join(output_dir, "chunk_metadata.json")
        metadata = {
            str(c["id"]): {
                "text": c.get("text", ""),
                "narration": c["narration"],
                "type": c.get("type", ""),
                "section_summary": c.get("section_summary", ""),
            }
            for c in chunks
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info("chunk_metadata_saved", path=meta_path)

    def run(self, pdf_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Executes full GPP: parse -> chunk -> narrate -> enhance -> index.
        Returns parse output dict augmented with `chunks` for downstream processes.
        """
        parsed = self.parse_pdf(pdf_path, output_dir)
        blocks = parsed.get("blocks", [])
        chunks = self.chunk_blocks(blocks)
        # assigning ID's to chuncks for traceability
        for idx, chunk in enumerate(chunks):
            chunk["id"] = idx
        self.narrate_multimodal(chunks)
        chunks = self.deduplicate(chunks)
        self.coref_resolution(chunks)
        self.metadata_summarization(chunks)
        self.build_bm25(chunks)
        self.compute_and_store(chunks, output_dir)
        parsed["chunks"] = chunks
        logger.info("GPP pipeline complete.")
        return parsed
