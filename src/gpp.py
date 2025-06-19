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
import numpy as np

from src import EmbeddingConfig, GPPConfig, logger, get_embedder, get_chroma_client
from src.utils import OpenAIEmbedder, LLMClient

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
        self.text_embedder = get_embedder()
        self.chroma_client = get_chroma_client()

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
            narrations = [c.get("narration", "") for c in chunks]
            embs = self.text_embedder.embed(narrations)
            
            # Simple cosine similarity check
            keep_indices = []
            for i in range(len(embs)):
                is_duplicate = False
                for j_idx in keep_indices:
                    sim = np.dot(embs[i], embs[j_idx]) / (np.linalg.norm(embs[i]) * np.linalg.norm(embs[j_idx]))
                    if sim > self.config.DEDUP_SIM_THRESHOLD:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    keep_indices.append(i)

            deduped = [chunks[i] for i in keep_indices]
            logger.info(f"Deduplicated: {len(chunks)} -> {len(deduped)}")
            return deduped
        except Exception as e:
            logger.error(f"Deduplication failed: {e}")
            return chunks

    def coref_resolution(self, chunks: List[Dict[str, Any]]) -> None:
        for idx, c in enumerate(chunks):
            start = max(0, idx - self.config.COREF_CONTEXT_SIZE)
            ctx = "\n".join(chunks[i].get("narration", "") for i in range(start, idx))
            prompt = f"Context:\n{ctx}\nRewrite pronouns in:\n{c.get('narration', '')}\n\n give only the rewritten text, no other text"
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
                summ = LLMClient.generate(f"Summarize this section:\n{blob}\n\n give only the summarized text, no other text")
                for i in items:
                    i.setdefault("metadata", {})["section_summary"] = summ
            except Exception as e:
                logger.error(f"Metadata summarization failed for section {sec}: {e}")

    def store_in_chroma(self, chunks: List[Dict[str, Any]], collection_name: str) -> None:
        """
        Computes embeddings and stores the chunks in a ChromaDB collection.
        """
        if not chunks:
            logger.warning("No chunks to store in ChromaDB.")
            return

        collection = self.chroma_client.get_or_create_collection(name=collection_name)
        
        # Prepare data for ChromaDB
        documents = [c['narration'] for c in chunks]
        metadatas = []
        for chunk in chunks:
            # metadata can only contain str, int, float, bool
            meta = {k: v for k, v in chunk.items() if k not in ['narration', 'text', 'id'] and type(v) in [str, int, float, bool]}
            meta['text'] = chunk.get('text', '') # Add original text to metadata
            metadatas.append(meta)
        
        ids = [str(c['id']) for c in chunks]
        
        logger.info(f"Storing {len(chunks)} chunks in ChromaDB collection '{collection_name}'...")
        try:
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            logger.info("Successfully stored chunks in ChromaDB.")
        except Exception as e:
            logger.error(f"Failed to store chunks in ChromaDB: {e}")
            raise

    def run(self, pdf_path: str, output_dir: str, collection_name: str) -> Dict[str, Any]:
        """
        Executes a streamlined GPP: parse -> chunk -> narrate -> store.
        Heavy enhancement steps are bypassed for maximum efficiency.
        """
        parsed_output = self.parse_pdf(pdf_path, output_dir)
        blocks = parsed_output.get("blocks", [])
        
        chunks = self.chunk_blocks(blocks)
        for idx, chunk in enumerate(chunks):
            chunk["id"] = idx
            
        self.narrate_multimodal(chunks)
        
        # NOTE: Heavy enhancement steps are disabled for performance.
        # To re-enable, uncomment the following lines:
        # chunks = self.deduplicate(chunks)
        # self.coref_resolution(chunks)
        # self.metadata_summarization(chunks)
        
        self.store_in_chroma(chunks, collection_name)
        
        parsed_output["chunks"] = chunks
        parsed_output["collection_name"] = collection_name
        logger.info("GPP pipeline complete. Data stored in ChromaDB.")
        return parsed_output
