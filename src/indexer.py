"""Indexer Agent: 构建高效向量索引，支持相似度检索"""

import json
from pathlib import Path


class IndexerAgent:
    def build_index(self, model_path: str, project_path: str) -> dict:
        """构建向量索引"""
        config_path = Path(project_path) / "config.json"
        config = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

        index_type = config.get("index_type", "hnsw")
        dimension = config.get("dimension", 768)

        print(f"    索引类型: {index_type}")
        print(f"    向量维度: {dimension}")

        # 加载语料并生成向量
        corpus_path = Path(project_path) / "corpus.jsonl"
        documents = self._load_corpus(corpus_path)
        print(f"    文档数量: {len(documents)}")

        # 生成向量
        embeddings = self._generate_embeddings(documents, model_path)
        print(f"    生成向量: {len(embeddings)} 个")

        # 构建索引
        index_path = str(Path(project_path) / "output" / f"vector_index.{index_type}")
        output_dir = Path(project_path) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        index_info = self._build_faiss_index(
            embeddings, index_type, dimension, index_path
        )

        # 保存文档映射
        doc_map_path = output_dir / "document_map.json"
        with open(doc_map_path, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)

        # 保存索引元数据
        meta_path = output_dir / "index_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "index_type": index_type,
                "dimension": dimension,
                "num_vectors": len(embeddings),
                "model_path": model_path,
                "index_path": index_path,
                "documents_path": str(doc_map_path),
            }, f, ensure_ascii=False, indent=2)

        return {
            "index_type": index_type,
            "dimension": dimension,
            "num_vectors": len(embeddings),
            "index_size_mb": index_info["size_mb"],
            "index_path": index_path,
            "doc_map_path": str(doc_map_path),
            "build_time": index_info["build_time"],
        }

    def _load_corpus(self, corpus_path: Path) -> list:
        """加载语料文档"""
        documents = []
        if not corpus_path.exists():
            return documents

        with open(corpus_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    documents.append({
                        "id": i,
                        "text": item.get("text", item.get("sentence", line)),
                        "metadata": item.get("metadata", {}),
                    })
                except json.JSONDecodeError:
                    documents.append({
                        "id": i,
                        "text": line,
                        "metadata": {},
                    })

        return documents

    def _generate_embeddings(self, documents: list, model_path: str) -> list:
        """为文档生成嵌入向量"""
        # 实际项目中会执行:
        # from sentence_transformers import SentenceTransformer
        # model = SentenceTransformer(model_path)
        # texts = [doc["text"] for doc in documents]
        # embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
        # return embeddings.tolist()

        # 模拟生成
        dimension = 768
        import random
        embeddings = []
        for _ in documents:
            vec = [random.gauss(0, 0.02) for _ in range(dimension)]
            norm = sum(x ** 2 for x in vec) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]
            embeddings.append(vec)
        return embeddings

    def _build_faiss_index(self, embeddings: list, index_type: str,
                           dimension: int, index_path: str) -> dict:
        """构建FAISS索引"""
        # 实际项目中会执行:
        # import faiss
        # import numpy as np
        # vectors = np.array(embeddings, dtype=np.float32)
        # if index_type == "hnsw":
        #     index = faiss.IndexHNSWFlat(dimension, 32)
        #     index.hnsw.efSearch = 128
        # elif index_type == "ivf":
        #     nlist = min(len(vectors) // 10, 256)
        #     quantizer = faiss.IndexFlatIP(dimension)
        #     index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        #     index.train(vectors)
        # else:
        #     index = faiss.IndexFlatIP(dimension)
        # index.add(vectors)
        # faiss.write_index(index, index_path)

        import time
        start = time.time()
        num_vectors = len(embeddings)
        # 模拟索引大小
        size_mb = num_vectors * dimension * 4 / (1024 * 1024)
        build_time = time.time() - start

        return {
            "size_mb": size_mb,
            "build_time": f"{build_time:.2f}s",
        }

    def search(self, query: str, model_path: str, index_path: str,
               doc_map_path: str, top_k: int = 5) -> list:
        """检索相似文档"""
        # 实际项目中会执行:
        # import faiss
        # import numpy as np
        # from sentence_transformers import SentenceTransformer
        # model = SentenceTransformer(model_path)
        # index = faiss.read_index(index_path)
        # with open(doc_map_path, "r") as f:
        #     documents = json.load(f)
        # query_vec = model.encode([query])
        # distances, indices = index.search(np.array(query_vec, dtype=np.float32), top_k)
        # results = []
        # for dist, idx in zip(distances[0], indices[0]):
        #     if idx < len(documents):
        #         results.append({
        #             "id": documents[idx]["id"],
        #             "text": documents[idx]["text"],
        #             "score": float(dist),
        #         })
        # return results

        return []
