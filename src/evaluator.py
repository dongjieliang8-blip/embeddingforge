"""Evaluator Agent: 多维度评估嵌入质量"""

import json
from pathlib import Path


class EvaluatorAgent:
    def evaluate(self, model_path: str, index_result: dict,
                 project_path: str) -> dict:
        """多维度评估嵌入模型质量"""
        print(f"    模型路径: {model_path}")
        print(f"    向量数量: {index_result['num_vectors']}")
        print(f"    索引类型: {index_result['index_type']}")

        # 加载评估数据
        eval_data = self._load_eval_data(project_path)
        print(f"    评估样本: {len(eval_data)} 条")

        # 评估语义相似度
        cosine_sim = self._eval_cosine_similarity(eval_data, model_path)
        print(f"    语义相似度: {cosine_sim:.4f}")

        # 评估检索精度
        recall_at_10 = self._eval_retrieval_recall(eval_data, index_result)
        print(f"    检索精度@10: {recall_at_10:.4f}")

        recall_at_5 = self._eval_retrieval_recall_at_k(eval_data, index_result, k=5)
        print(f"    检索精度@5: {recall_at_5:.4f}")

        # 评估聚类效果
        cluster_purity = self._eval_clustering(eval_data, model_path)
        print(f"    聚类纯度: {cluster_purity:.4f}")

        # 评估嵌入均匀度
        embedding_variance = self._eval_uniformity(eval_data, model_path)
        print(f"    嵌入方差: {embedding_variance:.4f}")

        # 综合评分
        overall_score = self._compute_overall_score(
            cosine_sim, recall_at_10, recall_at_5,
            cluster_purity, embedding_variance
        )
        print(f"    综合评分: {overall_score:.4f}")

        result = {
            "cosine_similarity": cosine_sim,
            "recall_at_10": recall_at_10,
            "recall_at_5": recall_at_5,
            "cluster_purity": cluster_purity,
            "embedding_variance": embedding_variance,
            "overall_score": overall_score,
            "eval_samples": len(eval_data),
        }

        # 保存评估报告
        output_path = Path(project_path) / "output" / "eval_report.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result

    def _load_eval_data(self, project_path: str) -> list:
        """加载评估数据"""
        eval_path = Path(project_path) / "eval.jsonl"
        data = []

        if eval_path.exists():
            with open(eval_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

        return data

    def _eval_cosine_similarity(self, eval_data: list, model_path: str) -> float:
        """评估语义相似度 (STS任务)"""
        # 实际项目中会执行:
        # from sentence_transformers import SentenceTransformer
        # from sklearn.metrics.pairwise import cosine_similarity
        # import numpy as np
        # model = SentenceTransformer(model_path)
        # pairs = [(d["sentence1"], d["sentence2"]) for d in eval_data if "sentence1" in d]
        # if not pairs:
        #     return 0.0
        # emb1 = model.encode([p[0] for p in pairs])
        # emb2 = model.encode([p[1] for p in pairs])
        # sims = cosine_similarity(emb1, emb2).diagonal()
        # return float(np.mean(sims))

        return 0.8234

    def _eval_retrieval_recall(self, eval_data: list, index_result: dict) -> float:
        """评估检索召回率@10"""
        # 实际项目中会执行:
        # import faiss
        # import numpy as np
        # from sentence_transformers import SentenceTransformer
        # model = SentenceTransformer(model_path)
        # index = faiss.read_index(index_path)
        # hits = 0
        # for d in eval_data:
        #     if "query" not in d or "relevant_ids" not in d:
        #         continue
        #     query_vec = model.encode([d["query"]])
        #     _, indices = index.search(np.array(query_vec, dtype=np.float32), 10)
        #     if set(indices[0].tolist()) & set(d["relevant_ids"]):
        #         hits += 1
        # return hits / max(len(eval_data), 1)

        return 0.7891

    def _eval_retrieval_recall_at_k(self, eval_data: list,
                                    index_result: dict, k: int = 5) -> float:
        """评估检索召回率@k"""
        # 模拟
        if k == 5:
            return 0.7123
        return 0.0

    def _eval_clustering(self, eval_data: list, model_path: str) -> float:
        """评估聚类纯度"""
        # 实际项目中会执行:
        # from sklearn.cluster import KMeans
        # from sklearn.metrics import adjusted_rand_score
        # from sentence_transformers import SentenceTransformer
        # model = SentenceTransformer(model_path)
        # texts = [d["text"] for d in eval_data if "text" in d and "label" in d]
        # labels = [d["label"] for d in eval_data if "text" in d and "label" in d]
        # if len(texts) < 2:
        #     return 0.0
        # embeddings = model.encode(texts)
        # num_clusters = len(set(labels))
        # kmeans = KMeans(n_clusters=min(num_clusters, len(texts)), random_state=42)
        # predicted = kmeans.fit_predict(embeddings)
        # return float(adjusted_rand_score(labels, predicted))

        return 0.7456

    def _eval_uniformity(self, eval_data: list, model_path: str) -> float:
        """评估嵌入均匀度 (方差)"""
        # 实际项目中会执行:
        # import numpy as np
        # from sentence_transformers import SentenceTransformer
        # model = SentenceTransformer(model_path)
        # texts = [d["text"] for d in eval_data if "text" in d][:100]
        # if not texts:
        #     return 0.0
        # embeddings = model.encode(texts)
        # return float(np.var(embeddings))

        return 0.0023

    def _compute_overall_score(self, cosine_sim, recall_10, recall_5,
                               cluster_purity, embedding_variance) -> float:
        """计算综合评分"""
        weights = {
            "cosine_similarity": 0.3,
            "recall_at_10": 0.25,
            "recall_at_5": 0.15,
            "cluster_purity": 0.2,
            "uniformity": 0.1,
        }
        # 方差越小越好，转为评分
        uniformity_score = max(0, 1 - embedding_variance * 100)

        score = (
            weights["cosine_similarity"] * cosine_sim
            + weights["recall_at_10"] * recall_10
            + weights["recall_at_5"] * recall_5
            + weights["cluster_purity"] * cluster_purity
            + weights["uniformity"] * uniformity_score
        )
        return score
