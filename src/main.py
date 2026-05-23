"""
EmbeddingForge — 多Agent协作向量嵌入优化流水线

Analyzer -> Trainer -> Indexer -> Evaluator
"""

import json
import sys
from pathlib import Path

from .analyzer import AnalyzerAgent
from .trainer import TrainerAgent
from .indexer import IndexerAgent
from .evaluator import EvaluatorAgent


class EmbeddingForgePipeline:
    def __init__(self):
        self.analyzer = AnalyzerAgent()
        self.trainer = TrainerAgent()
        self.indexer = IndexerAgent()
        self.evaluator = EvaluatorAgent()

    def run(self, project_path: str):
        print("=" * 60)
        print("EmbeddingForge — 向量嵌入优化流水线")
        print("=" * 60)

        # Step 1: 语料分析
        print("\n[1/4] Analyzer Agent: 分析语料特征，推荐嵌入策略...")
        analysis = self.analyzer.analyze(project_path)
        print(f"  推荐模型: {analysis['embedding_model']}")
        print(f"  训练策略: {analysis['training_strategy']}")
        print(f"  语料规模: {analysis['corpus_size']} 条")

        # Step 2: 嵌入训练
        print("\n[2/4] Trainer Agent: 基于对比学习优化嵌入模型...")
        train_result = self.trainer.train(project_path, analysis)
        print(f"  训练损失: {train_result['train_loss']:.4f}")
        print(f"  模型路径: {train_result['model_path']}")
        print(f"  训练时长: {train_result['duration']}")

        # Step 3: 向量索引
        print("\n[3/4] Indexer Agent: 构建高效向量索引...")
        index_result = self.indexer.build_index(train_result['model_path'], project_path)
        print(f"  索引类型: {index_result['index_type']}")
        print(f"  向量维度: {index_result['dimension']}")
        print(f"  索引大小: {index_result['index_size_mb']:.2f} MB")

        # Step 4: 效果评估
        print("\n[4/4] Evaluator Agent: 多维度评估嵌入质量...")
        eval_result = self.evaluator.evaluate(
            train_result['model_path'],
            index_result,
            project_path
        )
        print(f"  语义相似度: {eval_result['cosine_similarity']:.4f}")
        print(f"  检索精度@10: {eval_result['recall_at_10']:.4f}")
        print(f"  聚类纯度: {eval_result['cluster_purity']:.4f}")

        # 输出完整报告
        report = {
            "analysis": analysis,
            "train_result": train_result,
            "index_result": index_result,
            "eval_result": eval_result,
        }

        output_path = Path(project_path) / "output" / "report.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 60)
        print(f"流水线完成! 报告已保存至: {output_path}")
        print("=" * 60)

        return report


def main():
    if len(sys.argv) < 3 or sys.argv[1] != "run":
        print("用法: python -m src.main run <project_path>")
        sys.exit(1)

    project_path = sys.argv[2]
    pipeline = EmbeddingForgePipeline()
    pipeline.run(project_path)


if __name__ == "__main__":
    main()
