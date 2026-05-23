"""Trainer Agent: 基于对比学习优化向量嵌入模型"""

import json
import time
from pathlib import Path


class TrainerAgent:
    def train(self, project_path: str, analysis: dict) -> dict:
        """训练嵌入模型"""
        print(f"    加载基座模型: {analysis['embedding_model']}")
        print(f"    训练策略: {analysis['training_strategy']}")
        print(f"    学习率: {analysis['learning_rate']}")
        print(f"    训练轮次: {analysis['num_epochs']}")

        # 加载训练数据
        corpus_path = Path(project_path) / "corpus.jsonl"
        pairs_path = Path(project_path) / "pairs.jsonl"
        queries_path = Path(project_path) / "queries.jsonl"

        train_data = self._load_training_data(corpus_path, pairs_path, queries_path)
        print(f"    训练样本: {len(train_data)} 条")

        # 执行训练 (模拟)
        start_time = time.time()
        result = self._execute_training(train_data, analysis)
        duration = time.time() - start_time

        # 保存模型路径
        model_path = str(Path(project_path) / "output" / "embedding_model")
        output_dir = Path(project_path) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存训练配置
        config_path = output_dir / "training_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({
                "base_model": analysis["embedding_model"],
                "strategy": analysis["training_strategy"],
                "dimension": analysis["dimension"],
                "learning_rate": analysis["learning_rate"],
                "num_epochs": analysis["num_epochs"],
                "batch_size": analysis["batch_size"],
                "num_negatives": analysis["num_negatives"],
                "temperature": analysis["temperature"],
                "max_seq_length": analysis["max_seq_length"],
                "train_samples": len(train_data),
            }, f, ensure_ascii=False, indent=2)

        return {
            "model_path": model_path,
            "model_name": analysis["embedding_model"],
            "strategy": analysis["training_strategy"],
            "dimension": analysis["dimension"],
            "train_loss": result["loss"],
            "val_loss": result["val_loss"],
            "duration": f"{duration:.1f}s",
            "num_samples": len(train_data),
            "output_dir": str(output_dir),
        }

    def _load_training_data(self, corpus_path, pairs_path, queries_path):
        """加载训练数据"""
        data = []

        # 从corpus加载
        if corpus_path.exists():
            with open(corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            data.append({"text": line})

        # 从pairs加载
        if pairs_path.exists():
            with open(pairs_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

        # 从queries加载
        if queries_path.exists():
            with open(queries_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

        return data

    def _execute_training(self, train_data: list, analysis: dict) -> dict:
        """执行训练逻辑 (实际项目中使用Sentence Transformers训练)"""
        # 模拟训练过程
        num_epochs = analysis["num_epochs"]
        total_steps = len(train_data) * num_epochs

        print(f"    总训练步数: {total_steps}")

        # 实际项目中会执行:
        # from sentence_transformers import SentenceTransformer, losses, InputExample
        # from torch.utils.data import DataLoader
        #
        # model = SentenceTransformer(analysis["embedding_model"])
        # train_examples = [InputExample(texts=[d["text"]], label=1.0) for d in train_data]
        # train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=analysis["batch_size"])
        # train_loss = losses.ContrastiveLoss(model=model)
        # model.fit(train_objectives=[(train_dataloader, train_loss)],
        #           epochs=num_epochs,
        #           warmup_steps=100,
        #           output_path=analysis["model_path"])

        return {
            "loss": 0.0234,
            "val_loss": 0.0312,
        }
