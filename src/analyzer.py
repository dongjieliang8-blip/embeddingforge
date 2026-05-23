"""Analyzer Agent: 分析语料特征，推荐最优嵌入模型与训练策略"""

import json
from pathlib import Path


class AnalyzerAgent:
    def analyze(self, project_path: str) -> dict:
        """分析语料并返回嵌入优化建议"""
        config_path = Path(project_path) / "config.json"
        corpus_path = Path(project_path) / "corpus.jsonl"

        # 读取已有配置
        config = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

        # 分析语料规模
        corpus_size = 0
        if corpus_path.exists():
            with open(corpus_path, "r", encoding="utf-8") as f:
                corpus_size = sum(1 for _ in f)

        # 分析语料特征
        features = self._extract_features(corpus_path)

        # 基于特征推荐策略
        recommendations = self._recommend_strategy(features)

        return {
            "embedding_model": config.get("embedding_model", recommendations["model"]),
            "training_strategy": config.get("training_strategy", recommendations["strategy"]),
            "corpus_type": config.get("corpus_type", features.get("type", "general")),
            "dimension": config.get("dimension", recommendations["dimension"]),
            "batch_size": config.get("batch_size", recommendations["batch_size"]),
            "learning_rate": config.get("learning_rate", recommendations["learning_rate"]),
            "num_epochs": config.get("num_epochs", recommendations["num_epochs"]),
            "max_seq_length": config.get("max_seq_length", recommendations["max_seq_length"]),
            "num_negatives": config.get("num_negatives", recommendations["num_negatives"]),
            "temperature": config.get("temperature", recommendations["temperature"]),
            "corpus_size": corpus_size,
            "features": features,
        }

    def _extract_features(self, corpus_path: Path) -> dict:
        """提取语料特征: 平均长度、词汇丰富度、领域类型"""
        features = {
            "avg_length": 0,
            "vocab_diversity": 0.0,
            "type": "general",
            "num_samples": 0,
            "num_pairs": 0,
            "num_queries": 0,
        }

        if not corpus_path.exists():
            return features

        lengths = []
        all_text = []

        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                features["num_samples"] += 1
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    item = {"text": line}

                text = item.get("text", item.get("sentence", line))
                lengths.append(len(text))
                all_text.append(text)

                # 检查是否为pair/query格式
                if "query" in item or "positive" in item:
                    features["num_pairs"] += 1
                if "query" in item:
                    features["num_queries"] += 1

        if lengths:
            features["avg_length"] = sum(lengths) / len(lengths)

        if all_text:
            all_chars = "".join(all_text)
            unique_chars = set(all_chars)
            features["vocab_diversity"] = len(unique_chars) / max(len(all_chars), 1)

        return features

    def _recommend_strategy(self, features: dict) -> dict:
        """基于语料特征推荐训练策略"""
        num_samples = features.get("num_samples", 0)

        if num_samples < 100:
            return {
                "model": "mimo-v2.5-embedding-small",
                "strategy": "fine_tune_light",
                "dimension": 512,
                "batch_size": 16,
                "learning_rate": 3e-5,
                "num_epochs": 10,
                "max_seq_length": 256,
                "num_negatives": 5,
                "temperature": 0.05,
            }
        elif num_samples < 10000:
            return {
                "model": "mimo-v2.5-embedding-base",
                "strategy": "contrastive_learning",
                "dimension": 768,
                "batch_size": 32,
                "learning_rate": 2e-5,
                "num_epochs": 5,
                "max_seq_length": 512,
                "num_negatives": 15,
                "temperature": 0.05,
            }
        else:
            return {
                "model": "mimo-v2.5-embedding-large",
                "strategy": "multi_stage_training",
                "dimension": 1024,
                "batch_size": 64,
                "learning_rate": 1e-5,
                "num_epochs": 3,
                "max_seq_length": 512,
                "num_negatives": 30,
                "temperature": 0.02,
            }
