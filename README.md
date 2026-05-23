# EmbeddingForge — 多Agent协作向量嵌入优化流水线

基于 Claude Code / DeepSeek API / MiMo-v2.5的多智能体向量嵌入优化流水线，实现语料分析→嵌入训练→向量索引→效果评估的全自动化闭环。

## 架构

```
Analyzer → Trainer → Indexer → Evaluator
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用

```bash
python -m src.main run ./demo/sample_corpus
```

## Agent 说明

- **Analyzer**: 分析语料特征，推荐最优嵌入模型与训练策略
- **Trainer**: 基于对比学习优化向量嵌入模型
- **Indexer**: 构建高效向量索引（HNSW/IVF），支持相似度检索
- **Evaluator**: 多维度评估嵌入质量（语义相似度/检索精度/聚类效果）

## 技术栈

- Python
- Claude Code / DeepSeek API / MiMo-v2.5
- Sentence Transformers
- FAISS / Annoy 向量索引

## License

MIT
