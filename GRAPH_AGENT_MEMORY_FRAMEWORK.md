# Graph Agent Memory Test Framework

这个框架现在已经整理成一个更独立的 `gam/` 项目包，用来统一测试和对比 Graph Agent Memory 方法。

## 已支持能力

- 用户自定义 LLM `base_url`、`api_key`、`model_name`
- 用户自定义 embedding backend 和 `embedding_model_name`
- 独立目录结构：`gam/config.py`、`gam/methods.py`、`gam/workspace.py`、`gam/reporting.py`、`gam/cli.py`
- 兼容入口保留：`run_graph_memory_benchmark.py`
- 在同一套评测流程里对比多种方法：
  - `graph_full`
  - `basic_retrieval`
  - `no_causal`
  - `no_temporal`
  - `flat_graph`
  - `vector_only`
  - `keyword_only`
  - `scan_only`

## 推荐用法

```bash
cd /home/lhw/clawhouse/MAMGA

python gam_cli.py \
  --dataset data/locomo10.json \
  --sample 0 \
  --methods graph_full,basic_retrieval,no_causal,no_temporal,flat_graph,vector_only,keyword_only \
  --model-name qwen3.5-plus \
  --base-url https://your-openai-compatible-endpoint/v1 \
  --api-key sk-your-key \
  --embedding-backend minilm \
  --max-questions 10 \
  --parallel
```

如果 embedding 也走 OpenAI 兼容接口：

```bash
python gam_cli.py \
  --dataset data/locomo10.json \
  --sample 0 \
  --methods graph_full,vector_only,keyword_only \
  --model-name Qwen/Qwen2.5-72B-Instruct \
  --base-url https://api.siliconflow.cn/v1 \
  --api-key sk-your-key \
  --embedding-backend openai \
  --embedding-model-name text-embedding-3-small \
  --embedding-base-url https://api.siliconflow.cn/v1 \
  --embedding-api-key sk-your-key
```

## 输出

结果会输出到 `benchmark_results/graph_memory_benchmark_*.json`，包含：

- 实验配置
- 每个 sample 的 memory 统计
- 每种方法的逐题结果
- overall / by-category 指标

## 说明

- 新框架层已经尽量独立于原始测试脚本的目录组织；目前仍通过适配层复用底层 memory 构建与查询能力。
- 旧脚本现在也可以通过环境变量 `OPENAI_BASE_URL` 自定义兼容接口。
- `embedding_backend=minilm` 时默认走本地句向量模型，不依赖远程 embedding API。
