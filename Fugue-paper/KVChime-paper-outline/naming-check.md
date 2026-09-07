# KVChime 命名检索记录

采用名称：**KVChime**。完整题目：**KVChime: Enabling Shared-KV Attention in PIM for Multi-Agent LLM Inference**。

KV 表示共享的 key/value；Chime 取协同发声的含义，对应多个 agent/query 围绕共同 KV 执行。

## arXiv 官方查询

- 检索日期：2026-09-07；官方响应时间为 `2026-09-07T17:01:19Z`。
- 查询条件：`all:"KVChime" OR all:"KV-Chime" OR all:"KV Chime"`。
- 查询范围：arXiv API 的 `all` 可检索元数据字段，包含标题、摘要等；不等同于逐篇扫描 PDF 正文。
- 结果：`opensearch:totalResults = 0`，没有论文条目。
- [官方 API 查询](https://export.arxiv.org/api/query?search_query=all%3A%22KVChime%22%20OR%20all%3A%22KV-Chime%22%20OR%20all%3A%22KV%20Chime%22&start=0&max_results=10)。
- [原始 XML 响应](KVChime-arxiv-name-check.xml)。

## 补充网页检索

检索 `"KVChime"`、`"KVChime" arxiv`、`site:arxiv.org "KV-Chime"` 和 `site:arxiv.org "KV Chime"`，均未返回使用该名称的结果。

采用依据：截至检索日，在上述 arXiv 查询及补充网页检索中未发现同名论文或系统名称。原始响应保留供复核。
