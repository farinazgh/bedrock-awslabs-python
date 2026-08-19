# Semantic vs Hybrid Retrieval with Amazon Bedrock Knowledge Bases

## Goal

Compare `SEMANTIC` and `HYBRID` retrieval using the exact identifier:

`BP-STAFF-003`

with the same document, query, and number of results.

## What We Discovered

### 1. Managed Knowledge Base

The first Knowledge Base was fully managed by Bedrock.

Using:

`vectorSearchConfiguration`

failed with:

`vectorSearchConfiguration is not supported for managed knowledge bases`

Managed Knowledge Bases use `managedSearchConfiguration` and automatically use hybrid retrieval, so they cannot be used to compare `SEMANTIC` vs `HYBRID`.

### 2. S3 Vectors Knowledge Base

A second Knowledge Base was created with:

- S3 Vectors
- Amazon Titan Text Embeddings V2

`SEMANTIC` search worked after the data source was synced.

`HYBRID` failed with:

`HYBRID search type is not supported`

S3 Vectors supports semantic/vector retrieval but not hybrid retrieval.

### 3. OpenSearch Serverless Knowledge Base

The Knowledge Base was recreated using OpenSearch Serverless.

Now both worked:

- `SEMANTIC`
- `HYBRID`

Using:

`numberOfResults = 20`

Bedrock returned 12 available chunks.

## Final Result

For the exact identifier `BP-STAFF-003`:

| Search Type | Exact Match Rank |
|---|---:|
| SEMANTIC | #11 |
| HYBRID | #1 |

## Lesson Learned

Semantic search relies on embedding similarity and can perform poorly for identifiers such as:

- Policy IDs
- SKUs
- Ticket numbers
- Error codes
- Product codes

Hybrid search combines semantic/vector retrieval with lexical text search.

In this experiment, the exact identifier moved from:

**Semantic: #11 → Hybrid: #1**

For comparing Bedrock `SEMANTIC` and `HYBRID` retrieval directly, **OpenSearch Serverless was the appropriate vector store**.