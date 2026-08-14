# Enterprise RAG Retrieval Framework

## Overview

Enterprise knowledge is usually distributed across heterogeneous sources such as Slack conversations, internal documents, meeting transcripts, pull requests, URLs, customer information, and employee metadata.

A traditional Retrieval-Augmented Generation (RAG) pipeline generally uses a single retrieval strategy for every query. However, different enterprise queries require different types of evidence.

For example:

* Exact names, IDs, and document titles may benefit from **lexical retrieval such as BM25**.
* Conceptual questions may benefit from **dense semantic retrieval**.
* Employee- or product-specific questions may require **metadata-aware retrieval**.
* Questions involving terms such as *previous release*, *latest version*, or *earlier discussion* require **temporal understanding**.
* Some questions require evidence from multiple connected sources and therefore need **multi-hop retrieval**.

This project investigates how combining multiple retrieval strategies can improve evidence retrieval for enterprise RAG systems.

---

# Objective

The main objective of this project is to develop a:

**Metadata-Aware Hybrid Retrieval and Adaptive Re-ranking Framework for Improving Enterprise RAG Accuracy**

The framework aims to improve retrieval by combining:

* Sparse / lexical retrieval
* Dense semantic retrieval
* Hybrid retrieval
* Enterprise metadata
* Temporal information
* Cross-source relationships
* Adaptive query-specific retrieval
* Re-ranking

Instead of assuming that one retrieval method works equally well for every query, the project explores selecting and combining retrieval strategies depending on the characteristics of the user query.

---

# Research Motivation

During initial experiments, pure dense retrieval often returned documents that were semantically related to a query but did not contain the evidence required to answer it.

For example, an employee-role query could retrieve general Slack messages mentioning employees in the same product instead of the documents associated with the requested role.

Similarly, questions about a **previous release** could retrieve more recent conversations because they contain semantically similar words such as the product name or customer references.

These observations motivate the need for a retrieval pipeline that considers more than vector similarity alone.

The project therefore studies whether combining lexical, semantic, metadata, temporal, and cross-document signals can improve retrieval quality.

---

# Project Architecture

```text
                    Enterprise Query
                           |
                           v
                   Query Understanding
                           |
                           v
                +----------------------+
                | Adaptive Retrieval   |
                +----------------------+
                           |
             +-------------+-------------+
             |                           |
             v                           v
       BM25 Retrieval              Dense Retrieval
      Lexical Matching          Semantic Similarity
                                         |
                                         v
                                      FAISS
             |                           |
             +-------------+-------------+
                           |
                           v
                    Hybrid Retrieval
                       RRF Fusion
                           |
                           v
                 Metadata-Aware Layer
                 - Product
                 - Source
                 - Author
                 - Employee Role
                 - Channel
                 - Date / Time
                           |
                           v
                Temporal / Multi-Hop
                      Retrieval
                           |
                           v
                       Re-ranking
                           |
                           v
                 Top Relevant Evidence
                           |
                           v
                    RAG / LLM Layer
                           |
                           v
                 Grounded Final Answer
```

The architecture is being developed incrementally so that each retrieval method can first be evaluated independently before being combined into the final adaptive framework.

---

# Dataset

This project uses the **HERB — Heterogeneous Enterprise RAG Benchmark** developed by Salesforce.

HERB represents enterprise information distributed across multiple types of sources, including:

* Slack conversations
* Internal documents
* Meeting transcripts
* Meeting chats
* Pull requests
* URLs
* Employee metadata
* Customer metadata

The local preprocessing pipeline currently produces approximately:

```text
Enterprise products       : 30
Unified retrieval records : 38,600
Generated chunks          : 41,790
Answerable questions      : 815
Unanswerable questions    : 699
```

Each retrieval chunk retains useful metadata such as:

```text
doc_id
chunk_id
parent_doc_id
source
product
author
user_id
channel
date / timestamp
```

This metadata is used in later retrieval experiments.

---

# Data Processing Pipeline

The current preprocessing pipeline follows:

```text
HERB Raw Dataset
       |
       v
loader.py
       |
       v
Unified Enterprise Records
       |
       v
chunker.py
       |
       v
Text Chunks
       |
       v
embeddings.py
       |
       v
MiniLM Embeddings
       |
       v
vector_store.py
       |
       v
FAISS Vector Index
```

### 1. Data Loading

`loader.py` converts heterogeneous HERB sources into a common retrieval representation.

### 2. Chunking

`chunker.py` divides long enterprise documents into smaller retrieval units while preserving their source metadata.

### 3. Embedding Generation

Sentence Transformer:

```text
all-MiniLM-L6-v2
```

is used to generate normalized dense embeddings.

Each embedding contains:

```text
384 dimensions
```

### 4. Vector Indexing

FAISS is used to index the dense vectors and perform semantic similarity search.

---

# Retrieval Methods

The project evaluates retrieval methods incrementally.

## 1. BM25 Retrieval

BM25 is used as the lexical retrieval baseline.

It is useful when the query contains important exact terms such as:

* Employee IDs
* Product names
* Document names
* Technical keywords
* Entity names

---

## 2. Dense Retrieval

Dense retrieval uses:

```text
SentenceTransformer
        +
all-MiniLM-L6-v2
        +
FAISS
```

The query is converted into an embedding and compared with enterprise chunk embeddings using vector similarity.

Dense retrieval is useful for identifying semantically related information even when the exact query words are not present.

---

## 3. Hybrid Retrieval

BM25 and Dense retrieval are combined to take advantage of both:

```text
BM25
Exact / lexical similarity
        +
Dense
Semantic similarity
        |
        v
Hybrid Retrieval
```

Reciprocal Rank Fusion (RRF) is being explored for combining the two rankings because BM25 and dense similarity scores operate on different numerical scales.

---

## 4. Metadata-Aware Retrieval

Enterprise metadata is incorporated to improve ranking.

Examples include:

```text
Product
Source
Author
Employee Role
Slack Channel
Document Date
Timestamp
```

For example, if a query asks for a **Marketing Research Analyst**, employee metadata can help distinguish relevant employees from other people merely mentioned in the same Slack channel.

---

## 5. Temporal Retrieval

**In Progress**

Some enterprise questions contain temporal relationships such as:

```text
previous release
latest release
before
after
earlier version
most recent
```

A purely semantic retriever may confuse current and historical information.

The temporal retrieval component aims to identify the relevant time or release context before selecting evidence.

---

## 6. Multi-Hop Retrieval

**In Progress**

Some HERB questions require information distributed across multiple artifacts.

Example:

```text
User Question
      |
      v
Slack discussion
      |
      v
Referenced document
      |
      v
Document author
      |
      v
Employee metadata
```

The framework therefore explores retrieving connected evidence rather than treating every chunk independently.

---

## 7. Adaptive Retrieval and Re-ranking

**Planned / Ongoing**

The final framework will identify characteristics of the query and adapt the retrieval strategy.

Example:

```text
General semantic query
        |
        v
Hybrid Retrieval


Entity / Role query
        |
        v
Metadata-Aware Hybrid Retrieval


Temporal query
        |
        v
Temporal + Multi-Hop Retrieval
```

The retrieved candidate set can then be re-ranked before being provided to the final RAG generation layer.

---

# Experimental Observation

Initial experiments show why using only one retrieval strategy can be insufficient.

For one HERB employee-role query, pure dense retrieval ranked two relevant market-research documents at approximately:

```text
Relevant Document 1 -> Dense Rank 52
Relevant Document 2 -> Dense Rank 144
```

A metadata-aware hybrid experiment was able to move these relevant documents to the top results.

However, temporal questions involving **previous-release customer feedback** still produced incorrect evidence, showing that temporal and multi-hop retrieval remain important areas for improvement.

These failure cases are being treated as part of the research evaluation rather than manually hard-coding answers.

---

# Evaluation Metrics

Retrieval methods will be compared using standard Information Retrieval metrics.

### Recall@K

Measures how many relevant documents are retrieved within the top K results.

### Precision@K

Measures how many retrieved documents within the top K are relevant.

### Mean Reciprocal Rank — MRR

Measures how highly the first relevant result appears in the ranking.

### nDCG@K

Measures ranking quality while giving greater importance to relevant results appearing near the top.

The planned comparison is:

```text
BM25
        vs
Dense
        vs
Hybrid
        vs
Metadata-Aware Hybrid
        vs
Temporal / Multi-Hop Hybrid
        vs
Adaptive Retrieval + Re-ranking
```

HERB benchmark citations will be used as retrieval ground truth during evaluation.

---

# Current Development Status

```text
HERB dataset integration          Completed
Unified data loader               Completed
Chunking pipeline                 Completed
Dense embedding generation        Completed
FAISS vector indexing             Completed
Dense retrieval baseline          Completed
BM25 experiments                  In Progress
Hybrid retrieval                  In Progress
Metadata-aware retrieval          Experimental
Temporal retrieval                Planned
Multi-hop retrieval               Planned
Adaptive query routing            Planned
Re-ranking                        Planned
Full HERB evaluation              Planned
LLM grounded answer generation    Planned
```

This project is currently an **ongoing research implementation**, and retrieval components are being evaluated incrementally.

---

# Evaluation Strategy

Rather than optimizing the system for individual sample questions, retrieval methods will be evaluated over multiple HERB benchmark questions.

The objective is to determine which retrieval method performs best for different query characteristics.

The planned experimental analysis includes:

```text
Query Type
    |
    +-- Semantic
    |
    +-- Entity / Role
    |
    +-- Temporal
    |
    +-- Multi-Hop
    |
    +-- Exact / Lexical
```

Performance across these categories will be used to guide the adaptive retrieval strategy.

---

# Project Structure

```text
Enterprise-RAG/
|
├── herb/
│   |
│   ├── loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   │
│   ├── retrievers/
│   │   ├── bm25_retriever.py
│   │   ├── dense_retriever.py
│   │   ├── hybrid_retriever.py
│   │   ├── metadata_retriever.py
│   │   ├── temporal_retriever.py
│   │   └── multihop_retriever.py
│   │
│   ├── adaptive_retriever.py
│   ├── reranker.py
│   ├── evaluate_retrieval.py
│   │
│   └── data/
│
├── data/
├── src/
├── main.py
├── requirements.txt
└── README.md
```

Some files shown above represent the target research architecture and are being implemented incrementally.

---

# How to Run

## 1. Activate the Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

## 2. Load and Normalize HERB Data

```bash
py herb\loader.py
```

## 3. Generate Chunks

```bash
py herb\chunker.py
```

## 4. Generate Dense Embeddings

```bash
py herb\embeddings.py
```

## 5. Build the FAISS Index

```bash
py herb\vector_store.py
```

Individual retrieval experiments can then be executed from the `herb/retrievers/` directory as they are implemented.

---

# Project Direction

The long-term goal of the project is to investigate the following question:

> **Can enterprise RAG accuracy be improved by dynamically combining lexical, semantic, metadata, temporal, and multi-hop retrieval strategies according to the characteristics of the user query?**

The project is being developed as an experimental framework for studying this problem on heterogeneous enterprise knowledge sources.
