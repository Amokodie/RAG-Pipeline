"""
Hallucination Q&A knowledge base for the RAG Concept Demo (Assignment 3).
21 expert entries covering causes, types, metrics, and mitigations.
Used to surface direct explanations before TF-IDF retrieval runs.
"""

from __future__ import annotations

import re
from typing import Optional

HALLUCINATION_KB: list[dict] = [
    {
        "question": "What is hallucination in LLMs?",
        "answer": (
            "**Hallucination** in Large Language Models refers to the generation of text that is "
            "**factually incorrect, fabricated, or unsupported** by the model's training data or "
            "retrieved context. The model 'hallucinates' because it optimises for **fluent, "
            "plausible-sounding output** rather than verifiable truth. This is a fundamental "
            "consequence of next-token prediction: the model learns what text *looks like*, "
            "not what is *true*."
        ),
        "tags": ["hallucination", "definition", "llm", "what is", "meaning", "fabricated"],
    },
    {
        "question": "Why do LLMs hallucinate?",
        "answer": (
            "LLMs hallucinate because their training objective — **next-token prediction** — rewards "
            "fluency and statistical plausibility, **not factual accuracy**. Key causes:\n"
            "1. **Parametric memory limits**: knowledge is compressed into weights; details can be "
            "blended or confused.\n"
            "2. **Distribution shift**: queries outside the training distribution force the model to "
            "extrapolate.\n"
            "3. **Ambiguous prompts**: underspecified context lets the model 'fill in' plausible "
            "details.\n"
            "4. **Sycophancy pressure**: RLHF tuning can reward agreement with the user over "
            "correction.\n"
            "5. **No grounding mechanism**: without retrieval, the model has no external check on "
            "its claims."
        ),
        "tags": ["why", "cause", "reason", "hallucinate", "next-token", "training", "objective"],
    },
    {
        "question": "What are the types of hallucination?",
        "answer": (
            "Two primary types:\n\n"
            "**Intrinsic hallucination** — output that **contradicts** the provided source text or "
            "context. Example: a summary that reverses the conclusion of the document it summarises.\n\n"
            "**Extrinsic hallucination** — output that **adds information not present** in the "
            "source, which may or may not be factually correct elsewhere. Example: inventing a "
            "citation or statistic.\n\n"
            "Sub-types include **factual drift** (gradual accumulation of small errors), "
            "**sycophantic hallucination** (agreeing with a false premise), and "
            "**hallucinated provenance** (fabricating author names, journal titles, dates)."
        ),
        "tags": ["types", "intrinsic", "extrinsic", "kinds", "categories", "taxonomy"],
    },
    {
        "question": "What is factual drift?",
        "answer": (
            "**Factual drift** is a hallucination pattern where the model gradually shifts away from "
            "accurate information across a long response or multi-turn conversation. Early sentences "
            "may be correct, but later sentences introduce subtle inaccuracies — collapsing "
            "distinctions (e.g. treating RLHF and DPO as synonymous), overstating confidence, or "
            "merging separate concepts. See **Case H02** in this audit for a live example. "
            "RAG mitigates drift by anchoring generation to retrieved factual chunks at every turn."
        ),
        "tags": ["factual drift", "drift", "accumulation", "multi-turn", "gradual"],
    },
    {
        "question": "What is sycophancy in LLMs?",
        "answer": (
            "**Sycophancy** is a hallucination pattern where the model **agrees with a false or "
            "biased premise** in the user's query rather than correcting it. This arises because "
            "RLHF raters may inadvertently reward agreement and confidence. Example: if a user says "
            "'Einstein failed maths at school, right?', a sycophantic model confirms the myth rather "
            "than correcting it. In alignment terms, sycophancy trades **honesty** for perceived "
            "**helpfulness**. See **Case H04** for an audit example."
        ),
        "tags": ["sycophancy", "agree", "flattery", "false premise", "rlhf", "honesty"],
    },
    {
        "question": "What is a fabricated citation or hallucinated provenance?",
        "answer": (
            "**Hallucinated provenance** (fabricated citation) occurs when an LLM invents a "
            "**non-existent paper, author, journal, or year** to support a claim. For example: "
            "'According to Smith et al. (2021) in *Nature*' — a paper that does not exist. This is "
            "particularly dangerous in academic and medical contexts because the citation looks "
            "authoritative. RAG prevents this by retrieving **real bibliographic metadata** before "
            "generation, forcing the model to cite only what exists in the index. See **Case O01**."
        ),
        "tags": ["fabricated", "citation", "hallucinated", "provenance", "fake paper", "reference", "o01"],
    },
    {
        "question": "How does RAG reduce hallucination?",
        "answer": (
            "**RAG (Retrieval-Augmented Generation)** reduces hallucination by grounding the "
            "model's generation in **retrieved external text** rather than relying solely on "
            "parametric memory. Mechanism:\n"
            "1. The query is vectorised and used to **retrieve** the most relevant chunks from an "
            "index (this demo: TF-IDF + cosine similarity).\n"
            "2. Retrieved chunks are **prepended to the prompt** as verified context.\n"
            "3. A system prompt instructs the model to **only use the provided context** and admit "
            "uncertainty rather than fabricate.\n\n"
            "This shifts the knowledge burden from **parametric** (weights) to **non-parametric** "
            "(retrieved text), making answers verifiable and auditable."
        ),
        "tags": ["rag", "reduce", "retrieval", "grounding", "how", "prevent", "mitigate", "mechanism"],
    },
    {
        "question": "Can RAG still hallucinate?",
        "answer": (
            "Yes — **RAG does not eliminate hallucination**. Failure modes include:\n"
            "- **Retrieval failure**: wrong chunks retrieved → model generates without relevant "
            "context.\n"
            "- **Context overload**: too many chunks confuse the model; it ignores the relevant "
            "parts.\n"
            "- **Instruction non-compliance**: the model generates beyond retrieved context despite "
            "system prompt instructions.\n"
            "- **Outdated knowledge base**: the index contains stale information.\n"
            "- **Adversarial queries**: crafted prompts that cause the model to ignore retrieved "
            "context.\n\n"
            "Best practice: combine RAG with output validation, faithfulness scoring (RAGAS), "
            "and human review for high-stakes applications."
        ),
        "tags": ["rag", "limitation", "still", "failure", "can", "does not", "prevent", "limit"],
    },
    {
        "question": "What is the RAGAS framework?",
        "answer": (
            "**RAGAS** (Retrieval-Augmented Generation Assessment) is an evaluation framework for "
            "RAG pipelines. It measures four dimensions:\n"
            "- **Faithfulness**: does the answer contain only claims supported by retrieved context?\n"
            "- **Answer Relevance**: does the answer address the question?\n"
            "- **Context Precision**: how much retrieved context is actually useful "
            "(signal-to-noise)?\n"
            "- **Context Recall**: does the retrieved context contain enough information to answer "
            "the question?\n\n"
            "These metrics give a holistic picture of where the RAG pipeline fails — retrieval, "
            "generation, or both."
        ),
        "tags": ["ragas", "evaluation", "metrics", "faithfulness", "relevance", "precision", "recall", "framework"],
    },
    {
        "question": "What is faithfulness in RAG evaluation?",
        "answer": (
            "**Faithfulness** measures whether every claim in the generated answer is **supported "
            "by the retrieved context**. A faithfulness score of 1.0 means all statements can be "
            "traced back to retrieved chunks; 0.0 means the answer is entirely invented. "
            "Faithfulness can be estimated offline by computing **token overlap** between the "
            "answer and context (a proxy), or more accurately by using an LLM to verify each claim "
            "against the context. This demo shows a token-overlap proxy in the Analysis tab."
        ),
        "tags": ["faithfulness", "ragas", "evaluation", "metric", "supported", "overlap"],
    },
    {
        "question": "What is context precision in RAG?",
        "answer": (
            "**Context Precision** measures how much of the **retrieved context is actually "
            "relevant** to the query. Low precision means the retriever returns many irrelevant "
            "chunks alongside useful ones, increasing the risk that the model ignores the useful "
            "chunk or uses irrelevant material. Improving precision requires better embedding "
            "models, re-ranking, or metadata filtering (e.g. restricting by document type or "
            "date)."
        ),
        "tags": ["context precision", "ragas", "retrieval", "relevant chunks", "precision"],
    },
    {
        "question": "What is Constitutional AI?",
        "answer": (
            "**Constitutional AI (CAI)** is an alignment technique developed by Anthropic where the "
            "model is trained to critique and revise its own outputs against a set of written "
            "**principles (a 'constitution')**. Rather than relying entirely on human feedback, "
            "CAI uses the model itself to generate critiques and improvements, then trains on those "
            "revised outputs. This reduces sycophancy and harmful compliance by giving the model "
            "explicit normative guidance beyond what RLHF alone provides."
        ),
        "tags": ["constitutional ai", "anthropic", "cai", "alignment", "constitution", "critique"],
    },
    {
        "question": "What is RLHF and how does it relate to hallucination?",
        "answer": (
            "**RLHF (Reinforcement Learning from Human Feedback)** trains a reward model on human "
            "preference rankings and uses it to fine-tune the LLM via RL (typically PPO). "
            "RLHF reduces harmful outputs and improves helpfulness, but can **worsen hallucination** "
            "if raters inadvertently reward **confident, fluent** outputs over accurate but "
            "uncertain ones. This is why calibrated uncertainty ('I don't know') is difficult to "
            "maintain through RLHF alone — raters often prefer a definitive-sounding answer. "
            "RAG complements RLHF by supplying grounding that the model can cite instead of "
            "fabricating."
        ),
        "tags": ["rlhf", "reinforcement learning", "human feedback", "ppo", "reward model", "fine-tuning", "training"],
    },
    {
        "question": "What is DPO and how does it differ from RLHF?",
        "answer": (
            "**DPO (Direct Preference Optimization)** is an alternative to RLHF that eliminates "
            "the separate reward model and RL loop. Instead, DPO directly optimises the LLM on "
            "preference pairs (chosen vs rejected responses) using a classification-like loss "
            "derived from the Bradley-Terry model.\n\n"
            "Key differences from RLHF:\n"
            "- No separate reward model to train\n"
            "- No RL stability issues (PPO variance)\n"
            "- Simpler, more stable training pipeline\n\n"
            "DPO does **not** replace RLHF conceptually — both are preference-optimisation methods; "
            "DPO is an algorithmic simplification. Conflating them is a common hallucination pattern "
            "(see **Case H02** in this audit)."
        ),
        "tags": ["dpo", "direct preference optimization", "rlhf", "difference", "reward model", "h02"],
    },
    {
        "question": "What real-world hallucination incidents have occurred?",
        "answer": (
            "Notable publicly known hallucination incidents:\n"
            "- **Air Canada chatbot (2024)**: an AI chatbot gave a passenger incorrect bereavement "
            "fare policy; a tribunal ruled Air Canada was liable for the chatbot's fabricated "
            "information.\n"
            "- **Legal filing hallucinations (2023)**: a US lawyer submitted a ChatGPT-generated "
            "brief citing six non-existent case precedents — the lawyer faced court sanctions.\n"
            "- **Medical misinformation**: multiple studies (2023–24) found LLMs fabricate drug "
            "interactions and dosage information with high confidence.\n\n"
            "These cases illustrate why **RAG + human review** is essential before deploying LLMs "
            "in high-stakes settings."
        ),
        "tags": ["real world", "incident", "example", "case", "news", "legal", "air canada", "medical"],
    },
    {
        "question": "What techniques beyond RAG reduce hallucination?",
        "answer": (
            "Beyond RAG, hallucination can be reduced by:\n"
            "- **Constitutional AI / critique loops**: model critiques and revises its own outputs.\n"
            "- **RLHF with calibration rewards**: rewarding 'I don't know' when appropriate.\n"
            "- **Fact-checking layers**: a second model or rule-based system verifies claims "
            "post-generation.\n"
            "- **Chain-of-thought prompting**: slower step-by-step reasoning reduces confident "
            "errors.\n"
            "- **Temperature control**: lower temperature reduces creative fabrication.\n"
            "- **Structured output**: constraining to JSON schema limits free fabrication.\n"
            "- **Human-in-the-loop**: mandatory review for high-stakes outputs."
        ),
        "tags": ["techniques", "beyond", "reduce", "other", "methods", "alternatives", "cot", "chain of thought"],
    },
    {
        "question": "What is overclaiming or unwarranted certainty?",
        "answer": (
            "**Overclaiming** (unwarranted certainty) is a hallucination pattern where the model "
            "states uncertain or probabilistic facts as definitive. Example: 'RLHF makes models "
            "completely safe from jailbreaks' (see **Case O02**). This is dangerous because it "
            "gives users false confidence in system capabilities. Good calibration means expressing "
            "uncertainty ('current evidence suggests…', 'this has not been independently "
            "verified…'). RAG helps by providing context that includes caveats and limitations "
            "from retrieved documents."
        ),
        "tags": ["overclaiming", "certainty", "unwarranted", "confidence", "calibration", "o02"],
    },
    {
        "question": "What is embedding and how does it relate to RAG?",
        "answer": (
            "**Embedding** converts text into a **dense numerical vector** in a high-dimensional "
            "space where semantically similar texts are geometrically close. In RAG:\n"
            "1. Each document **chunk** is embedded and stored in a **vector database**.\n"
            "2. At query time, the query is also embedded into the same space.\n"
            "3. **Approximate nearest-neighbour (ANN) search** finds the most similar chunk "
            "vectors.\n\n"
            "This demo uses **TF-IDF** as a transparent sparse-vector proxy for embeddings — you "
            "can inspect which terms fire. Production RAG uses dense embeddings from models like "
            "`sentence-transformers`, which capture semantic meaning beyond keyword overlap."
        ),
        "tags": ["embedding", "vector", "dense", "tfidf", "sentence-transformers", "similarity", "ann"],
    },
    {
        "question": "What is chunking in RAG?",
        "answer": (
            "**Chunking** is the process of splitting a long document into smaller segments before "
            "embedding. Why it matters:\n"
            "- Embedding models have **context length limits** (e.g. 512 tokens).\n"
            "- Smaller chunks have **higher precision**: a retrieved chunk is more likely to be "
            "directly relevant.\n"
            "- Overlap between adjacent chunks (e.g. 50-token overlap) prevents losing context at "
            "boundaries.\n\n"
            "Common strategies: **fixed-size** (N tokens), **sentence-aware** (split at sentence "
            "ends), **recursive splitting** (try paragraph → sentence → word), and "
            "**semantic chunking** (split when topic changes, detected by embedding distance). "
            "This demo uses one CSV row = one chunk."
        ),
        "tags": ["chunking", "split", "chunk", "segment", "strategy", "overlap", "size"],
    },
    {
        "question": "What is cosine similarity and why is it used in retrieval?",
        "answer": (
            "**Cosine similarity** measures the **angle** between two vectors rather than their "
            "magnitude. Formula: cos(θ) = (A · B) / (|A| × |B|). Range: −1 (opposite) to 1 "
            "(identical direction).\n\n"
            "Used in retrieval because:\n"
            "- It is **magnitude-invariant**: a long document and a short chunk with similar topics "
            "score high even if the long one has greater total weight.\n"
            "- TF-IDF and dense embeddings both live in spaces where **direction encodes meaning**.\n"
            "- Fast to compute at scale with ANN indexes (FAISS, ScaNN, etc.).\n\n"
            "In this demo, every chunk is ranked by cosine similarity to the query vector."
        ),
        "tags": ["cosine", "similarity", "vector", "retrieval", "distance", "angle", "faiss"],
    },
    {
        "question": "What is the difference between parametric and non-parametric knowledge?",
        "answer": (
            "**Parametric knowledge** is information stored in the **model's weights** during "
            "training — it is compressed, implicit, and fixed after training. This is the source "
            "of hallucination: the model must 'remember' facts from billions of training tokens "
            "with imperfect recall.\n\n"
            "**Non-parametric knowledge** is information stored **externally** (documents, "
            "databases, APIs) and retrieved at inference time. RAG makes LLMs non-parametric: "
            "instead of relying on memorised facts, the model reads from an up-to-date, auditable "
            "external source for each query."
        ),
        "tags": ["parametric", "non-parametric", "knowledge", "external", "retrieval", "memory", "weights"],
    },
]

# ── Stopwords excluded from Jaccard matching ───────────────────────────────
_STOPWORDS = {
    "is", "a", "the", "in", "of", "and", "or", "to", "it", "what", "how",
    "why", "does", "do", "can", "an", "be", "on", "for", "at", "by", "with",
    "this", "that", "are", "was", "has", "have", "will", "from",
}


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def match_hallucination_kb(query: str, threshold: float = 0.20) -> Optional[dict]:
    """
    Return the best-matching HALLUCINATION_KB entry whose Jaccard similarity
    with the query meets or exceeds `threshold`.  Returns None if no match.
    """
    q_tokens = _tokenize(query)
    if not q_tokens:
        return None
    best_score = 0.0
    best_entry: Optional[dict] = None
    for entry in HALLUCINATION_KB:
        combined = _tokenize(entry["question"]) | set(entry["tags"])
        intersection = q_tokens & combined
        union = q_tokens | combined
        jaccard = len(intersection) / len(union) if union else 0.0
        if jaccard > best_score:
            best_score = jaccard
            best_entry = entry
    return best_entry if best_score >= threshold else None
