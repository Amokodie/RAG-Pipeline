# Session 8 Classroom Exercise — Mini RAG Lab (30 Minutes)

## Theme
**Grounded retrieval, evidence checking, and error diagnosis**

## Learning Goal
In this exercise, you will not be rewarded for getting a fluent answer quickly.  
You will be rewarded for deciding **whether the answer is grounded in the right source**.

You must use:
- **NotebookLM** to understand the document set and judge source authority
- **Claude Code** to run a lightweight retrieval workflow and inspect top results

## Scenario
You are helping a drone maintenance team answer operational and maintenance questions about the **AeroFleet X200 battery cooling subsystem**.

The document set contains:
- current and outdated manuals
- service bulletins
- safety protocols
- incident reports
- reference notes that may confuse retrieval

Your task is to determine:
1. Which source is authoritative
2. Whether the retrieved evidence really supports the answer
3. What kind of retrieval failure happens when the system gets it wrong

## Time Budget
- **0–5 min**: Understand the corpus structure
- **5–15 min**: Answer evidence-based questions
- **15–25 min**: Run and inspect the retrieval baseline
- **25–30 min**: Write your conclusion and one improvement idea

## Required Deliverables
Submit **one short package** containing:

### Deliverable A — Evidence Table
Complete the worksheet for **4 required queries**:
- Q01
- Q03
- Q05
- Q08

For each query, include:
- your final answer
- the document ID(s) used
- one supporting sentence or phrase
- your confidence score (High / Medium / Low)
- whether there is a conflicting or misleading document

### Deliverable B — Retrieval Check
Run the starter notebook or Python file and record the **top-3 retrieved documents** for:
- Q04
- Q05
- Q10

Then write:
- Which result was correct?
- Which result was misleading?
- What failure mode do you observe?

### Deliverable C — Short Reflection (120–180 words)
Answer all three:
1. What is one place where the model or retriever could look convincing but still be wrong?
2. What is one change that could improve retrieval quality?
3. Why is “current authority” important in engineering RAG?

## What You Must NOT Do
- Do **not** submit a direct LLM answer without evidence
- Do **not** treat the longest or most technical source as automatically correct
- Do **not** ignore document version and status
- Do **not** paste the whole dataset into one giant prompt and stop there

## Suggested Workflow

### Step 1 — NotebookLM
Use NotebookLM to:
- map the document types
- identify which documents are current authority
- note one pair of conflicting documents
- identify one document that looks relevant but should not override the main answer

### Step 2 — Claude Code
Use the starter files to:
- load the corpus
- run retrieval for selected queries
- inspect the top results instead of trusting rank #1 automatically

### Step 3 — Judge the Output
For every answer, ask:
- Is the source current?
- Is the source actually answering the question?
- Is there a conflicting source?
- Is the wording in the query different from the wording in the source?

## Fast Finishers
If you finish early, test one improvement:
- change chunk size
- filter out outdated / informational-only documents
- compare TF-IDF retrieval with a sentence-transformer retriever
