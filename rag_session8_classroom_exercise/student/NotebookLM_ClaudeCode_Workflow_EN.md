# NotebookLM + Claude Code Workflow Notes

This file is intentionally **not** a ready-made prompt pack.

Use it as a thinking guide.

## In NotebookLM
Before asking detailed questions, first identify:
- which files are current authority
- which files are archived or informational only
- which files define terms or synonyms
- which files describe emergency action

Then ask focused questions such as:
- “Which source should be treated as the current authority for inspection intervals?”
- “Which sources could conflict on the same issue?”
- “Which term in the query may not match the exact wording in the source?”

## In Claude Code
Do not ask Claude Code only for the final answer.  
Ask it to show:
- loaded files
- chunk count
- top-3 retrieval results
- similarity scores or ranking order
- the exact evidence snippet used

## What to Inspect Manually
- version status
- whether the source directly answers the question
- whether the source is only background, not authority
- whether a synonym caused retrieval difficulty
- whether a misleading document was ranked too highly

## Good Engineering Habit
A good RAG answer should be:
- grounded
- source-aware
- conflict-aware
- explicit about uncertainty
