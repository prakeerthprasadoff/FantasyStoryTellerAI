# Multi-Agent Fantasy Storyteller
 
A multi-agent storytelling system where characters are **dynamically generated** from your story premise — no hardcoded cast. Built on GPT-2 with per-character LoRA adapters and retrieval-augmented lore injection.
 
> **Status:** Environment setup and dataset downloading implemented. Training and inference in progress.
 
## Planned Architecture
 
1. **You provide a story premise** — e.g. *"We are aboard a ghost ship drifting through endless fog..."*
2. **The orchestrator reasons about the premise** and mints 2–4 characters suited to that world
3. **LoRA adapters are trained** per character on top of a shared GPT-2 base
4. **A RAG vector store** injects relevant lore into each character's context window
5. **Characters respond** to each user turn, with smart routing deciding who speaks
## Setup
 
```bash
pip install transformers datasets peft accelerate faiss-cpu sentence-transformers huggingface_hub
```
 
## Datasets
 
Two sources are downloaded and combined into a single training corpus:
 
**LIGHT Dialogues** — fantasy character dialogues from Meta's LIGHT text adventure game, loaded from [`npc-engine/light-batch-summarize-dialogue`](https://huggingface.co/datasets/npc-engine/light-batch-summarize-dialogue) (~20k episodes).
 
**Character-LLM** — persona-grounded dialogue data from [`fnlp/character-llm-data`](https://huggingface.co/datasets/fnlp/character-llm-data), downloaded via `snapshot_download` due to mixed schemas in the repo.
 
The combined corpus is saved to `fantasy_project/data/combined_corpus.txt`.


Update 05/21 - Downloaeded and loaded dataset. Setting up models
