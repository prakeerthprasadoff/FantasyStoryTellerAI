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
Update 05/23 - Model set up and ran. I have ran a gpt2-large model and im trying to increase performance. Also trying out different ways to make the orchestrator perform better.

Update 05/23 - I am getting some generation from the gpt-small model. 
<img width="1004" height="398" alt="image" src="https://github.com/user-attachments/assets/8d1eb584-739f-4fc7-983a-973bf2cd1534" />

This is the benchmark and now I am going to train on gpt2-large and try to improve the performance. Also might have overlooked some architectural changes but I am going to look into it again and confirm.

Update 05/26 - I was able to train the gpt2-large model and got some results. But saving the model to disk has caused some I/O exception error. I am now not able to access the file that I used for training. Currently diagnosing and debugging.

<img width="711" height="198" alt="image" src="https://github.com/user-attachments/assets/6b393025-4c4e-45c7-9d72-25fae77b4127" />

Update 05/26 - The file got corrupted somehow and was not able to open it. I have rewritten the python notebook with 'nbformat' and now the file is working.

Update 05/26 - The file is working but I am getting some I/O error which says disk quota exceeded. Trying to solve the issue

<img width="666" height="30" alt="image" src="https://github.com/user-attachments/assets/76402ad6-8cb8-49d9-bb2e-b649aa192a4b" />

Update 05/26 - My issued quota of disk space seems to be full. I have fixed the issue by requesting for a larger RAM allocation in quest and keep the trained model in RAM after training and use it directly instead of saving the best model and loading trained weights.

Update 05/27 - The orchestrator/narrator hasn't been generating very good results in a narrator's point of view and I realised that the LIGHT dataset does not really have a lot of narrator prose in it. So I have expanded the dataset to include https://huggingface.co/datasets/euclaise/writingprompts which is a Reddit scraped dataset which gives the exact prose style narration for the agent for scene setting. I am now training the model on this dataset as well.

Update 05/27 - The model gpt2-large is bering trained (774M parameters). Project Gutenberg has also been initialized with 6 stories to get the fantasy prose style. The 6 stories are : 

    GUTENBERG_BOOKS = [
 
    (62,   "A Princess of Mars — Burroughs"),
    
    (170,  "The Wood Beyond the World — Morris"),
    
    (1251, "Le Morte d'Arthur — Malory"),
    
    (5670, "The Story of the Glittering Plain — Morris"),
    
    (9782, "Champions of the Round Table — Pyle"),
    
    (20776,"Beowulf — Anonymous"),

    ]

UPdate 05/28 - I have completed training the model overnight. Now training the LoRA adapters for the characters.
<img width="991" height="247" alt="image" src="https://github.com/user-attachments/assets/721f34e5-b401-475a-8583-889f6a5b29c1" />

Update 05/28 - The training dataset is only 31K samples and is sparse to learn the dataset on a 774M parameter model. Hence im moving to pretrained Mistral-7B model where i will do fine tuning of the weights on this corpus so that the model which already understands language is fine tuned to match the fantasy domain.

The model itself is not able to understand language from such a sparse dataset, let alone get navigated into the fantasy domain and with the lora adapters, the existing characters from CharacterLLM are bleeding into the context which will also be fixed in v2.

<img width="977" height="320" alt="image" src="https://github.com/user-attachments/assets/50b66213-e0ef-4c87-ac2f-7e39b04bf967" />



