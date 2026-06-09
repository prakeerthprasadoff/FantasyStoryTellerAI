# Project Write-Up - Multi-Agent Fantasy Storytelling System

---

## 1. GitHub URL

[https://github.com/prakeerthprasadoff/FantasyStoryTellerAI](https://github.com/prakeerthprasadoff/FantasyStoryTellerAI)

---

## 2. Project Name & Overview

**Multi-Agent Fantasy Storytelling with Hot-Swappable LoRA Adapters**

The idea started simple: what if you could run a collaborative fantasy story where each character actually sounds different - not just a single model responding with different names slapped on, but genuinely different fine-tuned voices? That's what this project tries to do.

The setup is a two-phase training pipeline on top of Mistral-7B-v0.1. Phase 1 fine-tunes the base model on a ~54k example corpus pulled from four sources - the LIGHT fantasy dialogue dataset, Character-LLM, WritingPrompts, and Project Gutenberg public domain fantasy texts - using LoRA (r=16) to teach it the `[TAG]`-based dialogue format and general fantasy vocabulary. That merged model becomes the starting point for Phase 2, where 12 separate character adapters (r=64) are trained, one per archetype: wizard, knight, rogue, oracle, bard, ranger, necromancer, paladin, barbarian, sea captain, druid, and assassin. Each adapter is trained on the 500 corpus examples most semantically similar to that character's persona description.

At inference time, all adapters live on a single Mistral-7B instance simultaneously via PEFT's multi-adapter support, and a `set_adapter()` call switches between them - so running 12 characters costs about the same VRAM as running one (~15GB), not 12× that. A `StoryOrchestrator` manages a narrator agent plus however many character agents the user picks, which respond one after another to each scene the user inputs, each seeing what the previous character said. A FAISS-based RAG index over 15 lore entries gets queried per prompt so relevant world-building context gets injected automatically. The whole thing is wrapped in a Gradio UI running on the HPC where you can pick characters from a dropdown, toggle lore entries on/off, add your own lore, and write custom descriptions without touching any code.

---

## 3. Extra Criteria Pursued

- **Fine-tuning with LoRA**: Two-phase approach - Phase 1 adapts the base model to the fantasy domain, Phase 2 specializes each character. Hot-swapping via `model.set_adapter()` at inference keeps memory usage flat regardless of how many adapters are loaded.

- **Retrieval-Augmented Generation (RAG)**: FAISS flat L2 index over lore documents, embedded with `all-MiniLM-L6-v2`. The query shifts per character - each agent retrieves based on the previous character's line, so the lore that gets injected evolves as the conversation progresses through the cast.

- **Multi-Agent System**: A narrator agent + up to 6 character agents, all sharing one model but with different active adapters. Characters respond sequentially and each one conditions on what the previous said, so there's actual back-and-forth rather than independent parallel responses. A `CharacterFactory` backed by TinyLlama-1.1B generates the cast automatically from a user's premise if they don't want to pick manually.

- **Interactive Deployment**: Gradio UI deployed directly from the Jupyter notebook on the HPC's H100, exposed via a public `gradio.live` link. Users can configure the full story setup - characters, lore, custom descriptions - through the interface.

---

## 4. Difficulties Faced

**The V1 approach of training from scratch completely failed.** The original plan was to train GPT-2 774M from scratch on the fantasy corpus. Loss went down during training, so on paper it looked like it was learning - but the actual generated text was basically incoherent. It had some fantasy-ish tokens in the right general ballpark, but no real sentence structure or narrative coherence. Looking back, this was pretty predictable: you can't teach a 774M parameter model how language works from 54k examples in a couple of epochs. We were asking the model to simultaneously learn grammar, narrative structure, and fantasy style, which is way too much to learn from that little data. Switching to Mistral-7B pretrained and only fine-tuning the adapter weights fixed this almost immediately - the model already knew how to construct sentences, so the fine-tuning could focus entirely on style and format.

**CUDA OOM and disk quota issues.** Even before we dropped GPT-2, there were memory issues during the forward pass. After switching to Mistral with LoRA, a different problem hit: the Phase 1 checkpoint saving was set to `save_steps=100`, which meant it was serializing the full model (embedding layers included, since the vocab was resized) every 100 steps. The disk quota ran out mid-training. Switched to `save_strategy='no'` so it only saves once at the end of training, which solved it.

**CUDA assertion errors during LoRA inference.** After Phase 2 adapters were trained and loaded, running inference would crash with a device-side assertion error. Tracked this down to a tokenizer/embedding size mismatch - we were adding character-specific special tokens like `[WIZARD]` and `[ROGUE]` after the model was already loaded, which expanded the tokenizer vocab but not the model's embedding matrix. One `model.resize_token_embeddings(len(tokenizer))` call after adding tokens fixed it.

**WritingPrompts data contaminating the character adapters.** This one took a while to figure out. The character adapter training pulled the top-500 most semantically similar corpus examples per character using sentence embeddings. The problem was that WritingPrompts makes up about 37% of the corpus, and those examples contain a lot of Reddit-specific metadata mixed in with the actual story text - upvote counts, usernames, moderator notes, even JavaScript code in some cases. When you query for "wizard-like" examples, Reddit posts about wizards rank high, and the model would then learn to generate Reddit formatting along with fantasy text. Fixed it by pre-computing a filter mask that removes any corpus example containing Reddit markers (`/r/`, `/u/`, `upvoted`, etc.) before the semantic retrieval step. Also added post-processing regex patterns to catch any bleed-through that still made it into generation.

**All characters defaulting to the same adapter.** This happened when the Phase 2 canonical adapters weren't on disk - they got lost during a disk quota incident mid-training. The fallback logic was designed to train a new adapter dynamically if needed, but once that first dynamic adapter was trained and loaded, the semantic fallback for all remaining characters would map to it (since it was the only thing in the loaded adapter pool). Every character ended up using the same adapter voice. Fixed this by changing the fallback behavior entirely - instead of training a new adapter when canonical ones aren't available, it now just runs on the bare Phase 1 merged model. That gives reasonable output without character differentiation, which is still better than spending 2.5 minutes training an adapter on contaminated data just to have all characters share it anyway.

Final UI
<img width="1728" height="984" alt="image" src="https://github.com/user-attachments/assets/2eb1a59a-0fd8-42f2-802f-542dcc6db113" />
