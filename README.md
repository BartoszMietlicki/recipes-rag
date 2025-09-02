# Recipes RAG

A **Retrieval-Augmented Generation (RAG)** system that recommends cooking recipes based on a user query.  
The project was developed in **Google Colab** and is also available on **Kaggle**.

- Kaggle (notebook): https://www.kaggle.com/code/bartekmietlicki/recipes-rag  
- ▶Colab (notebook): https://colab.research.google.com/drive/1UcibjTWtYVBQYuAVxIfq2Vs5p8e2doKU  
- Hugging Face Datasets (artifacts): https://huggingface.co/datasets/bartekmietlicki/recipes-rag

---

## 1. What is this project?

**Goal.** Build a clean, publishable RAG pipeline that **retrieves and composes a recipe answer** tailored to a user query.  
The system is powered by **scraped recipes from allrecipes.com** — **over 62,000** records (name, description, ingredients, directions).

**Core layers:**
- **Recipe summaries.** A lightweight fine-tune of **BART + LoRA** on ~200 hand-crafted examples, then summary generation for the full dataset.
- **Embeddings.** **`sentence-transformers/all-mpnet-base-v2`** (768-dim) computed on summaries.
- **Vector search.** **FAISS** index + ID→row mapping.
- **Answer generation (LLM).** Calls to **Groq API** (default model **`llama-3.3-70b-versatile`**) to produce a well-structured reply (intro, ingredients, steps, brief summary).
- **Topic filter.** A tiny LLM check to ensure queries are culinary; off-topic questions get a short refusal.

The notebook mirrors these ideas in two logical stages:

**Stage I – Build RAG components**  
fine-tuning BART+LoRA, generating summaries, computing `all-mpnet-base-v2` embeddings, and building FAISS;

**Stage II – RAG (inference)**  
vectorizing the user query, retrieving top-k from FAISS, and composing the final answer with Groq (with the culinary-only filter).

If you don’t want to run Stage I, you can use prebuilt artifacts from HF and jump straight to Stage II.

---

## 3. Artifacts (Stage I)

To **skip Stage I** and use a ready-to-run system:

- set `RAG_ONLY = True` in the notebook,  
- the notebook will download artifacts from HF: **bartekmietlicki/recipes-rag**.

Included artifacts:
- `recipes_sum.parquet` — summaries,  
- `embeddings.npy` — `all-mpnet-base-v2` embeddings,  
- `index.faiss` — FAISS index,  
- `id_list.json` — row mapping.

---

## 4. Keys & environment

- **Groq API:** set a secret named `GROQ_API_KEY` (used in sections 7–10).
  - **Colab:** *Tools → Secrets* → add `GROQ_API_KEY`.
  - **Kaggle:** *Add-ons → Secrets* → add `GROQ_API_KEY`, and enable *Settings → Internet = ON*.
- Locally you can export `GROQ_API_KEY` as an environment variable.

The notebook includes a loader that tries, in order: `ENV` → `Kaggle Secrets` → `Colab Secrets`.

---

## 5. How to run

### 5.1. Quick start (RAG-only)
1. Open the notebook (Colab or Kaggle).  
2. Set `RAG_ONLY = True`.  
3. Provide `GROQ_API_KEY`.  
4. Run sections 1 → 10 in order.  
   - §6: retriever, §7: generator (Groq), §8: culinary-only filter, §9: batch tests, §10: interactive mode.

Note: Kaggle **Commit / Run All** executes in a non-interactive environment, so use §9 (batch). For interactive input (§10), use **Colab** or **Kaggle Edit** mode.

### 5.2. Full pipeline (build everything)
1. Set `RAG_ONLY = False`.  
2. Execute Stage I (sections 1–5) — GPU recommended and it takes time.  
3. Then run Stage II (sections 6–10).

---

## 8. Groq rate-limit note (429)

Large models (e.g., `llama-3.3-70b-versatile`) consume daily token budget (TPD) faster.  
If you hit **`RateLimitError (429)`**:
- temporarily switch to `llama-3.1-8b-instant` or any available model of your choice,
- reduce `top_k` (e.g., to 3) and `max_tokens` (e.g., 384–512),
- try again later (rolling window).

---

## 10. Author / contact

Project maintained as **“Recipes RAG.”**  
Links:
- Kaggle: https://www.kaggle.com/code/bartekmietlicki/recipes-rag  
- ▶Colab: https://colab.research.google.com/drive/1UcibjTWtYVBQYuAVxIfq2Vs5p8e2doKU  
- Hugging Face (artifacts): https://huggingface.co/datasets/bartekmietlicki/recipes-rag
