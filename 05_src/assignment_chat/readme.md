# Multi-Service Chat App — Ask Peter

A single conversational Gradio interface where **Peter**, a scholarly AI companion, routes each message to the right back-end service automatically.
---

## How to Launch

### Option 1 — Command Line

```bash
cd 05_src/assignment_chat
python launch_chatapp.py
```

### Option 2 — VS Code

1. Open `launch_chatapp.py` in the editor
2. Click the **▷ Run Python File** button in the top-right toolbar, or press **F5**

### Default URL

**http://localhost:7860**

The app auto-detects a free port starting at 7860 and opens your browser automatically.

> **Expect ~10 seconds on first launch.** Gradio, ChromaDB, and LangChain take several seconds to initialize. My experience is that first load takes about 30 seconds. This is definitely not a production service. However, I definitely will look at optimizing the code at a later date. The app is ready when the URL is printed in the terminal. I have also now updated the code to launch the default browser.

---

## Setup

1. Requires a `.secrets` file in `05_src/assignment_chat/`.

2. Dependencies:
   All dependencies are listed in `pyproject.toml`. No other external packages are used in the code.

3. I assume that there is an existing PDF file at `02_activities/documents/managing_oneself.pdf` (relative to the repo root). This is the .pdf used in the course.

---

## Architecture

Every user message passes through a two-stage pipeline before reaching a service:

```
User message
    │
    ▼
┌─────────────────┐
│  Guardrail Check │  — blocks prompt injection & restricted topics
└────────┬────────┘
         │ safe
         ▼
┌─────────────────┐
│  Intent Router   │  — classifies intent, extracts clean parameter
└────────┬────────┘
         │
    ┌────┴──────────────┐
    │                   │
    ▼                   ▼                   ▼
Weather Service   Ask Drucker       Smart Converter
(Open-Meteo API)  (ChromaDB RAG)    (LangGraph ReAct)
```

For anything that doesn't match a service, Peter answers directly in his scholarly voice.

---

## File Structure
## Folder: /05_src/assignment_chat ##

| File                         | Purpose                                           |
|------------------------------|---------------------------------------------------|
| `__init__.py`                | OpenAI client initialization, `.secrets` loading, and the single source of truth for `API_GATEWAY_URL` — all services import it from here |
| `prompts.py`                 | All system prompts (guardrail, router, Peter's voice, 
                                  service prompts) |
| `guardrails.py`              | Pre-check: blocks prompt injection and restricted 
                                 topics |
| `router.py`                  | Intent classifier and service dispatcher — the 
                                 `peter_chat()` function |
| `weather_service.py`         | Service 1 — live weather via Open-Meteo, summarized by 
                                 OpenAI |
| `ask_drucker_service.py`     | Service 2 — RAG pipeline over the Drucker PDF via 
                                 ChromaDB |
| `smart_converter_service_agent.py` | Service 3 — unit conversion via a LangGraph 
                                 ReAct agent with the `convert_units` tool |
| `main.py`                    | Gradio UI — single `ChatInterface` backed by 
                                `peter_chat()` |
| `launch_chatapp.py`          | Entry point — run this to start the app |
| `chroma_db/`                 | ChromaDB persistent storage (auto-created on first 
                                 run) |
| `readme.md`                  | Project documentation — architecture, setup, and 
                                 service descriptions |

---

## Services

### Service 1 — Weather Info

Get current weather for any city using natural language.

**Example:** *"What's the weather like in Lisbon?"*

**How it works:**
1. The router extracts the city name from the user's message
2. The Open-Meteo geocoding API resolves the city to coordinates
3. Open-Meteo fetches live temperature, humidity, wind speed, and condition code
4. OpenAI rewrites the raw data as a friendly, conversational summary — not a verbatim dump
5. Unfortunately exact city name spelling is required (in English)

---

### Service 2 — Ask Drucker

Ask questions about Peter Drucker's *Managing Oneself* (HBR article) in a persistent conversation. Peter remembers the full conversation and builds on earlier points.

**Example:** *"How do I identify my strengths?"*, *"What does Drucker say about values?"*

**How it works:**
1. On first use, the PDF is loaded, chunked, embedded, and stored in ChromaDB
(I now include the chunks in the file chroma.db and it should launch faster.)
2. On subsequent runs, the collection loads from disk — no re-embedding
3. The user's question is embedded and the top-3 matching chunks are retrieved
4. The full conversation history + retrieved context are passed to OpenAI
5. OpenAI produces an answer from those passages

**Embedding process:**
- **Model:** `text-embedding-3-small` via `OpenAIEmbeddingFunction`
- **Chunking:** `RecursiveCharacterTextSplitter` — `chunk_size=1000`, `chunk_overlap=150`
- **Storage:** ChromaDB `PersistentClient` writing to `chroma_db/` (file-based persistence)
- **When it runs:** Only on the very first launch when the collection is empty. All subsequent runs load from disk with no embedding API calls. I will deploy with the chunks and so it should launch faster.
- **Source PDF:** `02_activities/documents/managing_oneself.pdf`

**Code** (`ask_drucker_service.py`):

```python
# 1. Define the embedding function
ef = OpenAIEmbeddingFunction(model_name="text-embedding-3-small")

# 2. Load and chunk the PDF
loader = PyPDFLoader(str(_PDF_PATH))
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)

# 3. Store chunks and their embeddings in ChromaDB
collection.add(
    documents=[c.page_content for c in chunks],
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    metadatas=[{"page": c.metadata.get("page", 0)} for c in chunks]
)
# The total number of embedding created are:
collection = _get_drucker_collection()
print(f"Embeddings in chroma_db: {collection.count()}")
# Embeddings in chroma_db: 64
# 4. At query time, embed the question and retrieve the top-3 matching chunks
results = collection.query(query_texts=[message], n_results=3)
```
This is an example of a query and matching chunks (top 3):

## Query: What does Drucker say about how to determine my strengths? ##

============================================================
Chunk 1  |  distance=0.6980  |  cosine_sim=0.7564  |  Good/moderate match  |  page=12
============================================================
ments or projects.
In today’s organizations, competence is mea-
sured less in terms of subject matter and 
more in terms of abilities—for example, em-
pathy and stamina under pressure. So it’s up 
to you to help others understand what you’re 
able to contribute to the overall project.
Drucker also notes that your role as an exec-
utive or manager has changed. You no 
longer manage a workforce; you manage in-
dividuals with a variety of skills. Your job, 
then, is to combine these skills in a variety of 
configurations to create the best results for 
your company.
How to Play to Your Strengths
 
by Laura Morgan Roberts, 
Gretchen Spreitzer, Jane Dutton, 
Robert Quinn, Emily Heaphy, and 
Brianna Barker
Harvard Business Review
January 2005
Product no. R0501G
Like Drucker, the authors of this article em-
phasize the importance of understanding 
and leveraging your strengths. They present 
a feedback tool called the Reflective Best Self 
(RBS) exercise, which offers a feedback expe-

============================================================
Chunk 2  |  distance=0.7228  |  cosine_sim=0.7388  |  Good/moderate match  |  page=2
============================================================
selves where we can make the greatest contri-
bution. And we will have to stay mentally alert
and engaged during a 50-year working life,
which means knowing how and when to
change the work we do.
 
What Are My Strengths?
 
Most people think they know what they are
good at. They are usually wrong. More often,
people know what they are not good at—and
even then more people are wrong than right.
And yet, a person can perform only from
strength. One cannot build performance on
weaknesses, let alone on something one can-
not do at all.
Throughout history, people had little
need to know their strengths. A person was
This document is authorized for use only by Sharon Brooks (SHARON@PRICE-ASSOCIATES.COM). Copying or posting is an infringement of copyright. Please contact 
customerservice@harvardbusiness.org or 800-988-0886 for additional copies.

============================================================
Chunk 3  |  distance=0.7771  |  cosine_sim=0.6980  |  Good/moderate match  |  page=1
============================================================
What are your most valuable strengths and 
most dangerous weaknesses? Equally im-
portant, how do you learn and work with 
others? What are your most deeply held val-
ues? And in what type of work environment 
can you make the greatest contribution?
The implication is clear: Only when you op-
erate from a combination of your strengths 
and self-knowledge can you achieve true—
and lasting—excellence.
To build a life of excellence, begin by asking yourself these questions:
 
“What are my strengths?”
 
To accurately identify your strengths, use 
 
feedback analysis
 
. Every time you make a key 
decision, write down the outcome you ex-
pect. Several months later, compare the actual 
results with your expected results. Look for 
patterns in what you’re seeing: What results 
are you skilled at generating? What abilities do 
you need to enhance in order to get the re-
sults you want? What unproductive habits are 
preventing you from creating the outcomes
============================================================

## All 3 chunks are a good match for the query and OpenAI ##
## incorporates these chunks in the final answer. ##
---

### Service 3 — Smart Converter

Convert between common units using plain English, powered by a **LangGraph ReAct agent** (`smart_converter_service_agent.py`).

**Example:** *"70 kg to pounds"*, *"100 Fahrenheit to Celsius"*, *"5 miles to kilometers"*

**Supported units:**
- Temperature: Celsius, Fahrenheit, Kelvin
- Length: kilometers, miles, meters, feet
- Weight: kilograms, pounds

**How it works — LangGraph agent loop:**

The agent is a compiled `StateGraph` with two nodes (`agent` and `tools`) and a conditional edge that loops until the LLM stops calling tools.

```
START
  │
  ▼
[agent]  ← LLM receives SystemMessage + HumanMessage
  │         decides to call convert_units(value, from_unit, to_unit)
  ▼
[tools]  ← ToolNode executes convert_units locally
  │         result appended to messages as a ToolMessage
  ▼
[agent]  ← LLM sees all messages, writes the natural-language answer
  │         no further tool calls → exits loop
  ▼
 END
```

For a general question the LLM can answer without conversion (no tool call needed):

```
START → [agent] → no tool_calls → END
```

The conditional edge `_should_use_tools` inspects the last message: if `tool_calls` is non-empty it routes to `[tools]`; otherwise it goes to `END`. This means the loop runs as many times as needed — for a chained conversion the agent can call `convert_units` more than once before producing its final answer.

The model writes the final response; no string templates are used.

---

## Guardrails

Every message is screened before reaching the router or any service.

### Prompt Injection Protection

The following are blocked:
- Requests to reveal, read, or repeat the system prompt or instructions
- Attempts to override, ignore, or jailbreak the system prompt (e.g. *"ignore previous instructions"*, *"act as DAN"*)

### Restricted Topics

Peter will not engage with questions about:
- Cats or dogs
- Horoscopes or Zodiac signs
- Taylor Swift
- Religion
- Politics
- Gender
- Social problems

**Implementation:** `guardrails.py` sends each message to `gpt-4o-mini` with a classifier prompt that returns `{"safe": true}` or `{"safe": false}`. Blocked messages receive a polite, in-character refusal from Peter. The check runs before any routing or service call.

---

## Memory

Conversation memory is maintained throughout the session via Gradio's built-in `history` list, which is passed to every service on each turn. There is no truncation — the full conversation is included in every call.

---

## Summary — Natural Language Query Approaches

| Service | How natural language is used |
|---|---|
| **Ask Drucker** | Semantic vector search — the query is embedded into a vector and matched against pre-embedded chunks of the Drucker PDF stored in ChromaDB. The LLM produces an answer from the 3 top-matching passages. |
| **Weather** | API lookup — the router extracts the city name from the query; the actual data comes from the Open-Meteo REST API. The LLM's only role is to rewrite the raw API response as a friendly summary. |
| **Smart Converter** | LangGraph ReAct agent — a `StateGraph` with `agent` and `tools` nodes loops until the LLM stops issuing tool calls. The LLM parses the intent and calls `convert_units(value, from_unit, to_unit)`; the `ToolNode` executes the conversion locally; the LLM writes the final plain-English result. |
