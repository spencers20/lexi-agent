# LexiAgent — Multi-Agent Document & Data Intelligence Backend

LexiAgent is the backend microservice powering **Lexifile**, an AI-powered document analysis platform. It's built as a multi-agent system where incoming requests are routed to a specialized agent depending on what the user is trying to do — querying documents, analyzing a CSV, or generating a personalized health journey/habit plan. Each agent is built with **LangGraph** as a stateful graph, so a single "question" can trigger multiple internal reasoning steps before a final answer is returned.

---

## Architecture Overview

LexiAgent is served through a single **FastAPI** app (`app.py`) that exposes two endpoints and delegates work to one of several LangGraph-based agents depending on the request payload. The whole service is Dockerized for consistent deployment.

```
                     ┌───────────────────┐
                     │  FastAPI (app.py) │
                     └─────────┬─────────┘
              ┌────────────────┼─────────────────┐
              │                │                  │
      namespace given?   file_path given?   habit/journey/
              │                │           prescription fields?
              ▼                ▼                  ▼
     ┌────────────────┐ ┌───────────────┐ ┌────────────────────┐
     │  Doc/Q&A Agent │ │ Data Analysis │ │   Journal Agent    │
     │   (agent.py)   │ │    Agent      │ │   (journal.py)     │
     └────────────────┘ └───────────────┘ └────────────────────┘
```

Each agent is its own **LangGraph `StateGraph`**, with typed state (`TypedDict`) flowing through nodes that call an LLM, retrieve context, run generated code, or classify intent — before converging on a final answer.

---

## The Agents

### 1. Document Q&A Agent (`agent.py`)
Handles general conversational queries and question-answering over documents stored in Pinecone. This is the "normal conversation" / RAG agent behind Lexifile's chat interface.

**Flow:**
1. **Intent generation** — takes the user's raw response plus any prior follow-up prompt and distills it into a clear, standalone intent statement (so vague follow-ups like "yes, that one" still resolve to a real question).
2. **Retrieval** — embeds the intent (Cohere `embed-english-v3.0`) and performs a similarity search against a Pinecone index, scoped to a `namespace` (i.e. a specific uploaded document/collection).
3. **Answer generation** — passes the retrieved context and the intent to the LLM to produce a concise answer (max 3 sentences), grounded strictly in retrieved context.
4. **Follow-up generation** — generates a natural, contextual follow-up prompt/suggestion and appends it to the answer, and feeds it back into state so the next turn can use it for intent resolution.

Runs as a compiled LangGraph sequence (`intent_generator → retriever_generator → follow_up_prompt_generator`) with `InMemorySaver` checkpointing per `thread_id`, so conversations maintain turn-by-turn context.

### 2. Data Analysis Agent (`data_analysis_agent.py`)
Takes an uploaded CSV and answers natural-language questions about it — including generating charts.

**Flow:**
1. **Schema generation** — profiles the CSV (shape, columns, dtypes, missing values, unique counts, numeric/categorical/date columns, sample rows, describe() stats) into a structured schema dict.
2. **Intent classification** — an LLM call (via structured output/Pydantic) decides whether the question is answerable from the DataFrame schema (`related_to_schema`) or should be treated as a general knowledge question (`not_related_to_schema`), and routes accordingly.
3. **Code generation & safe execution** — if schema-related, the LLM generates executable pandas/matplotlib/seaborn code against the schema. The code is cleaned, then executed in a sandboxed `exec()` context with `df`, `plt`, and `sns` injected.
4. **Visualization capture** — `plt.savefig` is monkey-patched at runtime to intercept any chart the generated code produces, encode it to base64 in-memory (no disk writes), and return it directly in the API response instead of saving a file.
5. **Explanation pass** — the textual output and any generated chart images (as base64 vision input) are sent back to the LLM to produce a plain-English explanation of what the analysis shows — patterns, correlations, trends — without jargon.
6. **Fallback path** — if the question isn't answerable from the schema, it's routed to a general-purpose LLM response instead.

Stdout is captured via a redirected `StringIO` buffer during code execution so printed output (tables, summary stats) is returned to the caller alongside any visuals.

### 3. Journal Agent (`journal.py`)
A health-focused agent (originally built for AfyaSphere) responsible for generating personalized day-by-day health journeys, habit suggestions, and parsing medication prescriptions into structured data.

**Flow (routes on which fields are present in the request):**
- **Journey generation** — given a `journey_title`, `journey_description`, and `number_of_days`, retrieves guideline context from a Pinecone namespace (health journey design principles, example day structures) and generates a day-by-day schedule via structured LLM output (`Field`/`focus`/`description`/`activities` per day), scaling to however many days are requested.
- **Habit suggestion** — retrieves habit-formation guidelines and generates a structured list of recommended habits with short descriptions.
- **Prescription parsing** — parses free-text medication instructions (e.g. `"2 tablets twice a day"`, `"30ml every 8 hours"`, shorthand like `2*3`) into structured `dosage` / `frequency` / `no_of_days` fields, handling unit conversion (teaspoons → ml, hours → daily frequency) and written-number parsing.

All three paths return structured (Pydantic-validated) output rather than free text, since the frontend renders these as day cards, habit lists, and dosage schedules.

---

## API Surface

Two POST endpoints, both accepting a single flexible `QueryInput` payload — the fields present determine which agent handles the request.

### `POST /agent`
Routes to either the Document Q&A agent or the Data Analysis agent.

| Field | Type | Purpose |
|---|---|---|
| `question` | string | The user's question |
| `namespace` | string (optional) | Pinecone namespace → routes to Doc Q&A agent |
| `file_path` | string (optional) | Path to an uploaded CSV → routes to Data Analysis agent |
| `thread_id` | string (optional) | Conversation/session ID for memory continuity; auto-generated if omitted |

Returns the final answer, plus `text_response` and `visuals` (base64 chart images) when the data analysis path is used.

### `POST /habit_and_journeys`
Routes to the Journal agent based on which of `habit_query`, `journey_title`/`journey_description`/`number_of_days`, or `prescription` are populated.

---

## Tech Stack

- **Framework:** FastAPI, Uvicorn
- **Agent orchestration:** LangGraph (`StateGraph`, conditional edges, checkpointing via `MemorySaver` / `InMemorySaver`)
- **LLMs:** Azure OpenAI (via `AzureChatOpenAI`), with Hugging Face Endpoint / Groq integrations scaffolded for model flexibility
- **Vector store:** Pinecone, namespace-partitioned per document/context type
- **Embeddings:** Cohere (`embed-english-v3.0`)
- **Data analysis:** pandas, NumPy, Matplotlib, Seaborn (sandboxed execution with in-memory chart capture)
- **Validation:** Pydantic (structured LLM outputs across all three agents)
- **Deployment:** Docker

---

## Notes on Design Decisions

- **Structured output over free text** wherever the frontend needs to render something specific (day-by-day cards, dosage fields, habit lists) — reduces parsing fragility versus asking the LLM to return formatted text.
- **In-memory chart capture** (monkey-patched `savefig`) avoids writing temp files to disk and keeps the data analysis agent stateless between requests.
- **Per-agent state graphs** rather than one monolithic graph — keeps each agent's flow independently testable and lets the FastAPI layer stay a thin router.
- **Thread-based checkpointing** on the Q&A agent specifically, since multi-turn document conversations need memory of prior intent/follow-ups; the data analysis and journal agents are largely single-turn by design.
