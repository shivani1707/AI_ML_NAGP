| Resource | URL |
|---|---|
| **GitHub Repository** | https://github.com/shivani1707/AI_ML_NAGP|
| **Demo Video** | https://nagarro-my.sharepoint.com/:v:/p/shivani_verma/IQCM6Xs0TrdoTZCl0wGZuCRGAWVRoPz5NcIUnCJjDGO1ON0?e=xEiHya&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D


# Singapore Travel Planning Assistant

A context-aware AI travel assistant for Singapore that combines a document-based
knowledge base (Retrieval-Augmented Generation) with live data from MCP tools
(weather forecasts and currency conversion).

## Architecture

```
User question
     |
     v
[LangGraph ReAct Agent + Gemini 3.6 Flash]
     |
     |--- decides which tool(s) to call, based on the question ---
     |
     +--> search_travel_knowledge_base (RAG tool)
     |        -> FAISS vector search over embedded destination-knowledge chunks
     |
     +--> get_weather_forecast (MCP tool, Open-Meteo API)
     |
     +--> convert_currency (MCP tool, Frankfurter API)
     |
     v
Final answer, labeled by source (knowledge base / live tool / recommendation)
```

The agent is a single LangGraph `create_react_agent`, given all three tools (one RAG
tool, two MCP tools) plus a system prompt that governs when to use each and how to
label the response. Conversation memory is handled by a LangGraph checkpointer keyed
by a `thread_id`, so multi-turn context is retained automatically.

## Knowledge Base

Sources (destination knowledge for RAG):

| Source | URL | License |
|---|---|---|
| Wikivoyage: Singapore Travel Guide | https://en.wikivoyage.org/wiki/Singapore | CC BY-SA |
| Visit Singapore: Essential Travel Information | https://www.visitsingapore.com/travel-tips/essential-travel-information/ | All rights reserved (used per site's stated reuse terms) |
| Visit Singapore: Sample Itineraries | https://www.visitsingapore.com/travel-tips/travelling-to-singapore/itineraries/ | All rights reserved |
| Visit Singapore: Things to Do | https://www.visitsingapore.com/things-to-do | All rights reserved |

The Wikivoyage source was fetched automatically (`load_sources.py`, using
LangChain's `WebBaseLoader`); the VisitSingapore.com pages render their content via
JavaScript, so they were captured manually (page text copied into the corresponding
markdown file) and are not re-fetched programmatically.

Each source file is stored as Markdown in `knowledge_base/`, with a small metadata
header:

```markdown
---
title: "Source Title Here"
source_url: https://example.com/page
---

[page content]
```

## RAG Workflow

1. **Load** - `load_sources.py` fetches/stores each source as a local Markdown file
   with title + URL metadata.
2. **Chunk** - `build_vector_store.py` splits each document into ~1000-character
   chunks (150-character overlap) using `RecursiveCharacterTextSplitter`.
3. **Embed** - each chunk is embedded using Google's `gemini-embedding-001` model
   (batched, with retry/backoff for free-tier rate limits).
4. **Store** - embeddings are stored in a local FAISS index (`faiss_index/`).
5. **Retrieve** - at query time, the `search_travel_knowledge_base` tool performs a
   similarity search (`k=4`) and returns the matching chunks, each tagged with its
   source title.
6. **Generate** - the LLM is instructed (via system prompt) to answer only from
   retrieved content, and to state clearly when the knowledge base doesn't have
   enough information rather than inventing facts.
7. **Cite** - the LLM is instructed to list the distinct source titles used at the
   end of any answer drawing on the knowledge base.

## MCP Tools

Two custom MCP servers, built with the official `mcp` Python SDK (`FastMCP`), run
as local subprocesses over stdio and are connected via
`langchain-mcp-adapters` (`MultiServerMCPClient`).

- **`weather_server.py`** - `get_weather_forecast(days)`: current conditions and a
  configurable-length forecast for Singapore, from the free Open-Meteo API (no key
  required).
- **`currency_server.py`** - `convert_currency(amount, from_currency, to_currency)`:
  live exchange-rate conversion via the free Frankfurter API (no key required).

Both tools return a clear error message (rather than raising an unhandled
exception) if the underlying API call fails, so the agent can report a failure
honestly instead of fabricating an answer.

## Prompt & Context Strategy

The system prompt instructs the model to:
- Use the knowledge-base tool for destination facts, and never invent them.
- Use MCP tools only for current/live data (weather, currency), never for
  destination facts already covered by the knowledge base.
- Clearly label each part of a combined answer as knowledge-base fact, live tool
  result, or LLM-generated recommendation.
- State plainly when there isn't enough information, rather than guessing.
- List the knowledge-base sources used, when applicable.

Conversation context is preserved using a LangGraph `InMemorySaver` checkpointer,
keyed by a per-session `thread_id`, so follow-up questions (e.g. "what if I only
have 2 days?") are answered with awareness of prior turns.

## Setup Instructions

```bash
# 1. Clone the repo and enter it
git clone <your-repo-url>
cd travel-assistant

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Gemini API key
echo GOOGLE_API_KEY=your_key_here > .env

# 5. Build the knowledge base (fetch sources, chunk, embed, store)
python load_sources.py
python build_vector_store.py

# 6. Run the app
streamlit run app.py
```

## Obtaining the Knowledge Base

The `knowledge_base/` folder in this repo contains the Wikivoyage source (openly
licensed). If you do not include the VisitSingapore.com sources due to their
copyright terms, recreate them by:
1. Visiting each URL listed in the Knowledge Base table above.
2. Copying the main article text (skip navigation/ads/footers).
3. Saving it as a `.md` file in `knowledge_base/` with the header format shown
   above.

# Sample Questions & Responses

Real outputs captured during development/testing.

---

## 1. Pure RAG — Destination Knowledge

**Q: How can a tourist travel around Singapore?**


According to the knowledge base:

Mass Rapid Transit (MRT) & Light Rail Transit (LRT):

The MRT and LRT form the core public transit network, covering most major attractions across Singapore.
Train stations are clean, equipped with public toilets, and feature platform safety doors. Cash is not accepted on the MRT/LRT.

Public Buses:

Public buses operate from around 5:30 AM to midnight and provide extensive island-wide coverage.
Cash is accepted on buses, but exact change must be provided as no change is given.
Passengers must tap their card/device both when boarding and when exiting the bus; otherwise, the maximum fare for the route will be charged.

Ways to Pay for Public Transit:

Contactless Bank Cards / Mobile Wallets: You can tap directly at MRT gantries and bus readers using international contactless credit/debit cards or mobile payment apps.
Singapore Tourist Pass (STP): Offers unlimited public transit rides for 1, 2, or 3 days (
17
,
17,24, or $29 SGD respectively).

Taxis & Private Hire Cars:

Taxis can be flagged on the street or picked up at designated taxi stands. Private hire cars can be booked via mobile apps.

*Grounded entirely in knowledge-base content; no MCP tool involved.*

---

## 2. Pure MCP — Weather

**Q: What's the weather like in Singapore right now?**

> Current temperature in Singapore: 31.9°C
>
> Forecast:
> - 2026-09-18: 25.5 to 33.0°C, 88% chance of rain
> - 2026-09-19: 25.0 to 30.7°C, 81% chance of rain
> - 2026-09-20: 25.0 to 30.4°C, 96% chance of rain

*Answered via the `get_weather_forecast` MCP tool; no
knowledge-base lookup triggered.*

---

## 3. Pure MCP — Currency Conversion

**Q: convert 1000 inr into singapore currency

> 1,000 INR = 13.33 SGD (at an exchange rate of 1 INR = 0.01333 SGD).

*Answered via the `convert_currency` MCP tool.*

---

## 4. Combined RAG + MCP (required scenario)

**Q: Create a three-day Singapore itinerary for next week and adjust it
according to the weather forecast.**

> **Live Weather Forecast** (according to the live weather forecast):
> - Day 1 (2026-09-18): 25.5°C to 33.0°C, 88% chance of rain
> - Day 2 (2026-09-19): 25.0°C to 30.7°C, 81% chance of rain
> - Day 3 (2026-09-20): 25.0°C to 30.4°C, 96% chance of rain
>
> Due to the very high likelihood of rain across all three days, the itinerary
> below emphasizes indoor, covered, and air-conditioned attractions.
>
> **Destination Facts** (according to the knowledge base): Civic District &
> Riverside (National Gallery Singapore, CHIJMES, Clarke Quay), Marina Bay
> (Marina Bay Sands, ArtScience Museum, Gardens by the Bay's covered
> conservatories), Orchard Road (air-conditioned malls), Kampong Gelam,
> Chinatown, Sentosa, Jewel Changi Airport.
>
> **My Recommendation** (weather-adjusted 3-day itinerary):
> - **Day 1:** National Gallery Singapore (morning) → ArtScience Museum /
>   Marina Bay Sands Shoppes (afternoon) → Gardens by the Bay conservatories
>   (evening, weather-sheltered).
> - **Day 2:** Chinatown & Kampong Gelam (morning, sheltered walkways) →
>   Orchard Road malls (afternoon, indoor) → dinner at CHIJMES (evening).
> - **Day 3:** Sentosa indoor attractions if raining, outdoor (Skyline Luge)
>   if clear (morning) → Museum of Ice Cream or Jewel Changi's indoor Rain
>   Vortex (afternoon) → dinner at Jewel Changi (evening).



---



