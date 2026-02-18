# Plan: QMD with External Embedding Inference

Get QMD running with external API providers for embeddings, reranking, and query
expansion, then validate end-to-end against the ~/vault collection. Target:
sub-2-second search latency.

## Current State

The API backend (`src/api.ts`) is implemented and tested:
- 12 contract tests pass (mocked fetch)
- Live provider tests exist for OpenAI, Cohere, Voyage, OpenRouter
- `QMD_LLM_BACKEND=api` routes through `ApiLLM` via the factory in `src/llm.ts`
- `PassthroughLLMSession` wraps API calls with abort/timeout semantics
- Char-based chunking fallback works when tokenizer is unavailable

The vault collection is indexed (429 files) but embeddings are from the local
embeddinggemma model (384 dimensions). Switching to API embeddings (e.g.
text-embedding-3-small, 1536 dims) will auto-drop and recreate the vector table.

No GPU on this machine - local backend falls back to CPU (slow).
No QMD MCP server currently running. No cron/systemd jobs.

## Steps

### [*] 1. Validate contract tests pass

- [*] Contract tests: 12/12 pass

### [*] 2. Verify API keys are available

- [*] OPENAI_API_KEY (164 chars) and COHERE_API_KEY (40 chars) set in ~/.bashrc

### [*] 3. Run live provider tests

- [*] 5/5 pass: OpenAI embed+chat, Cohere embed+chat+rerank
- [*] 4 skipped (no OpenRouter/Voyage keys)

### [*] 4. Re-embed vault with API backend

- [*] 3609 chunks from 509 documents in 1m 22s (101.6 KB/s)
- [*] OpenAI text-embedding-3-small (1536 dims)

### [*] 5. Test query performance against vault

- [*] Cached query: pipeline 1040ms (wall 2.8s incl tsx startup)
- [*] Uncached query: pipeline 2699ms (expansion 1754ms dominates)
- [*] MCP server (no startup cost): 959ms for deep_search

Pipeline breakdown (uncached):
| Phase | Delta |
|-------|-------|
| bm25-probe | 78ms |
| expand | 1754ms |
| embed | 354ms |
| vsearch | 214ms |
| rerank | 251ms |
| blend | 48ms |

### [*] 6. Add query timing instrumentation

- [*] Added `QMD_DEBUG_TIMING=1` env var to `hybridQuery()` in `src/store.ts`
- [*] Instruments: bm25-probe, expand, embed, vsearch, rerank, blend

### [*] 7. Test MCP server with API backend

- [*] MCP HTTP server starts, initializes, and serves deep_search
- [*] deep_search returns structured results with docids, scores, snippets
- [*] 959ms response time via MCP HTTP (sub-2-second target met)

### [*] 8. Update CHANGELOG.md

- [*] Added [Unreleased] entries for API backend feature

## Out of Scope

- Nightly vault index updating (cron/systemd setup) - do after validation
- Integration with other local services - do after core QMD works
- Building/compiling (per CLAUDE.md: never run `bun build --compile`)
