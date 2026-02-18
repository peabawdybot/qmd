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

### [ ] 1. Validate contract tests pass

- [ ] Run `npx vitest run --reporter=verbose test/api.contract.test.ts` to confirm the
  baseline is green before making changes.

### [ ] 2. Verify API keys are available

- [ ] Check which provider keys are set in the environment (`QMD_EMBED_API_KEY`,
  `QMD_CHAT_API_KEY`, `QMD_RERANK_API_KEY`, or provider-specific keys like
  `OPENAI_API_KEY`, `COHERE_API_KEY`). If none are set, ask user to provide them.

- [ ] USER wants to use the default LLM providers and models. Ask user for OPENAI and COHERE keys. Others, if needed. Explain providers, endpoints and model implications of the configured solution.

### [ ] 3. Run live provider tests

- [ ] With API keys available, run `npx vitest run test/api.live.test.ts` to confirm
  the external providers respond correctly.

### [ ] 4. Re-embed vault with API backend

- [ ] Print the command for the user to run manually (per CLAUDE.md: never run
  `qmd embed` automatically):

```sh
QMD_LLM_BACKEND=api \
QMD_EMBED_API_KEY=<key> \
npx tsx src/qmd.ts embed -c vault
```

This will:
- [ ] Detect dimension mismatch (384 -> 1536)
- [ ] Drop and recreate `vectors_vec` table
- [ ] Re-embed all 429 vault documents via the API

### [ ] 5. Test query performance against vault

- [ ] Run a timed search to verify sub-2-second target:

```sh
time QMD_LLM_BACKEND=api \
  QMD_EMBED_API_KEY=<key> \
  QMD_CHAT_API_KEY=<key> \
  QMD_RERANK_API_KEY=<key> \
  npx tsx src/qmd.ts query "meeting notes" -c vault
```

The pipeline is: expand(chat) + embed(queries) + BM25 + vsearch + rerank.
With API providers, each step is a single HTTP round-trip. BM25 is local/instant.

### [ ] 6. Add query timing instrumentation (if needed)

- [ ] If performance isn't meeting the target, add timing to the hybrid query pipeline
  in `src/store.ts` `hybridQuery()` to identify bottlenecks:
  - [ ] Query expansion latency
  - [ ] Embedding latency
  - [ ] Vector search latency
  - [ ] Reranking latency

### [ ] 7. Test MCP server with API backend

- [ ] Start the MCP server with API backend and verify deep_search works:

```sh
QMD_LLM_BACKEND=api \
  QMD_EMBED_API_KEY=<key> \
  QMD_CHAT_API_KEY=<key> \
  QMD_RERANK_API_KEY=<key> \
  npx tsx src/qmd.ts mcp --http --port 8181
```

### [ ] 8. Update CHANGELOG.md

- [ ] Add entries under `[Unreleased]` for the API backend feature:
  - [ ] feat: API-backed LLM providers for embeddings, query expansion, and reranking
  - [ ] Environment variables: QMD_LLM_BACKEND, QMD_EMBED_*, QMD_CHAT_*, QMD_RERANK_*

## Out of Scope

- Nightly vault index updating (cron/systemd setup) - do after validation
- Integration with other local services - do after core QMD works
- Building/compiling (per CLAUDE.md: never run `bun build --compile`)
