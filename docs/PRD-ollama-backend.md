# PRD: Ollama Backend for QMD

## Overview

Add Ollama as an alternative LLM backend to QMD, enabling GPU-accelerated inference on a remote host while running QMD itself on a CPU-only machine (e.g., VM, container, server).

## Problem Statement

QMD currently uses `node-llama-cpp` with local GGUF models. This works well when QMD runs on a machine with GPU access, but fails in common deployment scenarios:

- **VM/Container**: QMD runs in OrbStack/Docker without GPU passthrough
- **Server**: Headless Linux box without GPU
- **Separation of concerns**: Keep inference on a dedicated GPU machine

Ollama provides an HTTP API for LLM inference and runs efficiently on Apple Silicon (Metal) and NVIDIA GPUs. By supporting Ollama as a backend, QMD can offload inference to a GPU-equipped host.

## Goals

1. **Implement `OllamaLLM`** class that satisfies the existing `LLM` interface
2. **Zero changes to search/index logic** — backend is swapped at the LLM layer only
3. **Configuration-driven backend selection** — env vars or config file
4. **Maintain upstream compatibility** — no breaking changes to existing `LlamaCpp` backend

## Non-Goals

- Ollama server management (user installs/runs Ollama separately)
- Model management (user pulls models via `ollama pull`)
- Streaming responses (batch mode is sufficient for QMD's use cases)

---

## Technical Design

### 1. LLM Interface (existing)

```typescript
// src/llm.ts - already exists
export interface LLM {
  embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null>;
  generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null>;
  modelExists(model: string): Promise<ModelInfo>;
  expandQuery(query: string, options?: { context?: string, includeLexical?: boolean }): Promise<Queryable[]>;
  rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult>;
  dispose(): Promise<void>;
}
```

### 2. New File: `src/ollama.ts`

Implement `OllamaLLM` class:

```typescript
export type OllamaConfig = {
  baseUrl?: string;           // Default: "http://localhost:11434"
  embedModel?: string;        // Default: "nomic-embed-text"
  generateModel?: string;     // Default: "qwen2.5:1.5b" (small, fast)
  rerankModel?: string;       // Default: same as generateModel (see Reranking section)
  timeoutMs?: number;         // Default: 30000
};

export class OllamaLLM implements LLM {
  // Implementation details below
}
```

### 3. Ollama API Mapping

| QMD Operation | Ollama Endpoint | Notes |
|---------------|-----------------|-------|
| `embed()` | `POST /api/embeddings` | Direct mapping |
| `embedBatch()` | `POST /api/embeddings` (loop) | Ollama doesn't batch; loop with concurrency limit |
| `generate()` | `POST /api/generate` | Use `stream: false` |
| `expandQuery()` | Uses `generate()` internally | No change needed |
| `rerank()` | `POST /api/generate` (per-doc) | See Reranking section |
| `modelExists()` | `GET /api/tags` | Check if model is pulled |

### 4. Reranking Strategy

Ollama doesn't have a native reranking API. Two options:

**Option A: Prompt-based reranking (recommended)**
- For each document, generate a yes/no relevance judgment
- Parse logprobs if available, or use binary scoring
- Simpler, works with any model

**Option B: Cross-encoder style**
- Concatenate query + document and generate relevance score
- More accurate but requires specific model support

Recommend **Option A** for initial implementation. The existing `LlamaCpp.rerank()` uses `qwen3-reranker` which is a specialized model; for Ollama we'll use prompt-based scoring with the generate model.

### 5. Configuration & Backend Selection

#### Environment Variables

```bash
# Backend selection
QMD_LLM_BACKEND=ollama          # "llamacpp" (default) or "ollama"

# Ollama-specific
QMD_OLLAMA_URL=http://host.internal:11434
QMD_OLLAMA_EMBED_MODEL=nomic-embed-text
QMD_OLLAMA_GENERATE_MODEL=qwen2.5:1.5b
QMD_OLLAMA_RERANK_MODEL=qwen2.5:1.5b
```

#### Factory Function

```typescript
// src/llm.ts - add to existing file
export function createLLM(backend?: string): LLM {
  const b = backend || process.env.QMD_LLM_BACKEND || "llamacpp";
  
  if (b === "ollama") {
    return new OllamaLLM({
      baseUrl: process.env.QMD_OLLAMA_URL,
      embedModel: process.env.QMD_OLLAMA_EMBED_MODEL,
      generateModel: process.env.QMD_OLLAMA_GENERATE_MODEL,
      rerankModel: process.env.QMD_OLLAMA_RERANK_MODEL,
    });
  }
  
  return new LlamaCpp();
}
```

#### Update Singleton

```typescript
// Replace getDefaultLlamaCpp() usage with getDefaultLLM()
let defaultLLM: LLM | null = null;

export function getDefaultLLM(): LLM {
  if (!defaultLLM) {
    defaultLLM = createLLM();
  }
  return defaultLLM;
}
```

---

## Task List

### Phase 1: Core Implementation

- [ ] **T1.1** Create `src/ollama.ts` with `OllamaLLM` class skeleton
- [ ] **T1.2** Implement `embed()` using `/api/embeddings`
- [ ] **T1.3** Implement `embedBatch()` with concurrency limiting
- [ ] **T1.4** Implement `generate()` using `/api/generate`
- [ ] **T1.5** Implement `modelExists()` using `/api/tags`
- [ ] **T1.6** Implement `rerank()` using prompt-based scoring
- [ ] **T1.7** Implement `expandQuery()` (delegates to `generate()`, may need grammar handling)
- [ ] **T1.8** Implement `dispose()` (no-op for HTTP client)

### Phase 2: Integration

- [ ] **T2.1** Add `createLLM()` factory function to `src/llm.ts`
- [ ] **T2.2** Add `getDefaultLLM()` to replace `getDefaultLlamaCpp()`
- [ ] **T2.3** Update all call sites to use `getDefaultLLM()` instead of `getDefaultLlamaCpp()`
- [ ] **T2.4** Add env var parsing for Ollama configuration

### Phase 3: Testing

- [ ] **T3.1** Create `src/ollama.test.ts` with unit tests (mocked HTTP)
- [ ] **T3.2** Add integration test that hits real Ollama (skipped if unavailable)
- [ ] **T3.3** Verify existing `LlamaCpp` tests still pass
- [ ] **T3.4** End-to-end test: index → embed → search with Ollama backend

### Phase 4: Documentation

- [ ] **T4.1** Update README with Ollama backend instructions
- [ ] **T4.2** Update CLAUDE.md with Ollama configuration notes
- [ ] **T4.3** Add example `.env` file

---

## Acceptance Criteria

### Functional

1. **Embedding works**: `qmd embed` generates vectors using Ollama when `QMD_LLM_BACKEND=ollama`
2. **Search works**: `qmd vsearch` and `qmd query` return relevant results
3. **Reranking works**: `qmd query` reranks results (quality may differ from LlamaCpp)
4. **Model check works**: `qmd status` shows Ollama model availability
5. **Graceful fallback**: Clear error message if Ollama unreachable

### Non-Functional

1. **No regression**: All existing tests pass with default (LlamaCpp) backend
2. **Performance**: Embedding throughput within 2x of direct Ollama CLI
3. **Timeout handling**: Requests timeout gracefully (default 30s)

### Test Commands

```bash
# Run with Ollama backend
export QMD_LLM_BACKEND=ollama
export QMD_OLLAMA_URL=http://host.internal:11434

# Test embedding
qmd collection add ./test-docs --name test
qmd embed

# Test search
qmd search "test query"
qmd vsearch "semantic search test"
qmd query "hybrid search test"

# Verify status
qmd status
```

---

## Open Questions

1. **Grammar support**: `expandQuery()` uses GBNF grammar for structured output. Ollama supports grammars via the `format` parameter but syntax may differ. Need to test compatibility or implement fallback.

2. **Rerank quality**: Prompt-based reranking may be lower quality than dedicated reranker models. Consider making reranking optional or supporting external reranker services.

3. **Batch embedding**: Ollama doesn't support batch embeddings natively. Should we use concurrent requests (with limit) or sequential? Recommend concurrent with limit of 4.

4. **Model defaults**: What Ollama models should be defaults?
   - Embed: `nomic-embed-text` (768 dims, same as embeddinggemma)
   - Generate: `qwen2.5:1.5b` (fast, good for query expansion)
   - Or match user's pulled models?

---

## References

- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [QMD LLM Interface](../src/llm.ts)
- [node-llama-cpp docs](https://node-llama-cpp.withcat.ai/)
