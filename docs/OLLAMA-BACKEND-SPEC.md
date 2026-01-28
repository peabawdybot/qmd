# Ollama Backend for QMD — Implementation Spec

**Branch:** `feature/ollama-backend`
**Repo:** `git@github.com:peabawdybot/qmd.git`

---

## Overview

Add Ollama as an alternative LLM backend to QMD, enabling GPU-accelerated inference on a remote host while QMD runs on a CPU-only machine (VM, container, server).

**Problem:** QMD uses `node-llama-cpp` with local GGUF models. No GPU passthrough in VM = slow CPU inference.

**Solution:** Implement `OllamaLLM` class satisfying the existing `LLM` interface. Backend selected via env var.

---

## Architecture

### Existing LLM Interface (`src/llm.ts`)

```typescript
export interface LLM {
  embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null>;
  generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null>;
  modelExists(model: string): Promise<ModelInfo>;
  expandQuery(query: string, options?: { context?: string, includeLexical?: boolean }): Promise<Queryable[]>;
  rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult>;
  dispose(): Promise<void>;
}
```

### Ollama API Mapping

| QMD Method | Ollama Endpoint | Notes |
|------------|-----------------|-------|
| `embed()` | `POST /api/embeddings` | Direct mapping |
| `embedBatch()` | Loop over `embed()` | Ollama doesn't batch natively |
| `generate()` | `POST /api/generate` | Use `stream: false` |
| `expandQuery()` | Uses `generate()` | JSON format instead of GBNF grammar |
| `rerank()` | `POST /api/generate` per doc | Prompt-based yes/no scoring |
| `modelExists()` | `GET /api/tags` | Check if model is pulled |

### Configuration

```bash
QMD_LLM_BACKEND=ollama              # "llamacpp" (default) or "ollama"
QMD_OLLAMA_URL=http://host:11434    # Ollama server URL
QMD_OLLAMA_EMBED_MODEL=nomic-embed-text
QMD_OLLAMA_GENERATE_MODEL=qwen2.5:1.5b
QMD_OLLAMA_RERANK_MODEL=qwen2.5:1.5b
```

---

## Implementation

### File 1: `src/ollama.ts` (NEW)

```typescript
import type {
  LLM,
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  ModelInfo,
  Queryable,
  QueryType,
  RerankDocument,
  RerankResult,
  RerankOptions,
} from "./llm";

export type OllamaConfig = {
  baseUrl?: string;
  embedModel?: string;
  generateModel?: string;
  rerankModel?: string;
  timeoutMs?: number;
};

export class OllamaLLM implements LLM {
  private baseUrl: string;
  private embedModel: string;
  private generateModel: string;
  private rerankModel: string;
  private timeoutMs: number;

  constructor(config: OllamaConfig = {}) {
    this.baseUrl = config.baseUrl || process.env.QMD_OLLAMA_URL || "http://localhost:11434";
    this.embedModel = config.embedModel || process.env.QMD_OLLAMA_EMBED_MODEL || "nomic-embed-text";
    this.generateModel = config.generateModel || process.env.QMD_OLLAMA_GENERATE_MODEL || "qwen2.5:1.5b";
    this.rerankModel = config.rerankModel || process.env.QMD_OLLAMA_RERANK_MODEL || this.generateModel;
    this.timeoutMs = config.timeoutMs || 30000;
  }

  // ========== embed ==========
  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    try {
      const response = await fetch(`${this.baseUrl}/api/embeddings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: options?.model || this.embedModel,
          prompt: text,
        }),
        signal: AbortSignal.timeout(this.timeoutMs),
      });

      if (!response.ok) {
        console.error(`Ollama embed error: ${response.status}`);
        return null;
      }

      const data = await response.json();
      return {
        embedding: data.embedding,
        model: options?.model || this.embedModel,
      };
    } catch (error) {
      console.error("Ollama embed error:", error);
      return null;
    }
  }

  // ========== embedBatch ==========
  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    // Ollama doesn't support batch embeddings; run concurrently with limit
    const results: (EmbeddingResult | null)[] = [];
    const concurrencyLimit = 4;
    
    for (let i = 0; i < texts.length; i += concurrencyLimit) {
      const batch = texts.slice(i, i + concurrencyLimit);
      const batchResults = await Promise.all(batch.map(text => this.embed(text)));
      results.push(...batchResults);
    }
    
    return results;
  }

  // ========== generate ==========
  async generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null> {
    try {
      const response = await fetch(`${this.baseUrl}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: options?.model || this.generateModel,
          prompt,
          stream: false,
          options: {
            num_predict: options?.maxTokens ?? 150,
            temperature: options?.temperature ?? 0,
          },
        }),
        signal: AbortSignal.timeout(this.timeoutMs),
      });

      if (!response.ok) {
        console.error(`Ollama generate error: ${response.status}`);
        return null;
      }

      const data = await response.json();
      return {
        text: data.response,
        model: options?.model || this.generateModel,
        done: data.done,
      };
    } catch (error) {
      console.error("Ollama generate error:", error);
      return null;
    }
  }

  // ========== modelExists ==========
  async modelExists(model: string): Promise<ModelInfo> {
    try {
      const response = await fetch(`${this.baseUrl}/api/tags`, {
        signal: AbortSignal.timeout(this.timeoutMs),
      });

      if (!response.ok) {
        return { name: model, exists: false };
      }

      const data = await response.json();
      const models: { name: string }[] = data.models || [];
      
      // Check if model exists (handle :latest suffix)
      const exists = models.some(m => 
        m.name === model || 
        m.name === `${model}:latest` ||
        m.name.startsWith(`${model}:`)
      );

      return { name: model, exists };
    } catch (error) {
      console.error("Ollama modelExists error:", error);
      return { name: model, exists: false };
    }
  }

  // ========== expandQuery ==========
  async expandQuery(
    query: string,
    options: { context?: string; includeLexical?: boolean } = {}
  ): Promise<Queryable[]> {
    const includeLexical = options.includeLexical ?? true;
    const context = options.context;

    const prompt = `You are a search query optimization expert. Rewrite the query for better retrieval.

Original Query: ${query}
${context ? `\nContext: ${context}` : ""}

Return a JSON object with these arrays:
- "lex": 1-3 keyword variations (single terms/phrases for BM25)
- "vec": 1-3 semantic query variations (natural language for vector search)  
- "hyde": 0-1 hypothetical document passages that would answer the query

Example response format:
{"lex":["keyword1","keyword2"],"vec":["semantic query 1","semantic query 2"],"hyde":["A hypothetical passage that answers the query..."]}

Respond with only valid JSON, no explanation.`;

    try {
      const result = await this.generate(prompt, { 
        maxTokens: 500, 
        temperature: 0.7,
        model: this.generateModel,
      });

      if (!result?.text) {
        return this.fallbackQueryables(query, includeLexical);
      }

      // Parse JSON response
      const jsonMatch = result.text.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        return this.fallbackQueryables(query, includeLexical);
      }

      const parsed = JSON.parse(jsonMatch[0]);
      const queryables: Queryable[] = [];

      if (includeLexical && Array.isArray(parsed.lex)) {
        for (const text of parsed.lex) {
          queryables.push({ type: "lex" as QueryType, text });
        }
      }

      if (Array.isArray(parsed.vec)) {
        for (const text of parsed.vec) {
          queryables.push({ type: "vec" as QueryType, text });
        }
      }

      if (Array.isArray(parsed.hyde)) {
        for (const text of parsed.hyde) {
          queryables.push({ type: "hyde" as QueryType, text });
        }
      }

      return queryables.length > 0 ? queryables : this.fallbackQueryables(query, includeLexical);
    } catch (error) {
      console.error("Ollama expandQuery error:", error);
      return this.fallbackQueryables(query, includeLexical);
    }
  }

  private fallbackQueryables(query: string, includeLexical: boolean): Queryable[] {
    const result: Queryable[] = [{ type: "vec" as QueryType, text: query }];
    if (includeLexical) {
      result.unshift({ type: "lex" as QueryType, text: query });
    }
    return result;
  }

  // ========== rerank ==========
  async rerank(
    query: string,
    documents: RerankDocument[],
    options?: RerankOptions
  ): Promise<RerankResult> {
    const results: { file: string; score: number; index: number }[] = [];

    // Score each document with prompt-based relevance check
    await Promise.all(
      documents.map(async (doc, index) => {
        const prompt = `Is this document relevant to the query? Answer only "yes" or "no".

Query: ${query}

Document: ${doc.text.slice(0, 500)}

Answer:`;

        try {
          const result = await this.generate(prompt, {
            maxTokens: 10,
            temperature: 0,
            model: options?.model || this.rerankModel,
          });

          const answer = result?.text?.toLowerCase().trim() || "";
          const score = answer.startsWith("yes") ? 1.0 : 0.0;

          results.push({ file: doc.file, score, index });
        } catch {
          results.push({ file: doc.file, score: 0, index });
        }
      })
    );

    // Sort by score descending
    results.sort((a, b) => b.score - a.score);

    return {
      results,
      model: options?.model || this.rerankModel,
    };
  }

  // ========== dispose ==========
  async dispose(): Promise<void> {
    // No resources to clean up for stateless HTTP client
  }
}
```

---

### File 2: Modify `src/llm.ts`

Add at the **top** (with other imports):

```typescript
import { OllamaLLM } from "./ollama";
```

Add at the **bottom** (after existing singleton code):

```typescript
// =============================================================================
// Backend Factory
// =============================================================================

/**
 * Create an LLM instance based on backend configuration.
 * Uses QMD_LLM_BACKEND env var: "llamacpp" (default) or "ollama"
 */
export function createLLM(backend?: string): LLM {
  const b = backend || process.env.QMD_LLM_BACKEND || "llamacpp";
  
  if (b === "ollama") {
    return new OllamaLLM();
  }
  
  return new LlamaCpp();
}

// =============================================================================
// Updated Singleton (backend-aware)
// =============================================================================

let defaultLLM: LLM | null = null;

/**
 * Get the default LLM instance (creates one if needed).
 * Respects QMD_LLM_BACKEND env var for backend selection.
 */
export function getDefaultLLM(): LLM {
  if (!defaultLLM) {
    defaultLLM = createLLM();
  }
  return defaultLLM;
}

/**
 * Set a custom default LLM instance (useful for testing)
 */
export function setDefaultLLM(llm: LLM | null): void {
  defaultLLM = llm;
}

/**
 * Dispose the default LLM instance if it exists.
 */
export async function disposeDefaultLLM(): Promise<void> {
  if (defaultLLM) {
    await defaultLLM.dispose();
    defaultLLM = null;
  }
}
```

---

### File 3: Update Call Sites

Search and replace throughout codebase:

| Find | Replace |
|------|---------|
| `getDefaultLlamaCpp()` | `getDefaultLLM()` |
| `disposeDefaultLlamaCpp()` | `disposeDefaultLLM()` |

Files likely affected:
- `src/qmd.ts`
- `src/mcp.ts`
- `src/search.ts` (if exists)
- Any test files

Also update imports:
```typescript
// Old
import { getDefaultLlamaCpp, disposeDefaultLlamaCpp } from "./llm";

// New
import { getDefaultLLM, disposeDefaultLLM } from "./llm";
```

---

### File 4: `src/ollama.test.ts` (NEW)

```typescript
import { describe, test, expect, mock, beforeEach, afterEach } from "bun:test";
import { OllamaLLM } from "./ollama";

// Save original fetch
const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe("OllamaLLM.embed", () => {
  test("returns embedding vector from Ollama API", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ embedding: [0.1, 0.2, 0.3] }),
      })
    ) as any;

    const ollama = new OllamaLLM({ baseUrl: "http://test:11434" });
    const result = await ollama.embed("test text");

    expect(result).not.toBeNull();
    expect(result!.embedding).toEqual([0.1, 0.2, 0.3]);
    expect(result!.model).toBe("nomic-embed-text");
  });

  test("returns null on API error", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve({ ok: false, status: 500 })
    ) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.embed("test");
    expect(result).toBeNull();
  });

  test("returns null on network error", async () => {
    globalThis.fetch = mock(() =>
      Promise.reject(new Error("ECONNREFUSED"))
    ) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.embed("test");
    expect(result).toBeNull();
  });
});

describe("OllamaLLM.generate", () => {
  test("generates text completion", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ response: "Hello world", done: true }),
      })
    ) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.generate("Say hello");

    expect(result).not.toBeNull();
    expect(result!.text).toBe("Hello world");
    expect(result!.done).toBe(true);
  });
});

describe("OllamaLLM.modelExists", () => {
  test("returns exists:true when model found", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve({
        ok: true,
        json: () =>
          Promise.resolve({
            models: [{ name: "nomic-embed-text:latest" }],
          }),
      })
    ) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.modelExists("nomic-embed-text");

    expect(result.exists).toBe(true);
  });

  test("returns exists:false when model not found", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ models: [] }),
      })
    ) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.modelExists("nonexistent");

    expect(result.exists).toBe(false);
  });
});

describe("OllamaLLM.rerank", () => {
  test("scores documents by relevance", async () => {
    let callCount = 0;
    globalThis.fetch = mock(() => {
      callCount++;
      // First doc relevant, second not
      const response = callCount === 1 ? "yes" : "no";
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ response, done: true }),
      });
    }) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.rerank("test query", [
      { file: "a.md", text: "relevant content" },
      { file: "b.md", text: "irrelevant content" },
    ]);

    expect(result.results.length).toBe(2);
    const scoreA = result.results.find((r) => r.file === "a.md")!.score;
    const scoreB = result.results.find((r) => r.file === "b.md")!.score;
    expect(scoreA).toBeGreaterThan(scoreB);
  });
});
```

---

## Verification Commands

```bash
# Run tests
bun test src/ollama.test.ts

# Run all tests (ensure no regression)
bun test

# Type check
tsc --noEmit

# Manual E2E test (after setup)
export QMD_LLM_BACKEND=ollama
export QMD_OLLAMA_URL=http://host.internal:11434

qmd status
qmd collection add /tmp/test-docs --name test
qmd embed
qmd vsearch "test query"
qmd query "test query"
```

---

## Acceptance Criteria

- [ ] `bun test src/ollama.test.ts` — all pass
- [ ] `bun test` — no regressions
- [ ] `tsc --noEmit` — no type errors
- [ ] `QMD_LLM_BACKEND=ollama qmd embed` — works
- [ ] `QMD_LLM_BACKEND=ollama qmd query "test"` — returns results
- [ ] Default behavior unchanged (llamacpp backend when env not set)

---

## Notes

1. **No GBNF grammar in Ollama** — `expandQuery` uses JSON format instead
2. **Prompt-based reranking** — Less accurate than dedicated reranker but works with any model
3. **Concurrency limit** — `embedBatch` uses limit of 4 to avoid overwhelming Ollama
4. **Timeout** — 30s default, configurable via `timeoutMs`
