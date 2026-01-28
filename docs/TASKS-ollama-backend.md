# Implementation Tasks: Ollama Backend

Branch: `feature/ollama-backend`

Follow TDD: write test first, then implement to pass.

---

## T1: Core OllamaLLM Implementation

### T1.1 — Skeleton & Types

**File**: `src/ollama.ts`

Create the class skeleton with types:

```typescript
import type { LLM, EmbedOptions, EmbeddingResult, GenerateOptions, GenerateResult, ModelInfo, Queryable, RerankDocument, RerankResult, RerankOptions } from "./llm";

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

  // Stub all methods with throw new Error("Not implemented")
}
```

**Acceptance**: File compiles, class implements LLM interface.

---

### T1.2 — embed()

**Test first** (`src/ollama.test.ts`):

```typescript
import { describe, test, expect, mock, beforeEach } from "bun:test";
import { OllamaLLM } from "./ollama";

describe("OllamaLLM.embed", () => {
  test("returns embedding vector from Ollama API", async () => {
    const mockFetch = mock(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ embedding: [0.1, 0.2, 0.3] })
    }));
    globalThis.fetch = mockFetch as any;

    const ollama = new OllamaLLM({ baseUrl: "http://test:11434" });
    const result = await ollama.embed("test text");

    expect(result).not.toBeNull();
    expect(result!.embedding).toEqual([0.1, 0.2, 0.3]);
    expect(result!.model).toBe("nomic-embed-text");
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test:11434/api/embeddings",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ model: "nomic-embed-text", prompt: "test text" })
      })
    );
  });

  test("returns null on API error", async () => {
    globalThis.fetch = mock(() => Promise.resolve({ ok: false, status: 500 })) as any;
    const ollama = new OllamaLLM();
    const result = await ollama.embed("test");
    expect(result).toBeNull();
  });

  test("returns null on network error", async () => {
    globalThis.fetch = mock(() => Promise.reject(new Error("ECONNREFUSED"))) as any;
    const ollama = new OllamaLLM();
    const result = await ollama.embed("test");
    expect(result).toBeNull();
  });
});
```

**Implementation**:

```typescript
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
```

**Acceptance**: All embed tests pass.

---

### T1.3 — embedBatch()

**Test**:

```typescript
describe("OllamaLLM.embedBatch", () => {
  test("embeds multiple texts with concurrency limit", async () => {
    let callCount = 0;
    globalThis.fetch = mock(() => {
      callCount++;
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ embedding: [callCount * 0.1] })
      });
    }) as any;

    const ollama = new OllamaLLM();
    const results = await ollama.embedBatch(["a", "b", "c"]);

    expect(results.length).toBe(3);
    expect(results.every(r => r !== null)).toBe(true);
  });

  test("returns null for failed embeddings in batch", async () => {
    let callCount = 0;
    globalThis.fetch = mock(() => {
      callCount++;
      if (callCount === 2) return Promise.resolve({ ok: false, status: 500 });
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ embedding: [0.1] }) });
    }) as any;

    const ollama = new OllamaLLM();
    const results = await ollama.embedBatch(["a", "b", "c"]);

    expect(results[0]).not.toBeNull();
    expect(results[1]).toBeNull();
    expect(results[2]).not.toBeNull();
  });
});
```

**Implementation**: Use `Promise.all` with individual `embed()` calls. Consider a concurrency limiter (p-limit pattern) if needed for large batches.

---

### T1.4 — generate()

**Test**:

```typescript
describe("OllamaLLM.generate", () => {
  test("generates text completion", async () => {
    globalThis.fetch = mock(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ response: "Generated text here", done: true })
    })) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.generate("Write a haiku");

    expect(result).not.toBeNull();
    expect(result!.text).toBe("Generated text here");
    expect(result!.done).toBe(true);
  });

  test("respects maxTokens and temperature", async () => {
    const mockFetch = mock(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ response: "ok", done: true })
    }));
    globalThis.fetch = mockFetch as any;

    const ollama = new OllamaLLM();
    await ollama.generate("test", { maxTokens: 100, temperature: 0.5 });

    const body = JSON.parse((mockFetch.mock.calls[0] as any)[1].body);
    expect(body.options.num_predict).toBe(100);
    expect(body.options.temperature).toBe(0.5);
  });
});
```

**Implementation**:

```typescript
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
```

---

### T1.5 — modelExists()

**Test**:

```typescript
describe("OllamaLLM.modelExists", () => {
  test("returns exists:true when model is in tags", async () => {
    globalThis.fetch = mock(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ models: [{ name: "nomic-embed-text:latest" }] })
    })) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.modelExists("nomic-embed-text");

    expect(result.exists).toBe(true);
    expect(result.name).toBe("nomic-embed-text");
  });

  test("returns exists:false when model not found", async () => {
    globalThis.fetch = mock(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ models: [] })
    })) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.modelExists("nonexistent-model");

    expect(result.exists).toBe(false);
  });
});
```

**Implementation**: Call `GET /api/tags`, search for model name (handle `:latest` suffix).

---

### T1.6 — rerank()

**Test**:

```typescript
describe("OllamaLLM.rerank", () => {
  test("scores documents by relevance using prompt-based method", async () => {
    // Mock generate to return "yes" for first doc, "no" for second
    let callCount = 0;
    globalThis.fetch = mock(() => {
      callCount++;
      const isRelevant = callCount === 1;
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ response: isRelevant ? "yes" : "no", done: true })
      });
    }) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.rerank("test query", [
      { file: "a.md", text: "relevant content" },
      { file: "b.md", text: "irrelevant content" },
    ]);

    expect(result.results.length).toBe(2);
    // First doc should have higher score
    const scoreA = result.results.find(r => r.file === "a.md")!.score;
    const scoreB = result.results.find(r => r.file === "b.md")!.score;
    expect(scoreA).toBeGreaterThan(scoreB);
  });
});
```

**Implementation approach**:

For each document, generate a prompt like:

```
Is this document relevant to the query?

Query: {query}

Document: {document_text}

Answer with only "yes" or "no".
```

Score: `yes` = 1.0, `no` = 0.0 (or use logprobs if available for nuanced scoring).

Return results sorted by score descending.

---

### T1.7 — expandQuery()

This method uses `generate()` internally with a grammar. Ollama supports JSON mode but not GBNF grammar directly.

**Options**:
1. Use `format: "json"` and restructure the prompt for JSON output
2. Parse freeform text output with regex (fallback)

**Recommended**: Try JSON format first, fall back to text parsing.

**Test**: Verify it returns `Queryable[]` with lex/vec/hyde entries.

---

### T1.8 — dispose()

**Implementation**: No-op (stateless HTTP client).

```typescript
async dispose(): Promise<void> {
  // No resources to clean up for HTTP client
}
```

---

## T2: Integration

### T2.1 — Factory Function

**File**: `src/llm.ts`

Add at bottom:

```typescript
import { OllamaLLM } from "./ollama";

export function createLLM(backend?: string): LLM {
  const b = backend || process.env.QMD_LLM_BACKEND || "llamacpp";
  
  if (b === "ollama") {
    return new OllamaLLM();
  }
  
  return new LlamaCpp();
}
```

### T2.2 — Update Singleton

Replace:
```typescript
let defaultLlamaCpp: LlamaCpp | null = null;
export function getDefaultLlamaCpp(): LlamaCpp { ... }
```

With:
```typescript
let defaultLLM: LLM | null = null;

export function getDefaultLLM(): LLM {
  if (!defaultLLM) {
    defaultLLM = createLLM();
  }
  return defaultLLM;
}

// Keep old function for backwards compat, but mark deprecated
/** @deprecated Use getDefaultLLM() instead */
export function getDefaultLlamaCpp(): LlamaCpp {
  return getDefaultLLM() as LlamaCpp;
}
```

### T2.3 — Update Call Sites

Search for `getDefaultLlamaCpp` and replace with `getDefaultLLM`:

```bash
grep -r "getDefaultLlamaCpp" src/
```

Files to update:
- `src/qmd.ts`
- `src/mcp.ts`
- `src/search.ts` (if exists)
- Any other files using the singleton

### T2.4 — Export from index

If there's an index.ts, export `OllamaLLM` and `createLLM`.

---

## T3: Testing

### T3.1 — Unit Tests

Create `src/ollama.test.ts` with mocked fetch (see tests above).

### T3.2 — Integration Test

```typescript
// src/ollama.integration.test.ts
import { describe, test, expect, beforeAll } from "bun:test";
import { OllamaLLM } from "./ollama";

const OLLAMA_URL = process.env.QMD_OLLAMA_URL || "http://localhost:11434";

describe.skipIf(!process.env.RUN_INTEGRATION_TESTS)("OllamaLLM Integration", () => {
  let ollama: OllamaLLM;

  beforeAll(() => {
    ollama = new OllamaLLM({ baseUrl: OLLAMA_URL });
  });

  test("embed returns real vectors", async () => {
    const result = await ollama.embed("Hello world");
    expect(result).not.toBeNull();
    expect(result!.embedding.length).toBeGreaterThan(0);
  });

  test("generate returns real text", async () => {
    const result = await ollama.generate("Say hello in one word:");
    expect(result).not.toBeNull();
    expect(result!.text.length).toBeGreaterThan(0);
  });
});
```

Run with: `RUN_INTEGRATION_TESTS=1 bun test src/ollama.integration.test.ts`

### T3.3 — Regression Tests

Ensure existing tests pass:

```bash
bun test src/llm.test.ts
bun test
```

### T3.4 — E2E Test

Manual verification:

```bash
export QMD_LLM_BACKEND=ollama
export QMD_OLLAMA_URL=http://host.internal:11434

# Create test collection
mkdir -p /tmp/qmd-test && echo "# Test\n\nThis is a test document about TypeScript." > /tmp/qmd-test/test.md
qmd collection add /tmp/qmd-test --name e2e-test

# Embed
qmd embed

# Search
qmd search "TypeScript"
qmd vsearch "programming language"
qmd query "What is this about?"

# Cleanup
qmd collection remove e2e-test
```

---

## T4: Documentation

### T4.1 — README Update

Add section after "Quick Start":

```markdown
### Using with Ollama (Remote GPU)

QMD can offload LLM inference to an Ollama server for GPU acceleration:

\`\`\`bash
# On your GPU machine
ollama serve
ollama pull nomic-embed-text
ollama pull qwen2.5:1.5b

# On the QMD machine
export QMD_LLM_BACKEND=ollama
export QMD_OLLAMA_URL=http://gpu-host:11434

qmd embed
qmd query "your search"
\`\`\`
```

### T4.2 — CLAUDE.md Update

Add under "## Development":

```markdown
## Ollama Backend

Set `QMD_LLM_BACKEND=ollama` to use Ollama instead of node-llama-cpp.

Environment variables:
- `QMD_OLLAMA_URL` - Ollama server URL (default: http://localhost:11434)
- `QMD_OLLAMA_EMBED_MODEL` - Embedding model (default: nomic-embed-text)
- `QMD_OLLAMA_GENERATE_MODEL` - Generation model (default: qwen2.5:1.5b)
```

---

## Definition of Done

- [ ] All unit tests pass (`bun test src/ollama.test.ts`)
- [ ] All existing tests pass (`bun test`)
- [ ] Integration test passes with real Ollama server
- [ ] `qmd embed` works with `QMD_LLM_BACKEND=ollama`
- [ ] `qmd query` returns relevant results
- [ ] README and CLAUDE.md updated
- [ ] No TypeScript errors (`bun run typecheck` or `tsc --noEmit`)
- [ ] Code follows existing patterns (error handling, logging)
