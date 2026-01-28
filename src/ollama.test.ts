/**
 * ollama.test.ts - Unit tests for OllamaLLM class
 *
 * Run with: bun test src/ollama.test.ts
 *
 * These tests mock the fetch API to test Ollama integration without a real server.
 */

import { describe, test, expect, mock, beforeEach, afterEach } from "bun:test";
import { OllamaLLM } from "./ollama";

// Save original fetch
const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

// =============================================================================
// Configuration Tests
// =============================================================================

describe("OllamaLLM Configuration", () => {
  test("uses default values when no config provided", () => {
    const ollama = new OllamaLLM();
    // Access private fields via any cast for testing
    expect((ollama as any).baseUrl).toBe("http://localhost:11434");
    expect((ollama as any).embedModel).toBe("embeddinggemma");
    expect((ollama as any).generateModel).toBe("qwen3:1.7b");
    expect((ollama as any).rerankModel).toBe("qwen3:1.7b");
  });

  test("uses config values when provided", () => {
    const ollama = new OllamaLLM({
      baseUrl: "http://custom:8080",
      embedModel: "custom-embed",
      generateModel: "custom-generate",
      rerankModel: "custom-rerank",
      timeoutMs: 60000,
    });
    expect((ollama as any).baseUrl).toBe("http://custom:8080");
    expect((ollama as any).embedModel).toBe("custom-embed");
    expect((ollama as any).generateModel).toBe("custom-generate");
    expect((ollama as any).rerankModel).toBe("custom-rerank");
    expect((ollama as any).timeoutMs).toBe(60000);
  });

  test("reads QMD_OLLAMA_URL from environment", () => {
    const originalEnv = process.env.QMD_OLLAMA_URL;
    process.env.QMD_OLLAMA_URL = "http://env-host:11434";
    try {
      const ollama = new OllamaLLM();
      expect((ollama as any).baseUrl).toBe("http://env-host:11434");
    } finally {
      if (originalEnv === undefined) {
        delete process.env.QMD_OLLAMA_URL;
      } else {
        process.env.QMD_OLLAMA_URL = originalEnv;
      }
    }
  });

  test("reads QMD_OLLAMA_EMBED_MODEL from environment", () => {
    const originalEnv = process.env.QMD_OLLAMA_EMBED_MODEL;
    process.env.QMD_OLLAMA_EMBED_MODEL = "env-embed-model";
    try {
      const ollama = new OllamaLLM();
      expect((ollama as any).embedModel).toBe("env-embed-model");
    } finally {
      if (originalEnv === undefined) {
        delete process.env.QMD_OLLAMA_EMBED_MODEL;
      } else {
        process.env.QMD_OLLAMA_EMBED_MODEL = originalEnv;
      }
    }
  });

  test("reads QMD_OLLAMA_GENERATE_MODEL from environment", () => {
    const originalEnv = process.env.QMD_OLLAMA_GENERATE_MODEL;
    process.env.QMD_OLLAMA_GENERATE_MODEL = "env-generate-model";
    try {
      const ollama = new OllamaLLM();
      expect((ollama as any).generateModel).toBe("env-generate-model");
    } finally {
      if (originalEnv === undefined) {
        delete process.env.QMD_OLLAMA_GENERATE_MODEL;
      } else {
        process.env.QMD_OLLAMA_GENERATE_MODEL = originalEnv;
      }
    }
  });

  test("reads QMD_OLLAMA_RERANK_MODEL from environment", () => {
    const originalEnv = process.env.QMD_OLLAMA_RERANK_MODEL;
    process.env.QMD_OLLAMA_RERANK_MODEL = "env-rerank-model";
    try {
      const ollama = new OllamaLLM();
      expect((ollama as any).rerankModel).toBe("env-rerank-model");
    } finally {
      if (originalEnv === undefined) {
        delete process.env.QMD_OLLAMA_RERANK_MODEL;
      } else {
        process.env.QMD_OLLAMA_RERANK_MODEL = originalEnv;
      }
    }
  });

  test("rerankModel defaults to generateModel if not specified", () => {
    const originalEnv = process.env.QMD_OLLAMA_RERANK_MODEL;
    delete process.env.QMD_OLLAMA_RERANK_MODEL;
    try {
      const ollama = new OllamaLLM({ generateModel: "custom-gen" });
      expect((ollama as any).rerankModel).toBe("custom-gen");
    } finally {
      if (originalEnv !== undefined) {
        process.env.QMD_OLLAMA_RERANK_MODEL = originalEnv;
      }
    }
  });
});

// =============================================================================
// embed() Tests
// =============================================================================

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
    expect(result!.model).toBe("embeddinggemma");
  });

  test("uses custom model from options", async () => {
    let capturedBody: any;
    globalThis.fetch = mock((url: string, options: any) => {
      capturedBody = JSON.parse(options.body);
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ embedding: [0.1] }),
      });
    }) as any;

    const ollama = new OllamaLLM();
    await ollama.embed("test", { model: "custom-model" });

    expect(capturedBody.model).toBe("custom-model");
  });

  test("returns null on API error (non-200 status)", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve({ ok: false, status: 500 })
    ) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.embed("test");

    expect(result).toBeNull();
  });

  test("returns null on network error (ECONNREFUSED)", async () => {
    globalThis.fetch = mock(() =>
      Promise.reject(new Error("ECONNREFUSED"))
    ) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.embed("test");

    expect(result).toBeNull();
  });

  test("returns null on timeout", async () => {
    globalThis.fetch = mock(() =>
      Promise.reject(new DOMException("Aborted", "AbortError"))
    ) as any;

    const ollama = new OllamaLLM({ timeoutMs: 100 });
    const result = await ollama.embed("test");

    expect(result).toBeNull();
  });

  test("calls correct API endpoint", async () => {
    let capturedUrl: string = "";
    globalThis.fetch = mock((url: string) => {
      capturedUrl = url;
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ embedding: [0.1] }),
      });
    }) as any;

    const ollama = new OllamaLLM({ baseUrl: "http://myhost:11434" });
    await ollama.embed("test");

    expect(capturedUrl).toBe("http://myhost:11434/api/embed");
  });
});

// =============================================================================
// embedBatch() Tests
// =============================================================================

describe("OllamaLLM.embedBatch", () => {
  test("processes multiple texts and returns results", async () => {
    // Use request body to determine response, ensuring deterministic results
    globalThis.fetch = mock((url: string, opts: any) => {
      const body = JSON.parse(opts.body);
      // Return embedding based on input text
      const textIndex = ["text1", "text2", "text3"].indexOf(body.input);
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ embeddings: [[(textIndex + 1) * 0.1]] }),
      });
    }) as any;

    const ollama = new OllamaLLM();
    const results = await ollama.embedBatch(["text1", "text2", "text3"]);

    expect(results).toHaveLength(3);
    // Each result should have an embedding array
    expect(results[0]).not.toBeNull();
    expect(results[1]).not.toBeNull();
    expect(results[2]).not.toBeNull();
    expect(results[0]!.embedding.length).toBe(1);
    expect(results[1]!.embedding.length).toBe(1);
    expect(results[2]!.embedding.length).toBe(1);
  });

  test("handles empty array", async () => {
    const ollama = new OllamaLLM();
    const results = await ollama.embedBatch([]);

    expect(results).toHaveLength(0);
  });

  test("respects concurrency limit of 4", async () => {
    let maxConcurrent = 0;
    let currentConcurrent = 0;

    globalThis.fetch = mock(async () => {
      currentConcurrent++;
      maxConcurrent = Math.max(maxConcurrent, currentConcurrent);
      await new Promise((r) => setTimeout(r, 10)); // Simulate latency
      currentConcurrent--;
      return {
        ok: true,
        json: () => Promise.resolve({ embedding: [0.1] }),
      };
    }) as any;

    const ollama = new OllamaLLM();
    // Process 8 texts - should batch in groups of 4
    await ollama.embedBatch(Array(8).fill("text"));

    expect(maxConcurrent).toBeLessThanOrEqual(4);
  });

  test("returns null for failed individual embeddings", async () => {
    let callCount = 0;
    globalThis.fetch = mock(() => {
      callCount++;
      if (callCount === 2) {
        return Promise.resolve({ ok: false, status: 500 });
      }
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ embedding: [0.1] }),
      });
    }) as any;

    const ollama = new OllamaLLM();
    const results = await ollama.embedBatch(["text1", "text2", "text3"]);

    expect(results).toHaveLength(3);
    expect(results[0]).not.toBeNull();
    expect(results[1]).toBeNull();
    expect(results[2]).not.toBeNull();
  });
});

// =============================================================================
// generate() Tests
// =============================================================================

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
    expect(result!.model).toBe("qwen3:1.7b");
  });

  test("uses custom model from options", async () => {
    let capturedBody: any;
    globalThis.fetch = mock((url: string, options: any) => {
      capturedBody = JSON.parse(options.body);
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ response: "ok", done: true }),
      });
    }) as any;

    const ollama = new OllamaLLM();
    await ollama.generate("test", { model: "custom-model" });

    expect(capturedBody.model).toBe("custom-model");
  });

  test("passes maxTokens and temperature options", async () => {
    let capturedBody: any;
    globalThis.fetch = mock((url: string, options: any) => {
      capturedBody = JSON.parse(options.body);
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ response: "ok", done: true }),
      });
    }) as any;

    const ollama = new OllamaLLM();
    await ollama.generate("test", { maxTokens: 500, temperature: 0.7 });

    expect(capturedBody.options.num_predict).toBe(500);
    expect(capturedBody.options.temperature).toBe(0.7);
  });

  test("uses stream: false", async () => {
    let capturedBody: any;
    globalThis.fetch = mock((url: string, options: any) => {
      capturedBody = JSON.parse(options.body);
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ response: "ok", done: true }),
      });
    }) as any;

    const ollama = new OllamaLLM();
    await ollama.generate("test");

    expect(capturedBody.stream).toBe(false);
  });

  test("returns null on API error", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve({ ok: false, status: 500 })
    ) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.generate("test");

    expect(result).toBeNull();
  });

  test("returns null on network error", async () => {
    globalThis.fetch = mock(() =>
      Promise.reject(new Error("Network error"))
    ) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.generate("test");

    expect(result).toBeNull();
  });

  test("calls correct API endpoint", async () => {
    let capturedUrl: string = "";
    globalThis.fetch = mock((url: string) => {
      capturedUrl = url;
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ response: "ok", done: true }),
      });
    }) as any;

    const ollama = new OllamaLLM({ baseUrl: "http://myhost:11434" });
    await ollama.generate("test");

    expect(capturedUrl).toBe("http://myhost:11434/api/generate");
  });
});

// =============================================================================
// modelExists() Tests
// =============================================================================

describe("OllamaLLM.modelExists", () => {
  test("returns exists:true when model found (exact match)", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve({
        ok: true,
        json: () =>
          Promise.resolve({
            models: [{ name: "embeddinggemma" }],
          }),
      })
    ) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.modelExists("embeddinggemma");

    expect(result.exists).toBe(true);
    expect(result.name).toBe("embeddinggemma");
  });

  test("returns exists:true when model found with :latest suffix", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve({
        ok: true,
        json: () =>
          Promise.resolve({
            models: [{ name: "embeddinggemma:latest" }],
          }),
      })
    ) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.modelExists("embeddinggemma");

    expect(result.exists).toBe(true);
  });

  test("returns exists:true when model found with any tag", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve({
        ok: true,
        json: () =>
          Promise.resolve({
            models: [{ name: "qwen3:1.7b" }],
          }),
      })
    ) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.modelExists("qwen3");

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
    expect(result.name).toBe("nonexistent");
  });

  test("returns exists:false on API error", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve({ ok: false, status: 500 })
    ) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.modelExists("test");

    expect(result.exists).toBe(false);
  });

  test("returns exists:false on network error", async () => {
    globalThis.fetch = mock(() =>
      Promise.reject(new Error("Network error"))
    ) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.modelExists("test");

    expect(result.exists).toBe(false);
  });

  test("calls correct API endpoint", async () => {
    let capturedUrl: string = "";
    globalThis.fetch = mock((url: string) => {
      capturedUrl = url;
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ models: [] }),
      });
    }) as any;

    const ollama = new OllamaLLM({ baseUrl: "http://myhost:11434" });
    await ollama.modelExists("test");

    expect(capturedUrl).toBe("http://myhost:11434/api/tags");
  });
});

// =============================================================================
// expandQuery() Tests
// =============================================================================

describe("OllamaLLM.expandQuery", () => {
  test("returns queryables from valid JSON response", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve({
        ok: true,
        json: () =>
          Promise.resolve({
            response: '{"lex":["keyword1","keyword2"],"vec":["semantic query"],"hyde":["hypothetical doc"]}',
            done: true,
          }),
      })
    ) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.expandQuery("test query");

    expect(result.length).toBeGreaterThanOrEqual(1);
    const types = result.map((q) => q.type);
    expect(types).toContain("lex");
    expect(types).toContain("vec");
    expect(types).toContain("hyde");
  });

  test("excludes lex entries when includeLexical is false", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve({
        ok: true,
        json: () =>
          Promise.resolve({
            response: '{"lex":["keyword"],"vec":["semantic query"],"hyde":[]}',
            done: true,
          }),
      })
    ) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.expandQuery("test", { includeLexical: false });

    const lexEntries = result.filter((q) => q.type === "lex");
    expect(lexEntries).toHaveLength(0);
  });

  test("falls back to simple queryables on invalid JSON", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve({
        ok: true,
        json: () =>
          Promise.resolve({
            response: "not valid json",
            done: true,
          }),
      })
    ) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.expandQuery("test query");

    expect(result.length).toBeGreaterThanOrEqual(1);
    // Should have at least a vec entry with the original query
    const vecEntries = result.filter((q) => q.type === "vec");
    expect(vecEntries.length).toBeGreaterThanOrEqual(1);
    expect(vecEntries[0]!.text).toBe("test query");
  });

  test("falls back to simple queryables on API error", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve({ ok: false, status: 500 })
    ) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.expandQuery("original query");

    expect(result.length).toBeGreaterThanOrEqual(1);
    const texts = result.map((q) => q.text);
    expect(texts).toContain("original query");
  });

  test("includes lex in fallback when includeLexical is true", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve({ ok: false, status: 500 })
    ) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.expandQuery("test", { includeLexical: true });

    const lexEntries = result.filter((q) => q.type === "lex");
    expect(lexEntries.length).toBeGreaterThanOrEqual(1);
  });

  test("excludes lex from fallback when includeLexical is false", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve({ ok: false, status: 500 })
    ) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.expandQuery("test", { includeLexical: false });

    const lexEntries = result.filter((q) => q.type === "lex");
    expect(lexEntries).toHaveLength(0);
  });

  test("extracts JSON from response with surrounding text", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve({
        ok: true,
        json: () =>
          Promise.resolve({
            response: 'Here is the result: {"lex":["kw"],"vec":["query"],"hyde":[]} Hope this helps!',
            done: true,
          }),
      })
    ) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.expandQuery("test");

    const types = result.map((q) => q.type);
    expect(types).toContain("lex");
    expect(types).toContain("vec");
  });
});

// =============================================================================
// rerank() Tests
// =============================================================================

describe("OllamaLLM.rerank", () => {
  test("scores documents by relevance (yes/no prompting)", async () => {
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

  test("returns sorted results (highest score first)", async () => {
    let callCount = 0;
    globalThis.fetch = mock(() => {
      callCount++;
      // Second doc is relevant, first is not
      const response = callCount === 2 ? "yes" : "no";
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ response, done: true }),
      });
    }) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.rerank("query", [
      { file: "first.md", text: "not relevant" },
      { file: "second.md", text: "relevant" },
    ]);

    // Second doc should be first in results (higher score)
    expect(result.results[0]!.file).toBe("second.md");
    expect(result.results[0]!.score).toBe(1.0);
    expect(result.results[1]!.file).toBe("first.md");
    expect(result.results[1]!.score).toBe(0.0);
  });

  test("handles empty document list", async () => {
    const ollama = new OllamaLLM();
    const result = await ollama.rerank("query", []);

    expect(result.results).toHaveLength(0);
  });

  test("handles single document", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ response: "yes", done: true }),
      })
    ) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.rerank("query", [{ file: "doc.md", text: "content" }]);

    expect(result.results).toHaveLength(1);
    expect(result.results[0]!.file).toBe("doc.md");
    expect(result.results[0]!.score).toBe(1.0);
  });

  test("preserves original file paths", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ response: "yes", done: true }),
      })
    ) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.rerank("query", [
      { file: "path/to/doc1.md", text: "content" },
      { file: "another/path/doc2.md", text: "content" },
    ]);

    const files = result.results.map((r) => r.file).sort();
    expect(files).toEqual(["another/path/doc2.md", "path/to/doc1.md"]);
  });

  test("preserves original index", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ response: "yes", done: true }),
      })
    ) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.rerank("query", [
      { file: "a.md", text: "content a" },
      { file: "b.md", text: "content b" },
      { file: "c.md", text: "content c" },
    ]);

    const aResult = result.results.find((r) => r.file === "a.md")!;
    const bResult = result.results.find((r) => r.file === "b.md")!;
    const cResult = result.results.find((r) => r.file === "c.md")!;

    expect(aResult.index).toBe(0);
    expect(bResult.index).toBe(1);
    expect(cResult.index).toBe(2);
  });

  test("handles API errors gracefully (scores 0)", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve({ ok: false, status: 500 })
    ) as any;

    const ollama = new OllamaLLM();
    const result = await ollama.rerank("query", [
      { file: "a.md", text: "content" },
    ]);

    expect(result.results).toHaveLength(1);
    expect(result.results[0]!.score).toBe(0);
  });

  test("returns model name in result", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ response: "yes", done: true }),
      })
    ) as any;

    const ollama = new OllamaLLM({ rerankModel: "custom-rerank" });
    const result = await ollama.rerank("query", [{ file: "a.md", text: "content" }]);

    expect(result.model).toBe("custom-rerank");
  });

  test("uses custom model from options", async () => {
    let capturedBody: any;
    globalThis.fetch = mock((url: string, options: any) => {
      capturedBody = JSON.parse(options.body);
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ response: "yes", done: true }),
      });
    }) as any;

    const ollama = new OllamaLLM();
    await ollama.rerank("query", [{ file: "a.md", text: "content" }], {
      model: "override-model",
    });

    expect(capturedBody.model).toBe("override-model");
  });

  test("truncates document text to 500 characters in prompt", async () => {
    let capturedBody: any;
    globalThis.fetch = mock((url: string, options: any) => {
      capturedBody = JSON.parse(options.body);
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ response: "yes", done: true }),
      });
    }) as any;

    const longText = "x".repeat(1000);
    const ollama = new OllamaLLM();
    await ollama.rerank("query", [{ file: "a.md", text: longText }]);

    // The prompt should contain truncated text
    expect(capturedBody.prompt.length).toBeLessThan(longText.length + 200); // prompt + template
  });
});

// =============================================================================
// dispose() Tests
// =============================================================================

describe("OllamaLLM.dispose", () => {
  test("completes without error", async () => {
    const ollama = new OllamaLLM();
    await expect(ollama.dispose()).resolves.toBeUndefined();
  });

  test("can be called multiple times safely", async () => {
    const ollama = new OllamaLLM();
    await ollama.dispose();
    await expect(ollama.dispose()).resolves.toBeUndefined();
  });
});

// =============================================================================
// LLM Interface Compliance Tests
// =============================================================================

describe("OllamaLLM implements LLM interface", () => {
  test("has all required methods", () => {
    const ollama = new OllamaLLM();

    expect(typeof ollama.embed).toBe("function");
    expect(typeof ollama.generate).toBe("function");
    expect(typeof ollama.modelExists).toBe("function");
    expect(typeof ollama.expandQuery).toBe("function");
    expect(typeof ollama.rerank).toBe("function");
    expect(typeof ollama.dispose).toBe("function");
  });
});
