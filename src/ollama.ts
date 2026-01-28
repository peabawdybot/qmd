/**
 * ollama.ts - Ollama LLM backend for QMD
 *
 * Provides embeddings, text generation, and reranking using a remote Ollama server.
 * This enables GPU-accelerated inference on a separate host while QMD runs on CPU-only machines.
 */

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

// =============================================================================
// Ollama API Response Types
// =============================================================================

type OllamaEmbedResponse = {
  embeddings?: number[][];
  embedding?: number[];
};

type OllamaGenerateResponse = {
  response: string;
  done: boolean;
};

type OllamaTagsResponse = {
  models?: { name: string }[];
};

// =============================================================================
// Configuration
// =============================================================================

export type OllamaConfig = {
  baseUrl?: string;
  embedModel?: string;
  generateModel?: string;
  rerankModel?: string;
  timeoutMs?: number;
};

// =============================================================================
// OllamaLLM Implementation
// =============================================================================

/**
 * LLM implementation using Ollama HTTP API
 */
export class OllamaLLM implements LLM {
  private baseUrl: string;
  private embedModel: string;
  private generateModel: string;
  private rerankModel: string;
  private timeoutMs: number;

  constructor(config: OllamaConfig = {}) {
    this.baseUrl = config.baseUrl || process.env.QMD_OLLAMA_URL || "http://localhost:11434";
    this.embedModel = config.embedModel || process.env.QMD_OLLAMA_EMBED_MODEL || "embeddinggemma";
    this.generateModel = config.generateModel || process.env.QMD_OLLAMA_GENERATE_MODEL || "qwen3:1.7b";
    this.rerankModel = config.rerankModel || process.env.QMD_OLLAMA_RERANK_MODEL || this.generateModel;
    this.timeoutMs = config.timeoutMs || 30000;
  }

  // ==========================================================================
  // Core API methods
  // ==========================================================================

  /**
   * Get embeddings for text using Ollama's /api/embed endpoint
   */
  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    try {
      const response = await fetch(`${this.baseUrl}/api/embed`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: options?.model || this.embedModel,
          input: text,
        }),
        signal: AbortSignal.timeout(this.timeoutMs),
      });

      if (!response.ok) {
        console.error(`Ollama embed error: ${response.status}`);
        return null;
      }

      const data = (await response.json()) as OllamaEmbedResponse;
      // Ollama returns embeddings array, take the first one
      const embedding = data.embeddings?.[0] || data.embedding;

      if (!embedding) {
        console.error("Ollama embed error: no embedding in response");
        return null;
      }

      return {
        embedding,
        model: options?.model || this.embedModel,
      };
    } catch (error) {
      console.error("Ollama embed error:", error);
      return null;
    }
  }

  /**
   * Batch embed multiple texts
   * Ollama doesn't support batch embeddings natively, so we run with concurrency limit
   */
  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];

    const results: (EmbeddingResult | null)[] = [];
    const concurrencyLimit = 4;

    for (let i = 0; i < texts.length; i += concurrencyLimit) {
      const batch = texts.slice(i, i + concurrencyLimit);
      const batchResults = await Promise.all(batch.map((text) => this.embed(text)));
      results.push(...batchResults);
    }

    return results;
  }

  /**
   * Generate text completion using Ollama's /api/generate endpoint
   */
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

      const data = (await response.json()) as OllamaGenerateResponse;
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

  /**
   * Check if a model exists in Ollama using /api/tags endpoint
   */
  async modelExists(model: string): Promise<ModelInfo> {
    try {
      const response = await fetch(`${this.baseUrl}/api/tags`, {
        signal: AbortSignal.timeout(this.timeoutMs),
      });

      if (!response.ok) {
        return { name: model, exists: false };
      }

      const data = (await response.json()) as OllamaTagsResponse;
      const models = data.models || [];

      // Check if model exists (handle :latest suffix)
      const exists = models.some(
        (m) =>
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

  /**
   * Expand a search query into multiple variations
   * Uses JSON format instead of GBNF grammar (Ollama doesn't support GBNF)
   */
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

      // Parse JSON response - extract JSON from potentially surrounding text
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

  /**
   * Fallback queryables when LLM fails
   */
  private fallbackQueryables(query: string, includeLexical: boolean): Queryable[] {
    const result: Queryable[] = [{ type: "vec" as QueryType, text: query }];
    if (includeLexical) {
      result.unshift({ type: "lex" as QueryType, text: query });
    }
    return result;
  }

  /**
   * Rerank documents by relevance using prompt-based yes/no scoring
   * Ollama doesn't have a dedicated reranker endpoint, so we use generate with prompts
   */
  async rerank(
    query: string,
    documents: RerankDocument[],
    options?: RerankOptions
  ): Promise<RerankResult> {
    if (documents.length === 0) {
      return { results: [], model: options?.model || this.rerankModel };
    }

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

  // ==========================================================================
  // Tokenization (approximation - Ollama doesn't expose native tokenizer)
  // ==========================================================================

  /**
   * Tokenize text - uses word-based approximation since Ollama doesn't expose tokenizer
   * Returns array of "tokens" (strings) for compatibility with LLM interface
   */
  async tokenize(text: string): Promise<readonly string[]> {
    // Simple word-based tokenization as approximation
    // Real tokenizers are model-specific but this gives reasonable estimates
    return text.split(/\s+/).filter((t) => t.length > 0);
  }

  /**
   * Count tokens in text
   */
  async countTokens(text: string): Promise<number> {
    const tokens = await this.tokenize(text);
    return tokens.length;
  }

  /**
   * Detokenize tokens back to text
   */
  async detokenize(tokens: readonly unknown[]): Promise<string> {
    // Simple join since we use word-based tokenization
    return (tokens as string[]).join(" ");
  }

  /**
   * Dispose of resources - no-op for stateless HTTP client
   */
  async dispose(): Promise<void> {
    // No resources to clean up for stateless HTTP client
  }
}
