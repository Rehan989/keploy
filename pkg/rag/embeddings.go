package rag

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

// EmbeddingResponse represents the response from OpenAI's embedding API
type EmbeddingResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
		Object    string    `json:"object"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

// EmbeddingRequest represents the request to OpenAI's embedding API
type EmbeddingRequest struct {
	Input []string `json:"input"`
	Model string   `json:"model"`
}

// EmbeddingGenerator handles the generation of embeddings using OpenAI's API
type EmbeddingGenerator struct {
	apiKey string
	model  string
}

// NewEmbeddingGenerator creates a new instance of EmbeddingGenerator
func NewEmbeddingGenerator(apiKey string) *EmbeddingGenerator {
	return &EmbeddingGenerator{
		apiKey: apiKey,
		model:  "text-embedding-ada-002", // Using OpenAI's recommended embedding model
	}
}

// GenerateEmbeddings generates embeddings for the given texts
func (g *EmbeddingGenerator) GenerateEmbeddings(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("no texts provided for embedding generation")
	}

	reqBody := EmbeddingRequest{
		Input: texts,
		Model: g.model,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %v", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.openai.com/v1/embeddings", bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", g.apiKey))

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API request failed with status code: %d", resp.StatusCode)
	}

	var embeddingResp EmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&embeddingResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}

	embeddings := make([][]float32, len(embeddingResp.Data))
	for i, data := range embeddingResp.Data {
		embeddings[i] = data.Embedding
	}

	return embeddings, nil
}
