package rag

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// CodeIndexer handles the indexing of code files
type CodeIndexer struct {
	store    *ChromaStore
	rootPath string
}

// NewCodeIndexer creates a new instance of CodeIndexer
func NewCodeIndexer(store *ChromaStore, rootPath string) *CodeIndexer {
	return &CodeIndexer{
		store:    store,
		rootPath: rootPath,
	}
}

// generateID creates a unique ID for a code snippet
func generateID(filePath string, content string) string {
	hash := sha256.New()
	hash.Write([]byte(filePath + content))
	return hex.EncodeToString(hash.Sum(nil))
}

// ProcessFile processes a single file and adds it to the vector store
func (i *CodeIndexer) ProcessFile(ctx context.Context, filePath string) error {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return fmt.Errorf("failed to read file %s: %v", filePath, err)
	}

	// Split content into chunks (you can adjust the chunk size based on your needs)
	chunks := i.splitIntoChunks(string(content), 1000)

	for idx, chunk := range chunks {
		// Create metadata for the chunk
		metadata := map[string]interface{}{
			"file_path":    filePath,
			"chunk_index":  idx,
			"total_chunks": len(chunks),
			"language":     strings.TrimPrefix(filepath.Ext(filePath), "."),
		}

		// Generate a unique ID for the chunk
		id := generateID(filePath, chunk)

		// Add the chunk to the vector store
		err = i.store.AddDocuments(ctx, []string{chunk}, []map[string]interface{}{metadata}, []string{id})
		if err != nil {
			return fmt.Errorf("failed to add chunk to vector store: %v", err)
		}
	}

	return nil
}

// splitIntoChunks splits text into chunks of approximately the specified size
func (i *CodeIndexer) splitIntoChunks(text string, chunkSize int) []string {
	var chunks []string
	lines := strings.Split(text, "\n")
	currentChunk := strings.Builder{}
	currentSize := 0

	for _, line := range lines {
		lineSize := len(line) + 1 // +1 for newline
		if currentSize+lineSize > chunkSize && currentSize > 0 {
			chunks = append(chunks, currentChunk.String())
			currentChunk.Reset()
			currentSize = 0
		}
		currentChunk.WriteString(line + "\n")
		currentSize += lineSize
	}

	if currentChunk.Len() > 0 {
		chunks = append(chunks, currentChunk.String())
	}

	return chunks
}

// IndexDirectory indexes all code files in a directory
func (i *CodeIndexer) IndexDirectory(ctx context.Context) error {
	return filepath.WalkDir(i.rootPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		// Skip directories and non-code files
		if d.IsDir() || !isCodeFile(path) {
			return nil
		}

		return i.ProcessFile(ctx, path)
	})
}

// isCodeFile checks if a file is a code file based on its extension
func isCodeFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	codeExtensions := map[string]bool{
		".go":   true,
		".py":   true,
		".js":   true,
		".ts":   true,
		".java": true,
		".cpp":  true,
		".c":    true,
		".h":    true,
		".hpp":  true,
		".rs":   true,
	}
	return codeExtensions[ext]
}

// Search performs a semantic search over the indexed code
func (i *CodeIndexer) Search(ctx context.Context, query string, limit int) ([]SearchResult, error) {
	docs, metadata, err := i.store.QuerySimilar(ctx, query, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to query vector store: %v", err)
	}

	results := make([]SearchResult, len(docs))
	for idx, doc := range docs {
		results[idx] = SearchResult{
			Content:  doc,
			Metadata: metadata[idx],
		}
	}

	return results, nil
}

// SearchResult represents a search result
type SearchResult struct {
	Content  string                 `json:"content"`
	Metadata map[string]interface{} `json:"metadata"`
}
