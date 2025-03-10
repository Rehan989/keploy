package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"go.keploy.io/server/v2/pkg/rag"
)

func main() {
	// Get OpenAI API key from environment variable
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is required")
	}

	// Create embedding generator
	embedder := rag.NewEmbeddingGenerator(apiKey)

	// Create ChromaDB store
	ctx := context.Background()
	persistDir := filepath.Join(os.TempDir(), "keploy-chromadb")
	store, err := rag.NewChromaStore(ctx, persistDir, "code-snippets", embedder)
	if err != nil {
		log.Fatalf("Failed to create ChromaDB store: %v", err)
	}
	defer store.Close()

	// Create code indexer
	indexer := rag.NewCodeIndexer(store, ".")

	// Index the current directory
	fmt.Println("Indexing code files...")
	if err := indexer.IndexDirectory(ctx); err != nil {
		log.Fatalf("Failed to index directory: %v", err)
	}
	fmt.Println("Indexing complete!")

	// Example search
	query := "find functions related to error handling"
	fmt.Printf("\nSearching for: %s\n", query)

	results, err := indexer.Search(ctx, query, 5)
	if err != nil {
		log.Fatalf("Failed to search: %v", err)
	}

	// Print results
	fmt.Println("\nSearch Results:")
	for i, result := range results {
		fmt.Printf("\n--- Result %d ---\n", i+1)
		fmt.Printf("File: %s\n", result.Metadata["file_path"])
		fmt.Printf("Language: %s\n", result.Metadata["language"])
		fmt.Printf("Chunk %d of %d\n", result.Metadata["chunk_index"].(int)+1, result.Metadata["total_chunks"].(int))
		fmt.Printf("Content:\n%s\n", result.Content)
	}
}
