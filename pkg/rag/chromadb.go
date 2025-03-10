package rag

import (
	"context"
	"fmt"
	"path/filepath"
	"sync"

	chroma "github.com/amikos-tech/go-chromadb"
)

// ChromaStore represents the ChromaDB vector store
type ChromaStore struct {
	client         *chroma.Client
	collection     *chroma.Collection
	collectionName string
	embedder       *EmbeddingGenerator
	mu             sync.RWMutex
}

// NewChromaStore creates a new instance of ChromaStore
func NewChromaStore(ctx context.Context, persistDir string, collectionName string, embedder *EmbeddingGenerator) (*ChromaStore, error) {
	// Ensure the persist directory exists
	persistDir = filepath.Clean(persistDir)

	// Initialize ChromaDB client
	cfg := chroma.Config{
		Path: persistDir,
	}

	client, err := chroma.NewClient(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create ChromaDB client: %v", err)
	}

	// Create or get collection
	collection, err := client.CreateCollection(ctx, chroma.CollectionConfig{
		Name: collectionName,
		Metadata: map[string]interface{}{
			"description": "Code snippets collection for RAG system",
		},
	})
	if err != nil {
		// If collection already exists, try to get it
		collection, err = client.GetCollection(ctx, collectionName)
		if err != nil {
			return nil, fmt.Errorf("failed to create/get collection: %v", err)
		}
	}

	return &ChromaStore{
		client:         client,
		collection:     collection,
		collectionName: collectionName,
		embedder:       embedder,
	}, nil
}

// AddDocuments adds documents to the vector store
func (s *ChromaStore) AddDocuments(ctx context.Context, documents []string, metadata []map[string]interface{}, ids []string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Generate embeddings for documents
	embeddings, err := s.embedder.GenerateEmbeddings(ctx, documents)
	if err != nil {
		return fmt.Errorf("failed to generate embeddings: %v", err)
	}

	// Add documents to ChromaDB
	err = s.collection.Add(ctx, chroma.AddConfig{
		Ids:        ids,
		Embeddings: embeddings,
		Documents:  documents,
		Metadatas:  metadata,
	})
	if err != nil {
		return fmt.Errorf("failed to add documents to ChromaDB: %v", err)
	}

	return nil
}

// QuerySimilar queries similar documents from the vector store
func (s *ChromaStore) QuerySimilar(ctx context.Context, query string, limit int) ([]string, []map[string]interface{}, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Generate embedding for query
	queryEmbeddings, err := s.embedder.GenerateEmbeddings(ctx, []string{query})
	if err != nil {
		return nil, nil, fmt.Errorf("failed to generate query embedding: %v", err)
	}

	// Query ChromaDB
	results, err := s.collection.Query(ctx, chroma.QueryConfig{
		QueryEmbeddings: queryEmbeddings[0],
		NResults:        limit,
		Include:         []string{"documents", "metadatas"},
	})
	if err != nil {
		return nil, nil, fmt.Errorf("failed to query ChromaDB: %v", err)
	}

	return results.Documents, results.Metadatas, nil
}

// DeleteDocuments deletes documents from the vector store
func (s *ChromaStore) DeleteDocuments(ctx context.Context, ids []string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	err := s.collection.Delete(ctx, chroma.DeleteConfig{
		Ids: ids,
	})
	if err != nil {
		return fmt.Errorf("failed to delete documents from ChromaDB: %v", err)
	}

	return nil
}

// Close closes the ChromaDB client
func (s *ChromaStore) Close() error {
	return s.client.Close()
}
