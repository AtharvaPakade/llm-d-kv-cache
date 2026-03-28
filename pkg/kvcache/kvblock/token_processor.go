/*
Copyright 2025 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package kvblock

import (
	"context"
	"fmt"
	"hash/fnv"

	"github.com/fxamacker/cbor/v2"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-kv-cache/pkg/utils"
)

// defaultBlockSize is the default number of tokens per block.
// 16 is the default value used by vLLM.
const defaultBlockSize = 16

// TokenProcessorConfig holds the configuration for the token processor.
type TokenProcessorConfig struct {
	BlockSize int `json:"blockSize"`
	// HashSeed is used to prefix initial hash chunks, similarly to vLLM's NONE_HASH.
	// This should be aligned with vLLM's `PYTHONHASHSEED` environment variable.
	// The system's deployer is responsible for aligning the vLLM deployments
	// with the same seed value.
	HashSeed string `json:"hashSeed"`
	initHash uint64 // cache once
}

type MMHash struct {
	Hash   string
	Offset int64
}

// MMHashWithPlaceholders represents a multi-modal hash entry with an absolute token offset
// and a length, before it has been chunked into per-block relative offsets.
// Use MMPlaceHolderToChunkedMMHashes to convert these into per-block [][]MMHash.
type MMHashWithPlaceholders struct {
	Hash   string
	Offset int64
	Length int64
}

// DefaultTokenProcessorConfig returns the default configuration for the token processor.
func DefaultTokenProcessorConfig() *TokenProcessorConfig {
	return &TokenProcessorConfig{
		BlockSize: defaultBlockSize,
		HashSeed:  "",
	}
}

// TokenProcessor defines the interface for converting tokens to
// KVBlockKeys.
type TokenProcessor interface {
	// TokensToKVBlockKeys converts tokens into kv_block.Keys.
	// It accepts an optional parentKey to continue a hash chain.
	// mmHashes is an optional per-block slice of multi-modal extra keys; nil entries mean no extra data for that block.
	// It returns a slice of generated Keys.
	TokensToKVBlockKeys(parentKey BlockHash, tokens []uint32, modelName string, chunkedMMHashes [][]MMHash) ([]BlockHash, error)
}

// chunkedTokenDatabase is a concrete implementation of TokenDatabase.
// It mimics the chunkedTokenDatabase in the Python code.
type chunkedTokenDatabase struct {
	TokenProcessorConfig
	encoder cbor.EncMode // cached CBOR encoder for interoperable encoding
}

var _ TokenProcessor = &chunkedTokenDatabase{}

// NewChunkedTokenDatabase creates a new instance with the given config and metadata.
func NewChunkedTokenDatabase(config *TokenProcessorConfig) (TokenProcessor, error) {
	if config == nil {
		config = DefaultTokenProcessorConfig()
	}

	if config.BlockSize <= 0 {
		return nil, fmt.Errorf("blockSize must be greater than 0, got %d", config.BlockSize)
	}

	if config.initHash == 0 {
		// Create initial hash
		h := fnv.New64a()
		_, _ = h.Write([]byte(config.HashSeed))
		config.initHash = h.Sum64()
	}

	encoder, err := cbor.CanonicalEncOptions().EncMode()
	if err != nil {
		return nil, fmt.Errorf("failed to create CBOR encoder: %w", err)
	}

	return &chunkedTokenDatabase{
		TokenProcessorConfig: *config,
		encoder:              encoder,
	}, nil
}

// getInitHash returns the initial hash for the given model name.
func (db *chunkedTokenDatabase) getInitHash(modelName string) uint64 {
	return db.hash(db.initHash, nil, modelName)
}

// hash computes the uint64 FNV-64a hash of the given parent, tokens,
// and extra keys.
//
// The hash is computed using FNV-64a over the CBOR canonical encoding of
// [parent, tokens, extra], ensuring deterministic results across runs and
// compatibility with vLLM's prefix caching algorithm.
//
// The extra parameter enables cache differentiation for LoRA adapters and
// multi-modal content. Supported types: nil, int, string, map[string]interface{}.
// Must be CBOR-serializable.
func (db *chunkedTokenDatabase) hash(parent uint64, tokens []uint32, extra interface{}) uint64 {
	payload := []interface{}{parent, tokens, extra}

	b, err := db.encoder.Marshal(payload)
	if err != nil {
		log.FromContext(context.Background()).Error(err, "failed to marshal payload to CBOR")
		return 0
	}

	h := fnv.New64a()
	_, _ = h.Write(b)
	return h.Sum64()
}

// prefixHashes returns a slice of uint64 hashes.
func (db *chunkedTokenDatabase) prefixHashes(parentHash uint64, tokenChunks [][]uint32, chunkedMMHashes [][]MMHash) []uint64 {
	prefix := parentHash
	hashes := make([]uint64, len(tokenChunks))
	for i, chunk := range tokenChunks {
		var mm []MMHash
		if chunkedMMHashes != nil && i < len(chunkedMMHashes) {
			mm = chunkedMMHashes[i]
		}
		prefix = db.hash(prefix, chunk, mm)
		hashes[i] = prefix
	}
	return hashes
}

// chunkTokens splits the input slice of tokens into chunks of size chunkSize.
func (db *chunkedTokenDatabase) chunkTokens(tokens []uint32) [][]uint32 {
	var chunks [][]uint32
	for i := 0; i < len(tokens); i += db.BlockSize {
		end := i + db.BlockSize
		if end > len(tokens) {
			break // no partial blocks
		}

		chunks = append(chunks, tokens[i:end])
	}

	return chunks
}

// MMPlaceHolderToChunkedMMHashes converts a flat list of MMHashWithPlaceholders into a
// per-block slice of []MMHash. Each element in the returned slice corresponds to one full
// token block; the offset stored in each MMHash is relative to the start of that block.
// Partial trailing blocks (len(tokens) % BlockSize != 0) are excluded.
func (db *chunkedTokenDatabase) MMPlaceHolderToChunkedMMHashes(tokens []uint32, mmHashesWithPlaceholder []MMHashWithPlaceholders) [][]MMHash {
	numBlocks := len(tokens) / db.BlockSize
	if numBlocks == 0 {
		return nil
	}

	mmHashes := make([][]MMHash, numBlocks)
	for _, mmHash := range mmHashesWithPlaceholder {
		firstBlock := int(mmHash.Offset) / db.BlockSize
		lastBlock := int(mmHash.Offset+mmHash.Length-1) / db.BlockSize

		if firstBlock < 0 {
			firstBlock = 0
		}
		if lastBlock >= numBlocks {
			lastBlock = numBlocks - 1
		}
		if firstBlock > lastBlock {
			continue
		}

		for b := firstBlock; b <= lastBlock; b++ {
			blockStart := int64(b * db.BlockSize)
			mmHashes[b] = append(mmHashes[b], MMHash{
				Hash:   mmHash.Hash,
				Offset: mmHash.Offset - blockStart, // relative offset, can be negative
			})
		}
	}

	return mmHashes
}

// TokensToKVBlockKeys converts tokens into kv_block.Keys.
func (db *chunkedTokenDatabase) TokensToKVBlockKeys(parentKey BlockHash, tokens []uint32, modelName string, chunkedMMHashes [][]MMHash) ([]BlockHash, error) {
	var currentParentHash uint64
	if parentKey != EmptyBlockHash {
		currentParentHash = uint64(parentKey)
	} else {
		currentParentHash = db.getInitHash(modelName)
	}

	chunks := db.chunkTokens(tokens)
	if len(chunks) == 0 {
		return nil, nil
	}

	if chunkedMMHashes == nil {
		chunkedMMHashes = make([][]MMHash, len(chunks))
	} else if len(chunks) != len(chunkedMMHashes) {
		return nil, fmt.Errorf("chunkedMMHashes length %d does not match token chunk count %d (blockSize=%d, tokens=%d)", len(chunkedMMHashes), len(chunks), db.BlockSize, len(tokens))
	}

	ph := db.prefixHashes(currentParentHash, chunks, chunkedMMHashes)

	return utils.SliceMap(ph, func(hashVal uint64) BlockHash {
		return BlockHash(hashVal)
	}), nil
}
