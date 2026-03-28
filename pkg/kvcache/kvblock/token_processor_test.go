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

package kvblock_test

import (
	"sync"
	"testing"

	"github.com/fxamacker/cbor/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
)

func TestNewChunkedTokenDatabase_Validation(t *testing.T) {
	tests := []struct {
		name      string
		config    *kvblock.TokenProcessorConfig
		wantErr   bool
		errSubstr string
	}{
		{
			name:      "BlockSize zero returns error",
			config:    &kvblock.TokenProcessorConfig{BlockSize: 0},
			wantErr:   true,
			errSubstr: "blockSize must be greater than 0, got 0",
		},
		{
			name:      "BlockSize negative returns error",
			config:    &kvblock.TokenProcessorConfig{BlockSize: -1},
			wantErr:   true,
			errSubstr: "blockSize must be greater than 0, got -1",
		},
		{
			name:    "BlockSize positive succeeds",
			config:  &kvblock.TokenProcessorConfig{BlockSize: 16},
			wantErr: false,
		},
		{
			name:    "nil config uses defaults and succeeds",
			config:  nil,
			wantErr: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			processor, err := kvblock.NewChunkedTokenDatabase(tc.config)
			if tc.wantErr {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.errSubstr)
				assert.Nil(t, processor)
			} else {
				require.NoError(t, err)
				assert.NotNil(t, processor)
			}
		})
	}
}

func TestGetInitHash_ConsistentHashesForSameModel(t *testing.T) {
	config := &kvblock.TokenProcessorConfig{
		BlockSize: 16,
		HashSeed:  "test-seed",
	}

	processor, err := kvblock.NewChunkedTokenDatabase(config)
	require.NoError(t, err)

	modelName := "test-model"
	tokens := []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16} // Full block
	chunkedMMHashes := [][]kvblock.MMHash{nil} // one nil entry per block

	// Get keys multiple times with no parent (should use init hash)
	keys1, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, modelName, chunkedMMHashes)
	require.NoError(t, err)
	keys2, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, modelName, chunkedMMHashes)
	require.NoError(t, err)
	keys3, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, modelName, chunkedMMHashes)
	require.NoError(t, err)

	require.NotEmpty(t, keys1, "Should generate keys")
	require.NotEmpty(t, keys2, "Should generate keys")
	require.NotEmpty(t, keys3, "Should generate keys")

	// All first keys should be identical (derived from same init hash)
	assert.Equal(t, keys1[0], keys2[0], "First key hash should be consistent across calls")
	assert.Equal(t, keys1[0], keys3[0], "First key hash should be consistent across calls")
	assert.NotEqual(t, keys1[0], kvblock.EmptyBlockHash, "Hash should not be zero")
}

func TestGetInitHash_DifferentHashesForDifferentModels(t *testing.T) {
	config := &kvblock.TokenProcessorConfig{
		BlockSize: 16,
		HashSeed:  "test-seed",
	}

	processor, err := kvblock.NewChunkedTokenDatabase(config)
	require.NoError(t, err)

	// Test different model names
	models := []string{
		"gpt-4",
		"llama-2-7b",
		"claude-3",
		"gemini-pro",
		"",  // empty string
		"a", // single character
		"very-long-model-name-with-special-characters-123!@#",
	}

	tokens := []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16} // Full block
	hashes := make(map[string]uint64)
	chunkedMMHashes := [][]kvblock.MMHash{nil}

	// Get first key hash for each model (derived from init hash)
	for _, modelName := range models {
		keys, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, modelName, chunkedMMHashes)
		require.NoError(t, err)
		require.NotEmpty(t, keys, "Should generate keys for model: %s", modelName)

		hashes[modelName] = uint64(keys[0])
		assert.NotZero(t, hashes[modelName], "Hash should not be zero for model: %s", modelName)
	}

	// Verify all hashes are different
	seenHashes := make(map[uint64]string)
	for modelName, hash := range hashes {
		if existingModel, exists := seenHashes[hash]; exists {
			t.Errorf("Hash collision detected: models '%s' and '%s' have the same initial key hash %d",
				modelName, existingModel, hash)
		}
		seenHashes[hash] = modelName
	}
}

func TestGetInitHash_DifferentSeedsProduceDifferentHashes(t *testing.T) {
	modelName := "test-model"
	tokens := []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	chunkedMMHashes := [][]kvblock.MMHash{nil}

	// Test with different seeds
	seeds := []string{
		"",
		"seed1",
		"seed2",
		"different-seed",
		"123456",
	}

	hashes := make(map[string]uint64)

	for _, seed := range seeds {
		config := &kvblock.TokenProcessorConfig{
			BlockSize: 16,
			HashSeed:  seed,
		}

		processor, err := kvblock.NewChunkedTokenDatabase(config)
		require.NoError(t, err)
		keys, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, modelName, chunkedMMHashes)
		require.NoError(t, err)
		require.NotEmpty(t, keys, "Should generate keys for seed: %s", seed)

		hashes[seed] = uint64(keys[0])
		assert.NotZero(t, hashes[seed], "Hash should not be zero for seed: %s", seed)
	}

	// Verify all hashes are different
	seenHashes := make(map[uint64]string)
	for seed, hash := range hashes {
		if existingSeed, exists := seenHashes[hash]; exists {
			t.Errorf("Hash collision detected: seeds '%s' and '%s' produce the same initial hash %d for model %s",
				seed, existingSeed, hash, modelName)
		}
		seenHashes[hash] = seed
	}
}

func TestGetInitHash_ConcurrentAccess(t *testing.T) {
	config := &kvblock.TokenProcessorConfig{
		BlockSize: 16,
		HashSeed:  "test-seed",
	}

	processor, err := kvblock.NewChunkedTokenDatabase(config)
	require.NoError(t, err)

	modelName := "concurrent-test-model"
	tokens := []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	chunkedMMHashes := [][]kvblock.MMHash{nil}
	numGoroutines := 100

	// Channel to collect results
	results := make(chan uint64, numGoroutines)
	var wg sync.WaitGroup

	// Start multiple goroutines calling TokensToKVBlockKeys (which calls getInitHash)
	for range numGoroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			keys, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, modelName, chunkedMMHashes)
			if err == nil && len(keys) > 0 {
				results <- uint64(keys[0])
			}
		}()
	}

	wg.Wait()
	close(results)

	// Collect all results
	hashes := make([]uint64, 0, numGoroutines)
	for hash := range results {
		hashes = append(hashes, hash)
	}

	require.Len(t, hashes, numGoroutines, "Should have received hash from all goroutines")

	// Verify all hashes are identical
	expectedHash := hashes[0]
	for i, hash := range hashes {
		assert.Equal(t, expectedHash, hash, "Hash mismatch at index %d", i)
	}

	assert.NotZero(t, expectedHash, "Hash should not be zero")
}

func TestGetInitHash_Deterministic(t *testing.T) {
	// Test that the same configuration always produces the same hash
	modelName := "deterministic-test"
	seed := "deterministic-seed"
	tokens := []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	chunkedMMHashes := [][]kvblock.MMHash{nil}

	var hashes []uint64

	// Create multiple instances with same config
	for i := 0; i < 5; i++ {
		config := &kvblock.TokenProcessorConfig{
			BlockSize: 16,
			HashSeed:  seed,
		}

		processor, err := kvblock.NewChunkedTokenDatabase(config)
		require.NoError(t, err)
		keys, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, modelName, chunkedMMHashes)
		require.NoError(t, err)
		require.NotEmpty(t, keys, "Should generate keys for instance %d", i)

		hashes = append(hashes, uint64(keys[0]))
	}

	// All instances should produce the same hash
	expectedHash := hashes[0]
	for i, hash := range hashes {
		assert.Equal(t, expectedHash, hash, "Hash should be deterministic across instances, mismatch at index %d", i)
	}

	assert.NotZero(t, expectedHash, "Hash should not be zero")
}

func TestHash_ExtraCBORDeterminism(t *testing.T) {
	// Create a canonical CBOR encoder
	// This ensures deterministic encoding (same value -> same bytes, always)
	encMode, err := cbor.CanonicalEncOptions().EncMode()
	require.NoError(t, err)

	testCases := []struct {
		name  string
		extra interface{}
	}{
		{"nil", nil},
		{"int_zero", 0},
		{"int_positive", 42},
		{"int_negative", -10},
		{"string_empty", ""},
		{"string_medium", "gpu"},
		{"string_long", "very-long-adapter-name-with-special-chars-123"},
		{"map_empty", map[string]interface{}{}},
		{"map_lora_only", map[string]interface{}{"lora_id": 42}},
		{"map_combined", map[string]interface{}{"lora_id": 42, "medium": "gpu"}},
		{"map_nested", map[string]interface{}{"meta": map[string]interface{}{"version": 1}}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Encode the same value 5 times
			var encodings [][]byte
			for i := 0; i < 5; i++ {
				bytes, err := encMode.Marshal(tc.extra)
				require.NoError(t, err)
				encodings = append(encodings, bytes)
			}

			// All encodings must be identical
			for i := 1; i < len(encodings); i++ {
				assert.Equal(t, encodings[0], encodings[i],
					"Encoding %d differs from encoding 0", i)
			}
		})
	}
}

func TestHash_ExtraMapKeyOrdering(t *testing.T) {
	// Create a canonical CBOR encoder
	encMode, err := cbor.CanonicalEncOptions().EncMode()
	require.NoError(t, err)

	testCases := []struct {
		name string
		maps []map[string]interface{}
		desc string
	}{
		{
			name: "two_keys_different_order",
			maps: []map[string]interface{}{
				{"lora_id": 42, "medium": "gpu"},
				{"medium": "gpu", "lora_id": 42},
			},
			desc: "Same keys inserted in different order",
		},
		{
			name: "three_keys_different_order",
			maps: []map[string]interface{}{
				{"lora_id": 42, "medium": "gpu", "version": 3},
				{"version": 3, "medium": "gpu", "lora_id": 42},
				{"medium": "gpu", "version": 3, "lora_id": 42},
			},
			desc: "Three keys with different permutations",
		},
		{
			name: "nested_maps",
			maps: []map[string]interface{}{
				{"outer": map[string]interface{}{"lora_id": 42, "medium": "gpu"}},
				{"outer": map[string]interface{}{"medium": "gpu", "lora_id": 42}},
			},
			desc: "Nested maps with different key order",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var encodings [][]byte

			// Encode each map
			for _, m := range tc.maps {
				bytes, err := encMode.Marshal(m)
				assert.NoError(t, err)
				encodings = append(encodings, bytes)
			}

			// All encodings must be identical
			expected := encodings[0]
			for i := 1; i < len(encodings); i++ {
				assert.Equal(t, expected, encodings[i],
					"Map %d encoding differs from map 0: %s", i, tc.desc)
			}
		})
	}
}

func TestHash_ExtraDifferentiation(t *testing.T) {
	// Create a canonical CBOR encoder
	encMode, err := cbor.CanonicalEncOptions().EncMode()
	require.NoError(t, err)

	testCases := []struct {
		name   string
		extra1 interface{}
		extra2 interface{}
		desc   string
	}{
		{
			name:   "nil_vs_zero",
			extra1: nil,
			extra2: 0,
			desc:   "nil should differ from zero",
		},
		{
			name:   "different_ints",
			extra1: 42,
			extra2: 99,
			desc:   "Different LoRA IDs",
		},
		{
			name:   "different_strings",
			extra1: "gpu",
			extra2: "cpu",
			desc:   "Different medium IDs",
		},
		{
			name:   "string_vs_int",
			extra1: "42",
			extra2: 42,
			desc:   "String vs int, type matters",
		},
		{
			name:   "map_different_values",
			extra1: map[string]interface{}{"lora_id": 42},
			extra2: map[string]interface{}{"lora_id": 99},
			desc:   "Maps with different values",
		},
		{
			name:   "map_different_keys",
			extra1: map[string]interface{}{"lora_id": 42},
			extra2: map[string]interface{}{"lora_adapter": 42},
			desc:   "Maps with different values but same values",
		},
		{
			name:   "map_vs_nil",
			extra1: map[string]interface{}{"lora_id": 42},
			extra2: nil,
			desc:   "Maps with LoRA ID vs nil (no LoRA ID)",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			bytes1, err := encMode.Marshal(tc.extra1)
			require.NoError(t, err)

			bytes2, err := encMode.Marshal(tc.extra2)
			require.NoError(t, err)

			// These must be different
			assert.NotEqual(t, bytes1, bytes2,
				"CBOR encodings should differ: %s", tc.desc)
		})
	}
}

func TestHash_ExtraVLLMCompatibility(t *testing.T) {
	encMode, err := cbor.CanonicalEncOptions().EncMode()
	require.NoError(t, err)

	testCases := []struct {
		name     string
		extra    interface{}
		scenario string
	}{
		{
			name:     "no_lora_no_multimodal",
			extra:    nil,
			scenario: "Standard text-only prompt without LoRA adapter",
		},
		{
			name:     "lora_v0_single_adapter",
			extra:    42,
			scenario: "vLLM v0: single LoRA adapter with hash(lora_int_id)",
		},
		{
			name:     "lora_v1_simple_tuple",
			extra:    map[string]interface{}{"lora_id": 42, "mm_hash": nil, "cache_salt": nil},
			scenario: "vLLM v1: LoRA only (lora_id, mm_hash=None, cache_salt=None)",
		},
		{
			name:     "lora_v1_with_multimodal",
			extra:    map[string]interface{}{"lora_id": 42, "mm_hash": "blake3_abc123", "cache_salt": "xyz"},
			scenario: "vLLM v1: LoRA + multi-modal content with Blake3 hash",
		},
		{
			name:     "medium_identifier",
			extra:    "gpu",
			scenario: "Custom medium identifier for cache segmentation",
		},
		{
			name:     "structured_metadata",
			extra:    map[string]interface{}{"lora_id": 42, "medium": "gpu", "version": 1},
			scenario: "Complex metadata combining multiple differentiation factors",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			bytes, err := encMode.Marshal(tc.extra)
			require.NoError(t, err, "Should successfully encode: %s", tc.scenario)
			assert.NotEmpty(t, bytes, "Encoded bytes should not be empty: %s", tc.scenario)
		})
	}
}

func TestHash_ExtraTypeSupport(t *testing.T) {
	encMode, err := cbor.CanonicalEncOptions().EncMode()
	require.NoError(t, err)

	testCases := []struct {
		name      string
		extra     interface{}
		shouldErr bool
	}{
		// Supported types that must work
		{"nil", nil, false},
		{"int", 42, false},
		{"int64", int64(9223372036854775807), false},
		{"string", "adapter-name", false},
		{"map_string_int", map[string]interface{}{"id": 42}, false},
		{"map_string_string", map[string]interface{}{"name": "lora"}, false},
		{"map_mixed", map[string]interface{}{"id": 42, "name": "lora"}, false},
		{"bool", true, false},
		{"float", 3.14, false},
		{"slice_int", []interface{}{1, 2, 3}, false},
		{"nested_map", map[string]interface{}{"meta": map[string]interface{}{"v": 1}}, false},

		// Edge cases that should still work
		{"empty_string", "", false},
		{"empty_map", map[string]interface{}{}, false},
		{"zero", 0, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			bytes, err := encMode.Marshal(tc.extra)

			if tc.shouldErr {
				assert.Error(t, err, "Expected encoding to fail")
			} else {
				require.NoError(t, err, "Expected encoding to succeed")
				assert.NotEmpty(t, bytes, "Encoded bytes should not be empty")
			}
		})
	}
}

func TestTokensToKVBlockKeys_ExtraKeysAffectHash(t *testing.T) {
	config := &kvblock.TokenProcessorConfig{
		BlockSize: 16,
		HashSeed:  "test-seed",
	}

	processor, err := kvblock.NewChunkedTokenDatabase(config)
	require.NoError(t, err)

	modelName := "test-model"
	tokens := []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16} // full block

	// nil chunked MM hashes (no multi-modal content)
	nilChunkedMMHashes := [][]kvblock.MMHash{nil}
	keysNil, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, modelName, nilChunkedMMHashes)
	require.NoError(t, err)
	require.Len(t, keysNil, 1)

	// non-nil chunked MM hashes with multi-modal content
	withChunkedMMHashes := [][]kvblock.MMHash{
		{{Hash: "abc123", Offset: 0}},
	}
	keysExtra, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, modelName, withChunkedMMHashes)
	require.NoError(t, err)
	require.Len(t, keysExtra, 1)

	// different chunkedMMHashes should produce different hashes
	assert.NotEqual(t, keysNil[0], keysExtra[0], "chunkedMMHashes should affect the resulting hash")

	// same chunkedMMHashes should produce the same hash (deterministic)
	keysExtra2, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, modelName, withChunkedMMHashes)
	require.NoError(t, err)
	assert.Equal(t, keysExtra[0], keysExtra2[0], "same chunkedMMHashes should produce the same hash")
}

func TestTokensToKVBlockKeys_ExtraKeysMismatchReturnsError(t *testing.T) {
	config := &kvblock.TokenProcessorConfig{
		BlockSize: 16,
		HashSeed:  "test-seed",
	}

	processor, err := kvblock.NewChunkedTokenDatabase(config)
	require.NoError(t, err)

	modelName := "test-model"
	// two full blocks → 2 chunks
	tokens := make([]uint32, 32)
	for i := range tokens {
		tokens[i] = uint32(i + 1)
	}

	// chunkedMMHashes length (1) does not match chunks length (2) — should error
	mismatchedChunkedMMHashes := [][]kvblock.MMHash{nil}
	keys, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, modelName, mismatchedChunkedMMHashes)
	require.Error(t, err, "mismatched chunkedMMHashes length should return an error")
	assert.Contains(t, err.Error(), "does not match token chunk count")
	assert.Nil(t, keys)
}

// TestTokensToKVBlockKeys_MixedNilAndMultiModalExtraKeys mirrors real vLLM BlockStored events
// where initial text blocks carry nil extra_keys and later image blocks carry multi-modal extra_keys.
// The real format per block: [(['image_hash', offset],)] — one entry per image region in the block.
func TestTokensToKVBlockKeys_MixedNilAndMultiModalExtraKeys(t *testing.T) {
	config := &kvblock.TokenProcessorConfig{
		BlockSize: 16,
		HashSeed:  "",
	}

	processor, err := kvblock.NewChunkedTokenDatabase(config)
	require.NoError(t, err)

	modelName := "Qwen/Qwen2.5-VL-7B-Instruct"

	// 4 blocks: 2 pure-text (nil), 2 multi-modal (same image hash, offsets 3 and -13).
	// Offset pattern matches real data: first occurrence is positive (3),
	// subsequent occurrences decrease by block-size (3 - 16 = -13).
	const imageHash = "6ab3a7d0570817f1a4e9adaeda325c07c2466b252279a633ee2995cdba59ab25"
	tokens := make([]uint32, 64) // 4 × 16 tokens
	for i := range tokens {
		tokens[i] = uint32(i + 1)
	}

	chunkedMMHashes := [][]kvblock.MMHash{
		nil,                          // block 0: text only
		nil,                          // block 1: text only
		{{Hash: imageHash, Offset: 3}},   // block 2: first image region
		{{Hash: imageHash, Offset: -13}}, // block 3: second image region
	}

	keys, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, modelName, chunkedMMHashes)
	require.NoError(t, err)
	require.Len(t, keys, 4)

	// All four block hashes must be distinct (prefix chain + extra_keys both contribute).
	seen := make(map[kvblock.BlockHash]int)
	for i, k := range keys {
		if prev, exists := seen[k]; exists {
			t.Errorf("blocks %d and %d produced the same hash %d", prev, i, k)
		}
		seen[k] = i
	}

	// The two multi-modal blocks (same image hash, different offsets) must differ.
	assert.NotEqual(t, keys[2], keys[3],
		"blocks with same image hash but different offsets must produce different hashes")

	// Determinism: same inputs must always yield the same output.
	keys2, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, modelName, chunkedMMHashes)
	require.NoError(t, err)
	assert.Equal(t, keys, keys2, "TokensToKVBlockKeys must be deterministic")
}

// TestTokensToKVBlockKeys_SameImageHashDifferentOffsetsDiffer checks that two otherwise-identical
// blocks whose only difference is the multi-modal offset produce distinct hashes.
// In real vLLM events the offset decreases by block_size for each successive image block:
// first block offset=3, next offset=-13 (= 3 - 16), then -29, etc.
func TestTokensToKVBlockKeys_SameImageHashDifferentOffsetsDiffer(t *testing.T) {
	config := &kvblock.TokenProcessorConfig{
		BlockSize: 16,
		HashSeed:  "",
	}

	processor, err := kvblock.NewChunkedTokenDatabase(config)
	require.NoError(t, err)

	modelName := "Qwen/Qwen2.5-VL-7B-Instruct"
	const imageHash = "6ab3a7d0570817f1a4e9adaeda325c07c2466b252279a633ee2995cdba59ab25"
	tokens := []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}

	offsets := []int64{3, -13, -29, -45, -61}
	hashes := make([]kvblock.BlockHash, len(offsets))
	for i, offset := range offsets {
		ek := [][]kvblock.MMHash{
			{{Hash: imageHash, Offset: offset}},
		}
		keys, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, modelName, ek)
		require.NoError(t, err)
		require.Len(t, keys, 1)
		hashes[i] = keys[0]
	}

	seen := make(map[kvblock.BlockHash]int64)
	for i, h := range hashes {
		if prevOffset, exists := seen[h]; exists {
			t.Errorf("offsets %d and %d produced the same block hash — offset must be part of the hash input",
				prevOffset, offsets[i])
		}
		seen[h] = offsets[i]
	}
}

// mmPlaceholderConverter exposes MMPlaceHolderToChunkedMMHashes via type assertion
// since chunkedTokenDatabase is unexported.
type mmPlaceholderConverter interface {
	MMPlaceHolderToChunkedMMHashes(tokens []uint32, input []kvblock.MMHashWithPlaceholders) [][]kvblock.MMHash
}

func TestMMPlaceHolderToChunkedMMHashes(t *testing.T) {
	db, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{BlockSize: 16})
	require.NoError(t, err)
	conv := db.(mmPlaceholderConverter)

	const imageHash1 = "6ab3a7d0570817f1a4e9adaeda325c07c2466b252279a633ee2995cdba59ab25"
	const imageHash2 = "e950785918bdef0f88ec349d3f65a2ed0b1d448c854333ea1e71bfedce1fe252"

	// image1: offset=51, length=952 → ends at 1002.
	// firstBlock = 51/16 = 3, so blocks 0-2 are nil.
	// Covers blocks 3 (offset 3) through 62 (offset -941). Block 63 is nil.
	//
	// image2: offset=1030, length=850 → ends at 1879.
	// Covers blocks 64 (offset 6) through 117 (offset -842). Block 118 is nil.
	//
	// 2432 tokens → 152 blocks.
	tokens := make([]uint32, 2432)
	input := []kvblock.MMHashWithPlaceholders{
		{Hash: imageHash1, Offset: 51, Length: 952},
		{Hash: imageHash2, Offset: 1030, Length: 850},
	}

	result := conv.MMPlaceHolderToChunkedMMHashes(tokens, input)
	require.Len(t, result, 152)
	// Blocks 0-2: text only → nil (51/16 = 3, so image1 starts at block 3)
	assert.Nil(t, result[0])
	assert.Nil(t, result[1])
	assert.Nil(t, result[2], "image1 starts at block 3, block 2 must be nil")

	// Block 3 [48,64): image1 starts here, relative offset = 51 - 48 = 3
	require.Len(t, result[3], 1)
	assert.Equal(t, kvblock.MMHash{Hash: imageHash1, Offset: 3}, result[3][0])

	// Block 4 [64,80): relative offset = 51 - 64 = -13
	require.Len(t, result[4], 1)
	assert.Equal(t, kvblock.MMHash{Hash: imageHash1, Offset: -13}, result[4][0])

	// Block 62 [992,1008): last block of image1, relative offset = 51 - 992 = -941
	require.Len(t, result[62], 1)
	assert.Equal(t, kvblock.MMHash{Hash: imageHash1, Offset: -941}, result[62][0])

	// Block 63 [1008,1024): image1 ends at 1002, so block 63 is nil
	assert.Nil(t, result[63])

	// Block 64 [1024,1040): image2 starts here, relative offset = 1030 - 1024 = 6
	require.Len(t, result[64], 1)
	assert.Equal(t, kvblock.MMHash{Hash: imageHash2, Offset: 6}, result[64][0])

	// Block 65 [1040,1056): relative offset = 1030 - 1040 = -10
	require.Len(t, result[65], 1)
	assert.Equal(t, kvblock.MMHash{Hash: imageHash2, Offset: -10}, result[65][0])

	// Block 117 [1872,1888): last block of image2, relative offset = 1030 - 1872 = -842
	require.Len(t, result[117], 1)
	assert.Equal(t, kvblock.MMHash{Hash: imageHash2, Offset: -842}, result[117][0])

	// Block 118 [1888,1904): image2 ends at 1879, so block 118 is nil
	assert.Nil(t, result[118])
}

// TestMMPlaceHolderToChunkedMMHashes_EmptyInput verifies that passing no images
// produces a slice of the correct length with all nil entries.
func TestMMPlaceHolderToChunkedMMHashes_EmptyInput(t *testing.T) {
	db, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{BlockSize: 16})
	require.NoError(t, err)
	conv := db.(mmPlaceholderConverter)

	// 32 tokens → 2 blocks, no images
	tokens := make([]uint32, 32)
	result := conv.MMPlaceHolderToChunkedMMHashes(tokens, nil)

	require.Len(t, result, 2)
	assert.Nil(t, result[0])
	assert.Nil(t, result[1])
}

// TestMMPlaceHolderToChunkedMMHashes_ExactBlockBoundaryStart verifies that an image
// whose offset lands exactly on a block boundary gets relative offset 0 in that block,
// and the preceding block remains nil.
func TestMMPlaceHolderToChunkedMMHashes_ExactBlockBoundaryStart(t *testing.T) {
	db, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{BlockSize: 16})
	require.NoError(t, err)
	conv := db.(mmPlaceholderConverter)

	const imageHash = "aabbcc"
	// 64 tokens → 4 blocks. image: offset=32 (= 2×16), length=16 → ends at 47.
	// firstBlock = 32/16 = 2, lastBlock = 47/16 = 2 → only block 2.
	tokens := make([]uint32, 64)
	input := []kvblock.MMHashWithPlaceholders{
		{Hash: imageHash, Offset: 32, Length: 16},
	}

	result := conv.MMPlaceHolderToChunkedMMHashes(tokens, input)
	require.Len(t, result, 4)

	assert.Nil(t, result[0])
	assert.Nil(t, result[1], "block 1 ends at 32 which is the image start, must be nil")

	// Block 2 [32,48): relative offset = 32 - 32 = 0
	require.Len(t, result[2], 1)
	assert.Equal(t, kvblock.MMHash{Hash: imageHash, Offset: 0}, result[2][0])

	assert.Nil(t, result[3])
}

// TestMMPlaceHolderToChunkedMMHashes_SingleBlockImage verifies that an image
// entirely within one block appears only in that block.
func TestMMPlaceHolderToChunkedMMHashes_SingleBlockImage(t *testing.T) {
	db, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{BlockSize: 16})
	require.NoError(t, err)
	conv := db.(mmPlaceholderConverter)

	const imageHash = "ddeeff"
	// 32 tokens → 2 blocks. image: offset=5, length=8 → ends at 12.
	// firstBlock = 5/16 = 0, lastBlock = 12/16 = 0 → only block 0.
	tokens := make([]uint32, 32)
	input := []kvblock.MMHashWithPlaceholders{
		{Hash: imageHash, Offset: 5, Length: 8},
	}

	result := conv.MMPlaceHolderToChunkedMMHashes(tokens, input)
	require.Len(t, result, 2)

	// Block 0 [0,16): relative offset = 5 - 0 = 5
	require.Len(t, result[0], 1)
	assert.Equal(t, kvblock.MMHash{Hash: imageHash, Offset: 5}, result[0][0])

	assert.Nil(t, result[1])
}

// TestMMPlaceHolderToChunkedMMHashes_TwoImagesInSameBlock verifies that when two
// images both fall within the same block, the block accumulates two MMHash entries.
func TestMMPlaceHolderToChunkedMMHashes_TwoImagesInSameBlock(t *testing.T) {
	db, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{BlockSize: 16})
	require.NoError(t, err)
	conv := db.(mmPlaceholderConverter)

	const hash1 = "img1"
	const hash2 = "img2"
	// 32 tokens → 2 blocks.
	// image1: offset=2, length=5 → ends at 6  → only block 0, relative offset = 2
	// image2: offset=9, length=4 → ends at 12 → only block 0, relative offset = 9
	tokens := make([]uint32, 32)
	input := []kvblock.MMHashWithPlaceholders{
		{Hash: hash1, Offset: 2, Length: 5},
		{Hash: hash2, Offset: 9, Length: 4},
	}

	result := conv.MMPlaceHolderToChunkedMMHashes(tokens, input)
	require.Len(t, result, 2)

	require.Len(t, result[0], 2, "both images fall in block 0")
	assert.Equal(t, kvblock.MMHash{Hash: hash1, Offset: 2}, result[0][0])
	assert.Equal(t, kvblock.MMHash{Hash: hash2, Offset: 9}, result[0][1])

	assert.Nil(t, result[1])
}

// TestMMPlaceHolderToChunkedMMHashes_ImageStartsAtTokenZero verifies correct relative
// offsets when an image begins at the very first token.
func TestMMPlaceHolderToChunkedMMHashes_ImageStartsAtTokenZero(t *testing.T) {
	db, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{BlockSize: 16})
	require.NoError(t, err)
	conv := db.(mmPlaceholderConverter)

	const imageHash = "zero-start"
	// 48 tokens → 3 blocks. image: offset=0, length=20 → ends at 19.
	// firstBlock=0, lastBlock=1.
	tokens := make([]uint32, 48)
	input := []kvblock.MMHashWithPlaceholders{
		{Hash: imageHash, Offset: 0, Length: 20},
	}

	result := conv.MMPlaceHolderToChunkedMMHashes(tokens, input)
	require.Len(t, result, 3)

	// Block 0 [0,16): relative offset = 0 - 0 = 0
	require.Len(t, result[0], 1)
	assert.Equal(t, kvblock.MMHash{Hash: imageHash, Offset: 0}, result[0][0])

	// Block 1 [16,32): relative offset = 0 - 16 = -16
	require.Len(t, result[1], 1)
	assert.Equal(t, kvblock.MMHash{Hash: imageHash, Offset: -16}, result[1][0])

	// Block 2: image ends at 19, so block 2 [32,48) is nil
	assert.Nil(t, result[2])
}

// TestMMPlaceHolderToChunkedMMHashes_PartialLastBlockExcluded verifies that tokens
// not divisible by blockSize produce no entry for the partial trailing block.
func TestMMPlaceHolderToChunkedMMHashes_PartialLastBlockExcluded(t *testing.T) {
	db, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{BlockSize: 16})
	require.NoError(t, err)
	conv := db.(mmPlaceholderConverter)

	const imageHash = "partial"
	// 25 tokens → numBlocks = 25/16 = 1 (the trailing 9 tokens are a partial block).
	// image covers tokens 0-19, but only block 0 [0,16) is a full block.
	tokens := make([]uint32, 25)
	input := []kvblock.MMHashWithPlaceholders{
		{Hash: imageHash, Offset: 0, Length: 20},
	}

	result := conv.MMPlaceHolderToChunkedMMHashes(tokens, input)
	require.Len(t, result, 1, "partial trailing block must not produce an entry")

	// Block 0 [0,16): relative offset = 0
	require.Len(t, result[0], 1)
	assert.Equal(t, kvblock.MMHash{Hash: imageHash, Offset: 0}, result[0][0])
}

// TestMMPlaceHolderToChunkedMMHashes_MatchesPoolPath verifies that the placeholder
// conversion path produces identical TokensToKVBlockKeys output as the pool.go path,
// where ExtraKeys are already chunked per block with block-relative offsets.
func TestMMPlaceHolderToChunkedMMHashes_MatchesPoolPath(t *testing.T) {
	db, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{BlockSize: 16})
	require.NoError(t, err)
	conv := db.(mmPlaceholderConverter)

	const imageHash = "6ab3a7d0570817f1a4e9adaeda325c07c2466b252279a633ee2995cdba59ab25"
	modelName := "Qwen/Qwen2.5-VL-7B-Instruct"

	// image starts at token 51, spans 32 tokens → covers blocks 3 [48,64) and 4 [64,80).
	// 5 blocks total (80 tokens).
	tokens := make([]uint32, 80)
	for i := range tokens {
		tokens[i] = uint32(i + 1)
	}

	// Path A: placeholder → chunked (what a client calling MMPlaceHolderToChunkedMMHashes would do).
	placeholders := []kvblock.MMHashWithPlaceholders{
		{Hash: imageHash, Offset: 51, Length: 32},
	}
	chunkedViaPlaceholder := conv.MMPlaceHolderToChunkedMMHashes(tokens, placeholders)

	// Path B: manually built per-block, mirroring how pool.go reads ExtraKeys from the engine.
	// vLLM reports block-relative offsets: block 3 offset=3 (51-48), block 4 offset=-13 (51-64).
	chunkedViaPoolPath := [][]kvblock.MMHash{
		nil,                            // block 0: text only
		nil,                            // block 1: text only
		nil,                            // block 2: text only
		{{Hash: imageHash, Offset: 3}}, // block 3: image starts 3 tokens in
		{{Hash: imageHash, Offset: -13}}, // block 4: image started 13 tokens before this block
	}

	require.Equal(t, chunkedViaPoolPath, chunkedViaPlaceholder,
		"MMPlaceHolderToChunkedMMHashes must produce the same chunked structure as pool.go")

	// Both must also yield identical block hashes from TokensToKVBlockKeys.
	keysA, err := db.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, modelName, chunkedViaPlaceholder)
	require.NoError(t, err)

	keysB, err := db.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, modelName, chunkedViaPoolPath)
	require.NoError(t, err)

	assert.Equal(t, keysB, keysA, "both paths must produce identical block hashes")
}