// Copyright 2026 Litix
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package anthropic

import (
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/anthropic/internal/converters"
)

// CustomMetadata keys under which responses from this model report Anthropic cache-write token counts.
// genai.GenerateContentResponseUsageMetadata has no cache-write field — the usage mapping folds cache-creation
// tokens into PromptTokenCount — so the per-class breakdown needed for cost accounting travels in
// model.LLMResponse.CustomMetadata under these keys; read them with CacheCreationInputTokensFromResponse.
const (
	// CacheCreationInputTokensKey holds the total input tokens written to cache.
	CacheCreationInputTokensKey = converters.CacheCreationInputTokensKey
	// CacheCreationEphemeral5mInputTokensKey holds the input tokens written to 5-minute-TTL cache entries.
	CacheCreationEphemeral5mInputTokensKey = converters.CacheCreationEphemeral5mInputTokensKey
	// CacheCreationEphemeral1hInputTokensKey holds the input tokens written to 1-hour-TTL cache entries.
	CacheCreationEphemeral1hInputTokensKey = converters.CacheCreationEphemeral1hInputTokensKey
)

// CacheCreationInputTokens is the cache-write portion of an Anthropic response's token usage, with the per-TTL
// breakdown. Cache writes are billed at a TTL-dependent premium over uncached input tokens, so cost accounting
// needs them separated from the PromptTokenCount total.
type CacheCreationInputTokens struct {
	// TotalInputTokens is the total input tokens written to cache across all TTLs.
	TotalInputTokens int64
	// Ephemeral5mInputTokens is the input tokens written to 5-minute-TTL cache entries.
	Ephemeral5mInputTokens int64
	// Ephemeral1hInputTokens is the input tokens written to 1-hour-TTL cache entries.
	Ephemeral1hInputTokens int64
}

// CacheCreationInputTokensFromResponse reports the cache-write token counts a response carries in its
// CustomMetadata, or ok=false when the response wrote nothing to cache (or did not come from this model).
// Values are accepted as int64 or float64: CustomMetadata that has round-tripped through a JSON-based session
// store comes back as float64.
func CacheCreationInputTokensFromResponse(resp *model.LLMResponse) (CacheCreationInputTokens, bool) {
	if resp == nil || resp.CustomMetadata == nil {
		return CacheCreationInputTokens{}, false
	}
	total, ok := metadataTokenCount(resp.CustomMetadata[CacheCreationInputTokensKey])
	if !ok {
		return CacheCreationInputTokens{}, false
	}
	ephemeral5m, _ := metadataTokenCount(resp.CustomMetadata[CacheCreationEphemeral5mInputTokensKey])
	ephemeral1h, _ := metadataTokenCount(resp.CustomMetadata[CacheCreationEphemeral1hInputTokensKey])
	return CacheCreationInputTokens{
		TotalInputTokens:       total,
		Ephemeral5mInputTokens: ephemeral5m,
		Ephemeral1hInputTokens: ephemeral1h,
	}, true
}

// metadataTokenCount converts a CustomMetadata value to a token count, accepting the int64 written by the
// converter and the float64 it becomes after a JSON round-trip.
func metadataTokenCount(value any) (int64, bool) {
	switch count := value.(type) {
	case int64:
		return count, true
	case float64:
		return int64(count), true
	default:
		return 0, false
	}
}
