// Copyright 2026 Google LLC
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

package openai

import (
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/openai/internal/responses"
)

// CustomMetadata keys under which a response from this package reports the
// input tokens it wrote to the prompt cache. genai's usage metadata has no
// cache-write field, and PromptTokenCount counts these tokens among every
// other input token, so cost accounting reads them here. Read them with
// CacheWriteInputTokensFromResponse.
const (
	// CacheWriteInputTokensKey holds the total input tokens written to cache.
	CacheWriteInputTokensKey = responses.CacheWriteInputTokensKey
	// CacheWrite30mInputTokensKey holds the input tokens billed as 30-minute
	// cache writes.
	CacheWrite30mInputTokensKey = responses.CacheWrite30mInputTokensKey
	// CacheWrite12hInputTokensKey holds the input tokens billed as 12-hour
	// cache writes.
	CacheWrite12hInputTokensKey = responses.CacheWrite12hInputTokensKey
)

// CacheWriteInputTokens is the cache-write portion of a response's input
// tokens, by TTL. Cache writes are billed at a premium over uncached input
// that grows with the TTL.
//
// The API reports one total per response, so a response is filed under the
// longest TTL any of its request's breakpoints asked for: exact when the
// breakpoints share a TTL, and never an undercount when they do not.
type CacheWriteInputTokens struct {
	// TotalInputTokens is the input tokens written to cache across TTLs.
	TotalInputTokens int64
	// TTL30mInputTokens is the input tokens billed as 30-minute writes.
	TTL30mInputTokens int64
	// TTL12hInputTokens is the input tokens billed as 12-hour writes.
	TTL12hInputTokens int64
}

// CacheWriteInputTokensFromResponse reports the cache-write token counts a
// response carries in its CustomMetadata, or ok=false when it wrote nothing to
// cache or did not come from this package. Values are accepted as int64 or as
// the float64 they become after a JSON round trip through a session store.
func CacheWriteInputTokensFromResponse(resp *model.LLMResponse) (CacheWriteInputTokens, bool) {
	if resp == nil || resp.CustomMetadata == nil {
		return CacheWriteInputTokens{}, false
	}
	total, ok := metadataTokenCount(resp.CustomMetadata[CacheWriteInputTokensKey])
	if !ok {
		return CacheWriteInputTokens{}, false
	}
	ttl30m, _ := metadataTokenCount(resp.CustomMetadata[CacheWrite30mInputTokensKey])
	ttl12h, _ := metadataTokenCount(resp.CustomMetadata[CacheWrite12hInputTokensKey])
	return CacheWriteInputTokens{TotalInputTokens: total, TTL30mInputTokens: ttl30m, TTL12hInputTokens: ttl12h}, true
}

// metadataTokenCount converts a CustomMetadata value to a token count.
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
