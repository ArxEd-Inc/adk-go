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
	oaishared "github.com/openai/openai-go/v3/shared"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model/openai/internal/shared"
)

// Effort is the reasoning effort a model spends on each request (the
// Responses API's reasoning.effort). It is a type alias so callers configure
// effort without importing openai-go. Not every model accepts every level.
type Effort = oaishared.ReasoningEffort

// Effort levels, in increasing order of reasoning depth and token spend.
const (
	EffortNone    Effort = oaishared.ReasoningEffortNone
	EffortMinimal Effort = oaishared.ReasoningEffortMinimal
	EffortLow     Effort = oaishared.ReasoningEffortLow
	EffortMedium  Effort = oaishared.ReasoningEffortMedium
	EffortHigh    Effort = oaishared.ReasoningEffortHigh
	EffortXHigh   Effort = oaishared.ReasoningEffortXhigh
	EffortMax     Effort = oaishared.ReasoningEffortMax
)

// CacheTTL is the minimum lifetime of a prompt cache breakpoint.
type CacheTTL = shared.CacheTTL

// Cache TTLs. A cache write is billed at a premium that grows with the TTL;
// reads cost the same whichever TTL wrote the entry.
const (
	CacheTTL30m = shared.CacheTTL30m
	CacheTTL12h = shared.CacheTTL12h
)

// CacheBreakpoint configures one explicit prompt cache breakpoint. Its TTL,
// when empty, inherits the request's, which is 30 minutes.
type CacheBreakpoint = shared.CacheBreakpoint

// PromptCachingConfig places explicit prompt cache breakpoints on top of the
// API's implicit one, which ends at the conversation's tail on every request.
// See the fields for where each goes.
type PromptCachingConfig = shared.PromptCachingConfig

// MarkCacheBreakpoint marks part as the end of a prefix several conversations
// share, for the PromptCachingConfig.MarkedPart breakpoint. The mark rides on
// the part's PartMetadata, in memory only.
func MarkCacheBreakpoint(part *genai.Part) {
	if part == nil {
		return
	}
	if part.PartMetadata == nil {
		part.PartMetadata = map[string]any{}
	}
	part.PartMetadata[shared.PartCacheBreakpointMetadataKey] = true
}

// IsCacheBreakpointMarked reports whether part was marked with
// MarkCacheBreakpoint.
func IsCacheBreakpointMarked(part *genai.Part) bool {
	return shared.IsCacheBreakpointMarked(part)
}
