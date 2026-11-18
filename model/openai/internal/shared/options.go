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

package shared

import "google.golang.org/genai"

// CacheTTL is the minimum lifetime of a prompt cache breakpoint, as the
// Responses API's prompt_cache_breakpoint.ttl takes it.
type CacheTTL string

const (
	// CacheTTL30m keeps a breakpoint for at least 30 minutes, refreshed by
	// each read. It is the API's default.
	CacheTTL30m CacheTTL = "30m"
	// CacheTTL12h keeps a breakpoint for at least 12 hours.
	CacheTTL12h CacheTTL = "12h"
)

// CacheBreakpoint configures one explicit prompt cache breakpoint.
type CacheBreakpoint struct {
	// TTL is the breakpoint's minimum lifetime. Empty inherits the request's
	// TTL, which is 30 minutes.
	TTL CacheTTL
}

// PromptCachingConfig places explicit prompt cache breakpoints.
//
// The API's implicit mode stays on: it writes one breakpoint at the end of
// the latest user message, tool-response group or initial developer group,
// which covers a conversation's tail on every request. The breakpoints here
// cover the two shared-prefix points implicit mode never writes. A request can
// write at most four breakpoints, the implicit one included, so the two here
// never exceed it.
type PromptCachingConfig struct {
	// SystemInstruction places a breakpoint at the end of the system
	// instruction, caching the tools and the instruction for every
	// conversation that shares them. The API takes no breakpoint on its
	// top-level instructions field, so when this is set the system instruction
	// is sent as the request's first input item, a developer message, which
	// can carry one.
	SystemInstruction *CacheBreakpoint

	// MarkedPart places a breakpoint on the content block converted from the
	// last part marked with PartCacheBreakpointMetadataKey, so a prefix several
	// conversations share, such as one document set many conversations open
	// with, is cached once for all of them. A request with no marked part
	// places nothing.
	MarkedPart *CacheBreakpoint
}

// PartCacheBreakpointMetadataKey is the genai.Part.PartMetadata key that marks
// a part as the end of a shared prefix for PromptCachingConfig.MarkedPart.
const PartCacheBreakpointMetadataKey = "openai_cache_breakpoint"

// IsCacheBreakpointMarked reports whether part carries a true
// PartCacheBreakpointMetadataKey.
func IsCacheBreakpointMarked(part *genai.Part) bool {
	if part == nil || part.PartMetadata == nil {
		return false
	}
	marked, ok := part.PartMetadata[PartCacheBreakpointMetadataKey].(bool)
	return ok && marked
}
