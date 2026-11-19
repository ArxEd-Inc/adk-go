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

import (
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

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
// Unless ExplicitOnly is set, the API's implicit mode stays on: it writes one
// breakpoint at the end of the latest user message, tool-response group or
// initial developer group, which covers a conversation's tail on every
// request. The breakpoints here cover the two shared-prefix points implicit
// mode never writes. A request can write at most four breakpoints, the
// implicit one included, so the two here never exceed it.
type PromptCachingConfig struct {
	// ExplicitOnly sends every request in the API's explicit cache mode, in
	// which it places no implicit breakpoint and writes only the breakpoints
	// configured here. With no breakpoint configured, requests are not cached
	// at all: for a one-shot prompt nothing later reads, where a cache write
	// would only add its premium over the input price.
	ExplicitOnly bool

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

// DeferredLoadingTool is implemented by a request's dispatch entry (a value of
// model.LLMRequest.Tools, keyed by the tool's name) to keep the tool's
// declaration out of the request's tools until a function response names it
// (see the Responses converter's tool references). The signal rides on the
// dispatch entry rather than on the declaration, which is left as the tool
// built it.
type DeferredLoadingTool interface {
	// DeferLoading reports whether the tool's declaration is deferred.
	DeferLoading() bool
}

// IsToolLoadingDeferred reports whether the request's dispatch entry for the
// named tool implements DeferredLoadingTool and defers; nil-safe.
func IsToolLoadingDeferred(req *model.LLMRequest, name string) bool {
	if req == nil {
		return false
	}
	entry, ok := req.Tools[name].(DeferredLoadingTool)
	return ok && entry.DeferLoading()
}

// ToolReferenceNames returns the tool names a function response lists under
// key, reporting false when key is empty, absent from the response, or holds
// anything but a list of strings: a []string as a tool returns it, or a []any
// of strings as it reads back from a session store that round-trips the
// response through JSON.
func ToolReferenceNames(response map[string]any, key string) ([]string, bool) {
	if key == "" {
		return nil, false
	}
	switch value := response[key].(type) {
	case []string:
		return value, true
	case []any:
		names := make([]string, 0, len(value))
		for _, element := range value {
			name, ok := element.(string)
			if !ok {
				return nil, false
			}
			names = append(names, name)
		}
		return names, true
	default:
		return nil, false
	}
}

// FilesConfig configures sending a request's inline PDFs and images as Files
// API references instead of base64. Each distinct document (by content and
// MIME type) is uploaded once per process and API key, and reused while its
// file has at least ReuseMargin of life left; concurrent requests needing the
// same document share one upload. Upload state lives in memory only: another
// process uploads again, which costs the upload and nothing else. A document
// whose upload fails is sent inline, as it would be without this config.
type FilesConfig struct {
	// ExpiresIn is how long each uploaded file lives before the Files API
	// deletes it. Defaults to 24 hours when zero; must lie between one hour
	// and thirty days, the range the API accepts.
	ExpiresIn time.Duration

	// ReuseMargin is the least remaining life an uploaded file must have to be
	// referenced again, so a request never references a file that expires
	// before the request is served. Defaults to one hour when zero; must be
	// shorter than ExpiresIn.
	ReuseMargin time.Duration

	// MaxInlineBytes caps the raw bytes of media a request may still carry
	// inline after resolution: documents whose upload failed and media of a
	// type no reference stands in for. A request over the cap fails with an
	// error wrapping ErrRequestTooLarge before it is sent, rather than with a
	// body the API would refuse. Zero means no cap.
	MaxInlineBytes int
}
