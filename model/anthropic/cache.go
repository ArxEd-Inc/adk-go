// Copyright 2025 Alcova AI
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
//
// Modified by Litix from github.com/Alcova-AI/adk-anthropic-go (v0.1.18).

package anthropic

import (
	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model/anthropic/internal/converters"
)

// MarkCacheBreakpoint marks the given part as a prompt-cache breakpoint
// candidate for the PromptCachingConfig.MarkedPart breakpoint (see that field
// for when a marker is placed on it). It sets
// converters.PartCacheBreakpointMetadataKey in the part's PartMetadata,
// allocating the map when nil; a nil part is left alone.
func MarkCacheBreakpoint(part *genai.Part) {
	if part == nil {
		return
	}
	if part.PartMetadata == nil {
		part.PartMetadata = map[string]any{}
	}
	part.PartMetadata[converters.PartCacheBreakpointMetadataKey] = true
}

// IsCacheBreakpointMarked reports whether the given part was marked with
// MarkCacheBreakpoint; nil-safe. Exported so callers that key on the marker —
// a gate serializing concurrent writes of the shared prefix, say — need not
// know the metadata key.
func IsCacheBreakpointMarked(part *genai.Part) bool {
	return converters.IsCacheBreakpointMarked(part)
}

// applyCacheBreakpoints sets cache_control breakpoints on the request based
// on the provided configuration. Each breakpoint is independently optional.
// markedBlockOrdinals are the flat ordinals of the request's marked content
// blocks, as converters.ContentsToMessagesWithMarkedBlocks reports them for
// params.Messages; nil when no part is marked.
//
// Anthropic evaluates cache prefixes in order: tools → system → messages.
// When mixing TTLs, longer TTLs must appear before shorter TTLs in this order.
//
// The conversation-history breakpoint targets the newest message so that large
// fresh content (e.g. a big tool result) is written to the cache on the call
// where it first appears (a cache write costs 1.25x base input at the default
// 5m TTL, 2x at 1h) instead of being billed at full input price once and only
// cached on the following call. The write premium is wasted only when the
// conversation ends at that message, which agentic loops rarely do.
func applyCacheBreakpoints(
	params *anthropicsdk.MessageNewParams,
	cfg *PromptCachingConfig,
	markedBlockOrdinals []int,
) {
	// 1. Tools — end of the static prefix, on the named tool definition
	if cfg.ToolsStaticPrefixEnd != nil && cfg.ToolsStaticPrefixEndToolName != "" {
		for i := range params.Tools {
			if tool := params.Tools[i].OfTool; tool != nil && tool.Name == cfg.ToolsStaticPrefixEndToolName {
				tool.CacheControl = newCacheControl(cfg.ToolsStaticPrefixEnd)
				break
			}
		}
	}

	// 2. Tools — last tool definition. When the last tool is also the named
	// static-prefix tool, this assignment overwrites the one above, leaving a
	// single marker on the wire.
	if cfg.Tools != nil && len(params.Tools) > 0 {
		last := &params.Tools[len(params.Tools)-1]
		if last.OfTool != nil {
			last.OfTool.CacheControl = newCacheControl(cfg.Tools)
		}
	}

	// 3. System — last text block
	if cfg.SystemInstruction != nil && len(params.System) > 0 {
		params.System[len(params.System)-1].CacheControl = newCacheControl(cfg.SystemInstruction)
	}

	// 4. Conversation history — the previous turn's endpoint: the last
	// cacheable block at or before the second-most-recent user message,
	// falling back message-by-message toward the front when that message has
	// no cacheable block. Placed before ConversationHistory so that if both
	// ever land on the same block, the later assignment wins — a single
	// marker on the wire, same semantics as the tools pair above.
	if cfg.ConversationHistoryPrevTurn != nil {
		for i := secondNewestUserMessageIndex(params.Messages); i >= 0; i-- {
			if setLastCacheableBlock(params.Messages[i].Content, cfg.ConversationHistoryPrevTurn) {
				break
			}
		}
	}

	// 5. Marked part — the block converted from the last marked part, on a
	// single-user-message request only: with two or more user messages the
	// prev-turn marker above and the history marker below already cover the
	// block, and a marker here would spend one of the four on an entry those
	// subsume. Placed before ConversationHistory so that when the marked block
	// is also the newest cacheable block the later assignment wins — a single
	// marker on the wire, same semantics as the pairs above.
	if cfg.MarkedPart != nil && len(markedBlockOrdinals) > 0 && secondNewestUserMessageIndex(params.Messages) < 0 {
		lastMarkedOrdinal := markedBlockOrdinals[len(markedBlockOrdinals)-1]
		if messageIndex, blockIndex, ok := blockPosition(params.Messages, lastMarkedOrdinal); ok {
			if ccPtr := params.Messages[messageIndex].Content[blockIndex].GetCacheControl(); ccPtr != nil {
				*ccPtr = newCacheControl(cfg.MarkedPart)
			}
		}
	}

	// 6. Conversation history — the newest cacheable content block, searching
	// messages from last to first
	if cfg.ConversationHistory != nil {
		for i := len(params.Messages) - 1; i >= 0; i-- {
			if setLastCacheableBlock(params.Messages[i].Content, cfg.ConversationHistory) {
				break
			}
		}
	}

	// 7. Auto — top-level cache_control (applies a marker to the last
	// cacheable block in the request)
	if cfg.Auto != nil {
		params.CacheControl = newCacheControl(cfg.Auto)
	}
}

// blockPosition resolves a flat block ordinal — an index into the
// concatenation of every message's content blocks, as
// converters.ContentsToMessagesWithMarkedBlocks reports them — to the message
// and block indexes it names, reporting false for an ordinal outside the
// request.
func blockPosition(messages []anthropicsdk.MessageParam, ordinal int) (messageIndex, blockIndex int, ok bool) {
	if ordinal < 0 {
		return 0, 0, false
	}
	for i := range messages {
		if ordinal < len(messages[i].Content) {
			return i, ordinal, true
		}
		ordinal -= len(messages[i].Content)
	}
	return 0, 0, false
}

// secondNewestUserMessageIndex returns the index of the second-most-recent
// user-role message, or -1 when messages holds fewer than two.
func secondNewestUserMessageIndex(messages []anthropicsdk.MessageParam) int {
	users := 0
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == anthropicsdk.MessageParamRoleUser {
			users++
			if users == 2 {
				return i
			}
		}
	}
	return -1
}

// setLastCacheableBlock places bp on the last content block that can carry
// cache_control, reporting whether any block accepted it. Thinking and
// redacted-thinking blocks cannot carry cache_control and are skipped, so a
// message consisting only of those reports false.
func setLastCacheableBlock(content []anthropicsdk.ContentBlockParamUnion, bp *CacheBreakpoint) bool {
	for i := len(content) - 1; i >= 0; i-- {
		if ccPtr := content[i].GetCacheControl(); ccPtr != nil {
			*ccPtr = newCacheControl(bp)
			return true
		}
	}
	return false
}

// newCacheControl creates a CacheControlEphemeralParam from a breakpoint config.
func newCacheControl(bp *CacheBreakpoint) anthropicsdk.CacheControlEphemeralParam {
	cc := anthropicsdk.NewCacheControlEphemeralParam()
	if bp.TTL != "" {
		cc.TTL = bp.TTL
	}
	return cc
}
