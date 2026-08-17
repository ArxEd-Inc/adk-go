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

import anthropicsdk "github.com/anthropics/anthropic-sdk-go"

// applyCacheBreakpoints sets cache_control breakpoints on the request based
// on the provided configuration. Each breakpoint is independently optional.
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
func applyCacheBreakpoints(params *anthropicsdk.MessageNewParams, cfg *PromptCachingConfig) {
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

	// 5. Conversation history — the newest cacheable content block, searching
	// messages from last to first
	if cfg.ConversationHistory != nil {
		for i := len(params.Messages) - 1; i >= 0; i-- {
			if setLastCacheableBlock(params.Messages[i].Content, cfg.ConversationHistory) {
				break
			}
		}
	}

	// 6. Auto — top-level cache_control (applies a marker to the last
	// cacheable block in the request)
	if cfg.Auto != nil {
		params.CacheControl = newCacheControl(cfg.Auto)
	}
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
