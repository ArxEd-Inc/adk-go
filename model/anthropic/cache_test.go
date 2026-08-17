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
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
)

func toolParam(name string) anthropicsdk.ToolUnionParam {
	return anthropicsdk.ToolUnionParam{OfTool: &anthropicsdk.ToolParam{Name: name}}
}

// toolUseBlocks returns n tool_use blocks, mimicking a parallel fan-out turn.
func toolUseBlocks(n int) []anthropicsdk.ContentBlockParamUnion {
	blocks := make([]anthropicsdk.ContentBlockParamUnion, n)
	for i := range blocks {
		blocks[i] = anthropicsdk.NewToolUseBlock(fmt.Sprintf("use-%d", i), map[string]any{}, "someTool")
	}
	return blocks
}

// toolResultBlocks returns n tool_result blocks matching toolUseBlocks(n).
func toolResultBlocks(n int) []anthropicsdk.ContentBlockParamUnion {
	blocks := make([]anthropicsdk.ContentBlockParamUnion, n)
	for i := range blocks {
		blocks[i] = anthropicsdk.NewToolResultBlock(fmt.Sprintf("use-%d", i), "result", false)
	}
	return blocks
}

// markerCount returns the number of cache_control markers in v's marshaled
// JSON form — the ground truth for what actually reaches the wire.
func markerCount(t *testing.T, v any) int {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return strings.Count(string(b), `"cache_control"`)
}

// TestApplyCacheBreakpoints covers marker placement for each configured
// breakpoint across request shapes, asserting both the exact positions and the
// total marker count on the marshaled request.
func TestApplyCacheBreakpoints(t *testing.T) {
	cases := []struct {
		name   string
		cfg    *PromptCachingConfig
		params anthropicsdk.MessageNewParams

		// Expected marker positions: indexes into Tools/System, and
		// {message, block} index pairs into Messages.
		wantTools    []int
		wantSystem   []int
		wantMessages [][2]int
		wantTopLevel bool
	}{
		{
			name: "tools and system markers land on the last elements",
			cfg: &PromptCachingConfig{
				Tools:             &CacheBreakpoint{},
				SystemInstruction: &CacheBreakpoint{},
			},
			params: anthropicsdk.MessageNewParams{
				Tools:  []anthropicsdk.ToolUnionParam{toolParam("alpha"), toolParam("beta")},
				System: []anthropicsdk.TextBlockParam{{Text: "one"}, {Text: "two"}},
				Messages: []anthropicsdk.MessageParam{
					anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("hi")),
				},
			},
			wantTools:  []int{1},
			wantSystem: []int{1},
		},
		{
			name: "static prefix marker lands on the named mid-array tool",
			cfg: &PromptCachingConfig{
				ToolsStaticPrefixEnd:         &CacheBreakpoint{},
				ToolsStaticPrefixEndToolName: "loader",
			},
			params: anthropicsdk.MessageNewParams{
				Tools: []anthropicsdk.ToolUnionParam{toolParam("alpha"), toolParam("loader"), toolParam("beta")},
			},
			wantTools: []int{1},
		},
		{
			name: "static prefix marker skipped when the named tool is absent",
			cfg: &PromptCachingConfig{
				ToolsStaticPrefixEnd:         &CacheBreakpoint{},
				ToolsStaticPrefixEndToolName: "loader",
			},
			params: anthropicsdk.MessageNewParams{
				Tools: []anthropicsdk.ToolUnionParam{toolParam("alpha"), toolParam("beta")},
			},
		},
		{
			name: "static prefix marker skipped without a tool name",
			cfg: &PromptCachingConfig{
				ToolsStaticPrefixEnd: &CacheBreakpoint{},
			},
			params: anthropicsdk.MessageNewParams{
				Tools: []anthropicsdk.ToolUnionParam{toolParam("alpha"), toolParam("beta")},
			},
		},
		{
			name: "history marker lands on the last block of the last message",
			cfg:  &PromptCachingConfig{ConversationHistory: &CacheBreakpoint{}},
			params: anthropicsdk.MessageNewParams{
				Messages: []anthropicsdk.MessageParam{
					anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("hi")),
					anthropicsdk.NewAssistantMessage(
						anthropicsdk.NewTextBlock("part one"),
						anthropicsdk.NewTextBlock("part two"),
					),
				},
			},
			wantMessages: [][2]int{{1, 1}},
		},
		{
			name: "history marker skips a trailing thinking block",
			cfg:  &PromptCachingConfig{ConversationHistory: &CacheBreakpoint{}},
			params: anthropicsdk.MessageNewParams{
				Messages: []anthropicsdk.MessageParam{
					anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("hi")),
					anthropicsdk.NewAssistantMessage(
						anthropicsdk.NewTextBlock("answer"),
						anthropicsdk.NewThinkingBlock("sig", "hmm"),
					),
				},
			},
			wantMessages: [][2]int{{1, 0}},
		},
		{
			name: "history marker falls back past a thinking-only message",
			cfg:  &PromptCachingConfig{ConversationHistory: &CacheBreakpoint{}},
			params: anthropicsdk.MessageNewParams{
				Messages: []anthropicsdk.MessageParam{
					anthropicsdk.NewUserMessage(
						anthropicsdk.NewTextBlock("hi"),
						anthropicsdk.NewTextBlock("there"),
					),
					anthropicsdk.NewAssistantMessage(
						anthropicsdk.NewThinkingBlock("sig", "hmm"),
						anthropicsdk.NewRedactedThinkingBlock("data"),
					),
				},
			},
			wantMessages: [][2]int{{0, 1}},
		},
		{
			name: "history marker lands on a single-message request",
			cfg:  &PromptCachingConfig{ConversationHistory: &CacheBreakpoint{}},
			params: anthropicsdk.MessageNewParams{
				Messages: []anthropicsdk.MessageParam{
					anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("hi")),
				},
			},
			wantMessages: [][2]int{{0, 0}},
		},
		{
			name: "history marker skipped when no block is cacheable",
			cfg:  &PromptCachingConfig{ConversationHistory: &CacheBreakpoint{}},
			params: anthropicsdk.MessageNewParams{
				Messages: []anthropicsdk.MessageParam{
					anthropicsdk.NewAssistantMessage(anthropicsdk.NewThinkingBlock("sig", "hmm")),
				},
			},
		},
		{
			name: "prev-turn marker lands on the second-most-recent user message's last block",
			cfg:  &PromptCachingConfig{ConversationHistoryPrevTurn: &CacheBreakpoint{}},
			params: anthropicsdk.MessageNewParams{
				Messages: []anthropicsdk.MessageParam{
					anthropicsdk.NewUserMessage(
						anthropicsdk.NewTextBlock("question"),
						anthropicsdk.NewTextBlock("attachment"),
					),
					anthropicsdk.NewAssistantMessage(toolUseBlocks(1)...),
					anthropicsdk.NewUserMessage(toolResultBlocks(1)...),
				},
			},
			wantMessages: [][2]int{{0, 1}},
		},
		{
			name: "prev-turn marker survives a wide parallel fan-out turn",
			cfg:  &PromptCachingConfig{ConversationHistoryPrevTurn: &CacheBreakpoint{}},
			params: anthropicsdk.MessageNewParams{
				Messages: []anthropicsdk.MessageParam{
					anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("go")),
					anthropicsdk.NewAssistantMessage(toolUseBlocks(12)...),
					anthropicsdk.NewUserMessage(toolResultBlocks(12)...),
				},
			},
			wantMessages: [][2]int{{0, 0}},
		},
		{
			name: "prev-turn marker skipped on a single-message request",
			cfg:  &PromptCachingConfig{ConversationHistoryPrevTurn: &CacheBreakpoint{}},
			params: anthropicsdk.MessageNewParams{
				Messages: []anthropicsdk.MessageParam{
					anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("hi")),
				},
			},
		},
		{
			name: "prev-turn marker skipped with only one user message",
			cfg:  &PromptCachingConfig{ConversationHistoryPrevTurn: &CacheBreakpoint{}},
			params: anthropicsdk.MessageNewParams{
				Messages: []anthropicsdk.MessageParam{
					anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("hi")),
					anthropicsdk.NewAssistantMessage(anthropicsdk.NewTextBlock("answer")),
				},
			},
		},
		{
			name: "prev-turn marker anchors past an assistant-text-only turn",
			cfg:  &PromptCachingConfig{ConversationHistoryPrevTurn: &CacheBreakpoint{}},
			params: anthropicsdk.MessageNewParams{
				Messages: []anthropicsdk.MessageParam{
					anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("question")),
					anthropicsdk.NewAssistantMessage(anthropicsdk.NewTextBlock("interim answer")),
					anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("follow-up")),
				},
			},
			wantMessages: [][2]int{{0, 0}},
		},
		{
			// A user message with no cacheable block cannot occur in
			// production (thinking blocks only appear on assistant turns) but
			// pins the fallback walk, same as the "no block is cacheable"
			// case above.
			name: "prev-turn marker falls back past an uncacheable anchor message",
			cfg:  &PromptCachingConfig{ConversationHistoryPrevTurn: &CacheBreakpoint{}},
			params: anthropicsdk.MessageNewParams{
				Messages: []anthropicsdk.MessageParam{
					anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("question")),
					anthropicsdk.NewAssistantMessage(anthropicsdk.NewTextBlock("answer")),
					anthropicsdk.NewUserMessage(anthropicsdk.NewThinkingBlock("sig", "hmm")),
					anthropicsdk.NewAssistantMessage(anthropicsdk.NewTextBlock("more")),
					anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("follow-up")),
				},
			},
			wantMessages: [][2]int{{1, 0}},
		},
		{
			// ConversationHistory falls back onto the sole user message while
			// PrevTurn places nothing: the two history markers cannot stack
			// on one block even under fallback.
			name: "history and prev-turn markers cannot stack even under fallback",
			cfg: &PromptCachingConfig{
				ConversationHistoryPrevTurn: &CacheBreakpoint{},
				ConversationHistory:         &CacheBreakpoint{},
			},
			params: anthropicsdk.MessageNewParams{
				Messages: []anthropicsdk.MessageParam{
					anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("hi")),
					anthropicsdk.NewAssistantMessage(anthropicsdk.NewThinkingBlock("sig", "hmm")),
				},
			},
			wantMessages: [][2]int{{0, 0}},
		},
		{
			name: "no configured breakpoints places no markers",
			cfg:  &PromptCachingConfig{},
			params: anthropicsdk.MessageNewParams{
				Tools:  []anthropicsdk.ToolUnionParam{toolParam("alpha")},
				System: []anthropicsdk.TextBlockParam{{Text: "one"}},
				Messages: []anthropicsdk.MessageParam{
					anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("hi")),
				},
			},
		},
		{
			name: "auto sets the top-level cache control",
			cfg:  &PromptCachingConfig{Auto: &CacheBreakpoint{}},
			params: anthropicsdk.MessageNewParams{
				Messages: []anthropicsdk.MessageParam{
					anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("hi")),
				},
			},
			wantTopLevel: true,
		},
		{
			// A config spending the full marker budget: both tools
			// breakpoints plus both history breakpoints. With tool
			// definitions appended after the named static-prefix tool, all
			// four markers are distinct — exactly Anthropic's maximum,
			// leaving no room for a SystemInstruction or Auto breakpoint
			// alongside.
			name: "four-breakpoint config with appended tools places exactly four markers",
			cfg: &PromptCachingConfig{
				Tools:                        &CacheBreakpoint{},
				ConversationHistoryPrevTurn:  &CacheBreakpoint{},
				ConversationHistory:          &CacheBreakpoint{},
				ToolsStaticPrefixEnd:         &CacheBreakpoint{},
				ToolsStaticPrefixEndToolName: "loader",
			},
			params: anthropicsdk.MessageNewParams{
				Tools: []anthropicsdk.ToolUnionParam{
					toolParam("staticAlpha"),
					toolParam("staticBeta"),
					toolParam("loader"),
					toolParam("groupGamma"),
					toolParam("groupDelta"),
				},
				System: []anthropicsdk.TextBlockParam{{Text: "instruction"}, {Text: "catalog"}},
				Messages: []anthropicsdk.MessageParam{
					anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("hi")),
					anthropicsdk.NewAssistantMessage(
						anthropicsdk.NewThinkingBlock("sig", "hmm"),
						anthropicsdk.NewTextBlock("calling a tool"),
					),
					anthropicsdk.NewUserMessage(anthropicsdk.NewToolResultBlock("use-1", "result", false)),
				},
			},
			wantTools:    []int{2, 4},
			wantMessages: [][2]int{{0, 0}, {2, 0}},
		},
		{
			// The same config when the named static-prefix tool is still the
			// last tool: the two tools breakpoints collapse into one marker
			// and only three reach the wire. Collapse only ever shrinks the
			// count, so a four-breakpoint config never exceeds the budget in
			// either state.
			name: "four-breakpoint config with the prefix tool last collapses to three markers",
			cfg: &PromptCachingConfig{
				Tools:                        &CacheBreakpoint{},
				ConversationHistoryPrevTurn:  &CacheBreakpoint{},
				ConversationHistory:          &CacheBreakpoint{},
				ToolsStaticPrefixEnd:         &CacheBreakpoint{},
				ToolsStaticPrefixEndToolName: "loader",
			},
			params: anthropicsdk.MessageNewParams{
				Tools: []anthropicsdk.ToolUnionParam{
					toolParam("staticAlpha"),
					toolParam("loader"),
				},
				System: []anthropicsdk.TextBlockParam{{Text: "instruction"}, {Text: "catalog"}},
				Messages: []anthropicsdk.MessageParam{
					anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("hi")),
					anthropicsdk.NewAssistantMessage(
						anthropicsdk.NewThinkingBlock("sig", "hmm"),
						anthropicsdk.NewTextBlock("calling a tool"),
					),
					anthropicsdk.NewUserMessage(anthropicsdk.NewToolResultBlock("use-1", "result", false)),
				},
			},
			wantTools:    []int{1},
			wantMessages: [][2]int{{0, 0}, {2, 0}},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			applyCacheBreakpoints(&tc.params, tc.cfg)

			for i := range tc.params.Tools {
				want := slices.Contains(tc.wantTools, i)
				got := tc.params.Tools[i].OfTool.CacheControl.Type != ""
				if got != want {
					t.Errorf("tools[%d] marker = %v, want %v", i, got, want)
				}
			}
			for i := range tc.params.System {
				want := slices.Contains(tc.wantSystem, i)
				got := tc.params.System[i].CacheControl.Type != ""
				if got != want {
					t.Errorf("system[%d] marker = %v, want %v", i, got, want)
				}
			}
			for i := range tc.params.Messages {
				for j := range tc.params.Messages[i].Content {
					want := slices.Contains(tc.wantMessages, [2]int{i, j})
					ccPtr := tc.params.Messages[i].Content[j].GetCacheControl()
					got := ccPtr != nil && ccPtr.Type != ""
					if got != want {
						t.Errorf("messages[%d].content[%d] marker = %v, want %v", i, j, got, want)
					}
				}
			}
			if got := tc.params.CacheControl.Type != ""; got != tc.wantTopLevel {
				t.Errorf("top-level marker = %v, want %v", got, tc.wantTopLevel)
			}

			wantTotal := len(tc.wantTools) + len(tc.wantSystem) + len(tc.wantMessages)
			if tc.wantTopLevel {
				wantTotal++
			}
			if got := markerCount(t, tc.params); got != wantTotal {
				t.Errorf("marshaled marker count = %d, want %d", got, wantTotal)
			}
			// Anthropic allows at most 4 cache_control markers per request.
			if wantTotal > 4 {
				t.Errorf("test case expects %d markers, above Anthropic's maximum of 4", wantTotal)
			}
		})
	}
}

// TestApplyCacheBreakpointsBothToolBreakpoints: when the named static-prefix
// tool and the last tool are distinct, each carries its own breakpoint with its
// own TTL.
func TestApplyCacheBreakpointsBothToolBreakpoints(t *testing.T) {
	params := anthropicsdk.MessageNewParams{
		Tools: []anthropicsdk.ToolUnionParam{toolParam("loader"), toolParam("groupGamma")},
	}
	applyCacheBreakpoints(&params, &PromptCachingConfig{
		Tools:                        &CacheBreakpoint{},
		ToolsStaticPrefixEnd:         &CacheBreakpoint{TTL: anthropicsdk.CacheControlEphemeralTTLTTL1h},
		ToolsStaticPrefixEndToolName: "loader",
	})
	if got := markerCount(t, params.Tools); got != 2 {
		t.Errorf("tool marker count = %d, want 2", got)
	}
	if ttl := params.Tools[0].OfTool.CacheControl.TTL; ttl != anthropicsdk.CacheControlEphemeralTTLTTL1h {
		t.Errorf("static prefix TTL = %q, want 1h", ttl)
	}
	if ttl := params.Tools[1].OfTool.CacheControl.TTL; ttl != "" {
		t.Errorf("last tool TTL = %q, want default", ttl)
	}
}

// TestApplyCacheBreakpointsStaticPrefixEndCoincidesWithLastTool: when the named
// static-prefix tool IS the last tool, the two breakpoints collapse into a
// single marker carrying the Tools breakpoint's TTL.
func TestApplyCacheBreakpointsStaticPrefixEndCoincidesWithLastTool(t *testing.T) {
	params := anthropicsdk.MessageNewParams{
		Tools: []anthropicsdk.ToolUnionParam{toolParam("alpha"), toolParam("loader")},
	}
	applyCacheBreakpoints(&params, &PromptCachingConfig{
		Tools:                        &CacheBreakpoint{},
		ToolsStaticPrefixEnd:         &CacheBreakpoint{TTL: anthropicsdk.CacheControlEphemeralTTLTTL1h},
		ToolsStaticPrefixEndToolName: "loader",
	})
	if got := markerCount(t, params.Tools); got != 1 {
		t.Errorf("tool marker count = %d, want 1", got)
	}
	if ttl := params.Tools[1].OfTool.CacheControl.TTL; ttl != "" {
		t.Errorf("surviving TTL = %q, want the Tools breakpoint's default", ttl)
	}
}

// TestApplyCacheBreakpointsPrevTurnTTL: the PrevTurn breakpoint's TTL reaches
// the wire on its anchor block.
func TestApplyCacheBreakpointsPrevTurnTTL(t *testing.T) {
	params := anthropicsdk.MessageNewParams{
		Messages: []anthropicsdk.MessageParam{
			anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("question")),
			anthropicsdk.NewAssistantMessage(anthropicsdk.NewTextBlock("calling a tool")),
			anthropicsdk.NewUserMessage(anthropicsdk.NewToolResultBlock("use-1", "result", false)),
		},
	}
	applyCacheBreakpoints(&params, &PromptCachingConfig{
		ConversationHistoryPrevTurn: &CacheBreakpoint{TTL: CacheTTL1h},
	})
	ccPtr := params.Messages[0].Content[0].GetCacheControl()
	if ccPtr == nil || ccPtr.TTL != CacheTTL1h {
		t.Fatalf("prev-turn anchor cache control = %+v, want TTL 1h", ccPtr)
	}
	b, err := json.Marshal(params.Messages)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(b), `"ttl":"1h"`) {
		t.Errorf(`marshaled messages missing "ttl":"1h": %s`, b)
	}
}
