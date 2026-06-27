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

package converters

import (
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/genai"
)

func ptr[T any](v T) *T { return &v }

// TestThinkingConfigToAnthropic covers the thinking mapping in both modes:
// adaptive mode emits adaptive thinking (with an effort hint from the level) or
// off; budget mode emits a manual budget_tokens form (per level) or off. The
// three "off" guards (nil cfg, explicit zero budget, Minimal level) are shared.
func TestThinkingConfigToAnthropic(t *testing.T) {
	cases := []struct {
		name       string
		cfg        *genai.ThinkingConfig
		budgetMode bool
		wantOn     bool
		wantEffort anthropic.OutputConfigEffort
		wantBudget int64
	}{
		// Adaptive mode (the default, for adaptive-capable models).
		{name: "nil is off", cfg: nil, wantOn: false},
		{name: "minimal is off", cfg: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelMinimal}, wantOn: false},
		{name: "explicit zero budget is off", cfg: &genai.ThinkingConfig{ThinkingBudget: ptr(int32(0))}, wantOn: false},
		{name: "high is adaptive+high", cfg: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelHigh}, wantOn: true, wantEffort: anthropic.OutputConfigEffortHigh},
		{name: "medium is adaptive+medium", cfg: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelMedium}, wantOn: true, wantEffort: anthropic.OutputConfigEffortMedium},
		{name: "low is adaptive+low", cfg: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelLow}, wantOn: true, wantEffort: anthropic.OutputConfigEffortLow},
		{name: "nonzero budget is adaptive (no manual form)", cfg: &genai.ThinkingConfig{ThinkingBudget: ptr(int32(8000))}, wantOn: true},

		// Budget mode (for models that reject adaptive thinking + effort, e.g. Haiku 4.5).
		{name: "budget: nil is off", cfg: nil, budgetMode: true, wantOn: false},
		{name: "budget: minimal is off", cfg: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelMinimal}, budgetMode: true, wantOn: false},
		{name: "budget: explicit zero budget is off", cfg: &genai.ThinkingConfig{ThinkingBudget: ptr(int32(0))}, budgetMode: true, wantOn: false},
		{name: "budget: unspecified level is off", cfg: &genai.ThinkingConfig{}, budgetMode: true, wantOn: false},
		{name: "budget: high is enabled 24000", cfg: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelHigh}, budgetMode: true, wantOn: true, wantBudget: 24000},
		{name: "budget: medium is enabled 5000", cfg: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelMedium}, budgetMode: true, wantOn: true, wantBudget: 5000},
		{name: "budget: low is enabled 1024", cfg: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelLow}, budgetMode: true, wantOn: true, wantBudget: 1024},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ThinkingConfigToAnthropic(tc.cfg, tc.budgetMode)
			if tc.budgetMode {
				on := got.Thinking.OfEnabled != nil
				if on != tc.wantOn {
					t.Fatalf("budget thinking on = %v, want %v", on, tc.wantOn)
				}
				// Budget mode must never emit adaptive thinking or an effort hint.
				if got.Thinking.OfAdaptive != nil {
					t.Fatalf("budget mode emitted adaptive thinking; want budget_tokens")
				}
				if got.Effort != "" {
					t.Fatalf("budget mode set effort %q; want none", got.Effort)
				}
				if on && got.Thinking.OfEnabled.BudgetTokens != tc.wantBudget {
					t.Fatalf("budget_tokens = %d, want %d", got.Thinking.OfEnabled.BudgetTokens, tc.wantBudget)
				}
				return
			}
			on := got.Thinking.OfAdaptive != nil
			if on != tc.wantOn {
				t.Fatalf("thinking on = %v, want %v", on, tc.wantOn)
			}
			// Adaptive mode must never emit the manual/budget_tokens form.
			if got.Thinking.OfEnabled != nil {
				t.Fatalf("adaptive mode emitted budget_tokens thinking; want adaptive-only")
			}
			if got.Effort != tc.wantEffort {
				t.Fatalf("effort = %q, want %q", got.Effort, tc.wantEffort)
			}
		})
	}
}

// TestToolUseIDSanitizer covers the sanitizer: valid IDs pass through, invalid
// IDs map to stable fallbacks, and the same input always maps to the same output
// within one sanitizer (so a tool_use and its tool_result stay correlated).
func TestToolUseIDSanitizer(t *testing.T) {
	s := newToolUseIDSanitizer()

	if got := s.sanitize("toolu_01ABC-xyz"); got != "toolu_01ABC-xyz" {
		t.Fatalf("valid id rewritten to %q", got)
	}

	first := s.sanitize("call 1!") // space + bang are invalid
	if first == "call 1!" {
		t.Fatalf("invalid id not rewritten")
	}
	if again := s.sanitize("call 1!"); again != first {
		t.Fatalf("inconsistent sanitization: %q then %q", first, again)
	}

	// A different invalid id gets a different fallback.
	if other := s.sanitize("other id?"); other == first {
		t.Fatalf("distinct invalid ids collided to %q", other)
	}
}

// TestPartToContentBlockEmptyTextThought verifies the empty-text thinking guard:
// a thought block carried back with a signature but no text (display:"omitted")
// must still be replayed as a thinking block, not dropped.
func TestPartToContentBlockEmptyTextThought(t *testing.T) {
	sig := []byte("signature-bytes")
	part := &genai.Part{Thought: true, ThoughtSignature: sig, Text: ""}

	block, err := partToContentBlock(part, newToolUseIDSanitizer())
	if err != nil {
		t.Fatalf("partToContentBlock: %v", err)
	}
	if block == nil || block.OfThinking == nil {
		t.Fatalf("empty-text thought dropped; want a thinking block")
	}
	if want := base64.StdEncoding.EncodeToString(sig); block.OfThinking.Signature != want {
		t.Fatalf("signature = %q, want %q", block.OfThinking.Signature, want)
	}
}

// TestPartToContentBlockUnsignedThoughtErrors verifies that a thought part carrying neither a signature nor
// redacted-thinking data cannot be faithfully replayed and is surfaced as an error rather than silently dropped.
// MessageToLLMResponse never produces such a part (a thinking block always carries a signature, a redacted block the
// redacted marker), so reaching here means corrupted or foreign history.
func TestPartToContentBlockUnsignedThoughtErrors(t *testing.T) {
	part := &genai.Part{Thought: true, Text: "stray reasoning"}

	if _, err := partToContentBlock(part, newToolUseIDSanitizer()); err == nil {
		t.Fatalf("unsigned, unmarked thought did not error; want an error")
	}
}

// TestThoughtRoundTripThroughConverters verifies the full converter symmetry our multi-turn thinking + tool-use agents
// depend on: an Anthropic assistant message of [thinking, redacted_thinking, tool_use] converted to a genai response
// (MessageToLLMResponse) and back to Anthropic message params (ContentsToMessages) reproduces a
// [thinking, redacted_thinking, tool_use] assistant message in order, preserving the thinking signature and the
// redacted data.
func TestThoughtRoundTripThroughConverters(t *testing.T) {
	signature := base64.StdEncoding.EncodeToString([]byte("signature-bytes"))
	const redactedData = "ENCRYPTED-BLOB"
	raw := `{
		"id": "msg_1",
		"type": "message",
		"role": "assistant",
		"model": "claude-opus-4-8",
		"stop_reason": "tool_use",
		"content": [
			{"type": "thinking", "thinking": "let me reason", "signature": "` + signature + `"},
			{"type": "redacted_thinking", "data": "` + redactedData + `"},
			{"type": "tool_use", "id": "toolu_abc", "name": "lookup", "input": {"q": 1}}
		],
		"usage": {"input_tokens": 1, "output_tokens": 1}
	}`

	var msg anthropic.Message
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		t.Fatalf("unmarshal message: %v", err)
	}

	resp, err := MessageToLLMResponse(&msg, nil)
	if err != nil {
		t.Fatalf("MessageToLLMResponse: %v", err)
	}

	messages, err := ContentsToMessages([]*genai.Content{resp.Content})
	if err != nil {
		t.Fatalf("ContentsToMessages: %v", err)
	}
	if len(messages) != 1 {
		t.Fatalf("got %d messages, want 1 assistant message", len(messages))
	}

	blocks := messages[0].Content
	if len(blocks) != 3 {
		t.Fatalf("got %d blocks, want [thinking, redacted_thinking, tool_use]", len(blocks))
	}
	if blocks[0].OfThinking == nil {
		t.Fatalf("block 0 is not a thinking block: %+v", blocks[0])
	}
	if blocks[0].OfThinking.Thinking != "let me reason" {
		t.Errorf("thinking text = %q, want %q", blocks[0].OfThinking.Thinking, "let me reason")
	}
	if blocks[0].OfThinking.Signature != signature {
		t.Errorf("thinking signature = %q, want %q (did not round-trip)", blocks[0].OfThinking.Signature, signature)
	}
	if blocks[1].OfRedactedThinking == nil {
		t.Fatalf("block 1 is not a redacted_thinking block: %+v", blocks[1])
	}
	if blocks[1].OfRedactedThinking.Data != redactedData {
		t.Errorf("redacted data = %q, want %q (did not round-trip)", blocks[1].OfRedactedThinking.Data, redactedData)
	}
	if blocks[2].OfToolUse == nil {
		t.Fatalf("block 2 is not a tool_use block: %+v", blocks[2])
	}
}

// TestRedactedThinkingCarriedInThoughtSignature verifies that redacted thinking is carried in ThoughtSignature — the
// only opaque per-Part field the Vertex AI session backend persists — and not in PartMetadata, which that backend
// drops. Guards against a regression that would silently lose redacted thinking across turns.
func TestRedactedThinkingCarriedInThoughtSignature(t *testing.T) {
	const redactedData = "ENCRYPTED-BLOB"
	raw := `{"id":"m","type":"message","role":"assistant","model":"claude-opus-4-8","stop_reason":"end_turn",` +
		`"content":[{"type":"redacted_thinking","data":"` + redactedData + `"}],"usage":{"input_tokens":1,"output_tokens":1}}`

	var msg anthropic.Message
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		t.Fatalf("unmarshal message: %v", err)
	}
	resp, err := MessageToLLMResponse(&msg, nil)
	if err != nil {
		t.Fatalf("MessageToLLMResponse: %v", err)
	}
	if len(resp.Content.Parts) != 1 {
		t.Fatalf("got %d parts, want 1", len(resp.Content.Parts))
	}
	part := resp.Content.Parts[0]
	if part.PartMetadata != nil {
		t.Errorf("redacted data leaked into PartMetadata %v; the Vertex backend drops it", part.PartMetadata)
	}
	if data, ok := decodeRedactedThinking(part.ThoughtSignature); !ok || data != redactedData {
		t.Errorf("redacted data not recoverable from ThoughtSignature: got (%q, %v)", data, ok)
	}
}

// TestRedactedThinkingEncodeDecode verifies the marker carrier round-trips and that an ordinary signature is not
// mistaken for redacted data (marker collision guard).
func TestRedactedThinkingEncodeDecode(t *testing.T) {
	const data = "ENCRYPTED-BLOB"
	if got, ok := decodeRedactedThinking(encodeRedactedThinking(data)); !ok || got != data {
		t.Errorf("decode(encode(%q)) = (%q, %v), want (%q, true)", data, got, ok, data)
	}
	if _, ok := decodeRedactedThinking([]byte("an-ordinary-signature")); ok {
		t.Errorf("an ordinary signature was decoded as redacted data; marker collision")
	}
}

// TestFunctionDeclarationToToolResolvesRootRef verifies that a tool whose parameter schema is a
// root $ref into $defs (a common shape for schemas generated from typed request objects) is
// converted so the referenced object's properties and required fields appear at the top level, and
// $defs is carried through so nested $refs still resolve. Before this, the converter read only the
// empty root properties and sent Anthropic a parameterless tool.
func TestFunctionDeclarationToToolResolvesRootRef(t *testing.T) {
	fd := &genai.FunctionDeclaration{
		Name:        "searchBooks",
		Description: "searches books",
		ParametersJsonSchema: map[string]any{
			"$ref": "#/$defs/SearchBooksRequest",
			"$defs": map[string]any{
				"SearchBooksRequest": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"limit":          map[string]any{"type": "integer"},
						"include_ebooks": map[string]any{"type": "boolean"},
						"genres": map[string]any{
							"type":  "array",
							"items": map[string]any{"$ref": "#/$defs/Genre"},
						},
					},
					"required": []any{"limit", "include_ebooks"},
				},
				"Genre": map[string]any{"type": "string", "enum": []any{"GENRE_FICTION"}},
			},
		},
	}

	tool := FunctionDeclarationToTool(fd, map[string]string{})
	if tool.OfTool == nil {
		t.Fatal("OfTool is nil")
	}
	input := tool.OfTool.InputSchema

	properties, ok := input.Properties.(map[string]any)
	if !ok {
		t.Fatalf("input_schema.properties is %T, want map[string]any", input.Properties)
	}
	for _, name := range []string{"limit", "include_ebooks", "genres"} {
		if _, ok := properties[name]; !ok {
			t.Errorf("input_schema.properties missing %q", name)
		}
	}
	if len(input.Required) != 2 {
		t.Errorf("input_schema.required = %v, want the two required fields", input.Required)
	}

	// $defs must survive (so the nested $ref resolves on Anthropic's side) and the root $ref must be
	// inlined (Anthropic requires a concrete object at the top level).
	data, err := json.Marshal(input)
	if err != nil {
		t.Fatalf("marshal input schema: %v", err)
	}
	var marshalled map[string]any
	if err := json.Unmarshal(data, &marshalled); err != nil {
		t.Fatalf("unmarshal input schema: %v", err)
	}
	if _, ok := marshalled["$defs"]; !ok {
		t.Errorf("$defs not carried into input_schema: %s", data)
	}
	if _, ok := marshalled["$ref"]; ok {
		t.Errorf("root $ref was not inlined: %s", data)
	}
}

// TestFunctionDeclarationToToolAliasesLongTopLevelKey verifies that a top-level property key which
// violates Anthropic's key pattern (here, longer than 64 chars) is aliased to a conforming key, the
// alias is recorded for restoration, and required entries are mapped to the alias too.
func TestFunctionDeclarationToToolAliasesLongTopLevelKey(t *testing.T) {
	longKey := strings.Repeat("field_", 12) // 72 valid chars, over the 64-char limit
	fd := &genai.FunctionDeclaration{
		Name: "updateThing",
		ParametersJsonSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{longKey: map[string]any{"type": "string"}},
			"required":   []any{longKey},
		},
	}

	aliases := map[string]string{}
	tool := FunctionDeclarationToTool(fd, aliases)
	properties, ok := tool.OfTool.InputSchema.Properties.(map[string]any)
	if !ok {
		t.Fatalf("input_schema.properties is %T, want map[string]any", tool.OfTool.InputSchema.Properties)
	}

	if _, sentVerbatim := properties[longKey]; sentVerbatim {
		t.Errorf("long key %q sent verbatim; expected an alias", longKey)
	}
	if len(properties) != 1 {
		t.Fatalf("expected exactly one property, got %d", len(properties))
	}
	var alias string
	for key := range properties {
		alias = key
	}
	if len(alias) > 64 || !toolPropertyKeyPattern.MatchString(alias) {
		t.Errorf("alias %q does not satisfy the 64-char pattern", alias)
	}
	if aliases[alias] != longKey {
		t.Errorf("alias map = %v, want %q -> %q", aliases, alias, longKey)
	}
	if got := tool.OfTool.InputSchema.Required; len(got) != 1 || got[0] != alias {
		t.Errorf("required = %v, want [%q]", got, alias)
	}
}

// TestContentsToMessagesThoughtToolCallRoundTrip verifies a model turn of
// [Thought(+signature), FunctionCall] replays into a single assistant message
// of [thinking, tool_use], with the tool-use ID sanitized.
func TestContentsToMessagesThoughtToolCallRoundTrip(t *testing.T) {
	contents := []*genai.Content{
		{
			Role: "model",
			Parts: []*genai.Part{
				{Thought: true, ThoughtSignature: []byte("sig"), Text: "reasoning"},
				{FunctionCall: &genai.FunctionCall{ID: "bad id", Name: "do_thing", Args: map[string]any{"x": 1}}},
			},
		},
	}

	msgs, err := ContentsToMessages(contents)
	if err != nil {
		t.Fatalf("ContentsToMessages: %v", err)
	}
	if len(msgs) != 1 {
		t.Fatalf("got %d messages, want 1", len(msgs))
	}
	if msgs[0].Role != anthropic.MessageParamRoleAssistant {
		t.Fatalf("role = %q, want assistant", msgs[0].Role)
	}
	if len(msgs[0].Content) != 2 {
		t.Fatalf("got %d blocks, want 2", len(msgs[0].Content))
	}
	if msgs[0].Content[0].OfThinking == nil {
		t.Fatalf("first block is not a thinking block")
	}
	tu := msgs[0].Content[1].OfToolUse
	if tu == nil {
		t.Fatalf("second block is not a tool_use block")
	}
	if !toolUseIDPattern.MatchString(tu.ID) {
		t.Fatalf("tool_use id %q was not sanitized to Anthropic's shape", tu.ID)
	}
}

// TestContentsToMessagesToolCallResultCorrelation verifies that a tool call and its result stay correlated
// across the two contents they live in (the model's tool_use turn and the following tool_result turn).
// Anthropic pairs a tool_result to its tool_use solely by id, so the single per-request sanitizer must rewrite
// both ids to the same value — even when the originating id needs rewriting to Anthropic's shape (here a
// genai-style id containing spaces). A mismatch here 400s the turn after every tool call.
func TestContentsToMessagesToolCallResultCorrelation(t *testing.T) {
	const callID = "function call 1" // contains spaces: invalid for Anthropic, must be sanitized
	contents := []*genai.Content{
		{
			Role:  "model",
			Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: callID, Name: "lookup", Args: map[string]any{"q": 1}}}},
		},
		{
			Role:  "user",
			Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: callID, Name: "lookup", Response: map[string]any{"answer": 2}}}},
		},
	}

	msgs, err := ContentsToMessages(contents)
	if err != nil {
		t.Fatalf("ContentsToMessages: %v", err)
	}
	if len(msgs) != 2 {
		t.Fatalf("got %d messages, want [assistant tool_use, user tool_result]", len(msgs))
	}
	if msgs[0].Role != anthropic.MessageParamRoleAssistant || msgs[1].Role != anthropic.MessageParamRoleUser {
		t.Fatalf("roles = [%q, %q], want [assistant, user]", msgs[0].Role, msgs[1].Role)
	}

	toolUse := msgs[0].Content[0].OfToolUse
	if toolUse == nil {
		t.Fatalf("first message block is not a tool_use")
	}
	toolResult := msgs[1].Content[0].OfToolResult
	if toolResult == nil {
		t.Fatalf("second message block is not a tool_result")
	}
	if !toolUseIDPattern.MatchString(toolUse.ID) {
		t.Errorf("tool_use id %q was not sanitized to Anthropic's shape", toolUse.ID)
	}
	if toolResult.ToolUseID != toolUse.ID {
		t.Errorf("tool_result.tool_use_id = %q, want %q (lost correlation with the tool_use)", toolResult.ToolUseID, toolUse.ID)
	}
}

// TestContentsToMessagesParallelToolCallsCorrelate verifies that when the model issues several tool calls in one
// turn, every result still correlates to its call by id, and that tool_result turns split across consecutive
// contents are merged into the single user message Anthropic requires (roles must strictly alternate). The results
// are supplied in the reverse order of the calls to confirm correlation is by id, not position.
func TestContentsToMessagesParallelToolCallsCorrelate(t *testing.T) {
	contents := []*genai.Content{
		{
			Role: "model",
			Parts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: "toolu_a", Name: "lookup", Args: map[string]any{"q": 1}}},
				{FunctionCall: &genai.FunctionCall{ID: "toolu_b", Name: "lookup", Args: map[string]any{"q": 2}}},
			},
		},
		{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "toolu_b", Name: "lookup", Response: map[string]any{"answer": 2}}}}},
		{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "toolu_a", Name: "lookup", Response: map[string]any{"answer": 1}}}}},
	}

	msgs, err := ContentsToMessages(contents)
	if err != nil {
		t.Fatalf("ContentsToMessages: %v", err)
	}
	if len(msgs) != 2 {
		t.Fatalf("got %d messages, want [assistant (2 tool_use), user (2 tool_result)]", len(msgs))
	}
	if len(msgs[0].Content) != 2 {
		t.Fatalf("assistant message has %d blocks, want 2 tool_use", len(msgs[0].Content))
	}
	if len(msgs[1].Content) != 2 {
		t.Fatalf("user message has %d blocks, want 2 tool_result (consecutive tool_result turns not merged?)", len(msgs[1].Content))
	}

	callIDs := map[string]bool{}
	for _, block := range msgs[0].Content {
		if block.OfToolUse == nil {
			t.Fatalf("assistant block is not a tool_use: %+v", block)
		}
		callIDs[block.OfToolUse.ID] = true
	}
	for _, block := range msgs[1].Content {
		if block.OfToolResult == nil {
			t.Fatalf("user block is not a tool_result: %+v", block)
		}
		if !callIDs[block.OfToolResult.ToolUseID] {
			t.Errorf("tool_result id %q has no matching tool_use (correlation lost)", block.OfToolResult.ToolUseID)
		}
	}
	for _, id := range []string{"toolu_a", "toolu_b"} {
		if !callIDs[id] {
			t.Errorf("tool_use id %q missing from the assistant message", id)
		}
	}
}

// TestStopReasonToFinishReason covers the refusal mapping (and the common cases).
func TestStopReasonToFinishReason(t *testing.T) {
	cases := map[anthropic.StopReason]genai.FinishReason{
		anthropic.StopReasonRefusal:      genai.FinishReasonSafety,
		anthropic.StopReasonEndTurn:      genai.FinishReasonStop,
		anthropic.StopReasonToolUse:      genai.FinishReasonStop,
		anthropic.StopReasonMaxTokens:    genai.FinishReasonMaxTokens,
		anthropic.StopReasonStopSequence: genai.FinishReasonStop,
	}
	for sr, want := range cases {
		if got := StopReasonToFinishReason(sr); got != want {
			t.Errorf("StopReasonToFinishReason(%q) = %q, want %q", sr, got, want)
		}
	}
}
