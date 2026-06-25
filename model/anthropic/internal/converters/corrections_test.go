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
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/genai"
)

func ptr[T any](v T) *T { return &v }

// TestThinkingConfigToAnthropic covers the always-adaptive thinking mapping:
// thinking is either adaptive (with an effort hint from the level) or off, and
// there is never a budget_tokens (manual) form.
func TestThinkingConfigToAnthropic(t *testing.T) {
	cases := []struct {
		name       string
		cfg        *genai.ThinkingConfig
		wantOn     bool
		wantEffort anthropic.OutputConfigEffort
	}{
		{name: "nil is off", cfg: nil, wantOn: false},
		{name: "minimal is off", cfg: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelMinimal}, wantOn: false},
		{name: "explicit zero budget is off", cfg: &genai.ThinkingConfig{ThinkingBudget: ptr(int32(0))}, wantOn: false},
		{name: "high is adaptive+high", cfg: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelHigh}, wantOn: true, wantEffort: anthropic.OutputConfigEffortHigh},
		{name: "medium is adaptive+medium", cfg: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelMedium}, wantOn: true, wantEffort: anthropic.OutputConfigEffortMedium},
		{name: "low is adaptive+low", cfg: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelLow}, wantOn: true, wantEffort: anthropic.OutputConfigEffortLow},
		{name: "nonzero budget is adaptive (no manual form)", cfg: &genai.ThinkingConfig{ThinkingBudget: ptr(int32(8000))}, wantOn: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ThinkingConfigToAnthropic(tc.cfg)
			on := got.Thinking.OfAdaptive != nil
			if on != tc.wantOn {
				t.Fatalf("thinking on = %v, want %v", on, tc.wantOn)
			}
			// We must never emit the legacy manual/budget_tokens form.
			if got.Thinking.OfEnabled != nil {
				t.Fatalf("emitted budget_tokens thinking; want adaptive-only")
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

	tool := FunctionDeclarationToTool(fd)
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
