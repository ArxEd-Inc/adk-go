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
	"strings"
	"testing"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"

	"google.golang.org/adk/v2/model/anthropic/internal/converters"
)

// bareInfinityEvents streams a tool call whose input carries a bare Infinity literal —
// the shape Claude emits when a numeric field's documentation gives Infinity a meaning,
// split across two input_json_delta fragments the way a real stream delivers it.
func bareInfinityEvents() []string {
	return []string{
		`{"type": "message_start", "message": {"id": "msg_1", "role": "assistant", "model": "claude-opus-5"}}`,
		`{"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "set_limit", "input": {}}}`,
		`{"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"max_units\": "}}`,
		`{"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "Infinity}"}}`,
		`{"type": "content_block_stop", "index": 0}`,
		`{"type": "message_stop"}`,
	}
}

// accumulateEvents replays raw stream events through Accumulate the way generate and
// generateStream do, optionally with the repair applied before each event.
func accumulateEvents(t *testing.T, rawEvents []string, repair bool) (*anthropicsdk.Message, error) {
	t.Helper()
	message := anthropicsdk.Message{}
	for _, raw := range rawEvents {
		var event anthropicsdk.MessageStreamEventUnion
		if err := json.Unmarshal([]byte(raw), &event); err != nil {
			t.Fatalf("unmarshal event %s: %v", raw, err)
		}
		if repair {
			repairAccumulatedToolInput(&message, event)
		}
		if err := message.Accumulate(event); err != nil {
			return nil, err
		}
	}
	return &message, nil
}

// TestAccumulateBareInfinityFailsWithoutRepair pins the SDK failure the repair exists
// for: bare Infinity in accumulated tool input kills Accumulate at content_block_stop,
// before the tool call is ever materialized.
func TestAccumulateBareInfinityFailsWithoutRepair(t *testing.T) {
	_, err := accumulateEvents(t, bareInfinityEvents(), false)
	if err == nil {
		t.Fatal("expected an accumulate error for bare Infinity tool input")
	}
	if !strings.Contains(err.Error(), "invalid character 'I'") {
		t.Fatalf("expected the bare-Infinity syntax error, got: %v", err)
	}
}

// TestRepairedBareInfinityReachesFunctionCall runs the same poisoned stream with the
// repair in place and asserts the tool call comes out the far end carrying the sentinel
// in its quoted string form.
func TestRepairedBareInfinityReachesFunctionCall(t *testing.T) {
	message, err := accumulateEvents(t, bareInfinityEvents(), true)
	if err != nil {
		t.Fatalf("accumulate with repair: %v", err)
	}
	resp, err := converters.MessageToLLMResponse(message, nil)
	if err != nil {
		t.Fatalf("convert repaired message: %v", err)
	}
	if resp.Content == nil || len(resp.Content.Parts) != 1 || resp.Content.Parts[0].FunctionCall == nil {
		t.Fatalf("expected a single function-call part, got %+v", resp.Content)
	}
	call := resp.Content.Parts[0].FunctionCall
	if call.Name != "set_limit" {
		t.Fatalf("unexpected tool name %q", call.Name)
	}
	if got := call.Args["max_units"]; got != "Infinity" {
		t.Fatalf(`expected args to carry the string "Infinity", got %#v`, got)
	}
}

func TestQuoteBareJSONSentinels(t *testing.T) {
	for name, tc := range map[string]struct {
		in          string
		want        string
		wantChanged bool
	}{
		"bare Infinity value": {
			in:          `{"a": Infinity}`,
			want:        `{"a": "Infinity"}`,
			wantChanged: true,
		},
		"bare negative Infinity": {
			in:          `{"a": -Infinity}`,
			want:        `{"a": "-Infinity"}`,
			wantChanged: true,
		},
		"bare NaN": {
			in:          `{"a": NaN}`,
			want:        `{"a": "NaN"}`,
			wantChanged: true,
		},
		"array elements": {
			in:          `{"a": [Infinity, -Infinity, NaN, 1]}`,
			want:        `{"a": ["Infinity", "-Infinity", "NaN", 1]}`,
			wantChanged: true,
		},
		"string content untouched": {
			in:          `{"notes": "recorded as Infinity in section 12"}`,
			want:        `{"notes": "recorded as Infinity in section 12"}`,
			wantChanged: false,
		},
		"string with escaped quotes untouched": {
			in:          `{"n": "say \"Infinity\" aloud", "a": Infinity}`,
			want:        `{"n": "say \"Infinity\" aloud", "a": "Infinity"}`,
			wantChanged: true,
		},
		"longer bare word not split": {
			in:          `{"a": InfinityCap}`,
			want:        `{"a": InfinityCap}`,
			wantChanged: false,
		},
		"already quoted untouched": {
			in:          `{"a": "Infinity"}`,
			want:        `{"a": "Infinity"}`,
			wantChanged: false,
		},
	} {
		t.Run(name, func(t *testing.T) {
			got, changed := quoteBareJSONSentinels([]byte(tc.in))
			if string(got) != tc.want {
				t.Errorf("got %s, want %s", got, tc.want)
			}
			if changed != tc.wantChanged {
				t.Errorf("changed = %v, want %v", changed, tc.wantChanged)
			}
		})
	}
}

// stopEventAtIndexZero builds the content_block_stop event the repair keys on.
func stopEventAtIndexZero(t *testing.T) anthropicsdk.MessageStreamEventUnion {
	t.Helper()
	var event anthropicsdk.MessageStreamEventUnion
	if err := json.Unmarshal([]byte(`{"type": "content_block_stop", "index": 0}`), &event); err != nil {
		t.Fatalf("unmarshal stop event: %v", err)
	}
	return event
}

// TestRepairAccumulatedToolInputLeavesUnrepairableInput: input that stays invalid even
// after the rewrite (here: truncated) must remain untouched, so the accumulator's
// original error still surfaces instead of a half-mangled variant.
func TestRepairAccumulatedToolInputLeavesUnrepairableInput(t *testing.T) {
	truncated := `{"a": Infinity, "b":`
	message := anthropicsdk.Message{Content: []anthropicsdk.ContentBlockUnion{
		{Type: "tool_use", ID: "toolu_1", Name: "set_limit", Input: []byte(truncated)},
	}}
	repairAccumulatedToolInput(&message, stopEventAtIndexZero(t))
	if got := string(message.Content[0].Input); got != truncated {
		t.Fatalf("unrepairable input was modified: %s", got)
	}
}

func TestRepairAccumulatedToolInputSkipsValidInput(t *testing.T) {
	valid := `{"a": 1}`
	message := anthropicsdk.Message{Content: []anthropicsdk.ContentBlockUnion{
		{Type: "tool_use", ID: "toolu_1", Name: "set_limit", Input: []byte(valid)},
	}}
	repairAccumulatedToolInput(&message, stopEventAtIndexZero(t))
	if got := string(message.Content[0].Input); got != valid {
		t.Fatalf("valid input was modified: %s", got)
	}
}

func TestRepairAccumulatedToolInputSkipsNonToolUseBlocks(t *testing.T) {
	message := anthropicsdk.Message{Content: []anthropicsdk.ContentBlockUnion{
		{Type: "text", Text: "Infinity"},
	}}
	repairAccumulatedToolInput(&message, stopEventAtIndexZero(t))
	if message.Content[0].Text != "Infinity" {
		t.Fatalf("text block was modified: %+v", message.Content[0])
	}
}
