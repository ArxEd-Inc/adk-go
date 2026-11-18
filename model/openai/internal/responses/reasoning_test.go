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

package responses

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"

	oairesponses "github.com/openai/openai-go/v3/responses"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// toolTurnResponse is a completed response in which the model reasons, then
// calls a tool.
func toolTurnResponse(t *testing.T, modelName string) *oairesponses.Response {
	t.Helper()
	var resp oairesponses.Response
	body := `{"id":"resp_1","model":"` + modelName + `","status":"completed","output":[` +
		`{"type":"reasoning","id":"rs_1","encrypted_content":"opaque-1","summary":[{"type":"summary_text","text":"Look the order up."}]},` +
		`{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup_order","arguments":"{\"order_id\":\"42\"}"}]}`
	if err := json.Unmarshal([]byte(body), &resp); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}
	return &resp
}

// replayContents is the conversation a caller sends after content: the user's
// question, the model's turn, and the tool's answer.
func replayContents(content *genai.Content) []*genai.Content {
	return []*genai.Content{
		genai.NewContentFromText("Where is order 42?", genai.RoleUser),
		content,
		{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
			ID: "call_1", Name: "lookup_order", Response: map[string]any{"status": "shipped"},
		}}}},
	}
}

// inputTypes lists the type of each input item, a message's role standing in
// for its type.
func inputTypes(t *testing.T, params oairesponses.ResponseNewParams) []string {
	t.Helper()
	raw, err := json.Marshal(params.Input.OfInputItemList)
	if err != nil {
		t.Fatalf("marshal input: %v", err)
	}
	var items []map[string]any
	if err := json.Unmarshal(raw, &items); err != nil {
		t.Fatalf("unmarshal input: %v", err)
	}
	var types []string
	for _, item := range items {
		typ, _ := item["type"].(string)
		if role, ok := item["role"].(string); ok && (typ == "" || typ == "message") {
			typ = role
		}
		types = append(types, typ)
	}
	return types
}

func TestReasoning_CarriedOnTheFirstThoughtPart(t *testing.T) {
	got, err := convertResponse(toolTurnResponse(t, "gpt-test"))
	if err != nil {
		t.Fatalf("convertResponse() err = %v", err)
	}
	parts := got.Candidates[0].Content.Parts
	if len(parts) != 2 {
		t.Fatalf("got %d parts, want the thought and the call", len(parts))
	}
	thought := parts[0]
	if !thought.Thought || thought.Text != "Look the order up." {
		t.Errorf("parts[0] = %+v, want the summary as a thought", thought)
	}
	payload, ok := decodeReasoningSignature(thought.ThoughtSignature)
	if !ok {
		t.Fatalf("parts[0].ThoughtSignature = %q, want a reasoning signature", thought.ThoughtSignature)
	}
	want := reasoningPayload{Model: "gpt-test", ID: "rs_1", EncryptedContent: "opaque-1", Summary: []string{"Look the order up."}}
	if !equalPayload(payload, want) {
		t.Errorf("payload = %+v, want %+v", payload, want)
	}
}

func TestReasoning_ItemWithoutTextStillYieldsAPart(t *testing.T) {
	parts, err := convertOutputItems("gpt-test", []oairesponses.ResponseOutputItemUnion{
		{Type: "reasoning", ID: "rs_1", EncryptedContent: "opaque-1"},
		{Type: "message", Content: []oairesponses.ResponseOutputMessageContentUnion{{Type: "output_text", Text: "Shipped."}}},
	})
	if err != nil {
		t.Fatalf("convertOutputItems() err = %v", err)
	}
	if len(parts) != 2 || !parts[0].Thought || parts[0].Text != "" {
		t.Fatalf("parts = %+v, want an empty thought then the answer", parts)
	}
	if _, ok := decodeReasoningSignature(parts[0].ThoughtSignature); !ok {
		t.Errorf("the empty thought carries no reasoning signature")
	}
}

func TestReasoning_ReplayedAheadOfTheCallThatFollowedIt(t *testing.T) {
	got, err := convertResponse(toolTurnResponse(t, "gpt-test-2026-11-12"))
	if err != nil {
		t.Fatalf("convertResponse() err = %v", err)
	}
	// The response names a dated snapshot; the request, the alias.
	params, err := buildParamsWith("gpt-test", &model.LLMRequest{Contents: replayContents(got.Candidates[0].Content)}, Options{})
	if err != nil {
		t.Fatalf("buildParamsWith() err = %v", err)
	}
	if got, want := inputTypes(t, params), []string{"user", "reasoning", "function_call", "function_call_output"}; !slices.Equal(got, want) {
		t.Fatalf("input types = %v, want %v", got, want)
	}
	reasoning := params.Input.OfInputItemList[1].OfReasoning
	if reasoning.ID != "rs_1" || reasoning.EncryptedContent.Value != "opaque-1" {
		t.Errorf("reasoning item = %+v, want rs_1 with its encrypted content", reasoning)
	}
	if len(reasoning.Summary) != 1 || reasoning.Summary[0].Text != "Look the order up." {
		t.Errorf("reasoning summary = %+v, want the original summary", reasoning.Summary)
	}
	if !slices.Contains(params.Include, oairesponses.ResponseIncludableReasoningEncryptedContent) {
		t.Errorf("include = %v, want it to ask for encrypted reasoning", params.Include)
	}
}

func TestReasoning_DroppedWhenItCannotGoBack(t *testing.T) {
	signed := func(modelName string) *genai.Part {
		return &genai.Part{Thought: true, ThoughtSignature: reasoningSignature(modelName, oairesponses.ResponseOutputItemUnion{
			ID: "rs_1", EncryptedContent: "opaque-1",
		})}
	}
	call := &genai.Part{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "lookup_order", Args: map[string]any{}}}

	tests := []struct {
		name    string
		content *genai.Content
	}{
		{
			name:    "another model produced it",
			content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{signed("gpt-other"), call}},
		},
		{
			name:    "nothing that followed it survives",
			content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{call, signed("gpt-test")}},
		},
		{
			name: "another provider's signature",
			content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
				{Thought: true, Text: "pondering", ThoughtSignature: []byte("someone-else's")}, call,
			}},
		},
		{
			name: "a damaged payload",
			content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
				{Thought: true, ThoughtSignature: []byte(ReasoningSignaturePrefix + "{not json")}, call,
			}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			params, err := buildParamsWith("gpt-test", &model.LLMRequest{Contents: replayContents(tt.content)}, Options{})
			if err != nil {
				t.Fatalf("buildParamsWith() err = %v", err)
			}
			if got, want := inputTypes(t, params), []string{"user", "function_call", "function_call_output"}; !slices.Equal(got, want) {
				t.Errorf("input types = %v, want %v", got, want)
			}
		})
	}
}

func TestReasoning_UserTurnNeverSendsIt(t *testing.T) {
	part := &genai.Part{Thought: true, ThoughtSignature: reasoningSignature("gpt-test", oairesponses.ResponseOutputItemUnion{ID: "rs_1"})}
	params, err := buildParamsWith("gpt-test", &model.LLMRequest{Contents: []*genai.Content{
		{Role: genai.RoleUser, Parts: []*genai.Part{part, genai.NewPartFromText("Where is order 42?")}},
	}}, Options{})
	if err != nil {
		t.Fatalf("buildParamsWith() err = %v", err)
	}
	if got, want := inputTypes(t, params), []string{"user"}; !slices.Equal(got, want) {
		t.Errorf("input types = %v, want %v", got, want)
	}
}

func TestPhase_RoundTrips(t *testing.T) {
	parts, err := convertOutputItems("gpt-test", []oairesponses.ResponseOutputItemUnion{
		{Type: "message", Phase: "commentary", Content: []oairesponses.ResponseOutputMessageContentUnion{{Type: "output_text", Text: "Checking the warehouse."}}},
		{Type: "message", Phase: "final_answer", Content: []oairesponses.ResponseOutputMessageContentUnion{{Type: "output_text", Text: "It shipped."}}},
	})
	if err != nil {
		t.Fatalf("convertOutputItems() err = %v", err)
	}
	params, err := buildParamsWith("gpt-test", &model.LLMRequest{Contents: []*genai.Content{
		genai.NewContentFromText("Where is order 42?", genai.RoleUser),
		{Role: genai.RoleModel, Parts: parts},
		genai.NewContentFromText("Thanks.", genai.RoleUser),
	}}, Options{})
	if err != nil {
		t.Fatalf("buildParamsWith() err = %v", err)
	}
	var phases []string
	for _, item := range params.Input.OfInputItemList {
		if msg := item.OfOutputMessage; msg != nil {
			phases = append(phases, string(msg.Phase))
		}
	}
	if want := []string{"commentary", "final_answer"}; !slices.Equal(phases, want) {
		t.Errorf("assistant message phases = %v, want %v", phases, want)
	}
}

func TestReasoning_StreamedTurnCarriesIt(t *testing.T) {
	events := []string{
		`{"type":"response.created","response":{"id":"resp_1","model":"gpt-test","status":"in_progress","output":[]}}`,
		`{"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[]}}`,
		`{"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"summary_index":0,"delta":"Look the order up."}`,
		`{"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1","encrypted_content":"opaque-1","summary":[{"type":"summary_text","text":"Look the order up."}]}}`,
		`{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup_order","arguments":""}}`,
		`{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":1,"delta":"{\"order_id\":\"42\"}"}`,
		`{"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":1,"arguments":"{\"order_id\":\"42\"}"}`,
		`{"type":"response.completed","response":{"id":"resp_1","model":"gpt-test","status":"completed","output":[` +
			`{"type":"reasoning","id":"rs_1","encrypted_content":"opaque-1","summary":[{"type":"summary_text","text":"Look the order up."}]},` +
			`{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup_order","arguments":"{\"order_id\":\"42\"}"}]}}`,
	}
	responses, err := runStream(t, events...)
	if err != nil {
		t.Fatalf("runStream() err = %v", err)
	}
	final := responses[len(responses)-1]
	if final.Partial || !final.TurnComplete {
		t.Fatalf("last response is not the final turn: %+v", final)
	}
	var signed int
	for i, part := range final.Content.Parts {
		if _, ok := decodeReasoningSignature(part.ThoughtSignature); ok {
			signed++
			if i+1 >= len(final.Content.Parts) || final.Content.Parts[i+1].FunctionCall == nil {
				t.Errorf("the reasoning part is not followed by the call: %+v", final.Content.Parts)
			}
		}
	}
	if signed != 1 {
		t.Errorf("final turn carries %d reasoning signatures, want 1", signed)
	}
}

func equalPayload(a, b reasoningPayload) bool {
	return a.Model == b.Model && a.ID == b.ID && a.EncryptedContent == b.EncryptedContent &&
		strings.Join(a.Summary, "\x00") == strings.Join(b.Summary, "\x00")
}
