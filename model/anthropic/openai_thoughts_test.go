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
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model/anthropic/internal/converters"
	"google.golang.org/adk/v2/model/openai"
)

// TestOpenAIThoughtPrefixMatchesModelOpenAI pins the prefix the converter
// repeats to the one model/openai marks its thoughts with.
func TestOpenAIThoughtPrefixMatchesModelOpenAI(t *testing.T) {
	if converters.OpenAIThoughtSignaturePrefix != openai.ThoughtSignaturePrefix {
		t.Fatalf("converters.OpenAIThoughtSignaturePrefix = %q, model/openai marks its thoughts with %q",
			converters.OpenAIThoughtSignaturePrefix, openai.ThoughtSignaturePrefix)
	}
}

// TestOpenAIThoughtsAreDropped checks that a conversation moved from an OpenAI
// model to Claude converts: the OpenAI thoughts are left out, and the call
// beside them is kept.
func TestOpenAIThoughtsAreDropped(t *testing.T) {
	contents := []*genai.Content{
		genai.NewContentFromText("Where is order 42?", genai.RoleUser),
		{Role: genai.RoleModel, Parts: []*genai.Part{
			{Thought: true, Text: "Look the order up.", ThoughtSignature: []byte(openai.ReasoningSignaturePrefix + `{"id":"rs_1"}`)},
			{Thought: true, Text: "Then report it.", ThoughtSignature: []byte(openai.ThoughtSignaturePrefix + "thought/v1")},
			{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "lookup_order", Args: map[string]any{"order_id": "42"}}},
		}},
		{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
			ID: "call_1", Name: "lookup_order", Response: map[string]any{"status": "shipped"},
		}}}},
	}
	messages, err := converters.ContentsToMessages(contents)
	if err != nil {
		t.Fatalf("ContentsToMessages() err = %v", err)
	}
	if len(messages) != 3 {
		t.Fatalf("got %d messages, want 3", len(messages))
	}
	assistant := messages[1].Content
	if len(assistant) != 1 || assistant[0].OfToolUse == nil {
		t.Errorf("assistant message = %+v, want the tool call alone", assistant)
	}
}
