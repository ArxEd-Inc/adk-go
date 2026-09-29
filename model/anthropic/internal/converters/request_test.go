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
	"encoding/json"
	"slices"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/genai"
)

// markedPart returns the given part with PartCacheBreakpointMetadataKey set.
func markedPart(part *genai.Part) *genai.Part {
	part.PartMetadata = map[string]any{PartCacheBreakpointMetadataKey: true}
	return part
}

// TestContentsToMessagesWithMarkedBlocks_OrdinalsIndexTheWireBlocks pins that
// a marked part's reported ordinal indexes the block it converts to in the
// flattened wire messages: parts that yield no block (nil, empty) are skipped,
// and a same-role merge moves a later content's blocks into the preceding
// message without renumbering them.
func TestContentsToMessagesWithMarkedBlocks_OrdinalsIndexTheWireBlocks(t *testing.T) {
	contents := []*genai.Content{
		{Role: "user", Parts: []*genai.Part{
			nil,
			{Text: "Field guide, pages 12-14:"},
			{}, // yields no block
			markedPart(&genai.Part{InlineData: &genai.Blob{MIMEType: "application/pdf", Data: []byte("%PDF-1.4")}}),
		}},
		// Same role as the previous content: merged into it on the wire.
		{Role: "user", Parts: []*genai.Part{
			markedPart(&genai.Part{Text: "Specimen notes for plot 7."}),
			{Text: "Which species is this?"},
		}},
		{Role: "model", Parts: []*genai.Part{{Text: "Looking at the plates."}}},
	}

	messages, ordinals, err := ContentsToMessagesWithMarkedBlocks(contents, ContentsOptions{})
	if err != nil {
		t.Fatalf("ContentsToMessagesWithMarkedBlocks: %v", err)
	}
	if len(messages) != 2 {
		t.Fatalf("messages = %d, want 2 (the two user contents merged)", len(messages))
	}
	if got := len(messages[0].Content); got != 4 {
		t.Fatalf("merged user message blocks = %d, want 4", got)
	}
	if want := []int{1, 2}; !slices.Equal(ordinals, want) {
		t.Fatalf("marked block ordinals = %v, want %v", ordinals, want)
	}
	if messages[0].Content[1].OfDocument == nil {
		t.Errorf("ordinal 1 names block %+v, want the marked document block", messages[0].Content[1])
	}
	if text := messages[0].Content[2].OfText; text == nil || text.Text != "Specimen notes for plot 7." {
		t.Errorf("ordinal 2 names block %+v, want the marked notes text block", messages[0].Content[2])
	}
}

// TestContentsToMessagesWithMarkedBlocks_NoMarksReportsNil pins the nil
// ordinals of a request with no marked part, and that ContentsToMessages
// converts identically.
func TestContentsToMessagesWithMarkedBlocks_NoMarksReportsNil(t *testing.T) {
	contents := []*genai.Content{genai.NewContentFromText("Which species is this?", "user")}

	messages, ordinals, err := ContentsToMessagesWithMarkedBlocks(contents, ContentsOptions{})
	if err != nil {
		t.Fatalf("ContentsToMessagesWithMarkedBlocks: %v", err)
	}
	if ordinals != nil {
		t.Errorf("marked block ordinals = %v, want nil", ordinals)
	}
	plainMessages, err := ContentsToMessages(contents)
	if err != nil {
		t.Fatalf("ContentsToMessages: %v", err)
	}
	if len(plainMessages) != len(messages) || len(plainMessages[0].Content) != len(messages[0].Content) {
		t.Errorf("ContentsToMessages = %+v, want the same messages as the marked-blocks variant %+v",
			plainMessages, messages)
	}
}

// TestIsCacheBreakpointMarked pins the marker predicate's nil safety and its
// insistence on a true boolean under the key.
func TestIsCacheBreakpointMarked(t *testing.T) {
	cases := []struct {
		name string
		part *genai.Part
		want bool
	}{
		{name: "nil part", part: nil},
		{name: "no metadata", part: &genai.Part{Text: "x"}},
		{name: "other key", part: &genai.Part{PartMetadata: map[string]any{"other": true}}},
		{name: "false", part: &genai.Part{PartMetadata: map[string]any{PartCacheBreakpointMetadataKey: false}}},
		{name: "non-boolean", part: &genai.Part{PartMetadata: map[string]any{PartCacheBreakpointMetadataKey: "true"}}},
		{name: "true", part: &genai.Part{PartMetadata: map[string]any{PartCacheBreakpointMetadataKey: true}}, want: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsCacheBreakpointMarked(tc.part); got != tc.want {
				t.Errorf("IsCacheBreakpointMarked = %v, want %v", got, tc.want)
			}
		})
	}
}

// fieldGuideDeclaration returns a declaration with a one-property schema.
func fieldGuideDeclaration(name string) *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{
		Name:        name,
		Description: "Field guide tool " + name + ".",
		ParametersJsonSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{"plot": map[string]any{"type": "integer"}},
			"required":   []any{"plot"},
		},
	}
}

// TestToolsToAnthropicTools_DefersNamedDeclarations pins that the predicate
// sets defer_loading on the declarations it names and nothing else changes:
// every schema converts as it does without deferral, and a nil predicate
// defers nothing.
func TestToolsToAnthropicTools_DefersNamedDeclarations(t *testing.T) {
	tools := []*genai.Tool{{FunctionDeclarations: []*genai.FunctionDeclaration{
		fieldGuideDeclaration("lookupSpecies"),
		fieldGuideDeclaration("describeHabitat"),
	}}}

	deferred, _ := ToolsToAnthropicTools(tools, func(name string) bool { return name == "describeHabitat" })
	plain, _ := ToolsToAnthropicTools(tools, nil)
	if len(deferred) != 2 || len(plain) != 2 {
		t.Fatalf("converted %d and %d tools, want 2 each", len(deferred), len(plain))
	}
	if IsDeferredTool(deferred[0]) || !IsDeferredTool(deferred[1]) {
		t.Errorf("deferred = [%v %v], want [false true]", IsDeferredTool(deferred[0]), IsDeferredTool(deferred[1]))
	}
	if IsDeferredTool(plain[0]) || IsDeferredTool(plain[1]) {
		t.Errorf("a nil predicate deferred a tool: %+v", plain)
	}
	for i := range deferred {
		gotSchema, err := json.Marshal(deferred[i].OfTool.InputSchema)
		if err != nil {
			t.Fatalf("marshal schema: %v", err)
		}
		wantSchema, err := json.Marshal(plain[i].OfTool.InputSchema)
		if err != nil {
			t.Fatalf("marshal schema: %v", err)
		}
		if string(gotSchema) != string(wantSchema) {
			t.Errorf("tool %d schema = %s, want the undeferred %s", i, gotSchema, wantSchema)
		}
	}
	if got := DeferredToolNames(deferred); len(got) != 1 {
		t.Errorf("DeferredToolNames = %v, want only describeHabitat", got)
	} else if _, ok := got["describeHabitat"]; !ok {
		t.Errorf("DeferredToolNames = %v, want only describeHabitat", got)
	}
	if got := DeferredToolNames(plain); got != nil {
		t.Errorf("DeferredToolNames of undeferred tools = %v, want nil", got)
	}
}

// toolReferenceOptions defers describeHabitat and countSpecimens and reads
// references under "toolReferences".
var toolReferenceOptions = ContentsOptions{
	ToolReferencesResponseKey: "toolReferences",
	DeferredToolNames:         map[string]struct{}{"describeHabitat": {}, "countSpecimens": {}},
}

// toolResultContents returns a lookupSpecies call and its response.
func toolResultContents(response map[string]any) []*genai.Content {
	return []*genai.Content{
		genai.NewContentFromText("Which species live on plot 7?", "user"),
		{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "lookupSpecies"}}}},
		{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
			ID: "call_1", Name: "lookupSpecies", Response: response,
		}}}},
	}
}

// TestContentsToMessagesWithMarkedBlocks_ToolReferences pins how a function
// response carrying tool references converts: the names of deferred tools
// become a reference-only tool_result, the rest of the response a labeled text
// block after it, and every other shape a single JSON tool_result.
func TestContentsToMessagesWithMarkedBlocks_ToolReferences(t *testing.T) {
	cases := []struct {
		name           string
		opts           ContentsOptions
		response       map[string]any
		wantReferences []string
		// wantJSON is the tool_result's JSON text when no reference survives,
		// else the JSON after the continuation label.
		wantJSON string
	}{
		{
			name:           "references to deferred tools",
			opts:           toolReferenceOptions,
			response:       map[string]any{"guide": "Plot 7 is wetland.", "toolReferences": []string{"describeHabitat", "countSpecimens"}},
			wantReferences: []string{"describeHabitat", "countSpecimens"},
			wantJSON:       `{"guide":"Plot 7 is wetland."}`,
		},
		{
			name:           "references read back from a JSON round trip, with a duplicate",
			opts:           toolReferenceOptions,
			response:       map[string]any{"guide": "Plot 7 is wetland.", "toolReferences": []any{"countSpecimens", "countSpecimens"}},
			wantReferences: []string{"countSpecimens"},
			wantJSON:       `{"guide":"Plot 7 is wetland."}`,
		},
		{
			name:           "references to undefined and undeferred tools are dropped",
			opts:           toolReferenceOptions,
			response:       map[string]any{"guide": "g", "toolReferences": []string{"lookupSpecies", "describeHabitat", "retiredTool"}},
			wantReferences: []string{"describeHabitat"},
			wantJSON:       `{"guide":"g"}`,
		},
		{
			name:     "every reference dropped converts to plain JSON without the key",
			opts:     toolReferenceOptions,
			response: map[string]any{"guide": "g", "toolReferences": []string{"lookupSpecies", "retiredTool"}},
			wantJSON: `{"guide":"g"}`,
		},
		{
			name:           "references with nothing else yield no text block",
			opts:           toolReferenceOptions,
			response:       map[string]any{"toolReferences": []string{"describeHabitat"}},
			wantReferences: []string{"describeHabitat"},
		},
		{
			name:     "a response without the key is unchanged",
			opts:     toolReferenceOptions,
			response: map[string]any{"guide": "a < b & c"},
			wantJSON: `{"guide":"a < b & c"}`,
		},
		{
			name:     "a key holding anything but strings is left in the JSON",
			opts:     toolReferenceOptions,
			response: map[string]any{"guide": "g", "toolReferences": []any{"describeHabitat", 7}},
			wantJSON: `{"guide":"g","toolReferences":["describeHabitat",7]}`,
		},
		{
			name:     "no configured key leaves the references in the JSON",
			opts:     ContentsOptions{DeferredToolNames: toolReferenceOptions.DeferredToolNames},
			response: map[string]any{"toolReferences": []string{"describeHabitat"}},
			wantJSON: `{"toolReferences":["describeHabitat"]}`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			messages, _, err := ContentsToMessagesWithMarkedBlocks(toolResultContents(tc.response), tc.opts)
			if err != nil {
				t.Fatalf("ContentsToMessagesWithMarkedBlocks: %v", err)
			}
			if len(messages) != 3 {
				t.Fatalf("messages = %d, want 3", len(messages))
			}
			blocks := messages[2].Content
			result := blocks[0].OfToolResult
			if result == nil || result.ToolUseID != "call_1" {
				t.Fatalf("first block = %+v, want the call_1 tool_result", blocks[0])
			}

			if tc.wantReferences == nil {
				if len(blocks) != 1 || len(result.Content) != 1 || result.Content[0].OfText == nil {
					t.Fatalf("blocks = %+v, want one tool_result of one text block", blocks)
				}
				if got := result.Content[0].OfText.Text; got != tc.wantJSON {
					t.Errorf("tool_result JSON = %s, want %s", got, tc.wantJSON)
				}
				return
			}

			var references []string
			for _, content := range result.Content {
				if content.OfToolReference == nil {
					t.Fatalf("tool_result content %+v mixes a non-reference block with references", content)
				}
				references = append(references, content.OfToolReference.ToolName)
			}
			if !slices.Equal(references, tc.wantReferences) {
				t.Errorf("references = %v, want %v", references, tc.wantReferences)
			}
			if tc.wantJSON == "" {
				if len(blocks) != 1 {
					t.Errorf("blocks = %+v, want the tool_result alone", blocks)
				}
				return
			}
			if len(blocks) != 2 || blocks[1].OfText == nil {
				t.Fatalf("blocks = %+v, want the tool_result then a text block", blocks)
			}
			if want := "Tool result call_1, continued:\n" + tc.wantJSON; blocks[1].OfText.Text != want {
				t.Errorf("text block = %q, want %q", blocks[1].OfText.Text, want)
			}
		})
	}
}

// TestContentsToMessagesWithMarkedBlocks_ToolResultsPrecedeOtherBlocks pins
// that every tool_result of a user message comes before its other blocks, as
// Anthropic requires: a response's continuation text follows the tool results
// of its own content and of a later content merged into the same message.
func TestContentsToMessagesWithMarkedBlocks_ToolResultsPrecedeOtherBlocks(t *testing.T) {
	contents := []*genai.Content{
		genai.NewContentFromText("Survey plots 7 and 9.", "user"),
		{Role: "model", Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "lookupSpecies"}},
			{FunctionCall: &genai.FunctionCall{ID: "call_2", Name: "lookupSpecies"}},
			{FunctionCall: &genai.FunctionCall{ID: "call_3", Name: "lookupSpecies"}},
		}},
		{Role: "user", Parts: []*genai.Part{
			{FunctionResponse: &genai.FunctionResponse{ID: "call_1", Response: map[string]any{
				"guide": "Plot 7 is wetland.", "toolReferences": []string{"describeHabitat"},
			}}},
			{FunctionResponse: &genai.FunctionResponse{ID: "call_2", Response: map[string]any{"count": 3}}},
		}},
		// Same role as the previous content: merged into it on the wire.
		{Role: "user", Parts: []*genai.Part{
			{FunctionResponse: &genai.FunctionResponse{ID: "call_3", Response: map[string]any{"count": 5}}},
		}},
	}

	messages, _, err := ContentsToMessagesWithMarkedBlocks(contents, toolReferenceOptions)
	if err != nil {
		t.Fatalf("ContentsToMessagesWithMarkedBlocks: %v", err)
	}
	if len(messages) != 3 {
		t.Fatalf("messages = %d, want 3", len(messages))
	}
	var order []string
	for _, block := range messages[2].Content {
		switch {
		case block.OfToolResult != nil:
			order = append(order, block.OfToolResult.ToolUseID)
		case block.OfText != nil:
			order = append(order, "text")
		default:
			order = append(order, "other")
		}
	}
	if want := []string{"call_1", "call_2", "call_3", "text"}; !slices.Equal(order, want) {
		t.Errorf("user message blocks = %v, want %v", order, want)
	}
}

// TestContentsToMessagesWithMarkedBlocks_OrdinalsFollowToolResultsFirst pins
// that a marked block moved by the tool-results-first reordering reports its
// new position.
func TestContentsToMessagesWithMarkedBlocks_OrdinalsFollowToolResultsFirst(t *testing.T) {
	contents := []*genai.Content{
		genai.NewContentFromText("Survey plot 7.", "user"),
		{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "lookupSpecies"}}}},
		{Role: "user", Parts: []*genai.Part{markedPart(&genai.Part{Text: "Plot 7 photographs follow."})}},
		{Role: "user", Parts: []*genai.Part{
			{FunctionResponse: &genai.FunctionResponse{ID: "call_1", Response: map[string]any{"count": 3}}},
		}},
	}

	messages, ordinals, err := ContentsToMessagesWithMarkedBlocks(contents, ContentsOptions{})
	if err != nil {
		t.Fatalf("ContentsToMessagesWithMarkedBlocks: %v", err)
	}
	if len(messages) != 3 || len(messages[2].Content) != 2 {
		t.Fatalf("messages = %+v, want three with two blocks in the last", messages)
	}
	if messages[2].Content[0].OfToolResult == nil {
		t.Fatalf("last message blocks = %+v, want the tool_result first", messages[2].Content)
	}
	// The first two messages hold one block each, so the marked text, now the
	// last message's second block, is flat ordinal 3.
	if want := []int{3}; !slices.Equal(ordinals, want) {
		t.Fatalf("marked block ordinals = %v, want %v", ordinals, want)
	}
	if text := messages[2].Content[1].OfText; text == nil || text.Text != "Plot 7 photographs follow." {
		t.Errorf("ordinal 3 names block %+v, want the marked text block", messages[2].Content[1])
	}
}

// TestContentsToMessagesWithMarkedBlocks_RepeatedReferenceExpandsAtTheLatest
// pins that a tool two responses reference expands only at the later one: the
// earlier response keeps its other references, or converts to plain JSON
// without the key when it has none left.
func TestContentsToMessagesWithMarkedBlocks_RepeatedReferenceExpandsAtTheLatest(t *testing.T) {
	loadResponse := func(id string, response map[string]any) []*genai.Content {
		return []*genai.Content{
			{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: id, Name: "lookupSpecies"}}}},
			{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
				ID: id, Name: "lookupSpecies", Response: response,
			}}}},
		}
	}
	cases := []struct {
		name                string
		earlierReferences   []any
		wantEarlierBlocks   []string
		wantEarlierJSONText string
	}{
		{
			name:                "the earlier response keeps its other references",
			earlierReferences:   []any{"describeHabitat", "countSpecimens"},
			wantEarlierBlocks:   []string{"reference:countSpecimens", "text"},
			wantEarlierJSONText: "Tool result call_1, continued:\n" + `{"guide":"earlier survey"}`,
		},
		{
			name:                "the earlier response with no reference left converts to plain JSON",
			earlierReferences:   []any{"describeHabitat"},
			wantEarlierBlocks:   []string{"json"},
			wantEarlierJSONText: `{"guide":"earlier survey"}`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			contents := []*genai.Content{genai.NewContentFromText("Survey plot 7 again.", "user")}
			contents = append(contents, loadResponse("call_1", map[string]any{
				"guide": "earlier survey", "toolReferences": tc.earlierReferences,
			})...)
			contents = append(contents, loadResponse("call_2", map[string]any{
				"guide": "revised survey", "toolReferences": []string{"describeHabitat"},
			})...)

			messages, _, err := ContentsToMessagesWithMarkedBlocks(contents, toolReferenceOptions)
			if err != nil {
				t.Fatalf("ContentsToMessagesWithMarkedBlocks: %v", err)
			}
			if len(messages) != 5 {
				t.Fatalf("messages = %d, want 5", len(messages))
			}

			describe := func(blocks []anthropic.ContentBlockParamUnion) ([]string, string) {
				var shape []string
				var text string
				for _, block := range blocks {
					switch {
					case block.OfToolResult != nil:
						for _, content := range block.OfToolResult.Content {
							if content.OfToolReference != nil {
								shape = append(shape, "reference:"+content.OfToolReference.ToolName)
							} else if content.OfText != nil {
								shape = append(shape, "json")
								text = content.OfText.Text
							}
						}
					case block.OfText != nil:
						shape = append(shape, "text")
						text = block.OfText.Text
					}
				}
				return shape, text
			}

			earlierShape, earlierText := describe(messages[2].Content)
			if !slices.Equal(earlierShape, tc.wantEarlierBlocks) || earlierText != tc.wantEarlierJSONText {
				t.Errorf("earlier response = %v %q, want %v %q",
					earlierShape, earlierText, tc.wantEarlierBlocks, tc.wantEarlierJSONText)
			}
			laterShape, laterText := describe(messages[4].Content)
			wantLaterText := "Tool result call_2, continued:\n" + `{"guide":"revised survey"}`
			if want := []string{"reference:describeHabitat", "text"}; !slices.Equal(laterShape, want) ||
				laterText != wantLaterText {
				t.Errorf("later response = %v %q, want %v %q", laterShape, laterText, want, wantLaterText)
			}
		})
	}
}
