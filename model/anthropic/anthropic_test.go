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
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/anthropic/internal/converters"
)

func ptr[T any](v T) *T { return &v }

func userReq(cfg *genai.GenerateContentConfig) *model.LLMRequest {
	return &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", "user")},
		Config:   cfg,
	}
}

// TestConvertRequestPerModelEffort: a model configured with EffortXHigh, given a
// request with ThinkingLevelHigh (which enables thinking), must carry adaptive
// thinking with the model's xhigh effort — not the level-derived high.
func TestConvertRequestPerModelEffort(t *testing.T) {
	m := &anthropicModel{name: "claude-opus-4-8", defaultMaxTokens: 64000, effort: EffortXHigh}
	params, _, err := m.convertRequest(userReq(&genai.GenerateContentConfig{
		ThinkingConfig: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelHigh},
	}))
	if err != nil {
		t.Fatalf("convertRequest: %v", err)
	}
	if params.Thinking.OfAdaptive == nil {
		t.Fatalf("thinking not adaptive")
	}
	if params.OutputConfig.Effort != EffortXHigh {
		t.Fatalf("effort = %q, want xhigh", params.OutputConfig.Effort)
	}
}

// TestConvertRequestEffortFallsBackToLevel: with no per-model effort, the effort
// derives from the request's ThinkingLevel.
func TestConvertRequestEffortFallsBackToLevel(t *testing.T) {
	m := &anthropicModel{name: "claude-sonnet-4-6", defaultMaxTokens: 64000}
	params, _, err := m.convertRequest(userReq(&genai.GenerateContentConfig{
		ThinkingConfig: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelHigh},
	}))
	if err != nil {
		t.Fatalf("convertRequest: %v", err)
	}
	if params.OutputConfig.Effort != EffortHigh {
		t.Fatalf("effort = %q, want high", params.OutputConfig.Effort)
	}
}

// TestConvertRequestNoThinking: a request that sets no ThinkingConfig runs
// without thinking, and no effort is applied even though the model is configured
// with one.
func TestConvertRequestNoThinking(t *testing.T) {
	m := &anthropicModel{name: "claude-sonnet-4-6", defaultMaxTokens: 64000, effort: EffortHigh}
	params, _, err := m.convertRequest(userReq(&genai.GenerateContentConfig{}))
	if err != nil {
		t.Fatalf("convertRequest: %v", err)
	}
	if params.Thinking.OfAdaptive != nil {
		t.Fatalf("thinking should be off with no ThinkingConfig")
	}
	if params.OutputConfig.Effort != "" {
		t.Fatalf("effort should be unset when thinking is off, got %q", params.OutputConfig.Effort)
	}
}

// TestConvertRequestBudgetModeThinking: a model in ThinkingModeBudget (e.g. the
// Haiku integration-testing tier), given a request with ThinkingLevelHigh, must
// carry manual budget_tokens thinking and neither adaptive thinking nor an
// effort — both of which Haiku rejects with a 400.
func TestConvertRequestBudgetModeThinking(t *testing.T) {
	m := &anthropicModel{name: "claude-haiku-4-5", defaultMaxTokens: 64000, thinkingMode: ThinkingModeBudget}
	params, _, err := m.convertRequest(userReq(&genai.GenerateContentConfig{
		ThinkingConfig: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelHigh},
	}))
	if err != nil {
		t.Fatalf("convertRequest: %v", err)
	}
	if params.Thinking.OfEnabled == nil {
		t.Fatalf("thinking not in budget_tokens form")
	}
	if got := params.Thinking.OfEnabled.BudgetTokens; got != 24000 {
		t.Fatalf("budget_tokens = %d, want 24000", got)
	}
	if params.Thinking.OfAdaptive != nil {
		t.Fatalf("budget mode emitted adaptive thinking")
	}
	if params.OutputConfig.Effort != "" {
		t.Fatalf("budget mode set effort %q; want none", params.OutputConfig.Effort)
	}
}

// TestConvertRequestDropsSamplingParams: the latest models reject
// temperature/top_p/top_k, so they must never reach the wire.
func TestConvertRequestDropsSamplingParams(t *testing.T) {
	m := &anthropicModel{name: "claude-opus-4-8", defaultMaxTokens: 64000, effort: EffortXHigh}
	params, _, err := m.convertRequest(userReq(&genai.GenerateContentConfig{
		Temperature: ptr(float32(0.7)),
		TopP:        ptr(float32(0.9)),
	}))
	if err != nil {
		t.Fatalf("convertRequest: %v", err)
	}
	data, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal params: %v", err)
	}
	for _, field := range []string{"temperature", "top_p", "top_k"} {
		if strings.Contains(string(data), field) {
			t.Errorf("request JSON unexpectedly contains %q: %s", field, data)
		}
	}
}

// TestConvertRequestResponseJsonSchema: a request that sets ResponseJsonSchema (the raw JSON-schema
// form, as opposed to a structured ResponseSchema) carries it through as the output format schema,
// made strict (additionalProperties:false added by enforceStrictObjectSchema).
func TestConvertRequestResponseJsonSchema(t *testing.T) {
	m := &anthropicModel{name: "claude-sonnet-4-6", defaultMaxTokens: 64000}
	params, _, err := m.convertRequest(userReq(&genai.GenerateContentConfig{
		ResponseJsonSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{"name": map[string]any{"type": "string"}},
			"required":   []any{"name"},
		},
	}))
	if err != nil {
		t.Fatalf("convertRequest: %v", err)
	}

	schema := params.OutputConfig.Format.Schema
	if schema == nil {
		t.Fatalf("output format schema not set from ResponseJsonSchema")
	}
	if additionalProperties, ok := schema["additionalProperties"].(bool); !ok || additionalProperties {
		t.Errorf("additionalProperties = %v, want false (enforceStrictObjectSchema not applied)", schema["additionalProperties"])
	}
	if properties, ok := schema["properties"].(map[string]any); !ok || properties["name"] == nil {
		t.Errorf("output format schema lost its properties: %v", schema)
	}
}

// TestGenerateContentWrapsConversionFailureWithErrRequestConversion: a request
// the converter rejects — here inline data of a MIME type Anthropic doesn't
// accept — fails with an error wrapping ErrRequestConversion on both the
// streaming and non-streaming paths, so callers can recognize the failure as
// permanent rather than retrying it.
func TestGenerateContentWrapsConversionFailureWithErrRequestConversion(t *testing.T) {
	cases := []struct {
		name   string
		stream bool
	}{
		{"non-streaming", false},
		{"streaming", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := &anthropicModel{name: "claude-sonnet-4-6", defaultMaxTokens: 64000}
			req := &model.LLMRequest{
				Contents: []*genai.Content{{
					Role: "user",
					Parts: []*genai.Part{{
						InlineData: &genai.Blob{Data: []byte("<html></html>"), MIMEType: "text/html"},
					}},
				}},
			}

			var gotErr error
			for _, err := range m.GenerateContent(context.Background(), req, tc.stream) {
				gotErr = err
			}
			if !errors.Is(gotErr, ErrRequestConversion) {
				t.Fatalf("GenerateContent error = %v, want an error wrapping ErrRequestConversion", gotErr)
			}
		})
	}
}

// TestConvertRequestMarkedPartCarriesCacheControl: a part marked with
// MarkCacheBreakpoint, converted under a MarkedPart layout, yields its wire
// block — here a PDF's document block — carrying cache_control, and nothing
// else in the single-user-message request does.
func TestConvertRequestMarkedPartCarriesCacheControl(t *testing.T) {
	m := &anthropicModel{
		name:             "claude-opus-4-8",
		defaultMaxTokens: 64000,
		promptCaching:    &PromptCachingConfig{MarkedPart: &CacheBreakpoint{}},
	}
	seededDocument := &genai.Part{InlineData: &genai.Blob{MIMEType: "application/pdf", Data: []byte("%PDF-1.4")}}
	MarkCacheBreakpoint(seededDocument)
	params, _, err := m.convertRequest(&model.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{
			{Text: "Field guide, pages 12-14:"},
			seededDocument,
			{Text: "Specimen notes for plot 7."},
			{Text: "Which species is this?"},
		}}},
	})
	if err != nil {
		t.Fatalf("convertRequest: %v", err)
	}
	if len(params.Messages) != 1 || len(params.Messages[0].Content) != 4 {
		t.Fatalf("messages = %+v, want one user message of four blocks", params.Messages)
	}
	documentBlock := params.Messages[0].Content[1]
	if documentBlock.OfDocument == nil {
		t.Fatalf("block 1 = %+v, want the document block", documentBlock)
	}
	if documentBlock.OfDocument.CacheControl.Type == "" {
		t.Error("marked document block carries no cache_control")
	}
	if got := markerCount(t, params); got != 1 {
		t.Errorf("marshaled marker count = %d, want 1 (the marked block alone)", got)
	}
}

// TestMarkCacheBreakpoint pins the marker helper: it allocates a nil metadata
// map, preserves existing keys, and tolerates a nil part.
func TestMarkCacheBreakpoint(t *testing.T) {
	MarkCacheBreakpoint(nil)

	bare := &genai.Part{Text: "x"}
	if IsCacheBreakpointMarked(bare) {
		t.Errorf("unmarked part reads as marked: %+v", bare.PartMetadata)
	}
	MarkCacheBreakpoint(bare)
	if !converters.IsCacheBreakpointMarked(bare) || !IsCacheBreakpointMarked(bare) {
		t.Errorf("part with no prior metadata not marked: %+v", bare.PartMetadata)
	}

	withMetadata := &genai.Part{Text: "y", PartMetadata: map[string]any{"other": "kept"}}
	MarkCacheBreakpoint(withMetadata)
	if !converters.IsCacheBreakpointMarked(withMetadata) || withMetadata.PartMetadata["other"] != "kept" {
		t.Errorf("part with prior metadata = %+v, want marked with the other key kept", withMetadata.PartMetadata)
	}
}

// deferringTool is a dispatch entry that reports deferral.
type deferringTool struct{ deferred bool }

func (t deferringTool) DeferLoading() bool { return t.deferred }

// TestIsToolLoadingDeferred pins that only a dispatch entry implementing
// DeferredLoadingTool and reporting true defers its tool.
func TestIsToolLoadingDeferred(t *testing.T) {
	req := &model.LLMRequest{Tools: map[string]any{
		"lookupSpecies":   "a dispatch entry of another type",
		"describeHabitat": deferringTool{deferred: true},
		"countSpecimens":  deferringTool{deferred: false},
	}}
	cases := []struct {
		req  *model.LLMRequest
		name string
		want bool
	}{
		{req: nil, name: "describeHabitat"},
		{req: req, name: "absentTool"},
		{req: req, name: "lookupSpecies"},
		{req: req, name: "countSpecimens"},
		{req: req, name: "describeHabitat", want: true},
	}
	for _, tc := range cases {
		if got := IsToolLoadingDeferred(tc.req, tc.name); got != tc.want {
			t.Errorf("IsToolLoadingDeferred(%q) = %v, want %v", tc.name, got, tc.want)
		}
	}
}

// TestConvertRequestDeferredToolsAndReferences: a request whose dispatch
// entries defer two of three tools, whose last tool result references one of
// them, converts to deferred declarations, a reference-only tool_result
// followed by the rest of the response as text, and markers only where
// Anthropic accepts them: the loader (the static prefix end and the last
// undeferred tool, collapsed), the system block and the newest block.
func TestConvertRequestDeferredToolsAndReferences(t *testing.T) {
	m := &anthropicModel{
		name:             "claude-opus-4-8",
		defaultMaxTokens: 64000,
		promptCaching: &PromptCachingConfig{
			ToolsStaticPrefixEnd:         &CacheBreakpoint{TTL: CacheTTL1h},
			ToolsStaticPrefixEndToolName: "lookupSpecies",
			Tools:                        &CacheBreakpoint{TTL: CacheTTL1h},
			SystemInstruction:            &CacheBreakpoint{TTL: CacheTTL1h},
			ConversationHistory:          &CacheBreakpoint{TTL: CacheTTL1h},
		},
		toolReferencesResponseKey: "toolReferences",
	}
	declaration := func(name string) *genai.FunctionDeclaration {
		return &genai.FunctionDeclaration{
			Name:        name,
			Description: "Field guide tool " + name + ".",
			ParametersJsonSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"plot": map[string]any{"type": "integer"}},
			},
		}
	}
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("Which species live on plot 7?", "user"),
			{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
				ID: "call_1", Name: "lookupSpecies", Args: map[string]any{"plot": 7},
			}}}},
			{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
				ID: "call_1", Name: "lookupSpecies", Response: map[string]any{
					"guide":          "Plot 7 is wetland; describeHabitat reads its survey.",
					"toolReferences": []string{"describeHabitat"},
				},
			}}}},
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: genai.NewContentFromText("You catalogue field surveys.", "user"),
			Tools: []*genai.Tool{{FunctionDeclarations: []*genai.FunctionDeclaration{
				declaration("lookupSpecies"), declaration("describeHabitat"), declaration("countSpecimens"),
			}}},
		},
		Tools: map[string]any{
			"lookupSpecies":   deferringTool{deferred: false},
			"describeHabitat": deferringTool{deferred: true},
			"countSpecimens":  deferringTool{deferred: true},
		},
	}

	params, _, err := m.convertRequest(req)
	if err != nil {
		t.Fatalf("convertRequest: %v", err)
	}
	data, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal params: %v", err)
	}
	wire := string(data)
	if got := strings.Count(wire, `"defer_loading":true`); got != 2 {
		t.Errorf(`"defer_loading":true appears %d times, want 2: %s`, got, wire)
	}
	if !strings.Contains(wire, `{"tool_name":"describeHabitat","type":"tool_reference"}`) {
		t.Errorf("request JSON lacks the describeHabitat tool_reference: %s", wire)
	}
	if !strings.Contains(wire, `Tool result call_1, continued:\n{\"guide\":`) {
		t.Errorf("request JSON lacks the labeled continuation text: %s", wire)
	}
	if strings.Contains(wire, `"toolReferences"`) {
		t.Errorf("request JSON still carries the references key: %s", wire)
	}
	for _, tool := range params.Tools[1:] {
		if tool.OfTool.CacheControl.Type != "" {
			t.Errorf("deferred tool %s carries cache_control", tool.OfTool.Name)
		}
	}
	if got := markerCount(t, params); got != 3 {
		t.Errorf("marshaled marker count = %d, want 3 (loader, system, newest block)", got)
	}
}
