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
