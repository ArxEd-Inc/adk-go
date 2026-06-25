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

	"google.golang.org/genai"

	"google.golang.org/adk/model"
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
