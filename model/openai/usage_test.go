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

package openai

import (
	"encoding/json"
	"io"
	"net/http"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// usageBody is a completed response reporting reasoning, cache reads and
// cache writes.
const usageBody = `{"id":"r","object":"response","model":"gpt-test-2026-11-12","status":"completed",` +
	`"output":[{"type":"message","id":"m","role":"assistant","status":"completed",` +
	`"content":[{"type":"output_text","text":"hi","annotations":[]}]}],` +
	`"usage":{"input_tokens":1000,"input_tokens_details":{"cached_tokens":600,"cache_write_tokens":300},` +
	`"output_tokens":100,"output_tokens_details":{"reasoning_tokens":60},"total_tokens":1100}}`

// finalResponse sends req through a model built from cfg against a server
// answering with body, and returns the last response.
func finalResponse(t *testing.T, cfg Config, req *model.LLMRequest, body string) *model.LLMResponse {
	t.Helper()
	server := newLoopbackServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	})
	cfg.APIKey = "test"
	cfg.BaseURL = server.URL + "/v1"
	cfg.HTTPClient = server.Client()
	cfg.Options = append(cfg.Options, blockingAsStream)
	llm, err := NewModel(t.Context(), "gpt-test", &cfg)
	if err != nil {
		t.Fatalf("NewModel() err = %v", err)
	}
	var last *model.LLMResponse
	for resp, err := range llm.GenerateContent(t.Context(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent() err = %v", err)
		}
		last = resp
	}
	if last == nil {
		t.Fatal("no response")
	}
	return last
}

func TestUsage_CountsEachTokenOnce(t *testing.T) {
	usage := finalResponse(t, Config{}, userText("hi"), usageBody).UsageMetadata
	if usage == nil {
		t.Fatal("UsageMetadata = nil")
	}
	if usage.PromptTokenCount != 1000 {
		t.Errorf("PromptTokenCount = %d, want every input token, 1000", usage.PromptTokenCount)
	}
	if usage.CachedContentTokenCount != 600 {
		t.Errorf("CachedContentTokenCount = %d, want 600", usage.CachedContentTokenCount)
	}
	if usage.CandidatesTokenCount != 40 || usage.ThoughtsTokenCount != 60 {
		t.Errorf("CandidatesTokenCount, ThoughtsTokenCount = %d, %d, want 40, 60: output_tokens already includes reasoning",
			usage.CandidatesTokenCount, usage.ThoughtsTokenCount)
	}
}

func TestUsage_CacheWritesByTTL(t *testing.T) {
	withInstruction := func() *model.LLMRequest {
		req := userText("hi")
		req.Config = &genai.GenerateContentConfig{SystemInstruction: genai.NewContentFromText("You track orders.", genai.RoleUser)}
		return req
	}
	tests := []struct {
		name string
		cfg  Config
		want CacheWriteInputTokens
	}{
		{
			name: "implicit breakpoint only",
			cfg:  Config{},
			want: CacheWriteInputTokens{TotalInputTokens: 300, TTL30mInputTokens: 300},
		},
		{
			name: "a 12-hour breakpoint",
			cfg:  Config{PromptCaching: &PromptCachingConfig{SystemInstruction: &CacheBreakpoint{TTL: CacheTTL12h}}},
			want: CacheWriteInputTokens{TotalInputTokens: 300, TTL12hInputTokens: 300},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := finalResponse(t, tt.cfg, withInstruction(), usageBody)
			got, ok := CacheWriteInputTokensFromResponse(resp)
			if !ok || got != tt.want {
				t.Errorf("CacheWriteInputTokensFromResponse() = %+v, %v, want %+v, true", got, ok, tt.want)
			}

			// A session store round-trips CustomMetadata through JSON.
			raw, err := json.Marshal(resp.CustomMetadata)
			if err != nil {
				t.Fatalf("marshal metadata: %v", err)
			}
			restored := &model.LLMResponse{}
			if err := json.Unmarshal(raw, &restored.CustomMetadata); err != nil {
				t.Fatalf("unmarshal metadata: %v", err)
			}
			if got, ok := CacheWriteInputTokensFromResponse(restored); !ok || got != tt.want {
				t.Errorf("after a JSON round trip = %+v, %v, want %+v, true", got, ok, tt.want)
			}
		})
	}

	t.Run("no writes reports none", func(t *testing.T) {
		if got, ok := CacheWriteInputTokensFromResponse(finalResponse(t, Config{}, userText("hi"), completedBody)); ok {
			t.Errorf("CacheWriteInputTokensFromResponse() = %+v, true, want false", got)
		}
	})
}
