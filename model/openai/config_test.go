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
	"errors"
	"io"
	"net/http"
	"sync/atomic"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// sendAndCapture sends req through a model built from cfg and returns the
// JSON body the server received. The server answers with completedBody.
func sendAndCapture(t *testing.T, cfg Config, req *model.LLMRequest) map[string]any {
	t.Helper()
	var body map[string]any
	server := newLoopbackServer(t, func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
		}
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Errorf("decode request body: %v", err)
		}
		writeCompleted(w, r)
	})
	cfg.APIKey = "test"
	cfg.BaseURL = server.URL + "/v1"
	cfg.HTTPClient = server.Client()
	llm, err := NewModel(t.Context(), "gpt-test", &cfg)
	if err != nil {
		t.Fatalf("NewModel() err = %v", err)
	}
	for _, err := range llm.GenerateContent(t.Context(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent() err = %v", err)
		}
	}
	if body == nil {
		t.Fatal("no request reached the server")
	}
	return body
}

func userText(text string) *model.LLMRequest {
	return &model.LLMRequest{Contents: []*genai.Content{genai.NewContentFromText(text, genai.RoleUser)}}
}

func TestNewModel_RequiresAPIKey(t *testing.T) {
	// The environment variable openai-go would otherwise fall back to must
	// not stand in for a missing key.
	t.Setenv("OPENAI_API_KEY", "from-the-environment")
	for name, cfg := range map[string]*Config{"nil config": nil, "empty key": {}} {
		t.Run(name, func(t *testing.T) {
			if _, err := NewModel(t.Context(), "gpt-test", cfg); !errors.Is(err, ErrAPIKeyRequired) {
				t.Fatalf("NewModel() err = %v, want %v", err, ErrAPIKeyRequired)
			}
		})
	}
}

func TestNewModel_DoesNotRetry(t *testing.T) {
	var calls atomic.Int32
	server := newLoopbackServer(t, func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	})
	llm, err := NewModel(t.Context(), "gpt-test", &Config{
		APIKey:     "test",
		BaseURL:    server.URL + "/v1",
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatalf("NewModel() err = %v", err)
	}
	var gotErr error
	for _, err := range llm.GenerateContent(t.Context(), userText("hi"), false) {
		gotErr = err
	}
	if gotErr == nil {
		t.Fatal("GenerateContent() err = nil, want the server's 500")
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("server saw %d requests, want 1: openai-go retried", got)
	}
}

func TestRequest_NeverStored(t *testing.T) {
	body := sendAndCapture(t, Config{}, userText("hi"))
	if got, ok := body["store"]; !ok || got != false {
		t.Errorf("store = %v (present %v), want false", got, ok)
	}
}

func TestRequest_EffortOverridesTheRequest(t *testing.T) {
	req := userText("hi")
	req.Config = &genai.GenerateContentConfig{ThinkingConfig: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelLow}}

	tests := []struct {
		name   string
		effort Effort
		want   string
	}{
		{name: "unset follows the request", effort: "", want: "low"},
		{name: "xhigh", effort: EffortXHigh, want: "xhigh"},
		{name: "max", effort: EffortMax, want: "max"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := sendAndCapture(t, Config{Effort: tt.effort}, req)
			reasoning, _ := body["reasoning"].(map[string]any)
			if got := reasoning["effort"]; got != tt.want {
				t.Errorf("reasoning.effort = %v, want %q", got, tt.want)
			}
		})
	}

	t.Run("applies to a request without a thinking config", func(t *testing.T) {
		body := sendAndCapture(t, Config{Effort: EffortHigh}, userText("hi"))
		reasoning, _ := body["reasoning"].(map[string]any)
		if got := reasoning["effort"]; got != "high" {
			t.Errorf("reasoning.effort = %v, want %q", got, "high")
		}
	})
}

func TestRequest_DefaultMaxTokens(t *testing.T) {
	t.Run("caps a request that sets none", func(t *testing.T) {
		body := sendAndCapture(t, Config{DefaultMaxTokens: 128000}, userText("hi"))
		if got := body["max_output_tokens"]; got != float64(128000) {
			t.Errorf("max_output_tokens = %v, want 128000", got)
		}
	})
	t.Run("yields to the request's own", func(t *testing.T) {
		req := userText("hi")
		req.Config = &genai.GenerateContentConfig{MaxOutputTokens: 500}
		body := sendAndCapture(t, Config{DefaultMaxTokens: 128000}, req)
		if got := body["max_output_tokens"]; got != float64(500) {
			t.Errorf("max_output_tokens = %v, want 500", got)
		}
	})
	t.Run("unset leaves the request uncapped", func(t *testing.T) {
		body := sendAndCapture(t, Config{}, userText("hi"))
		if got, ok := body["max_output_tokens"]; ok {
			t.Errorf("max_output_tokens = %v, want absent", got)
		}
	})
}

func TestRequest_SystemInstructionBreakpoint(t *testing.T) {
	req := userText("Where is order 42?")
	req.Config = &genai.GenerateContentConfig{
		SystemInstruction: genai.NewContentFromText("You track orders.", genai.RoleUser),
	}

	t.Run("without a breakpoint the instruction stays top-level", func(t *testing.T) {
		body := sendAndCapture(t, Config{}, req)
		if got := body["instructions"]; got != "You track orders." {
			t.Errorf("instructions = %v, want the system instruction", got)
		}
	})

	t.Run("with one it leads the input as a developer message", func(t *testing.T) {
		body := sendAndCapture(t, Config{PromptCaching: &PromptCachingConfig{
			SystemInstruction: &CacheBreakpoint{TTL: CacheTTL12h},
		}}, req)
		if got, ok := body["instructions"]; ok {
			t.Errorf("instructions = %v, want absent", got)
		}
		input, _ := body["input"].([]any)
		if len(input) != 2 {
			t.Fatalf("input has %d items, want the developer message and the user's", len(input))
		}
		first, _ := input[0].(map[string]any)
		if first["role"] != "developer" {
			t.Errorf("input[0].role = %v, want developer", first["role"])
		}
		content, _ := first["content"].([]any)
		if len(content) != 1 {
			t.Fatalf("developer message has %d content blocks, want 1", len(content))
		}
		block, _ := content[0].(map[string]any)
		if block["text"] != "You track orders." {
			t.Errorf("developer text = %v, want the system instruction", block["text"])
		}
		wantBreakpoint := map[string]any{"mode": "explicit", "ttl": "12h"}
		if got, _ := block["prompt_cache_breakpoint"].(map[string]any); !equalJSON(got, wantBreakpoint) {
			t.Errorf("prompt_cache_breakpoint = %v, want %v", got, wantBreakpoint)
		}
	})
}

func TestRequest_MarkedPartBreakpoint(t *testing.T) {
	newRequest := func() *model.LLMRequest {
		first := genai.NewPartFromText("Catalog page 1")
		second := genai.NewPartFromText("Catalog page 2")
		MarkCacheBreakpoint(first)
		MarkCacheBreakpoint(second)
		return &model.LLMRequest{Contents: []*genai.Content{
			{Role: genai.RoleUser, Parts: []*genai.Part{first, second, genai.NewPartFromText("Which SKU ships first?")}},
		}}
	}
	breakpoints := func(body map[string]any) []any {
		var got []any
		input, _ := body["input"].([]any)
		for _, item := range input {
			content, _ := item.(map[string]any)["content"].([]any)
			for _, block := range content {
				got = append(got, block.(map[string]any)["prompt_cache_breakpoint"])
			}
		}
		return got
	}

	t.Run("only the last marked part carries one", func(t *testing.T) {
		body := sendAndCapture(t, Config{PromptCaching: &PromptCachingConfig{MarkedPart: &CacheBreakpoint{}}}, newRequest())
		got := breakpoints(body)
		if len(got) != 3 {
			t.Fatalf("got %d content blocks, want 3", len(got))
		}
		if got[0] != nil || got[2] != nil {
			t.Errorf("breakpoints = %v, want one on the second block only", got)
		}
		if !equalJSON(got[1], map[string]any{"mode": "explicit"}) {
			t.Errorf("second block's breakpoint = %v, want an explicit one inheriting the request TTL", got[1])
		}
	})

	t.Run("not configured places none", func(t *testing.T) {
		for i, bp := range breakpoints(sendAndCapture(t, Config{}, newRequest())) {
			if bp != nil {
				t.Errorf("block %d carries breakpoint %v, want none", i, bp)
			}
		}
	})
}

func equalJSON(a, b any) bool {
	ja, errA := json.Marshal(a)
	jb, errB := json.Marshal(b)
	return errA == nil && errB == nil && string(ja) == string(jb)
}

// writeCompleted answers a request with a minimal completed response.
func writeCompleted(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = io.WriteString(w, completedBody)
}
