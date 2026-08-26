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
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// messageStartTestUsage is the usage block the canned SSE stream reports at message_start.
var messageStartTestUsage = MessageStartUsage{
	InputTokens:              7,
	CacheReadInputTokens:     330818,
	CacheCreationInputTokens: 432805,
}

// newMessageStartTestServer serves one canned, accumulatable SSE response: a message_start carrying the
// three input-side usage counts, one text delta, and the closing events.
func newMessageStartTestServer(t *testing.T) *httptest.Server {
	t.Helper()
	body := "event: message_start\n" +
		fmt.Sprintf(`data: {"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":%d,"cache_read_input_tokens":%d,"cache_creation_input_tokens":%d,"output_tokens":1}}}`,
			messageStartTestUsage.InputTokens, messageStartTestUsage.CacheReadInputTokens, messageStartTestUsage.CacheCreationInputTokens) + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":0}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":3}}` + "\n\n" +
		"event: message_stop\n" +
		`data: {"type":"message_stop"}` + "\n\n"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(server.Close)
	return server
}

func newMessageStartTestModel(t *testing.T, baseURL string) model.LLM {
	t.Helper()
	llm, err := NewModel(t.Context(), "claude-test", &Config{
		Variant: VariantAnthropicAPI,
		APIKey:  "test-key",
		BaseURL: baseURL,
	})
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}
	return llm
}

func messageStartTestRequest() *model.LLMRequest {
	return &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", "user")},
	}
}

func TestMessageStartCallbackContextRoundTrip(t *testing.T) {
	if _, ok := MessageStartCallbackFromContext(context.Background()); ok {
		t.Fatal("bare context unexpectedly carries a MessageStartCallback")
	}
	var invoked bool
	ctx := ContextWithMessageStartCallback(context.Background(), func(MessageStartUsage) { invoked = true })
	callback, ok := MessageStartCallbackFromContext(ctx)
	if !ok {
		t.Fatal("MessageStartCallbackFromContext did not return the installed callback")
	}
	callback(MessageStartUsage{})
	if !invoked {
		t.Fatal("returned callback is not the installed one")
	}
}

func TestGenerateStreamInvokesMessageStartCallbackBeforeFirstPartial(t *testing.T) {
	server := newMessageStartTestServer(t)
	llm := newMessageStartTestModel(t, server.URL)

	// The callback fires synchronously on the consuming goroutine, so an ordered event log needs no lock.
	var events []string
	var gotUsage []MessageStartUsage
	ctx := ContextWithMessageStartCallback(t.Context(), func(usage MessageStartUsage) {
		events = append(events, "callback")
		gotUsage = append(gotUsage, usage)
	})

	var responses []*model.LLMResponse
	for resp, err := range llm.GenerateContent(ctx, messageStartTestRequest(), true) {
		if err != nil {
			t.Fatalf("GenerateContent yielded error: %v", err)
		}
		if resp.Partial {
			events = append(events, "partial:"+resp.Content.Parts[0].Text)
		} else {
			events = append(events, "final")
		}
		responses = append(responses, resp)
	}

	if len(gotUsage) != 1 {
		t.Fatalf("callback invoked %d times, want exactly 1", len(gotUsage))
	}
	if gotUsage[0] != messageStartTestUsage {
		t.Errorf("callback usage = %+v, want %+v", gotUsage[0], messageStartTestUsage)
	}
	// The callback must precede the first partial, and the yielded stream shape must be unchanged by its
	// presence: one text partial, then the final complete response.
	wantEvents := []string{"callback", "partial:Hello", "final"}
	if len(events) != len(wantEvents) {
		t.Fatalf("event order = %v, want %v", events, wantEvents)
	}
	for i := range wantEvents {
		if events[i] != wantEvents[i] {
			t.Fatalf("event order = %v, want %v", events, wantEvents)
		}
	}
	if !responses[len(responses)-1].TurnComplete {
		t.Error("final response is not marked TurnComplete")
	}
}

func TestGenerateInvokesMessageStartCallback(t *testing.T) {
	server := newMessageStartTestServer(t)
	llm := newMessageStartTestModel(t, server.URL)

	var gotUsage []MessageStartUsage
	ctx := ContextWithMessageStartCallback(t.Context(), func(usage MessageStartUsage) {
		gotUsage = append(gotUsage, usage)
	})

	var responses []*model.LLMResponse
	for resp, err := range llm.GenerateContent(ctx, messageStartTestRequest(), false) {
		if err != nil {
			t.Fatalf("GenerateContent yielded error: %v", err)
		}
		responses = append(responses, resp)
	}

	if len(gotUsage) != 1 {
		t.Fatalf("callback invoked %d times, want exactly 1", len(gotUsage))
	}
	if gotUsage[0] != messageStartTestUsage {
		t.Errorf("callback usage = %+v, want %+v", gotUsage[0], messageStartTestUsage)
	}
	if len(responses) != 1 || !responses[0].TurnComplete {
		t.Fatalf("non-streaming call yielded %d responses, want 1 complete response", len(responses))
	}
}

func TestGenerateStreamWithoutCallbackIsUnchanged(t *testing.T) {
	server := newMessageStartTestServer(t)
	llm := newMessageStartTestModel(t, server.URL)

	var responses []*model.LLMResponse
	for resp, err := range llm.GenerateContent(t.Context(), messageStartTestRequest(), true) {
		if err != nil {
			t.Fatalf("GenerateContent yielded error: %v", err)
		}
		responses = append(responses, resp)
	}

	if len(responses) != 2 {
		t.Fatalf("stream yielded %d responses, want partial + final", len(responses))
	}
	if !responses[0].Partial || responses[0].Content.Parts[0].Text != "Hello" {
		t.Errorf("first response = %+v, want partial with text %q", responses[0], "Hello")
	}
	if !responses[1].TurnComplete {
		t.Error("final response is not marked TurnComplete")
	}
}
