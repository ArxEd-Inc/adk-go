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
	"fmt"
	"io"
	"net/http"
	"testing"

	"google.golang.org/adk/v2/model"
)

// textStream is a stream answering "Shipped." in two deltas.
var textStream = []string{
	`{"type":"response.created","response":{"id":"r","model":"gpt-test","status":"in_progress","output":[]}}`,
	`{"type":"response.output_text.delta","item_id":"m","output_index":0,"content_index":0,"delta":"Ship"}`,
	`{"type":"response.output_text.delta","item_id":"m","output_index":0,"content_index":0,"delta":"ped."}`,
	`{"type":"response.completed","response":{"id":"r","model":"gpt-test","status":"completed","output":[` +
		`{"type":"message","id":"m","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Shipped.","annotations":[]}]}]}}`,
}

func TestNonStreamingCallerGetsTheStreamsFinalResponse(t *testing.T) {
	var sentStream any
	server := newLoopbackServer(t, func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		sentStream = body["stream"]
		w.Header().Set("Content-Type", "text/event-stream")
		for _, event := range append(textStream, "[DONE]") {
			_, _ = fmt.Fprintf(w, "data: %s\n\n", event)
		}
	})
	llm, err := NewModel(t.Context(), "gpt-test", &Config{APIKey: "test", BaseURL: server.URL + "/v1", HTTPClient: server.Client()})
	if err != nil {
		t.Fatalf("NewModel() err = %v", err)
	}

	var got []*model.LLMResponse
	for resp, err := range llm.GenerateContent(t.Context(), userText("Where is order 42?"), false) {
		if err != nil {
			t.Fatalf("GenerateContent() err = %v", err)
		}
		got = append(got, resp)
	}
	if sentStream != true {
		t.Errorf("request stream = %v, want true", sentStream)
	}
	if len(got) != 1 {
		t.Fatalf("got %d responses, want only the final one", len(got))
	}
	if got[0].Partial || !got[0].TurnComplete {
		t.Errorf("response Partial, TurnComplete = %v, %v, want false, true", got[0].Partial, got[0].TurnComplete)
	}
	if text := got[0].Content.Parts[0].Text; text != "Shipped." {
		t.Errorf("text = %q, want %q", text, "Shipped.")
	}
}

func TestNonStreamingCallerIsToldWhenTheStreamSaysNothing(t *testing.T) {
	// A JSON body to a streamed request decodes as no events at all.
	server := newLoopbackServer(t, writeCompleted)
	llm, err := NewModel(t.Context(), "gpt-test", &Config{APIKey: "test", BaseURL: server.URL + "/v1", HTTPClient: server.Client()})
	if err != nil {
		t.Fatalf("NewModel() err = %v", err)
	}
	var gotErr error
	for resp, err := range llm.GenerateContent(t.Context(), userText("hi"), false) {
		if err == nil {
			t.Fatalf("GenerateContent() yielded %+v, want an error", resp)
		}
		gotErr = err
	}
	if !errors.Is(gotErr, ErrEmptyResponse) {
		t.Errorf("err = %v, want %v", gotErr, ErrEmptyResponse)
	}
}
