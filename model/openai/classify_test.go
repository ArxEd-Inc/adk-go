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
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
	"testing"

	openaisdk "github.com/openai/openai-go/v3"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// serveEvents answers every request with the given stream events.
func serveEvents(events ...string) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, event := range append(events, "[DONE]") {
			_, _ = fmt.Fprintf(w, "data: %s\n\n", event)
		}
	}
}

// generate sends one request to a model talking to handler and returns the
// final response or the error.
func generate(t *testing.T, handler http.HandlerFunc) (*model.LLMResponse, error) {
	t.Helper()
	server := newLoopbackServer(t, handler)
	llm, err := NewModel(t.Context(), "gpt-test", &Config{APIKey: "test", BaseURL: server.URL + "/v1", HTTPClient: server.Client()})
	if err != nil {
		t.Fatalf("NewModel() err = %v", err)
	}
	var final *model.LLMResponse
	for resp, err := range llm.GenerateContent(t.Context(), userText("Where is order 42?"), false) {
		if err != nil {
			return nil, err
		}
		final = resp
	}
	return final, nil
}

func TestErrors_HTTPErrorKeepsItsTypeAndCode(t *testing.T) {
	_, err := generate(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":{"type":"invalid_request_error","code":"context_length_exceeded","message":"Your input exceeds the context window of this model.","param":"input"}}`)
	})
	var apiErr *openaisdk.Error
	if !errors.As(err, &apiErr) || apiErr.StatusCode != http.StatusBadRequest {
		t.Fatalf("err = %v, want an *openai.Error with status 400", err)
	}
	if !IsContextWindowError(err) {
		t.Errorf("IsContextWindowError(%v) = false, want true", err)
	}
	if IsRequestTooLargeError(err) {
		t.Errorf("IsRequestTooLargeError(%v) = true, want false", err)
	}
}

func TestErrors_RequestTooLarge(t *testing.T) {
	_, err := generate(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusRequestEntityTooLarge)
	})
	if !IsRequestTooLargeError(err) {
		t.Errorf("IsRequestTooLargeError(%v) = false, want true", err)
	}
	if IsContextWindowError(err) {
		t.Errorf("IsContextWindowError(%v) = true, want false", err)
	}
}

func TestErrors_StreamErrorEventKeepsItsCode(t *testing.T) {
	_, err := generate(t, serveEvents(
		`{"type":"response.created","response":{"id":"r","model":"gpt-test","status":"in_progress","output":[]}}`,
		`{"type":"error","code":"context_length_exceeded","message":"Your input exceeds the context window of this model.","param":"input","sequence_number":1}`,
	))
	var serverErr *ServerError
	if !errors.As(err, &serverErr) || serverErr.Code != "context_length_exceeded" {
		t.Fatalf("err = %v, want a ServerError with the event's code", err)
	}
	if !IsContextWindowError(err) {
		t.Errorf("IsContextWindowError(%v) = false, want true", err)
	}
}

func TestErrors_FailedResponseKeepsItsCode(t *testing.T) {
	_, err := generate(t, serveEvents(
		`{"type":"response.failed","response":{"id":"r","model":"gpt-test","status":"failed","error":{"code":"server_error","message":"The server had an error."},"output":[]}}`,
	))
	var serverErr *ServerError
	if !errors.As(err, &serverErr) || serverErr.Code != "server_error" {
		t.Fatalf("err = %v, want a ServerError with code server_error", err)
	}
	if !errors.Is(err, ErrResponseFailed) {
		t.Errorf("err = %v, want it to match ErrResponseFailed", err)
	}
	if IsContextWindowError(err) {
		t.Errorf("IsContextWindowError(%v) = true, want false", err)
	}
}

func TestRefusal_IsASafetyStop(t *testing.T) {
	const body = `{"id":"r","model":"gpt-test","status":"completed","output":[{"type":"message","id":"m","role":"assistant","status":"completed",` +
		`"content":[{"type":"refusal","refusal":"I can't help with that."}]}]}`
	final, err := generate(t, serveEvents(
		`{"type":"response.refusal.delta","item_id":"m","output_index":0,"content_index":0,"delta":"I can't help with that."}`,
		`{"type":"response.completed","response":`+body+`}`,
	))
	if err != nil {
		t.Fatalf("GenerateContent() err = %v", err)
	}
	if final.FinishReason != genai.FinishReasonSafety || final.ErrorCode != string(genai.FinishReasonSafety) {
		t.Errorf("FinishReason, ErrorCode = %q, %q, want SAFETY for both", final.FinishReason, final.ErrorCode)
	}
	if want := "OpenAI refused the request: I can't help with that."; final.ErrorMessage != want {
		t.Errorf("ErrorMessage = %q, want %q", final.ErrorMessage, want)
	}
	if final.Content == nil || len(final.Content.Parts) != 1 || final.Content.Parts[0].Text != "I can't help with that." {
		t.Errorf("Content = %+v, want the refusal kept as text", final.Content)
	}
}

func TestUnknownOutputItem_IsSkippedAndListed(t *testing.T) {
	const body = `{"id":"r","model":"gpt-test","status":"completed","output":[` +
		`{"type":"web_search_call","id":"ws","status":"completed"},` +
		`{"type":"message","id":"m","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Shipped.","annotations":[]}]}]}`
	final, err := generate(t, serveEvents(`{"type":"response.completed","response":`+body+`}`))
	if err != nil {
		t.Fatalf("GenerateContent() err = %v", err)
	}
	if final.Content == nil || len(final.Content.Parts) != 1 || final.Content.Parts[0].Text != "Shipped." {
		t.Errorf("Content = %+v, want the answer alone", final.Content)
	}
	if got, _ := final.CustomMetadata[SkippedOutputItemTypesKey].([]string); !slices.Equal(got, []string{"web_search_call"}) {
		t.Errorf("CustomMetadata[%q] = %v, want [web_search_call]", SkippedOutputItemTypesKey, final.CustomMetadata[SkippedOutputItemTypesKey])
	}
}
