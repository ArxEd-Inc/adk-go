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
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// filesServer fakes the Files and Responses endpoints and records what each
// was sent.
type filesServer struct {
	mu sync.Mutex
	// uploads holds each upload's form values.
	uploads []map[string][]string
	// requests holds each Responses request's input content blocks.
	requests [][]map[string]any

	// failUploads makes every upload fail.
	failUploads bool
	// expiresIn is the life the fake gives each file; zero reports none.
	expiresIn time.Duration
	// missingFileResponses is how many Responses requests to refuse as
	// referencing a missing file.
	missingFileResponses int
}

func (s *filesServer) handle(t *testing.T) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		s.mu.Lock()
		defer s.mu.Unlock()
		switch r.URL.Path {
		case "/v1/files":
			if err := r.ParseMultipartForm(1 << 20); err != nil {
				t.Errorf("parse upload: %v", err)
			}
			s.uploads = append(s.uploads, r.MultipartForm.Value)
			if s.failUploads {
				w.WriteHeader(http.StatusServiceUnavailable)
				return
			}
			expiresAt := int64(0)
			if s.expiresIn > 0 {
				expiresAt = time.Now().Add(s.expiresIn).Unix()
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprintf(w, `{"id":"file-%d","object":"file","bytes":1,"created_at":%d,"filename":"f","purpose":"user_data","status":"processed","expires_at":%d}`,
				len(s.uploads), time.Now().Unix(), expiresAt)
		case "/v1/responses":
			var body struct {
				Input []struct {
					Content []map[string]any `json:"content"`
				} `json:"input"`
			}
			raw, _ := io.ReadAll(r.Body)
			if err := json.Unmarshal(raw, &body); err != nil {
				t.Errorf("decode request: %v", err)
			}
			var blocks []map[string]any
			for _, item := range body.Input {
				blocks = append(blocks, item.Content...)
			}
			s.requests = append(s.requests, blocks)
			if s.missingFileResponses > 0 {
				s.missingFileResponses--
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusNotFound)
				_, _ = io.WriteString(w, `{"error":{"type":"invalid_request_error","code":null,"message":"File not found.","param":"input"}}`)
				return
			}
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprintf(w, "data: {\"type\":\"response.completed\",\"response\":%s}\n\ndata: [DONE]\n\n", completedBody)
		default:
			t.Errorf("unexpected path %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}
}

func (s *filesServer) model(t *testing.T, files *FilesConfig) model.LLM {
	t.Helper()
	server := newLoopbackServer(t, s.handle(t))
	llm, err := NewModel(t.Context(), "gpt-test", &Config{
		APIKey: "test", BaseURL: server.URL + "/v1", HTTPClient: server.Client(), Files: files,
	})
	if err != nil {
		t.Fatalf("NewModel() err = %v", err)
	}
	return llm
}

// packingSlip is a request carrying a PDF, an image and a question.
func packingSlip() *model.LLMRequest {
	return &model.LLMRequest{Contents: []*genai.Content{{Role: genai.RoleUser, Parts: []*genai.Part{
		{InlineData: &genai.Blob{MIMEType: "application/pdf", Data: []byte("%PDF-1.7 packing slip for order 42")}},
		{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("\x89PNG shipping label")}},
		genai.NewPartFromText("Which SKUs shipped?"),
	}}}}
}

func run(t *testing.T, llm model.LLM, req *model.LLMRequest) error {
	t.Helper()
	var last error
	for _, err := range llm.GenerateContent(t.Context(), req, false) {
		last = err
	}
	return last
}

func TestFiles_SendsDocumentsByReference(t *testing.T) {
	fake := &filesServer{}
	llm := fake.model(t, &FilesConfig{})
	if err := run(t, llm, packingSlip()); err != nil {
		t.Fatalf("GenerateContent() err = %v", err)
	}

	if len(fake.uploads) != 2 {
		t.Fatalf("%d uploads, want the PDF and the image", len(fake.uploads))
	}
	for _, form := range fake.uploads {
		if got := form["purpose"]; len(got) != 1 || got[0] != "user_data" {
			t.Errorf("purpose = %v, want user_data", got)
		}
		var seconds string
		for key, values := range form {
			if strings.Contains(key, "seconds") {
				seconds = values[0]
			}
		}
		if seconds != "86400" {
			t.Errorf("expiry seconds = %q in %v, want 86400, the 24-hour default", seconds, form)
		}
	}
	// The two uploads run concurrently, so which got which ID varies.
	blocks := fake.requests[0]
	pdfID, _ := blocks[0]["file_id"].(string)
	imageID, _ := blocks[1]["file_id"].(string)
	if blocks[0]["type"] != "input_file" || !strings.HasPrefix(pdfID, "file-") || blocks[0]["file_data"] != nil {
		t.Errorf("PDF block = %v, want a file reference with no data", blocks[0])
	}
	if blocks[1]["type"] != "input_image" || !strings.HasPrefix(imageID, "file-") || blocks[1]["image_url"] != nil {
		t.Errorf("image block = %v, want a file reference with no data", blocks[1])
	}
	if pdfID == imageID {
		t.Errorf("the PDF and the image both reference %s", pdfID)
	}

	// The same documents again reuse the uploads.
	if err := run(t, llm, packingSlip()); err != nil {
		t.Fatalf("second GenerateContent() err = %v", err)
	}
	if len(fake.uploads) != 2 {
		t.Errorf("%d uploads after a second request, want the first two reused", len(fake.uploads))
	}
}

func TestFiles_ReuploadsAFileNearItsExpiry(t *testing.T) {
	fake := &filesServer{expiresIn: 30 * time.Minute}
	llm := fake.model(t, &FilesConfig{ReuseMargin: time.Hour})
	for range 2 {
		if err := run(t, llm, packingSlip()); err != nil {
			t.Fatalf("GenerateContent() err = %v", err)
		}
	}
	if len(fake.uploads) != 4 {
		t.Errorf("%d uploads, want each document uploaded again once its file has under an hour left", len(fake.uploads))
	}
}

func TestFiles_FailedUploadSendsTheDocumentInline(t *testing.T) {
	fake := &filesServer{failUploads: true}
	if err := run(t, fake.model(t, &FilesConfig{}), packingSlip()); err != nil {
		t.Fatalf("GenerateContent() err = %v", err)
	}
	blocks := fake.requests[0]
	if data, _ := blocks[0]["file_data"].(string); !strings.HasPrefix(data, "data:application/pdf;base64,") {
		t.Errorf("PDF block = %v, want it inline", blocks[0])
	}
}

func TestFiles_InlineCapFailsBeforeSending(t *testing.T) {
	fake := &filesServer{failUploads: true}
	err := run(t, fake.model(t, &FilesConfig{MaxInlineBytes: 10}), packingSlip())
	if !errors.Is(err, ErrRequestTooLarge) || !IsRequestTooLargeError(err) {
		t.Fatalf("err = %v, want ErrRequestTooLarge", err)
	}
	if len(fake.requests) != 0 {
		t.Errorf("%d requests reached the API, want none", len(fake.requests))
	}
}

func TestFiles_MissingFileIsUploadedAgainOnce(t *testing.T) {
	t.Run("once recovers", func(t *testing.T) {
		fake := &filesServer{missingFileResponses: 1}
		if err := run(t, fake.model(t, &FilesConfig{}), packingSlip()); err != nil {
			t.Fatalf("GenerateContent() err = %v", err)
		}
		if len(fake.requests) != 2 || len(fake.uploads) != 4 {
			t.Errorf("%d requests and %d uploads, want the request made again with fresh uploads", len(fake.requests), len(fake.uploads))
		}
	})
	t.Run("twice fails", func(t *testing.T) {
		fake := &filesServer{missingFileResponses: 2}
		if err := run(t, fake.model(t, &FilesConfig{}), packingSlip()); err == nil {
			t.Fatal("GenerateContent() err = nil, want the second refusal")
		}
		if len(fake.requests) != 2 {
			t.Errorf("%d requests, want two attempts only", len(fake.requests))
		}
	})
}

func TestFiles_UnsetSendsEverythingInline(t *testing.T) {
	fake := &filesServer{}
	if err := run(t, fake.model(t, nil), packingSlip()); err != nil {
		t.Fatalf("GenerateContent() err = %v", err)
	}
	if len(fake.uploads) != 0 {
		t.Errorf("%d uploads, want none", len(fake.uploads))
	}
}

func TestFiles_InvalidConfig(t *testing.T) {
	for name, cfg := range map[string]FilesConfig{
		"expiry too short":         {ExpiresIn: time.Minute},
		"expiry too long":          {ExpiresIn: 31 * 24 * time.Hour},
		"margin as long as expiry": {ExpiresIn: time.Hour, ReuseMargin: time.Hour},
		"negative cap":             {MaxInlineBytes: -1},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := NewModel(t.Context(), "gpt-test", &Config{APIKey: "test", Files: &cfg}); err == nil {
				t.Error("NewModel() err = nil, want the config refused")
			}
		})
	}
}
