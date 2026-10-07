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
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/anthropic/internal/converters"
)

// filesTestSSE is a complete, accumulatable streamed reply.
const filesTestSSE = "event: message_start\n" +
	`data: {"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":1}}}` + "\n\n" +
	"event: content_block_start\n" +
	`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n" +
	"event: content_block_delta\n" +
	`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Done"}}` + "\n\n" +
	"event: content_block_stop\n" +
	`data: {"type":"content_block_stop","index":0}` + "\n\n" +
	"event: message_delta\n" +
	`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}` + "\n\n" +
	"event: message_stop\n" +
	`data: {"type":"message_stop"}` + "\n\n"

// filesTestServer fakes the Files and Messages endpoints, recording what each
// receives.
type filesTestServer struct {
	*httptest.Server

	mu sync.Mutex
	// uploads records each upload's filename and expires_in_seconds, in order.
	uploads []fakeUpload
	// messageBodies records each Messages request body, in order.
	messageBodies []string
	// deleted holds file IDs the Messages endpoint treats as missing.
	deleted map[string]bool

	// uploadStatus, when non-zero, is the status every upload fails with.
	uploadStatus int
	// uploadDelay holds each upload open this long, so concurrent uploads overlap.
	uploadDelay time.Duration
}

// fakeUpload is one upload the fake Files endpoint received.
type fakeUpload struct {
	filename         string
	expiresInSeconds string
}

func newFilesTestServer(t *testing.T) *filesTestServer {
	t.Helper()
	s := &filesTestServer{deleted: map[string]bool{}}
	s.Server = httptest.NewServer(http.HandlerFunc(s.serve))
	t.Cleanup(s.Close)
	return s
}

func (s *filesTestServer) serve(w http.ResponseWriter, r *http.Request) {
	switch {
	case r.Method == http.MethodPost && r.URL.Path == "/v1/files":
		s.serveUpload(w, r)
	case r.Method == http.MethodPost && r.URL.Path == "/v1/messages":
		body, _ := io.ReadAll(r.Body)
		s.mu.Lock()
		s.messageBodies = append(s.messageBodies, string(body))
		var missing string
		for id := range s.deleted {
			if strings.Contains(string(body), id) {
				missing = id
			}
		}
		s.mu.Unlock()
		if missing != "" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprintf(w, `{"type":"error","error":{"type":"not_found_error","message":"File not found: %s"}}`, missing)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, filesTestSSE)
	default:
		http.NotFound(w, r)
	}
}

func (s *filesTestServer) serveUpload(w http.ResponseWriter, r *http.Request) {
	if s.uploadDelay > 0 {
		time.Sleep(s.uploadDelay)
	}
	if s.uploadStatus != 0 {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(s.uploadStatus)
		_, _ = io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"upload refused"}}`)
		return
	}
	if err := r.ParseMultipartForm(1 << 20); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	_, header, err := r.FormFile("file")
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	s.uploads = append(s.uploads, fakeUpload{filename: header.Filename, expiresInSeconds: r.FormValue("expires_in_seconds")})
	id := fmt.Sprintf("file_%d", len(s.uploads))
	s.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, `{"id":%q,"type":"file","filename":%q,"mime_type":"application/pdf","size_bytes":1,"created_at":"2026-01-01T00:00:00Z","expires_at":%q}`,
		id, header.Filename, time.Now().Add(24*time.Hour).UTC().Format(time.RFC3339))
}

func (s *filesTestServer) uploadCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.uploads)
}

func (s *filesTestServer) messageCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.messageBodies)
}

func (s *filesTestServer) lastMessageBody(t *testing.T) string {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.messageBodies) == 0 {
		t.Fatal("no Messages request was received")
	}
	return s.messageBodies[len(s.messageBodies)-1]
}

// newFilesTestModel returns a model on server with Files enabled. Each test
// passes its own apiKey, which keys the process-wide memo, so tests do not
// see each other's uploads.
func newFilesTestModel(t *testing.T, server *filesTestServer, apiKey string, files FilesConfig, caching *PromptCachingConfig) *anthropicModel {
	t.Helper()
	llm, err := NewModel(t.Context(), "claude-test", &Config{
		Variant:       VariantAnthropicAPI,
		APIKey:        apiKey,
		BaseURL:       server.URL,
		Files:         &files,
		PromptCaching: caching,
	})
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}
	return llm.(*anthropicModel)
}

// reportRequest returns a request carrying the given report PDF bytes ahead of
// a question.
func reportRequest(report []byte) *model.LLMRequest {
	return &model.LLMRequest{Contents: []*genai.Content{{
		Role: "user",
		Parts: []*genai.Part{
			{InlineData: &genai.Blob{MIMEType: "application/pdf", Data: report, DisplayName: "report-1.pdf"}},
			{Text: "Summarize the report."},
		},
	}}}
}

func generate(t *testing.T, llm model.LLM, req *model.LLMRequest, stream bool) error {
	t.Helper()
	var lastErr error
	for _, err := range llm.GenerateContent(t.Context(), req, stream) {
		if err != nil {
			lastErr = err
		}
	}
	return lastErr
}

// firstDocumentSource returns the source of the first document block in a
// Messages request body.
func firstDocumentSource(t *testing.T, body string) map[string]any {
	t.Helper()
	var request struct {
		Messages []struct {
			Content []map[string]any `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal([]byte(body), &request); err != nil {
		t.Fatalf("decode Messages request: %v", err)
	}
	for _, message := range request.Messages {
		for _, block := range message.Content {
			if block["type"] == "document" {
				return block
			}
		}
	}
	t.Fatalf("no document block in %s", body)
	return nil
}

func TestFilesReferenceDocumentsAndReuseUploads(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", stream), func(t *testing.T) {
			server := newFilesTestServer(t)
			llm := newFilesTestModel(t, server, t.Name(), FilesConfig{}, nil)
			report := []byte("%PDF-1.7 quarterly report body")

			for range 2 {
				if err := generate(t, llm, reportRequest(append([]byte(nil), report...)), stream); err != nil {
					t.Fatalf("GenerateContent: %v", err)
				}
				document := firstDocumentSource(t, server.lastMessageBody(t))
				source := document["source"].(map[string]any)
				if source["type"] != "file" || source["file_id"] != "file_1" {
					t.Fatalf("document source = %v, want a reference to file_1", source)
				}
			}
			if got := server.uploadCount(); got != 1 {
				t.Errorf("uploads = %d, want 1: the second request must reuse the first upload", got)
			}
			server.mu.Lock()
			upload := server.uploads[0]
			server.mu.Unlock()
			if strings.Contains(upload.filename, "/") || !strings.HasSuffix(upload.filename, ".pdf") || strings.Contains(upload.filename, "report") {
				t.Errorf("upload filename = %q, want a content hash with a .pdf extension and nothing of the document's name", upload.filename)
			}
			if upload.expiresInSeconds != "86400" {
				t.Errorf("expires_in_seconds = %q, want the 24-hour default", upload.expiresInSeconds)
			}
		})
	}
}

func TestFilesReuploadWhenLessThanTheReuseMarginRemains(t *testing.T) {
	server := newFilesTestServer(t)
	llm := newFilesTestModel(t, server, t.Name(), FilesConfig{ExpiresIn: 2 * time.Hour, ReuseMargin: 30 * time.Minute}, nil)
	clock := time.Now()
	llm.files.now = func() time.Time { return clock }
	report := []byte("%PDF-1.7 annual report body")

	if err := generate(t, llm, reportRequest(report), false); err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	// The fake server stamps files to expire 24 hours out; 23 hours on, one
	// hour remains, which the 30-minute margin still reuses.
	clock = clock.Add(23 * time.Hour)
	if err := generate(t, llm, reportRequest(report), false); err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	if got := server.uploadCount(); got != 1 {
		t.Fatalf("uploads = %d after 23 hours, want 1", got)
	}
	clock = clock.Add(40 * time.Minute)
	if err := generate(t, llm, reportRequest(report), false); err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	if got := server.uploadCount(); got != 2 {
		t.Fatalf("uploads = %d with 20 minutes left, want a fresh upload", got)
	}
	source := firstDocumentSource(t, server.lastMessageBody(t))["source"].(map[string]any)
	if source["file_id"] != "file_2" {
		t.Errorf("document references %v, want the fresh upload file_2", source["file_id"])
	}
}

func TestFilesConcurrentRequestsShareOneUpload(t *testing.T) {
	server := newFilesTestServer(t)
	server.uploadDelay = 100 * time.Millisecond
	llm := newFilesTestModel(t, server, t.Name(), FilesConfig{}, nil)
	report := []byte("%PDF-1.7 shared appendix")

	var wg sync.WaitGroup
	errs := make([]error, 5)
	for i := range errs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for _, err := range llm.GenerateContent(t.Context(), reportRequest(report), false) {
				errs[i] = err
			}
		}()
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}
	if got := server.uploadCount(); got != 1 {
		t.Errorf("uploads = %d, want 1 shared by every concurrent request", got)
	}
}

func TestFilesAreNotSharedAcrossAPIKeys(t *testing.T) {
	server := newFilesTestServer(t)
	report := []byte("%PDF-1.7 board minutes")
	for _, apiKey := range []string{t.Name() + "-a", t.Name() + "-b"} {
		llm := newFilesTestModel(t, server, apiKey, FilesConfig{}, nil)
		if err := generate(t, llm, reportRequest(report), false); err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}
	if got := server.uploadCount(); got != 2 {
		t.Errorf("uploads = %d, want one per API key", got)
	}
}

func TestFilesReuploadOnceWhenAReferencedFileIsMissing(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", stream), func(t *testing.T) {
			server := newFilesTestServer(t)
			llm := newFilesTestModel(t, server, t.Name(), FilesConfig{}, nil)
			report := []byte("%PDF-1.7 field survey")

			if err := generate(t, llm, reportRequest(report), stream); err != nil {
				t.Fatalf("GenerateContent: %v", err)
			}
			server.mu.Lock()
			server.deleted["file_1"] = true
			server.mu.Unlock()

			if err := generate(t, llm, reportRequest(report), stream); err != nil {
				t.Fatalf("GenerateContent after the file vanished: %v", err)
			}
			if got := server.uploadCount(); got != 2 {
				t.Errorf("uploads = %d, want a re-upload of the missing file", got)
			}
			source := firstDocumentSource(t, server.lastMessageBody(t))["source"].(map[string]any)
			if source["file_id"] != "file_2" {
				t.Errorf("retried request references %v, want file_2", source["file_id"])
			}

			// A rejection that persists through the re-upload surfaces after
			// the one retry.
			server.mu.Lock()
			server.deleted["file_2"] = true
			server.deleted["file_3"] = true
			server.mu.Unlock()
			if err := generate(t, llm, reportRequest(report), stream); err == nil || !isMissingFileError(err) {
				t.Errorf("err = %v, want the missing-file error after one retry", err)
			}
			if got := server.uploadCount(); got != 3 {
				t.Errorf("uploads = %d, want exactly one re-upload per call", got)
			}
		})
	}
}

func TestFilesFailedUploadsStayInlineUnderTheCap(t *testing.T) {
	server := newFilesTestServer(t)
	server.uploadStatus = http.StatusBadRequest
	report := []byte("%PDF-1.7 maintenance log")

	llm := newFilesTestModel(t, server, t.Name(), FilesConfig{MaxInlineBytes: len(report)}, nil)
	if err := generate(t, llm, reportRequest(report), false); err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	source := firstDocumentSource(t, server.lastMessageBody(t))["source"].(map[string]any)
	if source["type"] != "base64" {
		t.Errorf("document source = %v, want base64 after the upload failed", source)
	}

	capped := newFilesTestModel(t, server, t.Name()+"-capped", FilesConfig{MaxInlineBytes: len(report) - 1}, nil)
	messageCount := server.messageCount()
	err := generate(t, capped, reportRequest(report), false)
	if err == nil || !strings.Contains(err.Error(), "could not be uploaded") {
		t.Fatalf("err = %v, want the inline-cap error", err)
	}
	if server.messageCount() != messageCount {
		t.Error("a request over the inline cap must not be sent")
	}
}

func TestFilesKeepCacheBreakpointMarks(t *testing.T) {
	server := newFilesTestServer(t)
	llm := newFilesTestModel(t, server, t.Name(), FilesConfig{}, &PromptCachingConfig{MarkedPart: &CacheBreakpoint{}})
	marked := &genai.Part{InlineData: &genai.Blob{MIMEType: "application/pdf", Data: []byte("%PDF-1.7 shared intro")}}
	MarkCacheBreakpoint(marked)
	req := &model.LLMRequest{Contents: []*genai.Content{{
		Role:  "user",
		Parts: []*genai.Part{marked, {Text: "Compare it with last year."}},
	}}}
	if err := generate(t, llm, req, false); err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	document := firstDocumentSource(t, server.lastMessageBody(t))
	if document["source"].(map[string]any)["type"] != "file" {
		t.Fatalf("document = %v, want a file reference", document)
	}
	if _, ok := document["cache_control"]; !ok {
		t.Errorf("document = %v, want the marked part's cache_control", document)
	}
}

func TestFilesConfigValidation(t *testing.T) {
	for name, cfg := range map[string]*Config{
		"vertex variant":        {Variant: VariantVertexAI, VertexProjectID: "p", VertexLocation: "l", Files: &FilesConfig{}},
		"expiry under an hour":  {Variant: VariantAnthropicAPI, APIKey: "k", Files: &FilesConfig{ExpiresIn: time.Minute}},
		"expiry over 90 days":   {Variant: VariantAnthropicAPI, APIKey: "k", Files: &FilesConfig{ExpiresIn: 91 * 24 * time.Hour}},
		"margin at the expiry":  {Variant: VariantAnthropicAPI, APIKey: "k", Files: &FilesConfig{ExpiresIn: 2 * time.Hour, ReuseMargin: 2 * time.Hour}},
		"negative inline cap":   {Variant: VariantAnthropicAPI, APIKey: "k", Files: &FilesConfig{MaxInlineBytes: -1}},
		"negative reuse margin": {Variant: VariantAnthropicAPI, APIKey: "k", Files: &FilesConfig{ReuseMargin: -time.Minute}},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := NewModel(t.Context(), "claude-test", cfg); err == nil {
				t.Error("NewModel accepted an invalid Files config")
			}
		})
	}
}

func TestInlineDataToBlockReferencesFiles(t *testing.T) {
	messages, _, err := converters.ContentsToMessagesWithMarkedBlocks([]*genai.Content{{
		Role: "user",
		Parts: []*genai.Part{
			{InlineData: &genai.Blob{MIMEType: "application/pdf", Data: []byte("%PDF-")}},
			{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("\x89PNG")}},
			{InlineData: &genai.Blob{MIMEType: "application/pdf", Data: []byte("%PDF- inline")}},
		},
	}}, converters.ContentsOptions{})
	if err != nil {
		t.Fatalf("convert without references: %v", err)
	}
	if messages[0].Content[0].OfDocument.Source.OfBase64 == nil {
		t.Fatal("without a file ID a PDF must stay base64")
	}

	contents := []*genai.Content{{
		Role: "user",
		Parts: []*genai.Part{
			{InlineData: &genai.Blob{MIMEType: "application/pdf", Data: []byte("%PDF-")}},
			{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("\x89PNG")}},
			{InlineData: &genai.Blob{MIMEType: "application/pdf", Data: []byte("%PDF- inline")}},
		},
	}}
	parts := contents[0].Parts
	messages, _, err = converters.ContentsToMessagesWithMarkedBlocks(contents, converters.ContentsOptions{
		FileIDByBlob: map[*genai.Blob]string{parts[0].InlineData: "file_pdf", parts[1].InlineData: "file_png"},
	})
	if err != nil {
		t.Fatalf("convert with references: %v", err)
	}
	blocks := messages[0].Content
	if source := blocks[0].OfDocument.Source.OfFile; source == nil || source.FileID != "file_pdf" {
		t.Errorf("PDF source = %+v, want a reference to file_pdf", blocks[0].OfDocument.Source)
	}
	if source := blocks[1].OfImage.Source.OfFile; source == nil || source.FileID != "file_png" {
		t.Errorf("image source = %+v, want a reference to file_png", blocks[1].OfImage.Source)
	}
	if blocks[2].OfDocument.Source.OfBase64 == nil {
		t.Error("a PDF without a file ID must stay base64 beside referenced ones")
	}
}
