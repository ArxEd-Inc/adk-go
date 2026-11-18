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
	"bytes"
	"fmt"
	"os"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// liveToolsKey is the function-response key the live test loads tools under.
const liveToolsKey = "loaded_tools"

// deferredEntry defers its tool's declaration.
type deferredEntry struct{}

func (deferredEntry) DeferLoading() bool { return true }

// TestLive runs one short tool-using conversation against the real API: a PDF
// sent by file reference with a breakpoint after it, a system instruction with
// a breakpoint of its own, reasoning carried across tool calls, and a deferred
// tool loaded by a function response. It costs a few cents, so it runs only
// when both OPENAI_API_KEY and OPENAI_LIVE_TEST_MODEL are set; the second
// names the model to run against.
func TestLive(t *testing.T) {
	apiKey, modelName := os.Getenv("OPENAI_API_KEY"), os.Getenv("OPENAI_LIVE_TEST_MODEL")
	if apiKey == "" || modelName == "" {
		t.Skip("set OPENAI_API_KEY and OPENAI_LIVE_TEST_MODEL to run against the real API")
	}
	newModel := func(baseURL string) model.LLM {
		llm, err := NewModel(t.Context(), modelName, &Config{
			APIKey:  apiKey,
			BaseURL: baseURL,
			Effort:  EffortLow,
			PromptCaching: &PromptCachingConfig{
				SystemInstruction: &CacheBreakpoint{TTL: CacheTTL30m},
				MarkedPart:        &CacheBreakpoint{TTL: CacheTTL30m},
			},
			ToolReferencesResponseKey: liveToolsKey,
			Files:                     &FilesConfig{},
		})
		if err != nil {
			t.Fatalf("NewModel() err = %v", err)
		}
		return llm
	}
	llm := newModel("https://api.openai.com/v1")

	slip := &genai.Part{InlineData: &genai.Blob{MIMEType: "application/pdf", Data: liveTestPDF("Order 42 ships SKU A-17 from bin 9.")}}
	MarkCacheBreakpoint(slip)
	contents := []*genai.Content{{Role: genai.RoleUser, Parts: []*genai.Part{
		slip, genai.NewPartFromText("Look order 42 up, then cancel it."),
	}}}

	// Turn 1: the model must call lookup_order.
	turn1 := liveTurn(t, llm, contents, "lookup_order")
	call1 := liveCall(t, turn1, "lookup_order")
	if !hasReasoningSignature(turn1) {
		t.Log("turn 1 carried no reasoning item; replay goes unexercised")
	}
	contents = append(contents, turn1.Content, liveToolResponse(call1, map[string]any{
		"status": "packed", liveToolsKey: []string{"cancel_order"},
	}))

	// Turn 2: cancel_order is deferred, so it is callable only because the
	// lookup's response loaded it; the request also replays turn 1's
	// reasoning ahead of its call.
	turn2 := liveTurn(t, llm, contents, "cancel_order")
	call2 := liveCall(t, turn2, "cancel_order")
	contents = append(contents, turn2.Content, liveToolResponse(call2, map[string]any{"status": "canceled"}))

	// Turn 3: a final answer, reading the earlier turns from cache.
	turn3 := liveTurn(t, llm, contents, "")
	if text := answerText(turn3); !strings.Contains(strings.ToLower(text), "cancel") {
		t.Errorf("final answer = %q, want it to report the cancellation", text)
	}
	if turn3.UsageMetadata == nil || turn3.UsageMetadata.CachedContentTokenCount == 0 {
		t.Errorf("turn 3 read nothing from cache: %+v", turn3.UsageMetadata)
	}

	// The same first turn through a model with another identity uploads the
	// PDF again under a new file ID; the prefix cached above should still be
	// read, since the cache keys a document on its content.
	again := liveTurn(t, newModel("https://api.openai.com/v1/"), contents[:1], "lookup_order")
	if again.UsageMetadata == nil || again.UsageMetadata.CachedContentTokenCount == 0 {
		t.Errorf("a re-uploaded document read nothing from cache: %+v", again.UsageMetadata)
	}
}

// liveTurn sends contents with the live test's tools and returns the final
// response. forceTool, when set, requires the model to call that tool.
func liveTurn(t *testing.T, llm model.LLM, contents []*genai.Content, forceTool string) *model.LLMResponse {
	t.Helper()
	declare := func(name, description string) *genai.FunctionDeclaration {
		return &genai.FunctionDeclaration{
			Name: name, Description: description,
			ParametersJsonSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"order_id": map[string]any{"type": "string"}},
				"required":   []string{"order_id"},
			},
		}
	}
	config := &genai.GenerateContentConfig{
		SystemInstruction: genai.NewContentFromText(liveSystemInstruction(), genai.RoleUser),
		Tools: []*genai.Tool{{FunctionDeclarations: []*genai.FunctionDeclaration{
			declare("lookup_order", "Looks an order up by ID and reports its status."),
			declare("cancel_order", "Cancels an order by ID."),
		}}},
	}
	if forceTool != "" {
		config.ToolConfig = &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{
			Mode: genai.FunctionCallingConfigModeAny, AllowedFunctionNames: []string{forceTool},
		}}
	}
	req := &model.LLMRequest{
		Contents: contents,
		Config:   config,
		Tools:    map[string]any{"cancel_order": deferredEntry{}},
	}
	var final *model.LLMResponse
	for resp, err := range llm.GenerateContent(t.Context(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent() err = %v", err)
		}
		final = resp
	}
	if final == nil || final.Content == nil {
		t.Fatalf("no final response: %+v", final)
	}
	usage := final.UsageMetadata
	writes, _ := CacheWriteInputTokensFromResponse(final)
	if usage != nil {
		t.Logf("turn: input %d (cached %d, written %d), answer %d, reasoning %d",
			usage.PromptTokenCount, usage.CachedContentTokenCount, writes.TotalInputTokens,
			usage.CandidatesTokenCount, usage.ThoughtsTokenCount)
	}
	return final
}

func liveCall(t *testing.T, resp *model.LLMResponse, name string) *genai.FunctionCall {
	t.Helper()
	for _, part := range resp.Content.Parts {
		if part.FunctionCall != nil && part.FunctionCall.Name == name {
			return part.FunctionCall
		}
	}
	t.Fatalf("no call to %s in %+v", name, resp.Content.Parts)
	return nil
}

func liveToolResponse(call *genai.FunctionCall, response map[string]any) *genai.Content {
	return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
		ID: call.ID, Name: call.Name, Response: response,
	}}}}
}

func hasReasoningSignature(resp *model.LLMResponse) bool {
	for _, part := range resp.Content.Parts {
		if part.Thought && bytes.HasPrefix(part.ThoughtSignature, []byte(ReasoningSignaturePrefix)) {
			return true
		}
	}
	return false
}

func answerText(resp *model.LLMResponse) string {
	var text strings.Builder
	for _, part := range resp.Content.Parts {
		if !part.Thought {
			text.WriteString(part.Text)
		}
	}
	return text.String()
}

// liveSystemInstruction is long enough for the prompt cache, which caches a
// prefix only past a minimum length.
func liveSystemInstruction() string {
	var b strings.Builder
	b.WriteString("You help a warehouse team with orders. Use the tools to look orders up and cancel them, and say what you did.\n\nCatalog:\n")
	for i := range 300 {
		fmt.Fprintf(&b, "SKU A-%d: storage bin %d, shelf %d.\n", i, i%40, i%7)
	}
	return b.String()
}

// liveTestPDF returns a one-page PDF showing text, with its cross-reference
// offsets computed so readers need not repair it.
func liveTestPDF(text string) []byte {
	stream := fmt.Sprintf("BT /F1 18 Tf 72 720 Td (%s) Tj ET", text)
	objects := []string{
		"<< /Type /Catalog /Pages 2 0 R >>",
		"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
		"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
		fmt.Sprintf("<< /Length %d >>\nstream\n%s\nendstream", len(stream), stream),
		"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
	}
	var pdf bytes.Buffer
	pdf.WriteString("%PDF-1.4\n")
	offsets := make([]int, len(objects))
	for i, object := range objects {
		offsets[i] = pdf.Len()
		fmt.Fprintf(&pdf, "%d 0 obj\n%s\nendobj\n", i+1, object)
	}
	xref := pdf.Len()
	fmt.Fprintf(&pdf, "xref\n0 %d\n0000000000 65535 f \n", len(objects)+1)
	for _, offset := range offsets {
		fmt.Fprintf(&pdf, "%010d 00000 n \n", offset)
	}
	fmt.Fprintf(&pdf, "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n", len(objects)+1, xref)
	return pdf.Bytes()
}
