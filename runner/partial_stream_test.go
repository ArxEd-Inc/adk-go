// Copyright 2025 Google LLC
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

package runner

import (
	"context"
	"iter"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// partialOnlyModel is a model.LLM whose stream yields one partial response and
// then ends: a stream truncated before its final aggregated message.
type partialOnlyModel struct{}

func (partialOnlyModel) Name() string { return "partial-only" }

func (partialOnlyModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{
			Content: genai.NewContentFromText("the start of a reply that never", genai.RoleModel),
			Partial: true,
		}, nil)
	}
}

// TestRun_StepEndingOnPartialEventDoesNotRaceTheConsumer runs an LLM agent as
// the root node with a model whose stream ends on a partial event. The flow
// ends the step cleanly and inspects the last event to decide so, while the
// runner, on another goroutine, has already received that event and writes
// its compaction record back onto it. Under the race detector this failed
// until the flow read what it needed from the event before yielding it.
func TestRun_StepEndingOnPartialEventDoesNotRaceTheConsumer(t *testing.T) {
	ctx := context.Background()
	const appName, userID, sessionID = "partial-stream", "user", "session"

	rootAgent, err := llmagent.New(llmagent.Config{
		Name:        "narrator",
		Description: "an agent whose model stream ends before its final message",
		Model:       partialOnlyModel{},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	sessionService := session.InMemoryService()
	if _, err := sessionService.Create(ctx, &session.CreateRequest{AppName: appName, UserID: userID, SessionID: sessionID}); err != nil {
		t.Fatalf("Create: %v", err)
	}
	r, err := New(Config{AppName: appName, Agent: rootAgent, SessionService: sessionService})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	var partials, finals int
	for ev, err := range r.Run(ctx, userID, sessionID, genai.NewContentFromText("go on", genai.RoleUser), agent.RunConfig{StreamingMode: agent.StreamingModeSSE}) {
		if err != nil {
			t.Fatalf("Run yielded an error for a truncated stream, want a clean end: %v", err)
		}
		if ev == nil {
			continue
		}
		if ev.LLMResponse.Partial {
			partials++
		} else if ev.Author == rootAgent.Name() {
			finals++
		}
	}
	if partials != 1 {
		t.Errorf("partial events = %d, want 1", partials)
	}
	if finals != 0 {
		t.Errorf("final agent events = %d, want 0: a truncated stream leaves no final response", finals)
	}
}
