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

package agent

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/artifact"
)

func TestCommonContext_ContextFallbackDelegation(t *testing.T) {
	t.Parallel()

	baseIC := &invocationContext{
		Context: t.Context(),
	}

	wantPath := "wf/child@123"
	wantAncestors := []string{"wf/root", "wf/parent"}
	runID := "123"
	var subScheduler DynamicSubScheduler = nil
	// Create a dynamic node context that explicitly populates path and outputForAncestors.
	delta := &CommonContextDelta{
		Path:               &wantPath,
		OutputForAncestors: &wantAncestors,
		RunID:              &runID,
		SubScheduler:       &subScheduler,
	}

	dynCtx := PromoteWithDelta(baseIC, delta)

	tests := []struct {
		name         string
		buildWrapped func(parent Context) Context
	}{
		{
			name: "Direct dynamic node context (fast path baseline)",
			buildWrapped: func(parent Context) Context {
				return parent
			},
		},
		{
			name: "NewToolContext wrapping branchOverride adapter (delegates fallback to c.Context)",
			buildWrapped: func(parent Context) Context {
				tc := NewToolContext(parent, "call-id-1", nil, nil)
				branch := "parallel-branch"
				return tc.WithDelta(&CommonContextDelta{InvocationContextDelta: &InvocationContextDelta{Branch: &branch}})
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gotCtx := tc.buildWrapped(dynCtx)

			if gotPath := gotCtx.Path(); gotPath != wantPath {
				t.Errorf("Path() = %q, want %q", gotPath, wantPath)
			}

			gotAncestors := gotCtx.OutputForAncestors()
			if diff := cmp.Diff(wantAncestors, gotAncestors); diff != "" {
				t.Errorf("OutputForAncestors() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// versionedArtifacts is an Artifacts whose Save returns, per file name,
// versions counting up from 1. If beforeReturn is set, Save calls it with the
// name and version it is about to return, so a test can control the order in
// which concurrent saves complete.
type versionedArtifacts struct {
	fakeArtifacts

	beforeReturn func(name string, version int64)

	mu            sync.Mutex
	versionByName map[string]int64
}

func (a *versionedArtifacts) Save(_ context.Context, name string, _ *genai.Part) (*artifact.SaveResponse, error) {
	a.mu.Lock()
	if a.versionByName == nil {
		a.versionByName = make(map[string]int64)
	}
	a.versionByName[name]++
	version := a.versionByName[name]
	a.mu.Unlock()

	if a.beforeReturn != nil {
		a.beforeReturn(name, version)
	}
	return &artifact.SaveResponse{Version: version}, nil
}

func TestToolContext_ConcurrentArtifactSavesRecordEveryName(t *testing.T) {
	t.Parallel()

	ic := &invocationContext{Context: t.Context(), artifacts: &versionedArtifacts{}}
	toolCtx := NewToolContext(ic, "call-1", nil, nil)

	const fileCount = 16
	var wg sync.WaitGroup
	for i := range fileCount {
		wg.Go(func() {
			name := fmt.Sprintf("report-%d.txt", i)
			if _, err := toolCtx.Artifacts().Save(toolCtx, name, genai.NewPartFromText("contents")); err != nil {
				t.Errorf("Save(%q) error = %v", name, err)
			}
		})
	}
	wg.Wait()

	want := make(map[string]int64, fileCount)
	for i := range fileCount {
		want[fmt.Sprintf("report-%d.txt", i)] = 1
	}
	if diff := cmp.Diff(want, toolCtx.Actions().ArtifactDelta); diff != "" {
		t.Errorf("ArtifactDelta mismatch (-want +got):\n%s", diff)
	}
}

func TestToolContext_ArtifactDeltaKeepsNewestVersionWhenSavesCompleteOutOfOrder(t *testing.T) {
	t.Parallel()

	firstSaveDrewVersion := make(chan struct{})
	releaseFirstSave := make(chan struct{})
	artifacts := &versionedArtifacts{
		beforeReturn: func(_ string, version int64) {
			if version == 1 {
				close(firstSaveDrewVersion)
				<-releaseFirstSave
			}
		},
	}
	toolCtx := NewToolContext(&invocationContext{Context: t.Context(), artifacts: artifacts}, "call-1", nil, nil)

	firstSaveErr := make(chan error, 1)
	go func() {
		_, err := toolCtx.Artifacts().Save(toolCtx, "summary.txt", genai.NewPartFromText("first"))
		firstSaveErr <- err
	}()
	<-firstSaveDrewVersion

	// The second save draws version 2 and completes while the first, holding
	// version 1, is still in flight.
	if _, err := toolCtx.Artifacts().Save(toolCtx, "summary.txt", genai.NewPartFromText("second")); err != nil {
		t.Fatalf("second Save() error = %v", err)
	}
	close(releaseFirstSave)
	if err := <-firstSaveErr; err != nil {
		t.Fatalf("first Save() error = %v", err)
	}

	if got := toolCtx.Actions().ArtifactDelta["summary.txt"]; got != 2 {
		t.Errorf("ArtifactDelta[%q] = %d, want 2 (the newest version)", "summary.txt", got)
	}
}
