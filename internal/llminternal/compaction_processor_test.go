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

package llminternal_test

import (
	"context"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/agent/compactionctx"
	"google.golang.org/adk/v2/internal/compactioninternal"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// seedWrappedSession stands in for the wrapper an agent installs over the real
// session when it hands a sub-agent a synthetic first turn. It decorates the
// session and is not a type any session service recognizes.
type seedWrappedSession struct {
	session.Session
}

func (w *seedWrappedSession) Unwrap() session.Session { return w.Session }

// fixedSummarizer returns one canned summary.
type fixedSummarizer struct{ calls int }

func (s *fixedSummarizer) SummarizeEvents(_ context.Context, events []*session.Event) (compaction.SummarizeResult, error) {
	s.calls++
	return compaction.SummarizeResult{Content: genai.NewContentFromText("SUMMARY", "model")}, nil
}

// tailRetentionFixture builds a stored session holding n exchanges, the last of
// which reports a prompt token count well past any threshold a test will set.
func tailRetentionFixture(t *testing.T, n int) (session.Service, session.Session) {
	t.Helper()

	svc := session.InMemoryService()
	created, err := svc.Create(t.Context(), &session.CreateRequest{
		AppName: "app", UserID: "u", SessionID: "s",
	})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	sess := created.Session

	base := time.Unix(1, 0)
	for i := range n {
		q := session.NewEvent(t.Context(), "inv")
		q.Author = "user"
		q.Timestamp = base.Add(time.Duration(2*i) * time.Second)
		q.LLMResponse.Content = genai.NewContentFromText("question", "user")
		if err := svc.AppendEvent(t.Context(), sess, q); err != nil {
			t.Fatalf("AppendEvent() error = %v", err)
		}

		a := session.NewEvent(t.Context(), "inv")
		a.Author = "assistant"
		a.Timestamp = base.Add(time.Duration(2*i+1) * time.Second)
		a.LLMResponse.Content = genai.NewContentFromText("answer", "model")
		a.LLMResponse.UsageMetadata = &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 5000}
		if err := svc.AppendEvent(t.Context(), sess, a); err != nil {
			t.Fatalf("AppendEvent() error = %v", err)
		}
	}
	return svc, sess
}

func runCompactionProcessor(t *testing.T, svc session.Service, sess session.Session, cfg *compaction.Config) error {
	t.Helper()

	ctx := compactionctx.ToContext(t.Context(), compactionctx.New(cfg, svc))
	testAgent := utils.Must(llmagent.New(llmagent.Config{Name: "assistant", Model: &testModel{}}))
	ictx := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
		Agent:   testAgent,
		Session: sess,
	})

	var gotErr error
	for ev, err := range llminternal.CompactionRequestProcessor(ictx, &model.LLMRequest{}, &llminternal.Flow{}) {
		if ev != nil {
			t.Fatal("CompactionRequestProcessor yielded an event, which it must not do")
		}
		if err != nil {
			gotErr = err
		}
	}
	return gotErr
}

func storedCompactions(t *testing.T, svc session.Service) []*session.Event {
	t.Helper()

	resp, err := svc.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	var out []*session.Event
	for ev := range resp.Session.Events().All() {
		if compactioninternal.HasUsableSummary(ev) {
			out = append(out, ev)
		}
	}
	return out
}

// TestCompactionProcessorAppendsThroughASessionWrapper checks that tail
// retention still works when the invocation carries a wrapped session.
//
// An agent hands a sub-agent a session wrapped to carry a synthetic first turn.
// Every session service type-asserts on its own concrete type, so appending a
// summary to the wrapper fails outright. The failure does not even surface as an
// error on the delegating path: it becomes a tool-error response and the
// coordinator answers on top of a broken delegation.
func TestCompactionProcessorAppendsThroughASessionWrapper(t *testing.T) {
	t.Parallel()

	svc, sess := tailRetentionFixture(t, 4)
	summarizer := &fixedSummarizer{}

	err := runCompactionProcessor(t, svc, &seedWrappedSession{Session: sess}, &compaction.Config{
		TokenThreshold:     100,
		EventRetentionSize: 2,
		Summarizer:         summarizer,
	})
	if err != nil {
		t.Fatalf("compaction through a wrapped session failed: %v", err)
	}
	if summarizer.calls == 0 {
		t.Fatal("the summarizer never ran, so this test proved nothing")
	}
	if got := len(storedCompactions(t, svc)); got != 1 {
		t.Errorf("stored %d compaction events, want 1: the summary never reached the session", got)
	}
}

// TestCompactionProcessorSkipsOnCancelledContext checks that a canceled turn
// does not spend a model call on compaction, nor write a summary the caller is
// no longer waiting for.
func TestCompactionProcessorSkipsOnCancelledContext(t *testing.T) {
	t.Parallel()

	svc, sess := tailRetentionFixture(t, 4)
	summarizer := &fixedSummarizer{}

	ctx, cancel := context.WithCancel(t.Context())
	ctx = compactionctx.ToContext(ctx, compactionctx.New(&compaction.Config{
		TokenThreshold:     100,
		EventRetentionSize: 2,
		Summarizer:         summarizer,
	}, svc))
	testAgent := utils.Must(llmagent.New(llmagent.Config{Name: "assistant", Model: &testModel{}}))
	ictx := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
		Agent:   testAgent,
		Session: sess,
	})
	cancel()

	for range llminternal.CompactionRequestProcessor(ictx, &model.LLMRequest{}, &llminternal.Flow{}) { //nolint:revive
	}

	if summarizer.calls != 0 {
		t.Errorf("summarizer ran %d time(s) on a canceled turn", summarizer.calls)
	}
	if got := len(storedCompactions(t, svc)); got != 0 {
		t.Errorf("stored %d compaction events on a canceled turn", got)
	}
}

// racingSummarizer appends an event inside the range it is about to summarize,
// standing in for a concurrent invocation landing during the model call.
type racingSummarizer struct {
	svc session.Service
	t   *testing.T

	// The event this summarizer landed mid-call.
	racedInvocation string
	racedTimestamp  time.Time
}

func (s *racingSummarizer) SummarizeEvents(ctx context.Context, events []*session.Event) (compaction.SummarizeResult, error) {
	// Land a new event inside the range the summary claims, through a separate
	// handle on the same stored session. Appending through the caller's handle
	// would update that handle too, which is exactly what a concurrent
	// invocation in another goroutine does not do.
	other, err := s.svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		s.t.Fatalf("racing Get() error = %v", err)
	}
	late := session.NewEvent(ctx, "other-invocation")
	late.Author = "user"
	// Inside the range the framework will derive from the window it handed over.
	late.Timestamp = events[0].Timestamp.Add(time.Millisecond)
	late.LLMResponse.Content = genai.NewContentFromText("CONCURRENT", "user")
	if err := s.svc.AppendEvent(ctx, other.Session, late); err != nil {
		s.t.Fatalf("racing AppendEvent() error = %v", err)
	}
	s.racedInvocation, s.racedTimestamp = late.InvocationID, late.Timestamp
	return compaction.SummarizeResult{Content: genai.NewContentFromText("SUMMARY", "model")}, nil
}

// TestCompactionProcessorKeepsARacedSummaryAndTheRacerRaw checks that a
// summary whose range gained an event while it was being produced is stored
// with that event named as a hole.
//
// A summary records the holes inside its range, and that list is computed from
// what the framework could see when it was built. An event that lands
// afterwards is inside the range and was summarized by nothing, so recorded as
// it is it would read as covered and be dropped from the prompt. Naming it
// keeps it raw; discarding the summary would protect it too, but throws away a
// model call whose result is correct for everything else.
func TestCompactionProcessorKeepsARacedSummaryAndTheRacerRaw(t *testing.T) {
	t.Parallel()

	svc, sess := tailRetentionFixture(t, 4)

	summarizer := &racingSummarizer{svc: svc, t: t}
	err := runCompactionProcessor(t, svc, sess, &compaction.Config{
		TokenThreshold:     100,
		EventRetentionSize: 2,
		Summarizer:         summarizer,
	})
	if err != nil {
		t.Fatalf("CompactionRequestProcessor failed: %v", err)
	}
	if summarizer.racedInvocation == "" {
		t.Fatal("the racing summarizer did not record what it appended")
	}
	assertStoredWithHoleKeptRaw(t, svc, summarizer.racedInvocation, summarizer.racedTimestamp)
}

// TestCompactionProcessorKeepsASummaryOverAnEventFromASecondHandle checks the
// same for an event that was stored before the window was chosen, through a
// second handle on the session.
//
// An application's background bookkeeping writer that appends through its own
// handle is not racing anything, but the handle the turn runs on never holds
// its events, so every pass whose range reaches one finds it in the store and
// not in its snapshot. Treated as a race, that discarded every such summary
// after it was paid for, and because nothing was stored the next model call
// paid for another.
func TestCompactionProcessorKeepsASummaryOverAnEventFromASecondHandle(t *testing.T) {
	t.Parallel()

	svc, sess := tailRetentionFixture(t, 4)

	bookkeeperHandle, err := svc.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	// Between the first two exchanges, so inside the range the window will
	// claim, in the fixture turn's invocation.
	record := session.NewEvent(t.Context(), "inv")
	record.Author = "bookkeeper"
	record.Timestamp = time.Unix(2, int64(500*time.Millisecond))
	if err := svc.AppendEvent(t.Context(), bookkeeperHandle.Session, record); err != nil {
		t.Fatalf("AppendEvent() error = %v", err)
	}

	summarizer := &fixedSummarizer{}
	if err := runCompactionProcessor(t, svc, sess, &compaction.Config{
		TokenThreshold:     100,
		EventRetentionSize: 2,
		Summarizer:         summarizer,
	}); err != nil {
		t.Fatalf("CompactionRequestProcessor failed: %v", err)
	}
	if summarizer.calls == 0 {
		t.Fatal("the summarizer never ran, so this test proved nothing")
	}
	assertStoredWithHoleKeptRaw(t, svc, record.InvocationID, record.Timestamp)
}

// assertStoredWithHoleKeptRaw checks that exactly one summary was stored, that
// it names the event with the given invocation and timestamp as a hole, and
// that prompt assembly keeps that event raw.
func assertStoredWithHoleKeptRaw(t *testing.T, svc session.Service, invocationID string, timestamp time.Time) {
	t.Helper()

	records := storedCompactions(t, svc)
	if len(records) != 1 {
		t.Fatalf("stored %d compaction events, want 1: the summary must be kept", len(records))
	}
	rng := records[0].Actions.Compaction
	if timestamp.Before(rng.StartTimestamp) || timestamp.After(rng.EndTimestamp) {
		t.Fatalf("the event at %v is outside the stored range [%v, %v], so this test is not exercising the check",
			timestamp, rng.StartTimestamp, rng.EndTimestamp)
	}
	var named bool
	for _, ref := range rng.ExcludedEvents {
		if ref.InvocationID == invocationID && ref.Timestamp.Equal(timestamp) {
			named = true
		}
	}
	if !named {
		t.Errorf("the stored summary's holes %v do not name the event it never saw", rng.ExcludedEvents)
	}

	got, err := svc.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	var stored []*session.Event
	for ev := range got.Session.Events().All() {
		stored = append(stored, ev)
	}
	for _, ev := range compactioninternal.Apply(stored) {
		if ev.InvocationID == invocationID && ev.Timestamp.Equal(timestamp) {
			return
		}
	}
	t.Errorf("the event the summary never saw was dropped from the prompt")
}

// competingSummarizer stores another compaction record over the range it is
// about to summarize, standing in for a concurrent invocation compacting the
// same history during the model call.
type competingSummarizer struct {
	svc session.Service
	t   *testing.T

	// competed is set once the competing record is stored.
	competed bool
}

func (s *competingSummarizer) SummarizeEvents(ctx context.Context, events []*session.Event) (compaction.SummarizeResult, error) {
	other, err := s.svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		s.t.Fatalf("competing Get() error = %v", err)
	}
	rival := session.NewEvent(ctx, "other-invocation")
	rival.Author = "user"
	rival.Actions.Compaction = &session.EventCompaction{
		StartTimestamp:   events[0].Timestamp,
		EndTimestamp:     events[len(events)-1].Timestamp,
		CompactedContent: genai.NewContentFromText("COMPETING SUMMARY", "model"),
	}
	if err := s.svc.AppendEvent(ctx, other.Session, rival); err != nil {
		s.t.Fatalf("competing AppendEvent() error = %v", err)
	}
	s.competed = true
	return compaction.SummarizeResult{Content: genai.NewContentFromText("SUMMARY", "model")}, nil
}

// TestCompactionProcessorDiscardsASummaryAnotherCompactionRaced checks that a
// summary is still thrown away when another compaction over the same events
// was stored while it was being produced.
//
// Two summaries whose ranges meet would each stand in for the same turns, so
// the same content would be materialized twice into one prompt. A hole cannot
// express that, so the later summary is abandoned and its model call wasted.
func TestCompactionProcessorDiscardsASummaryAnotherCompactionRaced(t *testing.T) {
	t.Parallel()

	svc, sess := tailRetentionFixture(t, 4)

	summarizer := &competingSummarizer{svc: svc, t: t}
	if err := runCompactionProcessor(t, svc, sess, &compaction.Config{
		TokenThreshold:     100,
		EventRetentionSize: 2,
		Summarizer:         summarizer,
	}); err != nil {
		t.Fatalf("CompactionRequestProcessor failed: %v", err)
	}
	if !summarizer.competed {
		t.Fatal("the competing summarizer did not store its record")
	}
	for _, ev := range storedCompactions(t, svc) {
		if ev.InvocationID != "other-invocation" {
			t.Errorf("stored a summary from %q beside a competing one over the same events", ev.InvocationID)
		}
	}
}

// straggleOnAppend lands an event inside the summary's range at the moment the
// summary itself is being stored, which is the one window the race guard cannot
// cover.
type straggleOnAppend struct {
	session.Service
	t    *testing.T
	done bool

	stragglerInvocation string
	stragglerTimestamp  time.Time
}

func (s *straggleOnAppend) AppendEvent(ctx context.Context, sess session.Session, ev *session.Event) error {
	if !s.done && ev.Actions.Compaction != nil {
		s.done = true
		// Through a separate handle, the way a concurrent invocation would.
		other, err := s.Service.Get(ctx, &session.GetRequest{AppName: "app", UserID: "u", SessionID: "s"})
		if err != nil {
			s.t.Fatalf("straggler Get() error = %v", err)
		}
		late := session.NewEvent(ctx, "straggler-invocation")
		late.Author = "user"
		late.Timestamp = ev.Actions.Compaction.StartTimestamp.Add(time.Millisecond)
		late.LLMResponse.Content = genai.NewContentFromText("PLEASE DO NOT LOSE ME", "user")
		if err := s.Service.AppendEvent(ctx, other.Session, late); err != nil {
			s.t.Fatalf("straggler AppendEvent() error = %v", err)
		}
		s.stragglerInvocation, s.stragglerTimestamp = late.InvocationID, late.Timestamp
	}
	return s.Service.AppendEvent(ctx, sess, ev)
}

// TestCompactionProcessorRepairsAStragglerStoredDuringTheAppend pins that the
// mid-turn path repairs what the race guard cannot cover.
//
// The guard runs immediately before the append and the append cannot be made
// conditional, so an event stored in between lands inside the recorded range
// named by no hole, and prompt assembly drops it with no summary standing in
// for it. The post-invocation path has repaired this since it was found; this
// path did not, and it is the one that needs it more, because it runs inside
// the invocation while sub-agents and tools are still producing events.
func TestCompactionProcessorRepairsAStragglerStoredDuringTheAppend(t *testing.T) {
	t.Parallel()

	base, sess := tailRetentionFixture(t, 6)
	svc := &straggleOnAppend{Service: base, t: t}

	if err := runCompactionProcessor(t, svc, sess, &compaction.Config{
		TokenThreshold:     100,
		EventRetentionSize: 2,
		Summarizer:         &fixedSummarizer{},
	}); err != nil {
		t.Fatalf("CompactionRequestProcessor failed: %v", err)
	}
	if svc.stragglerInvocation == "" {
		t.Fatal("no straggler was landed, so this test is not exercising the window")
	}

	got, err := svc.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	var stored []*session.Event
	for ev := range got.Session.Events().All() {
		stored = append(stored, ev)
	}

	// The straggler must survive into the assembled prompt: no summary
	// describes it, so it has to be there raw.
	for _, ev := range compactioninternal.Apply(stored) {
		if ev.InvocationID == svc.stragglerInvocation {
			return
		}
	}
	t.Errorf("the event stored during the append was dropped from the prompt and no summary describes it")
}
