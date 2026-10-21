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

// Package compactionctx carries the context-compaction runtime from the runner
// down to the request processors that need it.
//
// Those processors need the compaction config, and in one case the session
// service, and [agent.InvocationContext] exposes neither. Adding them to that
// interface would break every external implementation of it, so the runtime
// rides on the context.Context instead, the same way parentmap, runconfig and
// plugininternal already do.
package compactionctx

import (
	"context"
	"sync"
	"sync/atomic"

	"google.golang.org/adk/v2/internal/compactioninternal"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// Runtime is everything compaction needs that the invocation context does not
// already provide.
type Runtime struct {
	// config is the resolved compaction config, with its summarizer filled in.
	//
	// Unexported, with accessors, because one Runtime is shared by every
	// goroutine in an invocation. An exported pointer field invites a caller to
	// swap it mid-turn, and the config it points at is shared across every
	// invocation of the runner, so a mutation would leak between turns.
	config *compaction.Config
	// sessionService persists the summary events the compactor produces.
	sessionService session.Service

	// compacted records that a compaction already ran in this invocation. A
	// Runtime is built per invocation, so it is the right scope for this, and
	// it is atomic because sub-agents running in parallel share one.
	compacted atomic.Bool

	// gates holds the per-agent progress state, keyed by the agent's scope.
	//
	// One invocation can run several agents, and they do not share a prompt: a
	// loop agent alternating a large-context worker with a small-context critic
	// is two different prompt sizes, and parallel sub-agents are as many as
	// there are children. A single marker for all of them was wrong in both
	// directions. It let every parallel sibling read the gate as open before
	// any of them closed it, so all of them summarized, and it let one agent's
	// compaction suppress another's.
	mu    sync.Mutex
	gates map[string]*gateState
}

// gateState is one agent's progress within an invocation.
type gateState struct {
	// lastCompactionTokens is the baseline for the most recent compaction
	// that has not yet been shown to work, or 0 when there is none: the prompt
	// size it was recorded at, lowered by any smaller reading since.
	lastCompactionTokens int64
	// failures counts attempts that produced nothing storable. Distinct from a
	// successful compaction because Recovered and growth can reopen that one
	// and neither reopens this: the prompt never dropped, so nothing will
	// report recovery, and a summarizer that keeps failing has no reason to
	// succeed on a longer transcript.
	//
	// A few are allowed rather than one. Retrying without limit made a
	// persistently failing summarizer cost a model call before every model call
	// for the life of the turn, 29 across a 30-round tool loop. Stopping after
	// the first swapped that for never recovering from one transient error, and
	// a turn long enough to need compaction is long enough for a provider to
	// blip once. Neither is the useful answer.
	failures int
}

// maxFailuresPerInvocation is how many unproductive attempts one agent makes in
// one invocation before giving up on compacting it.
const maxFailuresPerInvocation = 3

// reArmFraction sets how far a prompt still over the threshold after a
// compaction must grow before another is attempted in the same invocation: by
// TokenThreshold/reArmFraction. Smaller spends summarizer calls on a turn that
// is only creeping; larger lets the prompt run further past the threshold
// before it is caught. At an eighth, a turn whose prompt compacting cannot
// bring down at all makes at most eight more attempts before the prompt reaches
// twice the threshold.
const reArmFraction = 8

// Gate is a Runtime's progress gate scoped to one agent.
//
// The scope has to identify the agent, not just its branch. A sequential agent,
// a loop agent and an agent transfer all pass the invocation context through
// unchanged, so every agent in those shapes shares a branch: keying on branch
// alone gave a loop agent's worker and critic one gate, which is the very
// example this scoping exists for. Only a parallel agent gets distinct
// branches. The isolation scope belongs in the key for the same reason it
// belongs in the prompt filter: two agents that cannot see each other's events
// do not have the same prompt.
type Gate struct {
	rt    *Runtime
	scope string
}

// GateFor returns the progress gate for one agent, identified by its name,
// branch and isolation scope together.
func (rt *Runtime) GateFor(agentName, branch, isolationScope string) *Gate {
	if rt == nil {
		return nil
	}
	// NUL-joined. That is not injective on its own, since a NUL inside one part
	// can imitate a separator, but none of the three can contain one: an agent
	// name is validated, and a branch and an isolation scope are built by the
	// framework from agent names. Length-prefixing would be, if that ever stops
	// holding.
	return &Gate{rt: rt, scope: agentName + "\x00" + branch + "\x00" + isolationScope}
}

func (g *Gate) state() *gateState {
	if g.rt.gates == nil {
		g.rt.gates = make(map[string]*gateState)
	}
	st, ok := g.rt.gates[g.scope]
	if !ok {
		st = &gateState{}
		g.rt.gates[g.scope] = st
	}
	return st
}

// AllowAt reports whether another compaction is worth attempting for this agent
// in this invocation, with its prompt now at tokens.
//
// The caller has already found the prompt over the threshold by the time it
// asks, and calls Recovered the moment it drops back under, so what is left is
// whether a compaction earlier in this turn, which has not brought the prompt
// under, leaves anything worth another attempt. Straight afterwards it does
// not: the window past the retained tail is empty or nearly so, and another
// pass would summarize a little more and leave the prompt as far over. But the
// turn keeps growing, and every event that scrolls out of the retained tail as
// it does is compactable. Waiting for a recovery that only another compaction
// could bring shut compaction off for the rest of the turn, and the prompt
// grew unchecked to the model's context limit.
//
// So the gate reopens once the prompt has grown by TokenThreshold/reArmFraction
// since that compaction. That bounds the extra summarizer calls by growth, and
// it measures growth with the observed prompt count, which sees the tool
// declarations and documents that a window estimate cannot. The growth is
// measured from the lowest reading since the compaction rather than from the
// compaction itself, because RecordAt stores the prompt as it was before the
// pass: a pass that took the prompt from 12,000 to 9,000 against an 8,000
// threshold would otherwise reopen only at 13,000, not 10,000.
func (g *Gate) AllowAt(tokens int) bool {
	if g == nil || g.rt == nil {
		return false
	}
	g.rt.mu.Lock()
	defer g.rt.mu.Unlock()
	st := g.state()
	if st.lastCompactionTokens != 0 {
		// Kept non-zero for the reason RecordAt gives.
		st.lastCompactionTokens = min(st.lastCompactionTokens, max(int64(tokens), 1))
	}
	if st.failures >= maxFailuresPerInvocation {
		return false
	}
	if st.lastCompactionTokens == 0 {
		return true
	}
	reArm := g.rt.reArmGrowth()
	return reArm > 0 && int64(tokens)-st.lastCompactionTokens >= reArm
}

// reArmGrowth is how far a prompt left over the threshold by a compaction has
// to grow before AllowAt allows another, or 0 when it never does.
func (rt *Runtime) reArmGrowth() int64 {
	if rt.config == nil {
		return 0
	}
	return int64(rt.config.TokenThreshold) / reArmFraction
}

// RecordAt notes that a compaction was stored at this prompt size, which
// becomes the baseline AllowAt measures further growth from.
func (g *Gate) RecordAt(tokens int) {
	if g == nil || g.rt == nil {
		return
	}
	g.rt.mu.Lock()
	defer g.rt.mu.Unlock()
	// A compaction at a zero count would read as "nothing recorded yet", so the
	// marker is kept non-zero. A baseline one token high is immaterial against
	// the growth AllowAt waits for.
	g.state().lastCompactionTokens = max(int64(tokens), 1)
}

// Failed notes an attempt that produced nothing storable. After
// maxFailuresPerInvocation of them this agent stops attempting for the rest of
// the invocation.
func (g *Gate) Failed() {
	if g == nil || g.rt == nil {
		return
	}
	g.rt.mu.Lock()
	defer g.rt.mu.Unlock()
	g.state().failures++
}

// Recovered notes that this agent's prompt is back under the threshold,
// re-arming compaction for it.
func (g *Gate) Recovered() {
	if g == nil || g.rt == nil {
		return
	}
	g.rt.mu.Lock()
	defer g.rt.mu.Unlock()
	g.state().lastCompactionTokens = 0
}

// New builds a Runtime. A nil config yields a nil Runtime, which every method
// here tolerates, so callers do not have to branch.
func New(cfg *compaction.Config, svc session.Service) *Runtime {
	if cfg == nil {
		return nil
	}
	return &Runtime{config: cfg, sessionService: svc}
}

// Config returns the compaction config.
func (rt *Runtime) Config() *compaction.Config {
	if rt == nil {
		return nil
	}
	return rt.config
}

// SessionService returns the service that persists summaries.
func (rt *Runtime) SessionService() session.Service {
	if rt == nil {
		return nil
	}
	return rt.sessionService
}

// MarkCompacted records that a compaction ran during this invocation.
func (rt *Runtime) MarkCompacted() {
	if rt == nil {
		return
	}
	rt.compacted.Store(true)
}

// AlreadyCompacted reports whether a compaction ran during this invocation.
//
// The two strategies are independent triggers on the same history, so without
// this a turn that crossed the token threshold mid-flight would be summarized
// again by the sliding window the moment it ended, paying for a second model
// call to re-summarize what was just summarized. The reference implementation
// avoids it by evaluating the two in one place and returning early; the same
// effect is reached here by remembering, since the two run at different points
// in the turn.
func (rt *Runtime) AlreadyCompacted() bool {
	return rt != nil && rt.compacted.Load()
}

// Configured reports whether compaction is enabled for this run.
//
// Prompt assembly gates on this rather than simply honoring any compaction
// record it finds. A record instructs the prompt builder to drop a range of
// history and substitute content in its place, so acting on one that this
// runner did not ask for would turn a stored field into an erase-and-inject
// primitive, available even to an application that never enabled compaction.
func (rt *Runtime) Configured() bool {
	return rt != nil && rt.config != nil
}

// Enabled reports whether rt can actually run a tail-retention compaction.
func (rt *Runtime) Enabled() bool {
	return rt != nil && rt.sessionService != nil && compactioninternal.HasTailRetention(rt.config)
}

// ToContext returns a context carrying rt.
func ToContext(ctx context.Context, rt *Runtime) context.Context {
	return context.WithValue(ctx, runtimeCtxKey, rt)
}

// FromContext returns the [Runtime] carried by ctx, or nil when compaction is
// not configured.
func FromContext(ctx context.Context) *Runtime {
	rt, ok := ctx.Value(runtimeCtxKey).(*Runtime)
	if !ok {
		return nil
	}
	return rt
}

type ctxKey int

const runtimeCtxKey ctxKey = 0

// RecordAt notes that a compaction was performed at this prompt size.
//
