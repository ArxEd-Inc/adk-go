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

import "context"

// MessageStartUsage is the input-side token usage the provider reports at a response's message_start
// event: the final input and cache read/write counts for the request's prompt. These counts cannot be
// known before the provider has finished processing the prompt, so their arrival implies any prompt-cache
// entries the request writes have been written and are readable by concurrent requests.
type MessageStartUsage struct {
	// InputTokens is the input tokens neither read from nor written to cache (Anthropic reports the three
	// counts as separate additive fields).
	InputTokens int64
	// CacheReadInputTokens is the input tokens read from existing cache entries.
	CacheReadInputTokens int64
	// CacheCreationInputTokens is the input tokens written to new cache entries.
	CacheCreationInputTokens int64
}

// MessageStartCallback observes one GenerateContent call's message_start event — the moment the provider
// has finished processing the prompt, before any content is generated or yielded. It is invoked at most
// once per call, synchronously on the goroutine consuming the call's response sequence, in both streaming
// and non-streaming form; a call that fails before message_start never invokes it. Callers gating
// concurrent same-prefix requests use it to release waiters when the shared cache entry becomes readable
// rather than at the (possibly much later) first content yield.
type MessageStartCallback func(usage MessageStartUsage)

// messageStartCallbackContextKey keys a MessageStartCallback on the context.
type messageStartCallbackContextKey struct{}

// ContextWithMessageStartCallback returns ctx carrying the given callback, which this package's model
// invokes at the call's message_start event (see MessageStartCallback). The callback is per-request state
// rather than per-model configuration because its natural consumers key on the request (e.g. a
// prompt-prefix flight gate releasing that request's same-prefix waiters); a later installation shadows an
// earlier one, as with any context value.
func ContextWithMessageStartCallback(ctx context.Context, callback MessageStartCallback) context.Context {
	return context.WithValue(ctx, messageStartCallbackContextKey{}, callback)
}

// MessageStartCallbackFromContext returns the callback ctx carries, or ok=false when there is none.
// Exported so test fakes standing in for this model can honor the callback contract.
func MessageStartCallbackFromContext(ctx context.Context) (MessageStartCallback, bool) {
	callback, ok := ctx.Value(messageStartCallbackContextKey{}).(MessageStartCallback)
	return callback, ok
}
