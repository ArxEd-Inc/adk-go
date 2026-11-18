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

package responses

import (
	"bytes"
	"encoding/json"
	"regexp"

	"github.com/openai/openai-go/v3/packages/param"
	oairesponses "github.com/openai/openai-go/v3/responses"
	"google.golang.org/genai"
)

// ReasoningSignaturePrefix begins the ThoughtSignature of a thought part that
// carries a reasoning item for replay. A reasoning item travels in the
// signature rather than in PartMetadata because session stores persist the
// signature and some drop the metadata. The prefix tells such a part from a
// signature another provider wrote, which this package cannot replay and
// another provider's package must not.
const ReasoningSignaturePrefix = "adk-openai-reasoning/v1:"

// phaseSignaturePrefix begins the ThoughtSignature of an answer's text part
// that carries the phase of the assistant message it came from.
const phaseSignaturePrefix = "adk-openai-phase/v1:"

// reasoningPayload is what a reasoning item's signature holds: enough to send
// the item back as an input item on a later turn.
type reasoningPayload struct {
	// Model is the model that produced the item. Encrypted reasoning is
	// replayed only to the same model, which can read it.
	Model            string   `json:"model"`
	ID               string   `json:"id"`
	EncryptedContent string   `json:"encrypted_content,omitempty"`
	Summary          []string `json:"summary,omitempty"`
}

// reasoningSignature encodes a reasoning output item as a thought signature,
// or returns nil for an item with no ID, which the API could not match to
// anything on replay.
func reasoningSignature(modelName string, item oairesponses.ResponseOutputItemUnion) []byte {
	if item.ID == "" {
		return nil
	}
	payload := reasoningPayload{Model: modelName, ID: item.ID, EncryptedContent: item.EncryptedContent}
	for _, summary := range item.Summary {
		payload.Summary = append(payload.Summary, summary.Text)
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		// A struct of strings always marshals.
		return nil
	}
	return append([]byte(ReasoningSignaturePrefix), encoded...)
}

// decodeReasoningSignature returns the reasoning item a thought signature
// carries, or false when the signature is not one this package wrote.
func decodeReasoningSignature(signature []byte) (reasoningPayload, bool) {
	rest, ok := bytes.CutPrefix(signature, []byte(ReasoningSignaturePrefix))
	if !ok {
		return reasoningPayload{}, false
	}
	var payload reasoningPayload
	if err := json.Unmarshal(rest, &payload); err != nil || payload.ID == "" {
		return reasoningPayload{}, false
	}
	return payload, true
}

// reasoningInputItem is a decoded reasoning item as an input item.
func reasoningInputItem(payload reasoningPayload) oairesponses.ResponseInputItemUnionParam {
	item := &oairesponses.ResponseReasoningItemParam{
		ID: payload.ID,
		// The API requires the field, so an item that came with no summary
		// sends an empty list rather than none.
		Summary: []oairesponses.ResponseReasoningItemSummaryParam{},
	}
	for _, text := range payload.Summary {
		item.Summary = append(item.Summary, oairesponses.ResponseReasoningItemSummaryParam{Text: text})
	}
	if payload.EncryptedContent != "" {
		item.EncryptedContent = param.NewOpt(payload.EncryptedContent)
	}
	return oairesponses.ResponseInputItemUnionParam{OfReasoning: item}
}

// phaseSignature encodes an assistant message's phase as the signature of its
// text parts, or returns nil when the message stated none.
func phaseSignature(phase oairesponses.ResponseOutputMessagePhase) []byte {
	if phase == "" {
		return nil
	}
	return []byte(phaseSignaturePrefix + string(phase))
}

// decodePhaseSignature returns the phase a text part's signature carries, or
// "" when it carries none this package wrote.
func decodePhaseSignature(signature []byte) oairesponses.ResponseOutputMessagePhase {
	rest, ok := bytes.CutPrefix(signature, []byte(phaseSignaturePrefix))
	if !ok {
		return ""
	}
	return oairesponses.ResponseOutputMessagePhase(rest)
}

// replayableReasoning reports whether the reasoning part at parts[i], whose
// signature decoded to payload, can be sent back to modelName. The API accepts
// a reasoning item only together with the output items that followed it, so
// one whose turn kept none of them, as after a compaction rewrote the turn,
// is left out. So is one another model produced, which modelName cannot read.
func replayableReasoning(payload reasoningPayload, modelName string, parts []*genai.Part, i int) bool {
	if baseModelName(payload.Model) != baseModelName(modelName) {
		return false
	}
	for _, later := range parts[i+1:] {
		if later == nil {
			continue
		}
		if later.FunctionCall != nil || (later.Text != "" && !later.Thought) {
			return true
		}
	}
	return false
}

// snapshotSuffix is the "-YYYY-MM-DD" a response reports after the model name
// a request asked for when that name was an alias.
var snapshotSuffix = regexp.MustCompile(`-\d{4}-\d{2}-\d{2}$`)

// baseModelName strips a dated snapshot suffix, so a reasoning item produced
// under a model's alias and replayed under its snapshot name, or the other way
// round, counts as coming from the same model.
func baseModelName(name string) string {
	return snapshotSuffix.ReplaceAllString(name, "")
}
