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

// Package openai provides a client for OpenAI's Responses API.
//
// It is vendored from this module's experimental model/openaimodel package at
// upstream v2.8.0 and modified by Litix. Only the Responses path is carried;
// the Chat Completions path is left out. The package is meant to be deleted
// once openaimodel covers what it adds. To see what changed, diff this
// directory against the tag's model/openaimodel.
//
// It implements the model.LLM interface against POST /v1/responses.
//
// Every top-level field of genai.GenerateContentConfig is translated, rejected
// with an error naming it, or ignored because it has no effect on this API,
// the pre-existing errors being checked first
// so existing errors.Is call sites are unaffected. A field this package does
// not know, including one genai adds later, is rejected, as is an unknown field
// of HTTPOptions or ThinkingConfig. Rejection keys on presence,
// which is only observable where the zero value cannot itself be a setting, so
// a plain bool or string carrying its zero — AudioTimestamp false, CachedContent
// or MediaResolution empty — passes unremarked, as does HTTPOptions.Headers,
// ignored by design because headers addressed to another backend must not reach
// OpenAI.
//
//	Translated  Temperature, TopP, MaxOutputTokens, SystemInstruction,
//	            ResponseMIMEType, ResponseSchema, ResponseJsonSchema,
//	            ResponseLogprobs with Logprobs, Tools, ToolConfig,
//	            ThinkingConfig, ServiceTier, HTTPOptions.Timeout
//	Rejected    TopK, CandidateCount above one, Labels, SafetySettings, an
//	            unsupported ResponseMIMEType, CachedContent, ResponseModalities,
//	            MediaResolution, SpeechConfig, AudioTimestamp, ImageConfig,
//	            RoutingConfig, ModelSelectionConfig, ModelArmorConfig,
//	            EnableEnhancedCivicAnswers, AudioTranscriptionConfig,
//	            ContinuationToken, HTTPOptions apart from Timeout and Headers,
//	            a non-empty StopSequences, the penalties and Seed
//	Ignored     HTTPOptions.Headers; AutomaticContinuation, which has no
//	            effect because this API never pauses a generation with
//	            CONTINUATION
//
// Function tools are sent with strict parameter validation disabled, and that
// is not configurable, so tool call arguments are best effort rather than
// guaranteed to match the declared parameter schema.
//
// A successful fallback from a response.completed conversion failure records
// "openai.terminal_response_conversion_failed" on the active OpenTelemetry span
// for unsupported output items, unsupported message content, or invalid
// function-call arguments. The event contains only an "error.type" category.
//
// Model reasoning is reported to the caller as thought parts. Reasoning is not
// sent back on a later turn: the Responses API accepts it only as an input
// item referencing the id of the item that produced it, which ADK does not
// carry. Reasoning therefore informs the turn that produced it and no other; a
// caller that needs a conclusion to survive should have the model state it in
// the answer. What this package cannot police is reasoning that something
// upstream has already rendered as ordinary text: a peer agent's reply folded
// into context, a compacted transcript, or a memory recall all arrive with the
// thought marker gone, and reach every model package alike.
//
// Clients construct a Config and pass it to NewModel:
//
//	ctx := context.Background()
//	cfg := &openai.Config{APIKey: apiKey, Effort: openai.EffortHigh}
//	llm, err := openai.NewModel(ctx, "gpt-4o-mini", cfg)
//	if err != nil {
//		log.Fatal(err)
//	}
//
// The API key is required: the package never reads OPENAI_API_KEY from the
// environment. Setting BaseURL points the model at another server that
// implements the Responses API.
package openai
