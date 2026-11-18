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
// a plain bool or string carrying its zero — AudioTimestamp false or
// CachedContent empty — passes unremarked, as does HTTPOptions.Headers,
// ignored by design because headers addressed to another backend must not reach
// OpenAI.
//
//	Translated  Temperature, TopP, MaxOutputTokens, SystemInstruction,
//	            ResponseMIMEType, ResponseSchema, ResponseJsonSchema,
//	            ResponseLogprobs with Logprobs, Tools, ToolConfig,
//	            ThinkingConfig, ServiceTier, HTTPOptions.Timeout,
//	            MediaResolution
//	Rejected    TopK, CandidateCount above one, Labels, SafetySettings, an
//	            unsupported ResponseMIMEType, CachedContent, ResponseModalities,
//	            SpeechConfig, AudioTimestamp, ImageConfig,
//	            RoutingConfig, ModelSelectionConfig, ModelArmorConfig,
//	            EnableEnhancedCivicAnswers, AudioTranscriptionConfig,
//	            ContinuationToken, HTTPOptions apart from Timeout and Headers,
//	            a non-empty StopSequences, the penalties and Seed
//	Ignored     HTTPOptions.Headers; AutomaticContinuation, which has no
//	            effect because this API never pauses a generation with
//	            CONTINUATION
//
// A tool whose dispatch entry implements [DeferredLoadingTool] and defers is
// left out of the request's tools until a function response names it under
// [Config.ToolReferencesResponseKey]; an additional_tools input item then
// loads it right after that response, so the tools array, and the cached
// prefix behind it, stay unchanged as the set grows.
//
// Every request is streamed, whether or not the caller streams: a blocking
// request holds one silent HTTP response open for as long as the model takes,
// which at a high reasoning effort can outlast the idle timeouts between caller
// and API. A caller that does not stream receives only the stream's final,
// aggregated response, or an error wrapping [ErrEmptyResponse] when the stream
// ends without one.
//
// MediaResolution, the request's or a media part's own, sets the detail its
// images and files are sent at: high for high, low for low, auto otherwise.
//
// Function tools are sent with strict parameter validation disabled, and that
// is not configurable, so tool call arguments are best effort rather than
// guaranteed to match the declared parameter schema. A function's response
// schema, which a function tool has no field for, is appended to its
// description when the schema documents anything.
//
// A successful fallback from a response.completed conversion failure records
// "openai.terminal_response_conversion_failed" on the active OpenTelemetry span
// for unsupported message content or invalid function-call arguments. The
// event contains only an "error.type" category.
//
// An output item this package has no part for, such as a server-side tool's,
// is left out of the turn rather than failing it. The final response lists
// the types left out under [SkippedOutputItemTypesKey] in CustomMetadata, and
// "openai.output_items_skipped" is recorded on the active span with the same
// list.
//
// A refusal stays in the turn's content as text, and the final response is
// marked as a safety stop: FinishReason and ErrorCode SAFETY, with the refusal
// as ErrorMessage, as model/anthropic marks a refusal. A failure the server
// reports inside an HTTP 200, an error event in the stream or a response whose
// status is failed, is a [ServerError] carrying the server's error code.
//
// Model reasoning is reported to the caller as thought parts. Every request
// asks for its reasoning encrypted, and a reasoning item's id and encrypted
// content ride on the ThoughtSignature of the item's first thought part,
// behind [ReasoningSignaturePrefix]; an item that came with no text still
// yields one empty thought part to carry them. A later request sends the item
// back as an input item, ahead of the output items that followed it, so the
// model picks its reasoning up across tool calls instead of starting over. An
// item goes back only to the model that produced it, and only while one of the
// items that followed it is still in the turn; otherwise it is dropped, as are
// thoughts other providers wrote. A streamed turn carries its items when its
// terminal response restates what streamed, which OpenAI's does; a turn whose
// terminal response does not keeps its thoughts unsigned, and its reasoning is
// not sent back. The phase of an assistant message (commentary or final
// answer) rides the same way on its text parts and is sent back with them.
//
// What this package cannot police is reasoning that something upstream has
// already rendered as ordinary text: a peer agent's reply folded into context,
// a compacted transcript, or a memory recall all arrive with the thought
// marker gone, and reach every model package alike.
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
