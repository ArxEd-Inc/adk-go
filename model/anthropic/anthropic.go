// Copyright 2025 Alcova AI
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
//
// Modified by Litix from github.com/Alcova-AI/adk-anthropic-go (v0.1.18).

package anthropic

import (
	"context"
	"fmt"
	"iter"
	"os"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/vertex"
	"golang.org/x/oauth2/google"
	"google.golang.org/genai"

	"google.golang.org/adk/model"
	"google.golang.org/adk/model/anthropic/internal/converters"
)

const defaultMaxTokens = 16384

// cloudPlatformScope is the OAuth scope Vertex AI requires, passed explicitly when loading Application
// Default Credentials. Without an explicit scope, credentials that mint tokens by service-account
// impersonation request an empty scope set, which the IAM Credentials API rejects with HTTP 400.
// Metadata-server credentials already default to this scope, so passing it explicitly makes both paths work.
const cloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

type anthropicModel struct {
	client           anthropicsdk.Client
	name             anthropicsdk.Model
	variant          string
	defaultMaxTokens int
	effort           Effort
	promptCaching    *PromptCachingConfig
}

// NewModel returns [model.LLM], backed by Anthropic Claude.
//
// It creates an Anthropic client based on the provided configuration.
// If Variant is not specified, it checks the ANTHROPIC_USE_VERTEX environment variable.
//
// For direct Anthropic API, set APIKey in the config or the ANTHROPIC_API_KEY
// environment variable.
//
// For Vertex AI, set VertexProjectID and VertexLocation in the config or use
// GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables.
func NewModel(ctx context.Context, modelName anthropicsdk.Model, cfg *Config) (model.LLM, error) {
	if cfg == nil {
		cfg = &Config{}
	}

	variant := cfg.Variant
	if variant == "" {
		variant = GetVariant()
	}

	var client anthropicsdk.Client

	switch variant {
	case VariantVertexAI:
		projectID := cfg.VertexProjectID
		if projectID == "" {
			projectID = os.Getenv("GOOGLE_CLOUD_PROJECT")
		}
		if projectID == "" {
			return nil, fmt.Errorf("VertexProjectID is required for Vertex AI (set GOOGLE_CLOUD_PROJECT)")
		}

		location := cfg.VertexLocation
		if location == "" {
			location = os.Getenv("GOOGLE_CLOUD_LOCATION")
		}
		if location == "" {
			return nil, fmt.Errorf("VertexLocation is required for Vertex AI (set GOOGLE_CLOUD_LOCATION)")
		}

		var err error
		client, err = newVertexClient(ctx, cfg)
		if err != nil {
			return nil, err
		}
	default:
		client = newAPIClient(cfg)
	}

	maxTokens := cfg.DefaultMaxTokens
	if maxTokens == 0 {
		maxTokens = defaultMaxTokens
	}

	return &anthropicModel{
		client:           client,
		name:             modelName,
		variant:          variant,
		defaultMaxTokens: maxTokens,
		effort:           cfg.Effort,
		promptCaching:    cfg.PromptCaching,
	}, nil
}

// newAPIClient creates a client for the direct Anthropic API.
func newAPIClient(cfg *Config) anthropicsdk.Client {
	opts := []option.RequestOption{}

	apiKey := cfg.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("ANTHROPIC_API_KEY")
	}
	if apiKey != "" {
		opts = append(opts, option.WithAPIKey(apiKey))
	}

	if cfg.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(cfg.BaseURL))
	}

	return anthropicsdk.NewClient(opts...)
}

// newVertexClient creates a client for Anthropic via Vertex AI.
// Note: The caller must validate that projectID and region are set before calling this.
func newVertexClient(ctx context.Context, cfg *Config) (anthropicsdk.Client, error) {
	projectID := cfg.VertexProjectID
	if projectID == "" {
		projectID = os.Getenv("GOOGLE_CLOUD_PROJECT")
	}

	location := cfg.VertexLocation
	if location == "" {
		location = os.Getenv("GOOGLE_CLOUD_LOCATION")
	}

	// Load Application Default Credentials explicitly rather than via vertex.WithGoogleAuth, which
	// panics if credentials can't be resolved. Doing it here lets a credential failure — missing or
	// expired ADC, broken impersonation — surface as a returned error instead of crashing the caller.
	credentials, err := google.FindDefaultCredentials(ctx, cloudPlatformScope)
	if err != nil {
		return anthropicsdk.Client{}, fmt.Errorf("failed to load Google credentials for Vertex AI: %w", err)
	}

	return anthropicsdk.NewClient(
		vertex.WithCredentials(ctx, location, projectID, credentials),
	), nil
}

// Name returns the model name.
func (m *anthropicModel) Name() string {
	return string(m.name)
}

// GenerateContent calls the Anthropic model.
func (m *anthropicModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	m.maybeAppendUserContent(req)

	if stream {
		return m.generateStream(ctx, req)
	}

	return func(yield func(*model.LLMResponse, error) bool) {
		resp, err := m.generate(ctx, req)
		yield(resp, err)
	}
}

// generate calls the model synchronously.
func (m *anthropicModel) generate(ctx context.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
	params, toolKeyAliases, err := m.convertRequest(req)
	if err != nil {
		return nil, fmt.Errorf("failed to convert request: %w", err)
	}

	// Accumulate a streaming response rather than calling the non-streaming endpoint. The latter is
	// rejected ("streaming is required for operations that may take longer than 10 minutes") once
	// max_tokens is large — which our default is — so a non-streaming caller (e.g. structured-output
	// extraction) would otherwise fail. Streaming has no such limit and yields the same final message.
	stream := m.client.Messages.NewStreaming(ctx, params)
	message := anthropicsdk.Message{}
	for stream.Next() {
		if err := message.Accumulate(stream.Current()); err != nil {
			return nil, fmt.Errorf("failed to accumulate message: %w", err)
		}
	}
	if err := stream.Err(); err != nil {
		return nil, fmt.Errorf("failed to call model: %w", err)
	}

	resp, err := converters.MessageToLLMResponse(&message, toolKeyAliases)
	if err != nil {
		return nil, fmt.Errorf("failed to convert response: %w", err)
	}

	// A non-streaming response is the whole turn, so mark it complete — matching
	// the final response yielded by generateStream.
	resp.TurnComplete = true
	return resp, nil
}

// generateStream returns a stream of responses from the model.
func (m *anthropicModel) generateStream(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		params, toolKeyAliases, err := m.convertRequest(req)
		if err != nil {
			yield(nil, fmt.Errorf("failed to convert request: %w", err))
			return
		}

		stream := m.client.Messages.NewStreaming(ctx, params)
		message := anthropicsdk.Message{}

		for stream.Next() {
			event := stream.Current()

			// Accumulate the message
			if err := message.Accumulate(event); err != nil {
				yield(nil, fmt.Errorf("failed to accumulate message: %w", err))
				return
			}

			// Handle different event types for streaming
			switch ev := event.AsAny().(type) {
			case anthropicsdk.ContentBlockDeltaEvent:
				// Handle text deltas
				switch delta := ev.Delta.AsAny().(type) {
				case anthropicsdk.TextDelta:
					resp := converters.StreamDeltaToPartialResponse(delta.Text)
					if !yield(resp, nil) {
						return
					}
				case anthropicsdk.ThinkingDelta:
					resp := converters.StreamThinkingDeltaToPartialResponse(delta.Thinking)
					if !yield(resp, nil) {
						return
					}
				}
			}
		}

		if err := stream.Err(); err != nil {
			yield(nil, fmt.Errorf("stream error: %w", err))
			return
		}

		// Yield the final complete response
		finalResp, err := converters.MessageToLLMResponse(&message, toolKeyAliases)
		if err != nil {
			yield(nil, fmt.Errorf("failed to convert stream response: %w", err))
			return
		}
		finalResp.TurnComplete = true
		yield(finalResp, nil)
	}
}

// convertRequest converts an LLMRequest to Anthropic MessageNewParams.
func (m *anthropicModel) convertRequest(req *model.LLMRequest) (anthropicsdk.MessageNewParams, map[string]string, error) {
	messages, err := converters.ContentsToMessages(req.Contents)
	if err != nil {
		return anthropicsdk.MessageNewParams{}, nil, fmt.Errorf("failed to convert contents: %w", err)
	}

	// toolKeyAliases maps aliased top-level tool property keys back to their original names; it is
	// populated when tools are converted below and returned so the response parser can restore them.
	var toolKeyAliases map[string]string

	params := anthropicsdk.MessageNewParams{
		Model:     m.name,
		Messages:  messages,
		MaxTokens: int64(m.defaultMaxTokens),
	}

	if req.Config != nil {
		// System instruction
		if req.Config.SystemInstruction != nil {
			params.System = converters.SystemInstructionToSystem(req.Config.SystemInstruction)
		}

		// Sampling parameters (temperature/top_p/top_k) are intentionally not
		// forwarded: the latest Claude models reject them (HTTP 400) when used
		// with adaptive thinking, and depth is controlled via effort instead.
		if len(req.Config.StopSequences) > 0 {
			params.StopSequences = req.Config.StopSequences
		}
		if req.Config.MaxOutputTokens > 0 {
			params.MaxTokens = int64(req.Config.MaxOutputTokens)
		}

		// Tools
		if len(req.Config.Tools) > 0 {
			params.Tools, toolKeyAliases = converters.ToolsToAnthropicTools(req.Config.Tools)
		}

		// Tool choice from ToolConfig
		if req.Config.ToolConfig != nil {
			toolChoice, err := converters.ToolConfigToToolChoice(req.Config.ToolConfig)
			if err != nil {
				return anthropicsdk.MessageNewParams{}, nil, err
			}
			params.ToolChoice = toolChoice
		}

		// Structured output format. Anthropic structured outputs are GA on both
		// the direct API and Vertex AI (output_config.format with a json_schema,
		// no beta header), so the same path serves both variants. genai carries the
		// schema either as a structured ResponseSchema or a raw ResponseJsonSchema;
		// support both, mirroring the tool-parameter path. Anthropic resolves
		// $ref/$defs in the output schema (verified on Vertex), so a raw schema —
		// including a root $ref — is passed through as-is, only made strict below.
		//
		// Limitation: unlike tool input schemas, Anthropic's structured-output
		// format rejects JSON-schema validation keywords (minimum/maximum,
		// minLength/maxLength, minItems/maxItems, pattern), which SchemaToMap can
		// emit from a constrained genai.Schema. They are not stripped here, so a
		// ResponseSchema that sets any of them would be rejected; strip them on
		// this path if that need arises.
		var responseFormatSchema map[string]any
		switch {
		case req.Config.ResponseSchema != nil:
			responseFormatSchema = converters.SchemaToMap(req.Config.ResponseSchema)
		case req.Config.ResponseJsonSchema != nil:
			responseFormatSchema = converters.RawJSONSchemaToMap(req.Config.ResponseJsonSchema)
		}
		if responseFormatSchema != nil {
			enforceStrictObjectSchema(responseFormatSchema)
			params.OutputConfig.Format = anthropicsdk.JSONOutputFormatParam{Schema: responseFormatSchema}
		}
	}

	// Thinking config. We target only adaptive-capable models, so the converter
	// returns adaptive thinking (or off) — never a budget_tokens form, which the
	// latest models reject. When thinking is on, effort comes from the model's
	// configured Effort, falling back to the value derived from the request's
	// genai ThinkingLevel.
	var thinkingCfg *genai.ThinkingConfig
	if req.Config != nil {
		thinkingCfg = req.Config.ThinkingConfig
	}
	mapping := converters.ThinkingConfigToAnthropic(thinkingCfg)
	params.Thinking = mapping.Thinking
	if mapping.Thinking.OfAdaptive != nil {
		effort := m.effort
		if effort == "" {
			effort = mapping.Effort
		}
		params.OutputConfig.Effort = effort
	}

	// Anthropic rejects extended thinking (manual or adaptive) combined with
	// forced tool use (tool_choice.type = "tool" or "any"). When both are
	// requested, the API may either 400 or — worse — silently produce a
	// text/thinking response with no tool_use block, which looks to callers
	// like the model just refused to call the tool. The forced tool_choice
	// is the load-bearing semantic (the caller has pinned the response
	// shape), so drop the thinking parameter on this side of the wire.
	// Effort is meaningless without adaptive thinking, so clear it too.
	if converters.IsForcedToolUse(params.ToolChoice) {
		params.Thinking = anthropicsdk.ThinkingConfigParamUnion{}
		params.OutputConfig.Effort = ""
	}

	if m.promptCaching != nil {
		applyCacheBreakpoints(&params, m.promptCaching)
	}

	return params, toolKeyAliases, nil
}

// enforceStrictObjectSchema recursively sets additionalProperties:false on every
// object node of a JSON-schema map. Anthropic structured outputs reject an
// "object" schema that doesn't explicitly disallow additional properties
// ("output_config.format.schema: For 'object' type, 'additionalProperties' must
// be explicitly set to false"), and the genai→map conversion doesn't emit it.
func enforceStrictObjectSchema(node any) {
	switch n := node.(type) {
	case map[string]any:
		if t, ok := n["type"].(string); ok && t == "object" {
			if _, exists := n["additionalProperties"]; !exists {
				n["additionalProperties"] = false
			}
		}
		for _, v := range n {
			enforceStrictObjectSchema(v)
		}
	case []map[string]any:
		// SchemaToMap stores anyOf branches as []map[string]any, not []any.
		for _, v := range n {
			enforceStrictObjectSchema(v)
		}
	case []any:
		for _, v := range n {
			enforceStrictObjectSchema(v)
		}
	}
}

// maybeAppendUserContent ensures the conversation ends with a user message.
// Anthropic requires strictly alternating user/assistant turns.
func (m *anthropicModel) maybeAppendUserContent(req *model.LLMRequest) {
	if len(req.Contents) == 0 {
		req.Contents = append(req.Contents,
			genai.NewContentFromText("Handle the requests as specified in the System Instruction.", "user"))
		return
	}

	if last := req.Contents[len(req.Contents)-1]; last != nil && last.Role != "user" {
		req.Contents = append(req.Contents,
			genai.NewContentFromText("Continue processing previous requests as instructed.", "user"))
	}
}
