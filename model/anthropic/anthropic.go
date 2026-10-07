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
	"errors"
	"fmt"
	"iter"
	"os"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/vertex"
	"golang.org/x/oauth2/google"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/anthropic/internal/converters"
)

const defaultMaxTokens = 16384

// ErrRequestConversion marks a failure to convert a [model.LLMRequest] into Anthropic's request format, raised
// before any API call is made. Conversion is deterministic — the same request fails the same way on every attempt —
// so callers should treat errors wrapping it as permanent rather than retrying.
var ErrRequestConversion = errors.New("failed to convert request")

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
	thinkingMode     ThinkingMode
	promptCaching    *PromptCachingConfig

	toolReferencesResponseKey string

	// files resolves inline documents to Files API references; nil sends them
	// inline.
	files *filesResolver
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

	if cfg.Files != nil && variant != VariantAnthropicAPI {
		return nil, fmt.Errorf("Files requires the %s variant, not %s", VariantAnthropicAPI, variant)
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

	var files *filesResolver
	if cfg.Files != nil {
		apiKey := cfg.APIKey
		if apiKey == "" {
			apiKey = os.Getenv("ANTHROPIC_API_KEY")
		}
		var err error
		files, err = newFilesResolver(*cfg.Files, &client, apiKey, cfg.BaseURL)
		if err != nil {
			return nil, err
		}
	}

	return &anthropicModel{
		client:           client,
		name:             modelName,
		variant:          variant,
		defaultMaxTokens: maxTokens,
		effort:           cfg.Effort,
		thinkingMode:     cfg.ThinkingMode,
		promptCaching:    cfg.PromptCaching,

		toolReferencesResponseKey: cfg.ToolReferencesResponseKey,
		files:                     files,
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
	for attempt := 0; ; attempt++ {
		fileIDs, err := m.resolveFiles(ctx, req)
		if err != nil {
			return nil, err
		}
		resp, received, err := m.generateOnce(ctx, req, fileIDs)
		if err != nil && attempt == 0 && m.shouldReuploadFiles(err, received, fileIDs) {
			m.files.forget(fileIDs)
			continue
		}
		return resp, err
	}
}

// generateOnce is one attempt of generate with the given file references. It
// also reports whether any stream event arrived, which shouldReuploadFiles
// needs to tell a rejected request from a stream that failed partway.
func (m *anthropicModel) generateOnce(ctx context.Context, req *model.LLMRequest, fileIDs map[*genai.Blob]string) (*model.LLMResponse, bool, error) {
	params, toolKeyAliases, err := m.convertRequest(req, fileIDs)
	if err != nil {
		return nil, false, fmt.Errorf("%w: %w", ErrRequestConversion, err)
	}

	// Accumulate a streaming response rather than calling the non-streaming endpoint. The latter is
	// rejected ("streaming is required for operations that may take longer than 10 minutes") once
	// max_tokens is large — which our default is — so a non-streaming caller (e.g. structured-output
	// extraction) would otherwise fail. Streaming has no such limit and yields the same final message.
	stream := m.client.Messages.NewStreaming(ctx, params)
	message := anthropicsdk.Message{}
	messageStartCallback, _ := MessageStartCallbackFromContext(ctx)
	received := false
	for stream.Next() {
		received = true
		event := stream.Current()
		repairAccumulatedToolInput(&message, event)
		if err := toolInputErrorAtStop(&message, event); err != nil {
			return nil, received, fmt.Errorf("failed to accumulate message: %w", err)
		}
		if err := message.Accumulate(event); err != nil {
			return nil, received, fmt.Errorf("failed to accumulate message: %w", err)
		}
		switch ev := event.AsAny().(type) {
		case anthropicsdk.MessageStartEvent:
			if messageStartCallback != nil {
				messageStartCallback(MessageStartUsage{
					InputTokens:              ev.Message.Usage.InputTokens,
					CacheReadInputTokens:     ev.Message.Usage.CacheReadInputTokens,
					CacheCreationInputTokens: ev.Message.Usage.CacheCreationInputTokens,
				})
			}
		}
	}
	if err := stream.Err(); err != nil {
		return nil, received, fmt.Errorf("failed to call model: %w", err)
	}

	resp, err := converters.MessageToLLMResponse(&message, toolKeyAliases)
	if err != nil {
		return nil, received, fmt.Errorf("failed to convert response: %w", err)
	}

	// A non-streaming response is the whole turn, so mark it complete — matching
	// the final response yielded by generateStream.
	resp.TurnComplete = true
	return resp, received, nil
}

// resolveFiles returns the request's Files API references (see FilesConfig),
// or nil when the model sends documents inline.
func (m *anthropicModel) resolveFiles(ctx context.Context, req *model.LLMRequest) (map[*genai.Blob]string, error) {
	if m.files == nil {
		return nil, nil
	}
	return m.files.resolve(ctx, req)
}

// shouldReuploadFiles reports whether a failed attempt that referenced the
// given files should be retried once with fresh uploads: the API rejected the
// request outright (no stream event arrived) as referring to something that
// does not exist, which a file deleted before its expiry causes. The retry
// costs a re-upload when the rejection had another cause.
func (m *anthropicModel) shouldReuploadFiles(err error, received bool, fileIDs map[*genai.Blob]string) bool {
	return m.files != nil && len(fileIDs) > 0 && !received && isMissingFileError(err)
}

// generateStream returns a stream of responses from the model.
func (m *anthropicModel) generateStream(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		for attempt := 0; ; attempt++ {
			fileIDs, err := m.resolveFiles(ctx, req)
			if err != nil {
				yield(nil, err)
				return
			}
			received, err := m.generateStreamOnce(ctx, req, fileIDs, yield)
			if err == nil {
				return
			}
			if attempt == 0 && m.shouldReuploadFiles(err, received, fileIDs) {
				m.files.forget(fileIDs)
				continue
			}
			yield(nil, err)
			return
		}
	}
}

// generateStreamOnce is one attempt of generateStream with the given file
// references. It yields each partial response and the final one, and returns
// rather than yields a failure, with whether any stream event arrived before
// it, so the caller can retry a rejected request (see shouldReuploadFiles). A
// consumer that stops early ends the attempt with no error.
func (m *anthropicModel) generateStreamOnce(
	ctx context.Context,
	req *model.LLMRequest,
	fileIDs map[*genai.Blob]string,
	yield func(*model.LLMResponse, error) bool,
) (bool, error) {
	params, toolKeyAliases, err := m.convertRequest(req, fileIDs)
	if err != nil {
		return false, fmt.Errorf("%w: %w", ErrRequestConversion, err)
	}

	stream := m.client.Messages.NewStreaming(ctx, params)
	message := anthropicsdk.Message{}
	messageStartCallback, _ := MessageStartCallbackFromContext(ctx)
	received := false

	for stream.Next() {
		received = true
		event := stream.Current()

		// Accumulate the message
		repairAccumulatedToolInput(&message, event)
		if err := toolInputErrorAtStop(&message, event); err != nil {
			return received, fmt.Errorf("failed to accumulate message: %w", err)
		}
		if err := message.Accumulate(event); err != nil {
			return received, fmt.Errorf("failed to accumulate message: %w", err)
		}

		// Handle different event types for streaming
		switch ev := event.AsAny().(type) {
		case anthropicsdk.MessageStartEvent:
			if messageStartCallback != nil {
				messageStartCallback(MessageStartUsage{
					InputTokens:              ev.Message.Usage.InputTokens,
					CacheReadInputTokens:     ev.Message.Usage.CacheReadInputTokens,
					CacheCreationInputTokens: ev.Message.Usage.CacheCreationInputTokens,
				})
			}
		case anthropicsdk.ContentBlockDeltaEvent:
			// Handle text deltas
			switch delta := ev.Delta.AsAny().(type) {
			case anthropicsdk.TextDelta:
				resp := converters.StreamDeltaToPartialResponse(delta.Text)
				if !yield(resp, nil) {
					return received, nil
				}
			case anthropicsdk.ThinkingDelta:
				resp := converters.StreamThinkingDeltaToPartialResponse(delta.Thinking)
				if !yield(resp, nil) {
					return received, nil
				}
			}
		}
	}

	if err := stream.Err(); err != nil {
		return received, fmt.Errorf("stream error: %w", err)
	}

	// Yield the final complete response
	finalResp, err := converters.MessageToLLMResponse(&message, toolKeyAliases)
	if err != nil {
		return received, fmt.Errorf("failed to convert stream response: %w", err)
	}
	finalResp.TurnComplete = true
	yield(finalResp, nil)
	return received, nil
}

// convertRequest converts an LLMRequest to Anthropic MessageNewParams, sending
// each inline document with an entry in fileIDs as a reference to that file.
func (m *anthropicModel) convertRequest(req *model.LLMRequest, fileIDs map[*genai.Blob]string) (anthropicsdk.MessageNewParams, map[string]string, error) {
	// Tools convert first: the contents' tool references may name only the tools this request defers.
	// toolKeyAliases maps aliased top-level tool property keys back to their original names; it is
	// returned so the response parser can restore them.
	var tools []anthropicsdk.ToolUnionParam
	var toolKeyAliases map[string]string
	if req.Config != nil && len(req.Config.Tools) > 0 {
		tools, toolKeyAliases = converters.ToolsToAnthropicTools(req.Config.Tools, func(name string) bool {
			return IsToolLoadingDeferred(req, name)
		})
	}

	messages, markedBlockOrdinals, err := converters.ContentsToMessagesWithMarkedBlocks(
		req.Contents,
		converters.ContentsOptions{
			ToolReferencesResponseKey: m.toolReferencesResponseKey,
			DeferredToolNames:         converters.DeferredToolNames(tools),
			FileIDByBlob:              fileIDs,
		},
	)
	if err != nil {
		return anthropicsdk.MessageNewParams{}, nil, fmt.Errorf("failed to convert contents: %w", err)
	}

	params := anthropicsdk.MessageNewParams{
		Model:     m.name,
		Messages:  messages,
		MaxTokens: int64(m.defaultMaxTokens),
		Tools:     tools,
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

	// Thinking config. The converter emits adaptive thinking for adaptive-capable
	// models (Opus 4.8 / Sonnet 4.6 and newer) and a budget_tokens form for models
	// that reject adaptive thinking and effort (e.g. Haiku 4.5), selected by this
	// model's ThinkingMode. When adaptive thinking is on, effort comes from the
	// model's configured Effort, falling back to the value derived from the
	// request's genai ThinkingLevel; budget mode ignores effort.
	var thinkingCfg *genai.ThinkingConfig
	if req.Config != nil {
		thinkingCfg = req.Config.ThinkingConfig
	}
	mapping := converters.ThinkingConfigToAnthropic(thinkingCfg, m.thinkingMode == ThinkingModeBudget)
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
		applyCacheBreakpoints(&params, m.promptCaching, markedBlockOrdinals)
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
