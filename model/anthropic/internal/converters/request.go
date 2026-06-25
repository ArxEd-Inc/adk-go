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

// Package converters provides conversion functions between genai types and Anthropic SDK types.
package converters

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/genai"
)

// ContentsToMessages converts genai Contents to Anthropic MessageParams.
// It handles role mapping and content part conversion.
func ContentsToMessages(contents []*genai.Content) ([]anthropic.MessageParam, error) {
	if len(contents) == 0 {
		return nil, nil
	}

	// One sanitizer per request so a tool_use ID and its later tool_result ID
	// are rewritten consistently (see toolUseIDSanitizer).
	sanitizer := newToolUseIDSanitizer()

	var messages []anthropic.MessageParam
	for _, content := range contents {
		if content == nil {
			continue
		}

		msg, err := contentToMessage(content, sanitizer)
		if err != nil {
			return nil, fmt.Errorf("failed to convert content: %w", err)
		}
		if msg != nil {
			messages = append(messages, *msg)
		}
	}

	// Merge consecutive messages with the same role (Anthropic requires alternating roles)
	messages = mergeConsecutiveMessages(messages)

	return messages, nil
}

// contentToMessage converts a single genai.Content to an Anthropic MessageParam.
func contentToMessage(content *genai.Content, sanitizer *toolUseIDSanitizer) (*anthropic.MessageParam, error) {
	if content == nil || len(content.Parts) == 0 {
		return nil, nil
	}

	// Check if this content contains tool results (FunctionResponse).
	// Anthropic requires tool results to be in user messages.
	hasFunctionResponse := false
	hasFunctionCall := false
	for _, part := range content.Parts {
		if part != nil {
			if part.FunctionResponse != nil {
				hasFunctionResponse = true
			}
			if part.FunctionCall != nil {
				hasFunctionCall = true
			}
		}
	}

	// Determine the role - tool results must be user, tool calls must be assistant
	var role anthropic.MessageParamRole
	if hasFunctionResponse {
		// Tool results MUST be in user messages per Anthropic API requirements
		role = anthropic.MessageParamRoleUser
	} else if hasFunctionCall {
		// Tool calls (from model) MUST be in assistant messages
		role = anthropic.MessageParamRoleAssistant
	} else {
		var err error
		role, err = mapRole(content.Role)
		if err != nil {
			return nil, err
		}
	}

	var blocks []anthropic.ContentBlockParamUnion
	for _, part := range content.Parts {
		if part == nil {
			continue
		}
		block, err := partToContentBlock(part, sanitizer)
		if err != nil {
			return nil, fmt.Errorf("failed to convert part: %w", err)
		}
		if block != nil {
			blocks = append(blocks, *block)
		}
	}

	if len(blocks) == 0 {
		return nil, nil
	}

	msg := anthropic.MessageParam{
		Role:    role,
		Content: blocks,
	}
	return &msg, nil
}

// mapRole maps genai role to Anthropic MessageParamRole.
func mapRole(role string) (anthropic.MessageParamRole, error) {
	switch strings.ToLower(role) {
	case "user":
		return anthropic.MessageParamRoleUser, nil
	case "model", "assistant":
		return anthropic.MessageParamRoleAssistant, nil
	default:
		return "", fmt.Errorf("unsupported role: %s", role)
	}
}

// partToContentBlock converts a genai Part to an Anthropic ContentBlockParamUnion.
func partToContentBlock(part *genai.Part, sanitizer *toolUseIDSanitizer) (*anthropic.ContentBlockParamUnion, error) {
	if part == nil {
		return nil, nil
	}

	// Thinking block. A thought carried back into history must be replayed as a
	// thinking block with its signature so Anthropic accepts it on the same
	// model — including thoughts whose text is empty (which happens under
	// display:"omitted"). Handle this before the text gate below, which would
	// otherwise drop empty-text thoughts and lose the signature, risking a 400
	// on the following tool turn.
	if part.Thought && len(part.ThoughtSignature) > 0 {
		block := anthropic.ContentBlockParamUnion{
			OfThinking: &anthropic.ThinkingBlockParam{
				Thinking:  part.Text,
				Signature: base64.StdEncoding.EncodeToString(part.ThoughtSignature),
			},
		}
		return &block, nil
	}

	// A thought part with no signature can't be replayed as a thinking block (Anthropic requires the
	// signature) — e.g. a redacted-thinking marker or a persisted streaming partial. Drop it instead
	// of letting it fall through to the text gate below, which would re-enter the reasoning into
	// history as assistant text.
	if part.Thought {
		return nil, nil
	}

	// Text content
	if part.Text != "" {
		block := anthropic.NewTextBlock(part.Text)
		return &block, nil
	}

	// Inline binary data (images, PDFs)
	if part.InlineData != nil {
		return inlineDataToBlock(part.InlineData)
	}

	// File data (URI-based)
	if part.FileData != nil {
		return fileDataToBlock(part.FileData)
	}

	// Function response (tool result)
	if part.FunctionResponse != nil {
		return functionResponseToBlock(part.FunctionResponse, sanitizer)
	}

	// Function call - these appear in model responses replayed as history
	if part.FunctionCall != nil {
		return functionCallToBlock(part.FunctionCall, sanitizer)
	}

	// Executable code and CodeExecutionResult are Gemini-specific features
	// that don't have direct Anthropic equivalents
	if part.ExecutableCode != nil || part.CodeExecutionResult != nil {
		return nil, fmt.Errorf("ExecutableCode and CodeExecutionResult are not supported by Anthropic")
	}

	return nil, nil
}

// inlineDataToBlock converts inline binary data to an Anthropic content block.
func inlineDataToBlock(blob *genai.Blob) (*anthropic.ContentBlockParamUnion, error) {
	if blob == nil {
		return nil, nil
	}

	mimeType := strings.ToLower(blob.MIMEType)

	// Handle images
	if strings.HasPrefix(mimeType, "image/") {
		mediaType, err := mapImageMediaType(mimeType)
		if err != nil {
			return nil, err
		}
		block := anthropic.ContentBlockParamUnion{
			OfImage: &anthropic.ImageBlockParam{
				Source: anthropic.ImageBlockParamSourceUnion{
					OfBase64: &anthropic.Base64ImageSourceParam{
						Data:      base64.StdEncoding.EncodeToString(blob.Data),
						MediaType: mediaType,
					},
				},
			},
		}
		return &block, nil
	}

	// Handle PDFs (beta feature)
	if mimeType == "application/pdf" {
		block := anthropic.ContentBlockParamUnion{
			OfDocument: &anthropic.DocumentBlockParam{
				Source: anthropic.DocumentBlockParamSourceUnion{
					OfBase64: &anthropic.Base64PDFSourceParam{
						Data: base64.StdEncoding.EncodeToString(blob.Data),
					},
				},
			},
		}
		return &block, nil
	}

	return nil, fmt.Errorf("unsupported MIME type for inline data: %s", mimeType)
}

// mapImageMediaType maps MIME types to Anthropic Base64ImageSourceMediaType.
func mapImageMediaType(mimeType string) (anthropic.Base64ImageSourceMediaType, error) {
	switch mimeType {
	case "image/jpeg":
		return anthropic.Base64ImageSourceMediaTypeImageJPEG, nil
	case "image/png":
		return anthropic.Base64ImageSourceMediaTypeImagePNG, nil
	case "image/gif":
		return anthropic.Base64ImageSourceMediaTypeImageGIF, nil
	case "image/webp":
		return anthropic.Base64ImageSourceMediaTypeImageWebP, nil
	default:
		return "", fmt.Errorf("unsupported image media type: %s", mimeType)
	}
}

// fileDataToBlock converts URI-based file data to an Anthropic content block.
func fileDataToBlock(fileData *genai.FileData) (*anthropic.ContentBlockParamUnion, error) {
	if fileData == nil {
		return nil, nil
	}

	mimeType := strings.ToLower(fileData.MIMEType)

	// Handle images via URL
	if strings.HasPrefix(mimeType, "image/") {
		block := anthropic.ContentBlockParamUnion{
			OfImage: &anthropic.ImageBlockParam{
				Source: anthropic.ImageBlockParamSourceUnion{
					OfURL: &anthropic.URLImageSourceParam{
						URL: fileData.FileURI,
					},
				},
			},
		}
		return &block, nil
	}

	// Handle PDFs via URL (beta feature)
	if mimeType == "application/pdf" {
		block := anthropic.ContentBlockParamUnion{
			OfDocument: &anthropic.DocumentBlockParam{
				Source: anthropic.DocumentBlockParamSourceUnion{
					OfURL: &anthropic.URLPDFSourceParam{
						URL: fileData.FileURI,
					},
				},
			},
		}
		return &block, nil
	}

	return nil, fmt.Errorf("unsupported MIME type for file data: %s", mimeType)
}

// functionResponseToBlock converts a FunctionResponse to an Anthropic tool result block.
func functionResponseToBlock(resp *genai.FunctionResponse, sanitizer *toolUseIDSanitizer) (*anthropic.ContentBlockParamUnion, error) {
	if resp == nil {
		return nil, nil
	}

	// Convert the response to JSON string
	var content string
	if resp.Response != nil {
		jsonBytes, err := json.Marshal(resp.Response)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal function response: %w", err)
		}
		content = string(jsonBytes)
	}

	// Sanitize the tool-use ID to Anthropic's required shape, consistently with
	// the matching tool_use block so the result still correlates.
	block := anthropic.NewToolResultBlock(sanitizer.sanitize(resp.ID), content, false)
	return &block, nil
}

// functionCallToBlock converts a FunctionCall to an Anthropic tool use block.
// This is used when passing model responses back (e.g., in conversation history).
func functionCallToBlock(call *genai.FunctionCall, sanitizer *toolUseIDSanitizer) (*anthropic.ContentBlockParamUnion, error) {
	if call == nil {
		return nil, nil
	}

	// Mirror the top-level argument keys to the aliases used in the tool schema, so a replayed
	// tool_use matches the (aliased) input_schema Anthropic sees this turn. Anthropic also requires
	// input to be a dictionary, so always provide a valid map.
	var input any = aliasArgKeys(call.Args)
	if len(call.Args) == 0 {
		input = map[string]any{}
	}

	block := anthropic.NewToolUseBlock(sanitizer.sanitize(call.ID), input, call.Name)
	return &block, nil
}

// aliasArgKeys returns args with each top-level key replaced by aliasToolKey(key); keys that don't
// need aliasing pass through unchanged. Returns nil for empty input.
func aliasArgKeys(args map[string]any) map[string]any {
	if len(args) == 0 {
		return nil
	}
	aliased := make(map[string]any, len(args))
	for key, value := range args {
		aliased[aliasToolKey(key)] = value
	}
	return aliased
}

// SystemInstructionToSystem converts a genai SystemInstruction to Anthropic system text blocks.
func SystemInstructionToSystem(instruction *genai.Content) []anthropic.TextBlockParam {
	if instruction == nil || len(instruction.Parts) == 0 {
		return nil
	}

	var blocks []anthropic.TextBlockParam
	for _, part := range instruction.Parts {
		if part != nil && part.Text != "" {
			blocks = append(blocks, anthropic.TextBlockParam{
				Text: part.Text,
			})
		}
	}
	return blocks
}

// mergeConsecutiveMessages merges consecutive messages with the same role.
// Anthropic requires strictly alternating user/assistant messages.
func mergeConsecutiveMessages(messages []anthropic.MessageParam) []anthropic.MessageParam {
	if len(messages) <= 1 {
		return messages
	}

	var merged []anthropic.MessageParam
	for i, msg := range messages {
		if i == 0 {
			merged = append(merged, msg)
			continue
		}

		last := &merged[len(merged)-1]
		if last.Role == msg.Role {
			// Merge content blocks
			last.Content = append(last.Content, msg.Content...)
		} else {
			merged = append(merged, msg)
		}
	}
	return merged
}

// toolUseIDPattern is Anthropic's accepted shape for tool_use / tool_result IDs.
var toolUseIDPattern = regexp.MustCompile(`^[a-zA-Z0-9_-]+$`)

// toolUseIDSanitizer rewrites tool-call IDs that don't satisfy Anthropic's
// ^[a-zA-Z0-9_-]+$ requirement to stable fallback IDs, consistently within a
// single request so a tool_use ID and the tool_result that references it still
// map to the same value. genai/Gemini-style IDs (and IDs carried over from
// other providers' history) can contain characters Anthropic rejects, which
// would 400 the turn after a tool call. Mirrors Python ADK's _ToolUseIdSanitizer.
type toolUseIDSanitizer struct {
	mapping map[string]string
	next    int
}

// newToolUseIDSanitizer returns a sanitizer with an empty mapping. Create one
// per request (per ContentsToMessages call).
func newToolUseIDSanitizer() *toolUseIDSanitizer {
	return &toolUseIDSanitizer{mapping: map[string]string{}}
}

// sanitize returns id unchanged when it already satisfies Anthropic's shape,
// otherwise a stable toolu_fallback_N substitute (the same substitute for the
// same input within this sanitizer).
func (s *toolUseIDSanitizer) sanitize(id string) string {
	if id != "" && toolUseIDPattern.MatchString(id) {
		return id
	}
	if mapped, ok := s.mapping[id]; ok {
		return mapped
	}
	mapped := fmt.Sprintf("toolu_fallback_%d", s.next)
	s.next++
	s.mapping[id] = mapped
	return mapped
}

// ThinkingMapping bundles the Anthropic thinking parameter and the optional
// effort hint derived from a genai.ThinkingConfig's ThinkingLevel. Effort is
// empty unless thinking is on and a Low/Medium/High level was provided; the
// caller may override it with a per-model effort.
type ThinkingMapping struct {
	Thinking anthropic.ThinkingConfigParamUnion
	Effort   anthropic.OutputConfigEffort
}

// ThinkingConfigToAnthropic maps a genai.ThinkingConfig to Anthropic's thinking
// parameter plus an optional effort hint. This package targets only
// adaptive-capable Claude models (Opus 4.8 / Sonnet 4.6 and newer), so thinking
// is either adaptive or off — there is no legacy budget_tokens path (which those
// models reject with a 400). Depth is controlled by OutputConfig.Effort, which
// the caller sets per model; the level-derived Effort here is only a fallback.
//
// Mapping:
//   - nil cfg                                 → off (omit thinking)
//   - ThinkingBudget explicitly 0             → off
//   - ThinkingLevel == Minimal                → off
//   - anything else                           → adaptive (+ effort from Low/Medium/High)
//
// IncludeThoughts is ignored: in genai it governs whether thought summaries are
// returned, not whether the model thinks, and Anthropic returns thinking blocks
// whenever thinking is on regardless.
func ThinkingConfigToAnthropic(cfg *genai.ThinkingConfig) ThinkingMapping {
	if cfg == nil {
		return ThinkingMapping{}
	}
	if cfg.ThinkingBudget != nil && *cfg.ThinkingBudget == 0 {
		return ThinkingMapping{}
	}
	if cfg.ThinkingLevel == genai.ThinkingLevelMinimal {
		return ThinkingMapping{}
	}
	return ThinkingMapping{
		Thinking: adaptiveThinking(),
		Effort:   levelToEffort(cfg.ThinkingLevel),
	}
}

// adaptiveThinking returns the parameter union for Anthropic adaptive mode.
func adaptiveThinking() anthropic.ThinkingConfigParamUnion {
	return anthropic.ThinkingConfigParamUnion{
		OfAdaptive: &anthropic.ThinkingConfigAdaptiveParam{},
	}
}

// levelToEffort maps a genai ThinkingLevel to the matching Anthropic
// OutputConfigEffort. Returns the empty value for levels that don't map
// (Unspecified, Minimal).
func levelToEffort(level genai.ThinkingLevel) anthropic.OutputConfigEffort {
	switch level {
	case genai.ThinkingLevelLow:
		return anthropic.OutputConfigEffortLow
	case genai.ThinkingLevelMedium:
		return anthropic.OutputConfigEffortMedium
	case genai.ThinkingLevelHigh:
		return anthropic.OutputConfigEffortHigh
	}
	return ""
}
