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
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"regexp"
	"slices"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/genai"
)

// PartCacheBreakpointMetadataKey is the genai.Part.PartMetadata key that marks a
// part as a prompt-cache breakpoint candidate: a true value under it asks the
// model to place a cache_control marker on the content block the part converts
// to (see the PromptCachingConfig.MarkedPart breakpoint). Callers set it with
// MarkCacheBreakpoint rather than by hand. The metadata rides only on the
// in-memory part — it is not part of the wire request, and session backends
// need not persist it.
const PartCacheBreakpointMetadataKey = "anthropic.cacheBreakpoint"

// IsCacheBreakpointMarked reports whether the given part carries a true
// PartCacheBreakpointMetadataKey; nil-safe.
func IsCacheBreakpointMarked(part *genai.Part) bool {
	if part == nil || part.PartMetadata == nil {
		return false
	}
	marked, ok := part.PartMetadata[PartCacheBreakpointMetadataKey].(bool)
	return ok && marked
}

// ContentsOptions configures ContentsToMessagesWithMarkedBlocks. The zero value
// converts every function response to a tool_result of its JSON.
type ContentsOptions struct {
	// ToolReferencesResponseKey names the function-response key whose value, a
	// list of tool names, the converter sends as tool_reference blocks rather
	// than as JSON (see functionResponseToBlocks). Empty disables references.
	ToolReferencesResponseKey string

	// DeferredToolNames are the tools this request sends with defer_loading,
	// the only tools a tool_reference block may name: Anthropic rejects a
	// reference to a tool the request does not define, and documents
	// references only for deferred ones.
	DeferredToolNames map[string]struct{}

	// FileIDByBlob maps inline PDF and image data to the Files API file
	// holding the same bytes, which the converter references instead of
	// sending the data as base64. Keyed by the blob itself, so only the
	// caller, never the conversation's content, can name a file. Inline data
	// without an entry is sent as base64.
	FileIDByBlob map[*genai.Blob]string

	// latestReferenceByToolName maps each tool name some function response
	// lists under ToolReferencesResponseKey to the last such response in the
	// contents, filled by ContentsToMessagesWithMarkedBlocks: a tool referenced
	// by several responses is expanded only at the latest of them.
	latestReferenceByToolName map[string]*genai.FunctionResponse
}

// ContentsToMessages converts genai Contents to Anthropic MessageParams.
// It handles role mapping and content part conversion.
func ContentsToMessages(contents []*genai.Content) ([]anthropic.MessageParam, error) {
	messages, _, err := ContentsToMessagesWithMarkedBlocks(contents, ContentsOptions{})
	return messages, err
}

// ContentsToMessagesWithMarkedBlocks is ContentsToMessages that also reports,
// in ascending order, the flat block ordinals of the parts marked with
// PartCacheBreakpointMetadataKey — each ordinal indexing the concatenation of
// every returned message's content blocks. Ordinals rather than
// (message, block) pairs because the merge of same-role messages below moves
// blocks between messages; the conversion is order-preserving and drops
// nothing but parts that yield no block, and a merge only concatenates, so a
// block's flat ordinal is fixed once its part is converted, save for the
// tool-results-first reordering, which remaps the ordinals it moves. The
// ordinals are nil when no part is marked.
func ContentsToMessagesWithMarkedBlocks(
	contents []*genai.Content,
	opts ContentsOptions,
) ([]anthropic.MessageParam, []int, error) {
	if len(contents) == 0 {
		return nil, nil, nil
	}

	// One sanitizer per request so a tool_use ID and its later tool_result ID
	// are rewritten consistently (see toolUseIDSanitizer).
	sanitizer := newToolUseIDSanitizer()
	opts.latestReferenceByToolName = latestReferenceByToolName(contents, opts.ToolReferencesResponseKey)

	var messages []anthropic.MessageParam
	var markedBlockOrdinals []int
	var blockCount int
	for _, content := range contents {
		if content == nil {
			continue
		}

		msg, markedBlockIndexes, err := contentToMessage(content, sanitizer, opts)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to convert content: %w", err)
		}
		if msg != nil {
			messages = append(messages, *msg)
			for _, blockIndex := range markedBlockIndexes {
				markedBlockOrdinals = append(markedBlockOrdinals, blockCount+blockIndex)
			}
			blockCount += len(msg.Content)
		}
	}

	// Merge consecutive messages with the same role (Anthropic requires alternating roles)
	messages = mergeConsecutiveMessages(messages)
	markedBlockOrdinals = moveToolResultsFirst(messages, markedBlockOrdinals)

	return messages, markedBlockOrdinals, nil
}

// latestReferenceByToolName returns, for each tool name a function response in
// the given contents lists under the given key, the last response that lists
// it; nil when key is empty.
func latestReferenceByToolName(contents []*genai.Content, key string) map[string]*genai.FunctionResponse {
	if key == "" {
		return nil
	}
	latest := map[string]*genai.FunctionResponse{}
	for _, content := range contents {
		if content == nil {
			continue
		}
		for _, part := range content.Parts {
			if part == nil || part.FunctionResponse == nil {
				continue
			}
			names, ok := toolReferenceNames(part.FunctionResponse.Response, key)
			if !ok {
				continue
			}
			for _, name := range names {
				latest[name] = part.FunctionResponse
			}
		}
	}
	return latest
}

// moveToolResultsFirst reorders each user message holding a tool_result block
// so that every tool_result block precedes every other block, keeping the
// relative order within both groups, and returns markedBlockOrdinals remapped
// to the new positions, ascending. Anthropic rejects a user message with any
// block before one of its tool results, and two sources produce that order: a
// function response carrying tool references converts to a tool_result
// followed by a text block, and the merge above concatenates one content's
// blocks after the previous content's, so a text block can land before the next
// content's tool results.
func moveToolResultsFirst(messages []anthropic.MessageParam, markedBlockOrdinals []int) []int {
	remapped := slices.Clone(markedBlockOrdinals)
	offset := 0
	for i := range messages {
		content := messages[i].Content
		if messages[i].Role == anthropic.MessageParamRoleUser && toolResultsNotFirst(content) {
			order := make([]int, 0, len(content))
			for index, block := range content {
				if block.OfToolResult != nil {
					order = append(order, index)
				}
			}
			for index, block := range content {
				if block.OfToolResult == nil {
					order = append(order, index)
				}
			}
			newPositions := make([]int, len(content))
			reordered := make([]anthropic.ContentBlockParamUnion, len(content))
			for newPosition, oldPosition := range order {
				reordered[newPosition] = content[oldPosition]
				newPositions[oldPosition] = newPosition
			}
			messages[i].Content = reordered
			for j, ordinal := range remapped {
				if ordinal >= offset && ordinal < offset+len(content) {
					remapped[j] = offset + newPositions[ordinal-offset]
				}
			}
		}
		offset += len(content)
	}
	slices.Sort(remapped)
	return remapped
}

// toolResultsNotFirst reports whether any tool_result block follows a block
// that is not one.
func toolResultsNotFirst(content []anthropic.ContentBlockParamUnion) bool {
	sawOther := false
	for _, block := range content {
		if block.OfToolResult == nil {
			sawOther = true
		} else if sawOther {
			return true
		}
	}
	return false
}

// contentToMessage converts a single genai.Content to an Anthropic MessageParam,
// also returning the indexes into the message's content blocks of the parts
// marked with PartCacheBreakpointMetadataKey (nil when none is marked). A marked
// part that converts to more than one block marks the last of them.
func contentToMessage(
	content *genai.Content,
	sanitizer *toolUseIDSanitizer,
	opts ContentsOptions,
) (*anthropic.MessageParam, []int, error) {
	if content == nil || len(content.Parts) == 0 {
		return nil, nil, nil
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
			return nil, nil, err
		}
	}

	var blocks []anthropic.ContentBlockParamUnion
	var markedBlockIndexes []int
	for _, part := range content.Parts {
		if part == nil {
			continue
		}
		partBlocks, err := partToContentBlocks(part, sanitizer, opts)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to convert part: %w", err)
		}
		if len(partBlocks) > 0 {
			blocks = append(blocks, partBlocks...)
			if IsCacheBreakpointMarked(part) {
				markedBlockIndexes = append(markedBlockIndexes, len(blocks)-1)
			}
		}
	}

	if len(blocks) == 0 {
		return nil, nil, nil
	}

	msg := anthropic.MessageParam{
		Role:    role,
		Content: blocks,
	}
	return &msg, markedBlockIndexes, nil
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

// partToContentBlocks converts a genai Part to Anthropic content blocks: one
// block for every kind of part, save for a function response carrying tool
// references, which converts to two (see functionResponseToBlocks), and nil for
// a part that yields none.
func partToContentBlocks(
	part *genai.Part,
	sanitizer *toolUseIDSanitizer,
	opts ContentsOptions,
) ([]anthropic.ContentBlockParamUnion, error) {
	if part == nil {
		return nil, nil
	}

	// Redacted thinking carried back: response.go stashed the block's encrypted Data in ThoughtSignature
	// behind redactedThinkingMarker. Replay it faithfully as a redacted_thinking block so Anthropic accepts
	// it before a following tool_use. Checked before the normal-signature branch, since a carried redacted
	// block also has a non-empty ThoughtSignature.
	if part.Thought {
		if data, ok := decodeRedactedThinking(part.ThoughtSignature); ok {
			return []anthropic.ContentBlockParamUnion{anthropic.NewRedactedThinkingBlock(data)}, nil
		}
	}

	// Thinking block. A thought carried back into history must be replayed as a
	// thinking block with its signature so Anthropic accepts it on the same
	// model — including thoughts whose text is empty (which happens under
	// display:"omitted"). Handle this before the text gate below, which would
	// otherwise drop empty-text thoughts and lose the signature, risking a 400
	// on the following tool turn.
	if part.Thought && len(part.ThoughtSignature) > 0 {
		return []anthropic.ContentBlockParamUnion{{
			OfThinking: &anthropic.ThinkingBlockParam{
				Thinking:  part.Text,
				Signature: base64.StdEncoding.EncodeToString(part.ThoughtSignature),
			},
		}}, nil
	}

	// A thought part that is neither a signed thinking block nor recognized redacted thinking can't be
	// faithfully replayed (Anthropic requires a signature or the redacted Data). This shouldn't occur:
	// MessageToLLMResponse always sets a signature or the redacted marker, streamed partials aren't
	// persisted, and the session/contents layers preserve both — so reaching here means corrupted or
	// foreign history. Error rather than silently dropping reasoning, consistent with how this converter
	// treats other unconvertible content.
	if part.Thought {
		return nil, fmt.Errorf("thought part cannot be replayed: no signature and no redacted-thinking data")
	}

	// Text content
	if part.Text != "" {
		return []anthropic.ContentBlockParamUnion{anthropic.NewTextBlock(part.Text)}, nil
	}

	// Inline binary data (images, PDFs)
	if part.InlineData != nil {
		return singleBlock(inlineDataToBlock(part.InlineData, opts.FileIDByBlob[part.InlineData]))
	}

	// File data (URI-based)
	if part.FileData != nil {
		return singleBlock(fileDataToBlock(part.FileData))
	}

	// Function response (tool result)
	if part.FunctionResponse != nil {
		return functionResponseToBlocks(part.FunctionResponse, sanitizer, opts)
	}

	// Function call - these appear in model responses replayed as history
	if part.FunctionCall != nil {
		return singleBlock(functionCallToBlock(part.FunctionCall, sanitizer))
	}

	// Executable code and CodeExecutionResult are Gemini-specific features
	// that don't have direct Anthropic equivalents
	if part.ExecutableCode != nil || part.CodeExecutionResult != nil {
		return nil, fmt.Errorf("ExecutableCode and CodeExecutionResult are not supported by Anthropic")
	}

	return nil, nil
}

// singleBlock adapts a single-block converter's result to partToContentBlocks'.
func singleBlock(block *anthropic.ContentBlockParamUnion, err error) ([]anthropic.ContentBlockParamUnion, error) {
	if err != nil || block == nil {
		return nil, err
	}
	return []anthropic.ContentBlockParamUnion{*block}, nil
}

// inlineDataToBlock converts inline binary data to an Anthropic content block:
// a reference to the given file when fileID is non-empty, otherwise the data
// as base64.
func inlineDataToBlock(blob *genai.Blob, fileID string) (*anthropic.ContentBlockParamUnion, error) {
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
		if fileID != "" {
			block := anthropic.NewImageBlock(anthropic.FileImageSourceParam{FileID: fileID})
			return &block, nil
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
		if fileID != "" {
			block := anthropic.NewDocumentBlock(anthropic.FileDocumentSourceParam{FileID: fileID})
			return &block, nil
		}
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

// functionResponseToBlocks converts a FunctionResponse to Anthropic content
// blocks. When opts names a references key and the response holds a list of
// tool names under it, the names of tools the request defers become the
// tool_result's only content, as tool_reference blocks, which Anthropic expands
// in place into those tools' definitions so the model can call them; the rest
// of the response follows as a text block labeled with the result's ID, since a
// tool_result cannot mix tool_reference blocks with other content. A name the
// request does not defer is dropped, and so is a name a later response in the
// contents references again, so each tool expands once, at its latest
// reference — a caller that makes a tool callable again after withdrawing it
// leaves the earlier result converting as it did while the tool was
// withdrawn. A response left with no reference converts as one without the
// key would. Every other response converts to a single tool_result of its
// JSON.
func functionResponseToBlocks(
	resp *genai.FunctionResponse,
	sanitizer *toolUseIDSanitizer,
	opts ContentsOptions,
) ([]anthropic.ContentBlockParamUnion, error) {
	if resp == nil {
		return nil, nil
	}

	response := resp.Response
	var references []string
	if names, ok := toolReferenceNames(response, opts.ToolReferencesResponseKey); ok {
		response = make(map[string]any, len(resp.Response)-1)
		for key, value := range resp.Response {
			if key != opts.ToolReferencesResponseKey {
				response[key] = value
			}
		}
		for _, name := range names {
			if _, deferred := opts.DeferredToolNames[name]; !deferred || slices.Contains(references, name) {
				continue
			}
			if latest, tracked := opts.latestReferenceByToolName[name]; tracked && latest != resp {
				continue
			}
			references = append(references, name)
		}
	}

	// Sanitize the tool-use ID to Anthropic's required shape, consistently with
	// the matching tool_use block so the result still correlates.
	toolUseID := sanitizer.sanitize(resp.ID)

	if len(references) == 0 {
		var content string
		if response != nil {
			var err error
			if content, err = functionResponseJSON(response); err != nil {
				return nil, err
			}
		}
		return []anthropic.ContentBlockParamUnion{anthropic.NewToolResultBlock(toolUseID, content, false)}, nil
	}

	referenceBlocks := make([]anthropic.ToolResultBlockParamContentUnion, 0, len(references))
	for _, name := range references {
		referenceBlocks = append(referenceBlocks, anthropic.ToolResultBlockParamContentUnion{
			OfToolReference: &anthropic.ToolReferenceBlockParam{ToolName: name},
		})
	}
	blocks := []anthropic.ContentBlockParamUnion{{
		OfToolResult: &anthropic.ToolResultBlockParam{
			ToolUseID: toolUseID,
			Content:   referenceBlocks,
			IsError:   anthropic.Bool(false),
		},
	}}
	if len(response) > 0 {
		remainder, err := functionResponseJSON(response)
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, anthropic.NewTextBlock(fmt.Sprintf("Tool result %s, continued:\n%s", toolUseID, remainder)))
	}
	return blocks, nil
}

// toolReferenceNames returns the tool names a function response lists under
// key, reporting false when key is empty, absent from the response, or holds
// anything but a list of strings — a []string as a tool returns it, or a []any
// of strings as it reads back from a session store that round-trips the
// response through JSON.
func toolReferenceNames(response map[string]any, key string) ([]string, bool) {
	if key == "" {
		return nil, false
	}
	switch value := response[key].(type) {
	case []string:
		return value, true
	case []any:
		names := make([]string, 0, len(value))
		for _, element := range value {
			name, ok := element.(string)
			if !ok {
				return nil, false
			}
			names = append(names, name)
		}
		return names, true
	default:
		return nil, false
	}
}

// functionResponseJSON encodes a function response as compact JSON. It keeps
// `&`, `<`, and `>` literal: the JSON is read by a model rather than embedded
// in HTML, and the six-byte escape sequences json.Marshal applies to those
// characters by default inflate URL-heavy tool results.
func functionResponseJSON(response map[string]any) (string, error) {
	var jsonBuffer bytes.Buffer
	encoder := json.NewEncoder(&jsonBuffer)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(response); err != nil {
		return "", fmt.Errorf("failed to marshal function response: %w", err)
	}
	// Encode appends a trailing newline after the value.
	return strings.TrimSuffix(jsonBuffer.String(), "\n"), nil
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
// parameter plus an optional effort hint. budgetMode selects the wire form,
// since Claude models differ in capability: adaptive-capable models (Opus 4.8 /
// Sonnet 4.6 and newer) take adaptive thinking with depth controlled by
// OutputConfig.Effort, while models that reject adaptive thinking and effort
// (e.g. Haiku 4.5) take a manual budget_tokens form derived from ThinkingLevel.
// The caller selects the mode per model (mirroring the Python ADK, which keys
// off the budget value), so no model-ID list lives here.
//
// Mapping (both modes share the three "off" guards):
//   - nil cfg                      → off (omit thinking)
//   - ThinkingBudget explicitly 0  → off
//   - ThinkingLevel == Minimal     → off
//   - adaptive mode, otherwise     → adaptive (+ effort from Low/Medium/High)
//   - budget mode, otherwise       → enabled with budget_tokens (Low 1024 / Medium 5000 / High 24000); 0 → off
//
// IncludeThoughts is ignored: in genai it governs whether thought summaries are
// returned, not whether the model thinks, and Anthropic returns thinking blocks
// whenever thinking is on regardless.
func ThinkingConfigToAnthropic(cfg *genai.ThinkingConfig, budgetMode bool) ThinkingMapping {
	if cfg == nil {
		return ThinkingMapping{}
	}
	if cfg.ThinkingBudget != nil && *cfg.ThinkingBudget == 0 {
		return ThinkingMapping{}
	}
	if cfg.ThinkingLevel == genai.ThinkingLevelMinimal {
		return ThinkingMapping{}
	}
	if budgetMode {
		budget := levelToBudget(cfg.ThinkingLevel)
		if budget == 0 {
			return ThinkingMapping{}
		}
		return ThinkingMapping{
			Thinking: anthropic.ThinkingConfigParamOfEnabled(budget),
		}
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

// levelToBudget maps a genai ThinkingLevel to a manual extended-thinking
// budget_tokens value, used under budget mode for models that don't support
// adaptive thinking. Returns 0 for levels that don't map (Unspecified, Minimal),
// signaling thinking off. Low is Anthropic's minimum (1024) and Medium is the
// upstream Alcova default (5000); High is raised above the Alcova default of
// 10000 to 24000, giving budget-mode models (e.g. Haiku) more reasoning room for
// multi-step tool planning. All stay well below typical max_tokens.
func levelToBudget(level genai.ThinkingLevel) int64 {
	switch level {
	case genai.ThinkingLevelLow:
		return 1024
	case genai.ThinkingLevelMedium:
		return 5000
	case genai.ThinkingLevelHigh:
		return 24000
	}
	return 0
}
