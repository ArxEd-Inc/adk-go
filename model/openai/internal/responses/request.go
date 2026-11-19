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

package responses

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/openai/openai-go/v3/packages/param"
	oairesponses "github.com/openai/openai-go/v3/responses"
	oaishared "github.com/openai/openai-go/v3/shared"
	"github.com/openai/openai-go/v3/shared/constant"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"

	"google.golang.org/adk/v2/model/openai/internal/shared"
)

// buildParams converts a generic LLMRequest into the OpenAI-specific
// oairesponses.ResponseNewParams format, preparing it for an API call, with no
// model-wide options.
func buildParams(modelName string, req *model.LLMRequest) (oairesponses.ResponseNewParams, error) {
	return buildParamsWith(modelName, req, Options{})
}

// buildParamsWith is buildParams applying the model-wide options a Model
// carries.
func buildParamsWith(modelName string, req *model.LLMRequest, opts Options) (oairesponses.ResponseNewParams, error) {
	return buildParamsFor(modelName, req, opts, nil)
}

// buildParamsFor is buildParamsWith sending each inline document with an entry
// in fileIDs as a reference to that file.
func buildParamsFor(modelName string, req *model.LLMRequest, opts Options, fileIDs map[*genai.Blob]string) (oairesponses.ResponseNewParams, error) {
	if req == nil {
		return oairesponses.ResponseNewParams{}, shared.ErrRequestNil
	}

	params := oairesponses.ResponseNewParams{
		Model: oaishared.ResponsesModel(modelName),
		// Never stored: a conversation is resent whole on every request, so
		// nothing needs the response kept, and an unstored request keeps no
		// copy of what it carried beyond the provider's abuse-monitoring logs.
		Store: param.NewOpt(false),
	}
	if req.Model != "" {
		params.Model = oaishared.ResponsesModel(req.Model)
	}

	// Tools convert first: a function response can load only the tools this
	// request defers.
	tools, err := convertTools(req.Config)
	if err != nil {
		return oairesponses.ResponseNewParams{}, err
	}
	declared, deferred := splitDeferredTools(req, tools)

	contentOpts := contentOptions{
		modelName:         string(params.Model),
		mediaResolution:   requestMediaResolution(req),
		fileIDs:           fileIDs,
		toolReferencesKey: opts.ToolReferencesResponseKey,
		deferredTools:     deferred,
	}
	if opts.PromptCaching != nil && opts.PromptCaching.MarkedPart != nil {
		contentOpts.markedPart = lastMarkedPart(req.Contents)
		contentOpts.markedPartBreakpoint = opts.PromptCaching.MarkedPart
	}

	// We convert the generic content parts into OpenAI's input format.
	input, droppedReasoning, err := convertContentsWith(req.Contents, contentOpts)
	if err != nil {
		return oairesponses.ResponseNewParams{}, err
	}
	if len(input) == 0 {
		if droppedReasoning {
			// The drop is what emptied the request; don't report it as a
			// caller who sent nothing. Gated on a part actually having been
			// dropped, so a request that was empty on arrival still returns
			// the bare sentinel a caller may compare against directly.
			return oairesponses.ResponseNewParams{}, fmt.Errorf(
				"%w: every part was dropped as replayed reasoning", shared.ErrNoContents)
		}
		return oairesponses.ResponseNewParams{}, shared.ErrNoContents
	}
	params.Input = oairesponses.ResponseNewParamsInputUnion{
		OfInputItemList: input,
	}

	// Apply generation configuration settings like temperature and max output tokens.
	if err := applyGenerationConfig(&params, req.Config); err != nil {
		return oairesponses.ResponseNewParams{}, err
	}

	if len(declared) > 0 {
		params.Tools = declared
	}

	// Handle tool choice configuration, if provided.
	if cfg := req.Config; cfg != nil && cfg.ToolConfig != nil {
		choice, err := convertToolChoice(cfg.ToolConfig)
		if err != nil {
			return oairesponses.ResponseNewParams{}, err
		}
		if choice != nil {
			params.ToolChoice = *choice
		}
	}

	// Unstored reasoning comes back only encrypted, and only when asked for.
	// It is what a later turn sends back so the model can pick up its
	// reasoning instead of starting over (see convertContentsWith).
	params.Include = append(params.Include, oairesponses.ResponseIncludableReasoningEncryptedContent)
	if opts.Effort != "" {
		params.Reasoning.Effort = opts.Effort
	}
	if !params.MaxOutputTokens.Valid() && opts.DefaultMaxTokens > 0 {
		params.MaxOutputTokens = param.NewOpt(opts.DefaultMaxTokens)
	}
	if opts.PromptCaching != nil && opts.PromptCaching.SystemInstruction != nil && params.Instructions.Valid() {
		moveInstructionsToDeveloperMessage(&params, opts.PromptCaching.SystemInstruction)
	}
	if opts.PromptCaching != nil && opts.PromptCaching.ExplicitOnly {
		params.PromptCacheOptions = oairesponses.ResponseNewParamsPromptCacheOptions{Mode: "explicit"}
	}

	return params, nil
}

// moveInstructionsToDeveloperMessage sends the request's instructions as its
// first input item, a developer message whose text ends at the given
// breakpoint, since the top-level instructions field cannot carry one. The
// model reads a developer message first in the input as it reads the
// instructions field.
func moveInstructionsToDeveloperMessage(params *oairesponses.ResponseNewParams, bp *shared.CacheBreakpoint) {
	text := oairesponses.ResponseInputTextParam{
		Text:                  params.Instructions.Value,
		PromptCacheBreakpoint: textBreakpoint(bp),
	}
	developer := oairesponses.ResponseInputItemUnionParam{OfMessage: &oairesponses.EasyInputMessageParam{
		Role: oairesponses.EasyInputMessageRoleDeveloper,
		Type: oairesponses.EasyInputMessageTypeMessage,
		Content: oairesponses.EasyInputMessageContentUnionParam{
			OfInputItemContentList: oairesponses.ResponseInputMessageContentListParam{{OfInputText: &text}},
		},
	}}
	params.Input.OfInputItemList = append(oairesponses.ResponseInputParam{developer}, params.Input.OfInputItemList...)
	params.Instructions = param.Opt[string]{}
}

// requestMediaResolution is the MediaResolution req's config sets, or "".
func requestMediaResolution(req *model.LLMRequest) genai.MediaResolution {
	if req.Config == nil {
		return ""
	}
	return req.Config.MediaResolution
}

// splitDeferredTools separates the tools the request declares from those its
// dispatch entries defer, which are keyed by name for a function response to
// load (see contentOptions.deferredTools).
func splitDeferredTools(req *model.LLMRequest, tools []oairesponses.ToolUnionParam) ([]oairesponses.ToolUnionParam, map[string]oairesponses.ToolUnionParam) {
	var declared []oairesponses.ToolUnionParam
	var deferred map[string]oairesponses.ToolUnionParam
	for _, tool := range tools {
		if tool.OfFunction != nil && shared.IsToolLoadingDeferred(req, tool.OfFunction.Name) {
			if deferred == nil {
				deferred = map[string]oairesponses.ToolUnionParam{}
			}
			deferred[tool.OfFunction.Name] = tool
			continue
		}
		declared = append(declared, tool)
	}
	return declared, deferred
}

// contentOptions configures convertContentsWith. The zero value places no
// breakpoint and loads no tool.
type contentOptions struct {
	// mediaResolution is the request's MediaResolution, the detail of media
	// whose part sets none.
	mediaResolution genai.MediaResolution

	// fileIDs maps an inline document's blob to the uploaded file that stands
	// in for it; a blob with no entry is sent inline.
	fileIDs map[*genai.Blob]string

	// toolReferencesKey names the function-response key under which a tool
	// lists tools to make callable, or "" for none. Each listed tool the
	// request defers is loaded by an additional_tools item right after the
	// first function output that lists it, so the tool set grows only at the
	// end of the conversation and the prompt cache's prefix survives. The key
	// is left out of the output the model reads.
	toolReferencesKey string
	// deferredTools holds the definitions of the tools the request defers, by
	// name.
	deferredTools map[string]oairesponses.ToolUnionParam

	// modelName is the model the request goes to. A reasoning item is sent
	// back only to the model that produced it.
	modelName string

	// markedPart is the part whose content block ends at
	// markedPartBreakpoint; nil places no breakpoint.
	markedPart           *genai.Part
	markedPartBreakpoint *shared.CacheBreakpoint
}

// lastMarkedPart returns the last part of contents marked as the end of a
// shared prefix, or nil when none is.
func lastMarkedPart(contents []*genai.Content) *genai.Part {
	var last *genai.Part
	for _, content := range contents {
		if content == nil {
			continue
		}
		for _, part := range content.Parts {
			if shared.IsCacheBreakpointMarked(part) {
				last = part
			}
		}
	}
	return last
}

// convertContentsWith converts contents into Responses API input items,
// reporting separately whether any part was dropped as replayed reasoning. The
// caller needs that to tell a request the drop emptied from one that arrived
// empty.
func convertContentsWith(contents []*genai.Content, opts contentOptions) (oairesponses.ResponseInputParam, bool, error) {
	var (
		items            oairesponses.ResponseInputParam
		loaded           = map[string]bool{}
		tracker          shared.CallTracker
		buffered         []bufferedContent
		droppedReasoning bool
		curRole          genai.Role = genai.RoleUser
		// flushText is a helper function that takes any accumulated content
		// and converts it into a message, then appends it to our items.
		flushText = func() error {
			if len(buffered) == 0 {
				return nil
			}
			msgRole, err := normalizeRole(curRole)
			if err != nil {
				return err
			}
			// The Responses API rejects "input_text" for the assistant role, so
			// a replayed assistant turn goes out as an output message instead.
			if msgRole == oairesponses.EasyInputMessageRoleAssistant {
				msg, err := newOutputMessage(buffered)
				if err != nil {
					return err
				}
				if msg != nil {
					items = append(items, oairesponses.ResponseInputItemUnionParam{OfOutputMessage: msg})
				}
			} else {
				msg, err := newMessage(msgRole, buffered)
				if err != nil {
					return err
				}
				if msg != nil {
					items = append(items, oairesponses.ResponseInputItemUnionParam{OfMessage: msg})
				}
			}
			buffered = buffered[:0]
			return nil
		}
	)

	for _, content := range contents {
		if content == nil || len(content.Parts) == 0 {
			continue
		}
		curRole = genai.Role(content.Role)
		for i, part := range content.Parts {
			if part == nil {
				continue
			}
			// A reasoning item this package carried out of an earlier turn
			// goes back as an input item ahead of the output items that
			// followed it. One that cannot go back is dropped below with every
			// other replayed thought.
			if part.Thought && curRole == genai.RoleModel {
				if payload, ok := decodeReasoningSignature(part.ThoughtSignature); ok &&
					replayableReasoning(payload, opts.modelName, content.Parts, i) &&
					shared.UnsupportedPayload(part) == "" && part.FunctionCall == nil && part.FunctionResponse == nil {
					if err := flushText(); err != nil {
						return nil, false, err
					}
					items = append(items, reasoningInputItem(payload))
					continue
				}
			}
			// Reported before anything is emitted, so that a field this
			// package cannot send is named even when text or a call rides on
			// the same part and would otherwise have carried it out unnoticed.
			// Media is excluded from the report because this endpoint emits it
			// below, as Chat Completions does.
			if field := shared.UnsupportedPayload(part, shared.PartFieldMedia, shared.PartFieldCodeExecution, shared.PartFieldMediaResolution); field != "" {
				return nil, false, fmt.Errorf("openai: unsupported content part: %s", field)
			}
			// Same fenced text adk-python's Responses converter sends. Chat
			// Completions uses it too: dropping the parts there would hide
			// the code and its output from the model.
			codeText := shared.CodeExecutionText(part)
			// Classified before any of it is emitted, for the same reason the
			// check above runs first: media this endpoint cannot label must
			// fail the request rather than leave with the text beside it.
			media, err := shared.ClassifyMedia(part)
			if err != nil {
				return nil, false, err
			}
			// Text is read independently of a call or a response because one
			// part can carry both. A call and a response on the same part are
			// still alternatives, and the response is dropped, as on main.
			sendText := part.Text != "" && !part.Thought
			before := len(buffered)
			switch {
			case sendText:
				var phase oairesponses.ResponseOutputMessagePhase
				if curRole == genai.RoleModel {
					phase = decodePhaseSignature(part.ThoughtSignature)
					// One message states one phase, so text of another phase
					// starts a message of its own.
					if n := len(buffered); n > 0 && buffered[n-1].phase != phase {
						if err := flushText(); err != nil {
							return nil, false, err
						}
					}
				}
				buffered = append(buffered, bufferedContent{text: part.Text, phase: phase})
			case part.Text != "" || shared.ReplayedReasoning(part):
				// Dropping reasoning must not hide a bad role, so the check
				// still runs. The drop counts toward the emptied-request
				// report only when it suppressed text the model would
				// otherwise have seen, blank text being skipped either way.
				if strings.TrimSpace(part.Text) != "" {
					droppedReasoning = true
				}
				if _, err := normalizeRole(curRole); err != nil {
					return nil, false, err
				}
			}
			if codeText != "" {
				buffered = append(buffered, bufferedContent{text: codeText})
			}
			// After the part's own text, so that "describe this:" keeps its
			// place ahead of the image it introduces, and before the call
			// switch below, whose flush has to carry the media out with it.
			detail := mediaDetail(part.MediaResolution, opts.mediaResolution)
			if id, ok := opts.fileIDs[part.InlineData]; ok && part.InlineData != nil {
				// ClassifyMedia reports inline data first, so it is media[0].
				media[0].Source, media[0].FileID, media[0].Data = shared.MediaSourceFileID, id, nil
			}
			for _, m := range media {
				buffered = append(buffered, bufferedContent{media: &m, detail: detail})
			}
			if opts.markedPart != nil && part == opts.markedPart && len(buffered) > before {
				// The part's own content ends where its last piece does. A
				// marked part that buffered nothing, such as a function
				// response, has no input content block to carry a breakpoint.
				buffered[len(buffered)-1].breakpoint = opts.markedPartBreakpoint
			}
			switch {
			case part.FunctionCall != nil:
				// Flush first so buffered text keeps its place ahead of the call.
				if err := flushText(); err != nil {
					return nil, false, err
				}
				callParam, err := newFunctionCall(&tracker, part.FunctionCall)
				if err != nil {
					return nil, false, err
				}
				items = append(items, oairesponses.ResponseInputItemUnionParam{OfFunctionCall: callParam})
			case part.FunctionResponse != nil:
				// Similarly, for a function response, we flush text before adding the response.
				if err := flushText(); err != nil {
					return nil, false, err
				}
				response, references := splitToolReferences(part.FunctionResponse, opts.toolReferencesKey)
				respParam, err := newFunctionResponse(&tracker, response)
				if err != nil {
					return nil, false, err
				}
				items = append(items, oairesponses.ResponseInputItemUnionParam{OfFunctionCallOutput: respParam})
				if load := toolsToLoad(references, opts.deferredTools, loaded); len(load) > 0 {
					items = append(items, oairesponses.ResponseInputItemUnionParam{
						OfAdditionalTools: &oairesponses.ResponseInputItemAdditionalToolsParam{Tools: load},
					})
				}
			case !sendText && codeText == "" && len(media) == 0 && !shared.ReplayedReasoning(part):
				// Nothing in the part reaches the request. It keeps the
				// unsupported-content-part prefix the single message used
				// before, so a caller matching on that still matches here.
				return nil, false, errors.New("openai: unsupported content part: carries nothing to send")
			}
		}
		// After processing all parts in a content block, we flush any remaining text.
		if err := flushText(); err != nil {
			return nil, false, err
		}
	}

	return items, droppedReasoning, nil
}

// splitToolReferences returns a function response without the tool names it
// lists under key, and those names. A response listing none under key comes
// back as it is.
func splitToolReferences(resp *genai.FunctionResponse, key string) (*genai.FunctionResponse, []string) {
	names, ok := shared.ToolReferenceNames(resp.Response, key)
	if !ok {
		return resp, nil
	}
	stripped := *resp
	stripped.Response = make(map[string]any, len(resp.Response)-1)
	for k, v := range resp.Response {
		if k != key {
			stripped.Response[k] = v
		}
	}
	return &stripped, names
}

// toolsToLoad returns the definitions of the deferred tools names lists that
// no earlier function response loaded, in the order listed, and records them
// in loaded. A name the request does not defer is skipped: it is declared
// already, or not a tool at all, and the API refuses a second definition.
func toolsToLoad(names []string, deferred map[string]oairesponses.ToolUnionParam, loaded map[string]bool) []oairesponses.ToolUnionParam {
	var load []oairesponses.ToolUnionParam
	for _, name := range names {
		tool, ok := deferred[name]
		if !ok || loaded[name] {
			continue
		}
		loaded[name] = true
		load = append(load, tool)
	}
	return load
}

// bufferedContent is one piece of a message, held in the order it arrived:
// either text or a classified media value, never both. Text and media
// interleave inside a single message, so they are buffered together rather
// than text being collected on its own and media appended after it.
type bufferedContent struct {
	text  string
	media *shared.Media
	// breakpoint, when set, ends a prompt cache breakpoint at this piece.
	breakpoint *shared.CacheBreakpoint
	// phase is the phase of the assistant message the text came from, or ""
	// when it stated none.
	phase oairesponses.ResponseOutputMessagePhase
	// detail is the detail media is sent at; see mediaDetail.
	detail string
}

// mediaDetail is the detail a media part is sent at: the part's own
// MediaResolution when it sets one, else the request's. High asks for "high"
// and low for "low"; anything else, unset included, is "auto". ImageParam and
// FileParam tag Detail as required and omitzero at once, so an unset value
// would vanish from the request; "auto" is therefore sent explicitly.
func mediaDetail(part *genai.PartMediaResolution, request genai.MediaResolution) string {
	if part != nil && part.Level != "" {
		switch part.Level {
		case genai.PartMediaResolutionLevelMediaResolutionHigh, genai.PartMediaResolutionLevelMediaResolutionUltraHigh:
			return "high"
		case genai.PartMediaResolutionLevelMediaResolutionLow:
			return "low"
		default:
			return "auto"
		}
	}
	switch request {
	case genai.MediaResolutionHigh:
		return "high"
	case genai.MediaResolutionLow:
		return "low"
	default:
		return "auto"
	}
}

// newMessage builds an easy input message for an already-normalized role.
func newMessage(msgRole oairesponses.EasyInputMessageRole, buffered []bufferedContent) (*oairesponses.EasyInputMessageParam, error) {
	if len(buffered) == 0 {
		return nil, nil
	}
	contentList := make(oairesponses.ResponseInputMessageContentListParam, 0, len(buffered))
	for _, b := range buffered {
		if b.media != nil {
			// System and developer messages take input_text only.
			if msgRole == oairesponses.EasyInputMessageRoleSystem || msgRole == oairesponses.EasyInputMessageRoleDeveloper {
				return nil, fmt.Errorf("openai: media in a %s message is not supported: %s", msgRole, b.media.MIMEType)
			}
			content, err := newMediaContent(*b.media, b.detail)
			if err != nil {
				return nil, err
			}
			setBreakpoint(&content, b.breakpoint)
			contentList = append(contentList, content)
			continue
		}
		if strings.TrimSpace(b.text) == "" {
			continue
		}
		textParam := oairesponses.ResponseInputTextParam{
			Text: b.text,
			Type: constant.InputText("input_text"),
		}
		content := oairesponses.ResponseInputContentUnionParam{OfInputText: &textParam}
		setBreakpoint(&content, b.breakpoint)
		contentList = append(contentList, content)
	}
	if len(contentList) == 0 {
		return nil, nil
	}
	return &oairesponses.EasyInputMessageParam{
		Role: msgRole,
		Type: oairesponses.EasyInputMessageTypeMessage,
		Content: oairesponses.EasyInputMessageContentUnionParam{
			OfInputItemContentList: contentList,
		},
	}, nil
}

// setBreakpoint ends a prompt cache breakpoint at content; a nil bp sets none.
func setBreakpoint(content *oairesponses.ResponseInputContentUnionParam, bp *shared.CacheBreakpoint) {
	if bp == nil {
		return
	}
	// Built with the SDK's constructors, which set the constant mode: a
	// breakpoint whose fields are all zero is omitted from the request.
	switch {
	case content.OfInputText != nil:
		content.OfInputText.PromptCacheBreakpoint = textBreakpoint(bp)
	case content.OfInputImage != nil:
		image := oairesponses.NewResponseInputImagePromptCacheBreakpointParam()
		image.Ttl = string(bp.TTL)
		content.OfInputImage.PromptCacheBreakpoint = image
	case content.OfInputFile != nil:
		file := oairesponses.NewResponseInputFilePromptCacheBreakpointParam()
		file.Ttl = string(bp.TTL)
		content.OfInputFile.PromptCacheBreakpoint = file
	}
}

// textBreakpoint is bp as an input_text block's breakpoint.
func textBreakpoint(bp *shared.CacheBreakpoint) oairesponses.ResponseInputTextPromptCacheBreakpointParam {
	text := oairesponses.NewResponseInputTextPromptCacheBreakpointParam()
	text.Ttl = string(bp.TTL)
	return text
}

// newMediaContent puts one classified media value in the content field the
// Responses API has for it.
//
// Kind decides the field — input_image or input_file — and source decides
// which of that field's three ways of naming the bytes is used. The two vary
// independently: an image uploaded to the provider stays a file id rather than
// becoming a location for the server to fetch.
func newMediaContent(m shared.Media, detail string) (oairesponses.ResponseInputContentUnionParam, error) {
	if detail == "" {
		detail = "auto"
	}
	switch m.Kind {
	case shared.MediaKindImage:
		image := oairesponses.ResponseInputImageParam{Detail: oairesponses.ResponseInputImageDetail(detail)}
		switch m.Source {
		case shared.MediaSourceData:
			image.ImageURL = param.NewOpt(m.DataURL())
		case shared.MediaSourceFileID:
			image.FileID = param.NewOpt(m.FileID)
		case shared.MediaSourceURL:
			image.ImageURL = param.NewOpt(m.URL)
		default:
			return oairesponses.ResponseInputContentUnionParam{}, fmt.Errorf("openai: unsupported media source: %s", m.Source)
		}
		return oairesponses.ResponseInputContentUnionParam{OfInputImage: &image}, nil
	case shared.MediaKindFile:
		file := oairesponses.ResponseInputFileParam{Detail: oairesponses.ResponseInputFileDetail(detail)}
		switch m.Source {
		case shared.MediaSourceData:
			file.FileData = param.NewOpt(m.DataURL())
		case shared.MediaSourceFileID:
			file.FileID = param.NewOpt(m.FileID)
		case shared.MediaSourceURL:
			file.FileURL = param.NewOpt(m.URL)
		default:
			return oairesponses.ResponseInputContentUnionParam{}, fmt.Errorf("openai: unsupported media source: %s", m.Source)
		}
		// The filename travels only with inline data, as in adk-python: a
		// file id or a URL already names the file. The classifier supplies a
		// fallback name for inline bytes that have none.
		if m.Source == shared.MediaSourceData && m.Filename != "" {
			file.Filename = param.NewOpt(m.Filename)
		}
		return oairesponses.ResponseInputContentUnionParam{OfInputFile: &file}, nil
	default:
		// Audio and video are labeled by the classifier and refused here:
		// the Responses input has no field for either, and naming the type
		// the part declared says more than naming the field it arrived in.
		return oairesponses.ResponseInputContentUnionParam{}, fmt.Errorf("%w: %s", shared.ErrUnsupportedMIMEType, m.MIMEType)
	}
}

// newOutputMessage builds an assistant output message whose content uses the
// "output_text" type, as required when replaying a prior assistant turn to the
// OpenAI Responses API.
func newOutputMessage(buffered []bufferedContent) (*oairesponses.ResponseOutputMessageParam, error) {
	if len(buffered) == 0 {
		return nil, nil
	}
	contentList := make([]oairesponses.ResponseOutputMessageContentUnionParam, 0, len(buffered))
	for _, b := range buffered {
		if b.media != nil {
			// An output message's content is output_text or a refusal, so
			// there is nowhere to put an image a prior assistant turn
			// replayed. Refused rather than dropped, the caller being the
			// only one who can decide what to send instead.
			return nil, fmt.Errorf("%w: %s", shared.ErrMediaOnAssistantTurn, b.media.MIMEType)
		}
		if strings.TrimSpace(b.text) == "" {
			continue
		}
		contentList = append(contentList, oairesponses.ResponseOutputMessageContentUnionParam{
			OfOutputText: &oairesponses.ResponseOutputTextParam{
				Text: b.text,
				Type: constant.OutputText("output_text"),
			},
		})
	}
	if len(contentList) == 0 {
		return nil, nil
	}
	return &oairesponses.ResponseOutputMessageParam{
		Content: contentList,
		Status:  oairesponses.ResponseOutputMessageStatusCompleted,
		// Every piece of one buffer shares a phase; see convertContentsWith.
		Phase: buffered[0].phase,
	}, nil
}

func normalizeRole(role genai.Role) (oairesponses.EasyInputMessageRole, error) {
	switch role {
	case "", genai.RoleUser:
		return oairesponses.EasyInputMessageRoleUser, nil
	case genai.RoleModel:
		return oairesponses.EasyInputMessageRoleAssistant, nil
	case "system":
		return oairesponses.EasyInputMessageRoleSystem, nil
	case "developer":
		return oairesponses.EasyInputMessageRoleDeveloper, nil
	default:
		return "", fmt.Errorf("openai: unsupported role %q", role)
	}
}

// newFunctionCall converts a generic genai.FunctionCall into an OpenAI-specific
// ResponseFunctionToolCallParam. We generate a unique callID if one isn't
// provided, and then marshal the function arguments into a JSON string.
func newFunctionCall(t *shared.CallTracker, fc *genai.FunctionCall) (*oairesponses.ResponseFunctionToolCallParam, error) {
	if fc.Name == "" {
		return nil, shared.ErrFunctionCallMissingName
	}
	callID := t.TakeCallID(fc)
	args, err := shared.MarshalFunctionArgs(fc.Args)
	if err != nil {
		return nil, err
	}
	return &oairesponses.ResponseFunctionToolCallParam{
		Name:      fc.Name,
		CallID:    callID,
		Arguments: args,
		Type:      constant.FunctionCall("function_call"),
	}, nil
}

// newFunctionResponse converts a generic genai.FunctionResponse into an OpenAI-specific
// ResponseInputItemFunctionCallOutputParam. We try to match the response to a pending
// function call. If an explicit callID is provided, we find and remove it from our
// pending list. Otherwise, we assume it corresponds to the oldest pending call.
func newFunctionResponse(t *shared.CallTracker, fr *genai.FunctionResponse) (*oairesponses.ResponseInputItemFunctionCallOutputParam, error) {
	callID, err := t.ResolveResponseID(fr)
	if err != nil {
		return nil, err
	}
	payload, err := json.Marshal(fr.Response)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal function response: %w", err)
	}
	return &oairesponses.ResponseInputItemFunctionCallOutputParam{
		CallID: param.NewOpt(callID),
		Output: oairesponses.ResponseInputItemFunctionCallOutputOutputUnionParam{
			OfString: param.NewOpt(string(payload)),
		},
		Type: constant.FunctionCallOutput("function_call_output"),
	}, nil
}

// supportedConfigFields adds StopSequences to the fields both endpoints support,
// for the empty list alone: it asks for no stop sequences, which needs no wire
// field, while applyGenerationConfig refuses a non-empty one by name.
//
// MediaResolution is supported here alone: Responses maps it onto the detail
// its image and file inputs take (see mediaDetail).
var supportedConfigFields = append(slices.Clone(shared.SupportedConfigFields), "StopSequences", "MediaResolution")

// applyGenerationConfig translates our generic generation configuration into
// OpenAI-specific parameters. We also validate and return errors for features
// that are not supported by the OpenAI Responses API.
func applyGenerationConfig(params *oairesponses.ResponseNewParams, cfg *genai.GenerateContentConfig) error {
	if cfg == nil {
		return nil
	}
	if cfg.Temperature != nil {
		params.Temperature = param.NewOpt(float64(*cfg.Temperature))
	}
	if cfg.TopP != nil {
		params.TopP = param.NewOpt(float64(*cfg.TopP))
	}
	if cfg.TopK != nil {
		return shared.ErrTopKNotSupported
	}
	if cfg.MaxOutputTokens > 0 {
		params.MaxOutputTokens = param.NewOpt(int64(cfg.MaxOutputTokens))
	}
	if len(cfg.StopSequences) > 0 {
		return shared.ErrStopSequencesNotSupported
	}
	if cfg.CandidateCount > 1 {
		return shared.ErrMultipleCandidatesNotSupported
	}
	if cfg.FrequencyPenalty != nil || cfg.PresencePenalty != nil {
		return shared.ErrPenaltiesNotSupported
	}
	if cfg.ResponseLogprobs {
		if cfg.Logprobs != nil {
			params.TopLogprobs = param.NewOpt(int64(*cfg.Logprobs))
		} else {
			params.TopLogprobs = param.NewOpt(int64(1))
		}
		// Responses returns logprobs only when explicitly included.
		params.Include = append(params.Include, oairesponses.ResponseIncludableMessageOutputTextLogprobs)
	}
	if cfg.SystemInstruction != nil {
		inst, err := shared.FlattenContentText(cfg.SystemInstruction)
		if err != nil {
			return fmt.Errorf("openai: system instruction: %w", err)
		}
		if inst != "" {
			params.Instructions = param.NewOpt(inst)
		}
	}
	if cfg.ResponseMIMEType != "" && cfg.ResponseMIMEType != "text/plain" && cfg.ResponseMIMEType != "application/json" {
		return fmt.Errorf("%w: %s", shared.ErrUnsupportedMIMEType, cfg.ResponseMIMEType)
	}
	if cfg.ResponseMIMEType == "application/json" || cfg.ResponseSchema != nil || cfg.ResponseJsonSchema != nil {
		if cfg.ResponseSchema == nil && cfg.ResponseJsonSchema == nil {
			obj := oaishared.NewResponseFormatJSONObjectParam()
			params.Text = oairesponses.ResponseTextConfigParam{
				Format: oairesponses.ResponseFormatTextConfigUnionParam{
					OfJSONObject: &obj,
				},
			}
		} else {
			format, err := newJSONSchemaFormat(cfg)
			if err != nil {
				return err
			}
			params.Text = oairesponses.ResponseTextConfigParam{
				Format: oairesponses.ResponseFormatTextConfigUnionParam{
					OfJSONSchema: format,
				},
			}
		}
	}
	if cfg.Labels != nil {
		return shared.ErrLabelsNotSupported
	}
	if cfg.SafetySettings != nil {
		return shared.ErrSafetySettingsNotSupported
	}
	if err := applyThinkingConfig(params, cfg.ThinkingConfig); err != nil {
		return err
	}
	if cfg.ServiceTier != "" {
		tier, ok := serviceTiers[cfg.ServiceTier]
		if !ok {
			return fmt.Errorf("%w: ServiceTier %q", shared.ErrUnsupportedConfigField, cfg.ServiceTier)
		}
		params.ServiceTier = tier
	}
	if err := shared.RejectUntranslatableValues(cfg); err != nil {
		return err
	}
	// Last, so the named errors above win when a caller sets both.
	return shared.RejectUnsupportedConfigFields(cfg, supportedConfigFields)
}

// serviceTiers maps genai's processing tiers onto the Responses equivalents.
// "Unspecified" joins "standard" on default rather than auto, because genai
// documents it as "Default service tier, which is standard".
var serviceTiers = map[genai.ServiceTier]oairesponses.ResponseNewParamsServiceTier{
	genai.ServiceTierUnspecified: oairesponses.ResponseNewParamsServiceTierDefault,
	genai.ServiceTierStandard:    oairesponses.ResponseNewParamsServiceTierDefault,
	genai.ServiceTierFlex:        oairesponses.ResponseNewParamsServiceTierFlex,
	genai.ServiceTierPriority:    oairesponses.ResponseNewParamsServiceTierPriority,
}

// applyThinkingConfig maps genai's thinking config onto effort-based reasoning.
//
// Summary rides on IncludeThoughts because summaries need a verified OpenAI
// organization, so requesting one unprompted would fail an unverified org's
// every reasoning call.
func applyThinkingConfig(params *oairesponses.ResponseNewParams, cfg *genai.ThinkingConfig) error {
	if cfg == nil {
		return nil
	}
	effort, err := shared.ReasoningEffortFor(cfg)
	if err != nil {
		return err
	}
	// A reasoning block holding neither is the zero value, which omitzero drops
	// from the request, so an unasked-for one costs nothing on the wire.
	params.Reasoning = oaishared.ReasoningParam{Effort: effort}
	// IncludeThoughts alone leaves Effort unset, letting the model pick it, and
	// asks only for the summaries that response.go surfaces as thought parts.
	if cfg.IncludeThoughts {
		params.Reasoning.Summary = oaishared.ReasoningSummaryAuto
	}
	return nil
}

// newJSONSchemaFormat constructs an OpenAI-specific JSON schema format from our
// generic GenerateContentConfig. We handle cases where the schema is provided
// directly or needs to be converted, and assign a name to it.
func newJSONSchemaFormat(cfg *genai.GenerateContentConfig) (*oairesponses.ResponseFormatTextJSONSchemaConfigParam, error) {
	var (
		schema map[string]any
		err    error
	)
	switch {
	case cfg.ResponseJsonSchema != nil:
		schema, err = shared.NormalizeSchema(cfg.ResponseJsonSchema)
	case cfg.ResponseSchema != nil:
		schema, err = shared.SchemaToMap(cfg.ResponseSchema)
	default:
		return nil, fmt.Errorf("openai: json schema requested without schema")
	}
	if err != nil {
		return nil, err
	}
	shared.EnforceStrictOpenAISchema(schema)
	return &oairesponses.ResponseFormatTextJSONSchemaConfigParam{
		Name:   shared.ResponseFormatName(cfg.ResponseSchema),
		Schema: schema,
		Strict: param.NewOpt(true),
		Type:   constant.JSONSchema("json_schema"),
	}, nil
}
