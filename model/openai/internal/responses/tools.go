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
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"github.com/openai/openai-go/v3/packages/param"
	oairesponses "github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model/openai/internal/shared"
)

// convertTools takes our generic tool definitions and converts them into
// OpenAI's specific tool format. We ensure that only function tools are
// supported and properly declared.
func convertTools(cfg *genai.GenerateContentConfig) ([]oairesponses.ToolUnionParam, error) {
	if cfg == nil || len(cfg.Tools) == 0 {
		return nil, nil
	}
	var tools []oairesponses.ToolUnionParam
	for i, tool := range cfg.Tools {
		if err := shared.EnsureFunctionToolOnly(i, tool); err != nil {
			return nil, err
		}
		for _, decl := range tool.FunctionDeclarations {
			fn, err := convertFunctionDeclaration(decl)
			if err != nil {
				return nil, err
			}
			tools = append(tools, oairesponses.ToolUnionParam{OfFunction: fn})
		}
	}
	return tools, nil
}

// convertFunctionDeclaration takes a generic genai.FunctionDeclaration and
// converts it into an OpenAI-specific oairesponses.FunctionToolParam. We handle
// the function's name, description, and importantly, convert its parameters
// from a generic schema format to a map[string]any that the OpenAI API expects.
//
// Strict is pinned off rather than left unset: an absent flag lets the API pick
// the mode from the schema shape and add the caller's optional arguments to
// required. Pinning it on would do that permanently, so off is the only value
// that keeps an optional argument optional; adk-python's Responses path agrees.
func convertFunctionDeclaration(fn *genai.FunctionDeclaration) (*oairesponses.FunctionToolParam, error) {
	if fn == nil {
		return nil, fmt.Errorf("openai: nil function declaration")
	}
	if fn.Name == "" {
		return nil, fmt.Errorf("openai: function declaration missing name")
	}

	paramsMap, err := shared.SchemaToMap(fn.Parameters)
	if err != nil {
		return nil, err
	}
	if paramsMap == nil && fn.ParametersJsonSchema != nil {
		paramsMap, err = shared.NormalizeSchema(fn.ParametersJsonSchema)
		if err != nil {
			return nil, err
		}
	}
	if paramsMap == nil {
		// If no parameters are defined, we default to an empty object schema.
		paramsMap = map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		}
	}

	fnParam := &oairesponses.FunctionToolParam{
		Name:       fn.Name,
		Type:       constant.Function("function"),
		Parameters: paramsMap,
		Strict:     param.NewOpt(false),
	}
	if description := functionDescription(fn); description != "" {
		fnParam.Description = param.NewOpt(description)
	}
	return fnParam, nil
}

// responseSchemaLeadIn introduces a function's response schema where
// functionDescription appends it to the tool description.
const responseSchemaLeadIn = "The result conforms to this JSON Schema: "

// functionDescription returns the declaration's description with its response
// schema appended as compact JSON when that schema carries documentation. A
// function tool has no field for a response schema, so without this the
// documentation a declaration gives its result (what fields mean, which values
// occur) never reaches the model. A response schema with no description or
// title anywhere only restates the structure the result itself shows, so it is
// left out and the description is returned unchanged. Map keys encode sorted,
// so the description is byte-stable across requests, as prompt caching needs.
func functionDescription(fn *genai.FunctionDeclaration) string {
	schema := functionResponseSchemaMap(fn)
	if !schemaIsDocumented(schema) {
		return fn.Description
	}
	var schemaJSON bytes.Buffer
	encoder := json.NewEncoder(&schemaJSON)
	// Leave &, < and > readable, as they are in the rest of the description.
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(schema); err != nil {
		return fn.Description
	}
	rendered := responseSchemaLeadIn + strings.TrimSuffix(schemaJSON.String(), "\n")
	description := strings.TrimRight(fn.Description, " \t\n")
	if description == "" {
		return rendered
	}
	return description + "\n\n" + rendered
}

// functionResponseSchemaMap returns the declaration's response schema as a
// JSON Schema map, preferring ResponseJsonSchema over Response, or nil when it
// sets neither or the schema does not convert.
func functionResponseSchemaMap(fn *genai.FunctionDeclaration) map[string]any {
	var (
		schema map[string]any
		err    error
	)
	switch {
	case fn.ResponseJsonSchema != nil:
		if genaiSchema, ok := fn.ResponseJsonSchema.(*genai.Schema); ok {
			schema, err = shared.SchemaToMap(genaiSchema)
		} else {
			schema, err = shared.NormalizeSchema(fn.ResponseJsonSchema)
		}
	case fn.Response != nil:
		schema, err = shared.SchemaToMap(fn.Response)
	}
	if err != nil {
		return nil
	}
	return schema
}

var (
	// subschemaMapKeywords name the JSON Schema keywords whose values map
	// names to subschemas.
	subschemaMapKeywords = []string{"$defs", "definitions", "properties", "patternProperties", "dependentSchemas"}

	// subschemaArrayKeywords name the JSON Schema keywords whose values are
	// arrays of subschemas (items only in its array form; subschemaKeywords
	// covers the single-schema form).
	subschemaArrayKeywords = []string{"allOf", "anyOf", "oneOf", "prefixItems", "items"}

	// subschemaKeywords name the JSON Schema keywords whose values are single
	// subschemas.
	subschemaKeywords = []string{
		"items", "additionalProperties", "additionalItems", "unevaluatedProperties", "unevaluatedItems",
		"contains", "propertyNames", "not", "if", "then", "else", "contentSchema",
	}
)

// schemaIsDocumented reports whether the given JSON Schema, or any subschema
// it contains, has a non-empty description or title. Only subschema keywords
// are searched, so a property named "description" is a subschema rather than
// documentation, and the values under enum, const, default and examples are
// never mistaken for schemas. Boolean schemas carry no documentation.
func schemaIsDocumented(schema any) bool {
	schemaMap, ok := schema.(map[string]any)
	if !ok {
		return false
	}
	for _, keyword := range []string{"description", "title"} {
		if text, ok := schemaMap[keyword].(string); ok && text != "" {
			return true
		}
	}
	for _, keyword := range subschemaMapKeywords {
		if subschemas, ok := schemaMap[keyword].(map[string]any); ok {
			for _, subschema := range subschemas {
				if schemaIsDocumented(subschema) {
					return true
				}
			}
		}
	}
	for _, keyword := range subschemaArrayKeywords {
		switch subschemas := schemaMap[keyword].(type) {
		case []any:
			if slices.ContainsFunc(subschemas, schemaIsDocumented) {
				return true
			}
		case []map[string]any:
			for _, subschema := range subschemas {
				if schemaIsDocumented(subschema) {
					return true
				}
			}
		}
	}
	for _, keyword := range subschemaKeywords {
		if schemaIsDocumented(schemaMap[keyword]) {
			return true
		}
	}
	return false
}

// convertToolChoice takes our generic genai.ToolConfig and translates its
// FunctionCallingConfig into an OpenAI-specific tool choice parameter.
// We handle different function calling modes (Auto, None, Any) and
// incorporate any allowed function names into the appropriate OpenAI format.
func convertToolChoice(toolCfg *genai.ToolConfig) (*oairesponses.ResponseNewParamsToolChoiceUnion, error) {
	if toolCfg == nil || toolCfg.FunctionCallingConfig == nil {
		return nil, nil
	}
	cfg := toolCfg.FunctionCallingConfig
	choice := &oairesponses.ResponseNewParamsToolChoiceUnion{}
	switch cfg.Mode {
	case "", genai.FunctionCallingConfigModeUnspecified, genai.FunctionCallingConfigModeAuto:
		if len(cfg.AllowedFunctionNames) == 0 {
			// If no specific functions are allowed, we don't set a tool choice,
			// letting OpenAI decide (which is effectively 'auto').
			return nil, nil
		}
		// If specific functions are allowed in auto mode, we specify them.
		choice.OfAllowedTools = allowedToolParam(cfg.AllowedFunctionNames, oairesponses.ToolChoiceAllowedModeAuto)
	case genai.FunctionCallingConfigModeNone:
		// Explicitly disable tool calling.
		choice.OfToolChoiceMode = param.NewOpt(oairesponses.ToolChoiceOptionsNone)
	case genai.FunctionCallingConfigModeAny:
		if len(cfg.AllowedFunctionNames) == 0 {
			// If 'any' is specified without allowed names, it means the model
			// can call any tool.
			choice.OfToolChoiceMode = param.NewOpt(oairesponses.ToolChoiceOptionsRequired)
		} else {
			// If 'any' is specified with allowed names, the model must call
			// one of the allowed tools.
			choice.OfAllowedTools = allowedToolParam(cfg.AllowedFunctionNames, oairesponses.ToolChoiceAllowedModeRequired)
		}
	default:
		return nil, fmt.Errorf("openai: unsupported tool calling mode %q", cfg.Mode)
	}

	if !param.IsOmitted(choice.OfToolChoiceMode) || choice.OfAllowedTools != nil {
		return choice, nil
	}
	return nil, nil
}

func allowedToolParam(names []string, mode oairesponses.ToolChoiceAllowedMode) *oairesponses.ToolChoiceAllowedParam {
	tools := make([]map[string]any, 0, len(names))
	for _, name := range names {
		if name == "" {
			continue
		}
		tools = append(tools, map[string]any{
			"type": "function",
			"name": name,
		})
	}
	if len(tools) == 0 {
		return nil
	}
	return &oairesponses.ToolChoiceAllowedParam{
		Mode:  mode,
		Type:  constant.AllowedTools("allowed_tools"),
		Tools: tools,
	}
}
