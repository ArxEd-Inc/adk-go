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

package converters

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/genai"
)

// ToolsToAnthropicTools converts genai Tools to Anthropic ToolUnionParams. It also returns a map
// from each aliased top-level property key back to its original name (see aliasToolKey); the caller
// must pass that map to MessageToLLMResponse so tool-call arguments are restored to their original
// field names. The map is empty when no key needed aliasing.
func ToolsToAnthropicTools(tools []*genai.Tool) ([]anthropic.ToolUnionParam, map[string]string) {
	if len(tools) == 0 {
		return nil, nil
	}

	aliases := map[string]string{}
	var result []anthropic.ToolUnionParam
	for _, tool := range tools {
		if tool == nil || len(tool.FunctionDeclarations) == 0 {
			continue
		}
		for _, fd := range tool.FunctionDeclarations {
			if fd == nil {
				continue
			}
			result = append(result, FunctionDeclarationToTool(fd, aliases))
		}
	}
	return result, aliases
}

// FunctionDeclarationToTool converts a genai FunctionDeclaration to an Anthropic ToolUnionParam,
// recording any top-level property-key aliases it creates into aliases (alias -> original name).
func FunctionDeclarationToTool(fd *genai.FunctionDeclaration, aliases map[string]string) anthropic.ToolUnionParam {
	return anthropic.ToolUnionParam{
		OfTool: &anthropic.ToolParam{
			Name:        fd.Name,
			Description: anthropic.String(fd.Description),
			InputSchema: functionInputSchema(fd, aliases),
		},
	}
}

// functionInputSchema builds the Anthropic tool input schema from a FunctionDeclaration. It
// preserves the full JSON Schema — including $ref and $defs — so nested references still resolve,
// and inlines a root $ref against $defs so the top level is a concrete object (Anthropic requires
// input_schema.type to be "object", which a bare $ref isn't). properties and required are surfaced
// as the SDK's typed fields; any remaining top-level keywords (notably $defs) pass through
// ExtraFields, which the SDK merges into the marshalled input_schema.
func functionInputSchema(fd *genai.FunctionDeclaration, aliases map[string]string) anthropic.ToolInputSchemaParam {
	schema := functionSchemaMap(fd)
	if schema == nil {
		return anthropic.ToolInputSchemaParam{Properties: map[string]any{}}
	}

	if ref, ok := schema["$ref"].(string); ok {
		if defs, ok := schema["$defs"].(map[string]any); ok {
			if target := resolveDefRef(ref, defs); target != nil {
				delete(schema, "$ref")
				for key, value := range target {
					schema[key] = value
				}
			}
		}
	}

	// Anthropic validates top-level property keys against ^[a-zA-Z0-9_.-]{1,64}$ (but not $defs keys
	// or keys nested deeper). Alias any key that violates it to a stable substitute, recording
	// alias -> original so tool-call arguments can be restored (MessageToLLMResponse) and replayed
	// tool_use args re-aliased (functionCallToBlock). required is mapped through the same aliases.
	input := anthropic.ToolInputSchemaParam{Properties: map[string]any{}}
	if properties, ok := schema["properties"].(map[string]any); ok {
		aliasedProperties := make(map[string]any, len(properties))
		for key, value := range properties {
			aliasedKey := aliasToolKey(key)
			if aliasedKey != key {
				aliases[aliasedKey] = key
			}
			aliasedProperties[aliasedKey] = value
		}
		input.Properties = aliasedProperties
	}
	for _, required := range extractRequiredFields(schema["required"]) {
		input.Required = append(input.Required, aliasToolKey(required))
	}

	// type is fixed to "object" by the SDK and properties/required are set above; carry every other
	// keyword (e.g. $defs for nested $refs) through so the schema reaches Anthropic intact.
	var extraFields map[string]any
	for key, value := range schema {
		switch key {
		case "type", "properties", "required":
		default:
			if extraFields == nil {
				extraFields = map[string]any{}
			}
			extraFields[key] = value
		}
	}
	input.ExtraFields = extraFields

	return input
}

// functionSchemaMap returns the FunctionDeclaration's parameter schema as a JSON Schema map. It
// prefers the structured Parameters; otherwise it round-trips ParametersJsonSchema through JSON so
// any concrete schema type (e.g. *jsonschema.Schema) yields a faithful map that preserves $ref and
// $defs. Returns nil when no parameter schema is set.
func functionSchemaMap(fd *genai.FunctionDeclaration) map[string]any {
	switch {
	case fd.Parameters != nil:
		return SchemaToMap(fd.Parameters)
	case fd.ParametersJsonSchema != nil:
		return RawJSONSchemaToMap(fd.ParametersJsonSchema)
	default:
		return nil
	}
}

// RawJSONSchemaToMap converts a raw JSON-schema value (e.g. genai's ParametersJsonSchema or
// ResponseJsonSchema) to a map by round-tripping it through JSON, so any concrete schema type yields
// a faithful map that preserves $ref and $defs. Returns nil for nil input or on a conversion error.
func RawJSONSchemaToMap(raw any) map[string]any {
	if raw == nil {
		return nil
	}
	schemaBytes, err := json.Marshal(raw)
	if err != nil {
		return nil
	}
	var schema map[string]any
	if err := json.Unmarshal(schemaBytes, &schema); err != nil {
		return nil
	}
	return schema
}

// resolveDefRef returns the schema that a local "#/$defs/<name>" reference points to within defs,
// or nil if ref isn't such a reference or names a missing definition.
func resolveDefRef(ref string, defs map[string]any) map[string]any {
	const defsPrefix = "#/$defs/"
	if !strings.HasPrefix(ref, defsPrefix) {
		return nil
	}
	target, _ := defs[strings.TrimPrefix(ref, defsPrefix)].(map[string]any)
	return target
}

// toolPropertyKeyPattern is the shape Anthropic requires for top-level tool input_schema property
// keys. Keys that don't match (e.g. a proto field name longer than 64 characters) must be aliased.
var toolPropertyKeyPattern = regexp.MustCompile(`^[a-zA-Z0-9_.-]{1,64}$`)

// aliasToolKey returns key unchanged when it satisfies Anthropic's top-level property-key pattern,
// otherwise a deterministic substitute that does: a sanitized, truncated prefix of key plus a short
// hash of the full key for uniqueness. The mapping is stable, so the same original always yields the
// same alias across the tool schema, replayed history, and response parsing.
func aliasToolKey(key string) string {
	if toolPropertyKeyPattern.MatchString(key) {
		return key
	}

	sum := sha256.Sum256([]byte(key))
	suffix := hex.EncodeToString(sum[:])[:8]

	var prefix strings.Builder
	for _, r := range key {
		if prefix.Len() >= 64-1-len(suffix) {
			break
		}
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' || r == '.' || r == '-' {
			prefix.WriteRune(r)
		}
	}
	return prefix.String() + "_" + suffix
}

// extractRequiredFields extracts required field names from various input types.
// Supports []any (from JSON unmarshalling) and []string (from manual construction).
func extractRequiredFields(v any) []string {
	if v == nil {
		return nil
	}
	switch req := v.(type) {
	case []string:
		return req
	case []any:
		result := make([]string, 0, len(req))
		for _, r := range req {
			if s, ok := r.(string); ok {
				result = append(result, s)
			}
		}
		return result
	default:
		return nil
	}
}

// schemaPropertiesToMap converts genai Schema properties to a map for Anthropic.
func schemaPropertiesToMap(props map[string]*genai.Schema) map[string]any {
	if props == nil {
		return nil
	}

	result := make(map[string]any)
	for name, schema := range props {
		if schema == nil {
			continue
		}
		result[name] = SchemaToMap(schema)
	}
	return result
}

// SchemaToMap converts a genai.Schema to a map[string]any suitable for Anthropic.
func SchemaToMap(schema *genai.Schema) map[string]any {
	if schema == nil {
		return nil
	}

	result := make(map[string]any)

	// Type
	if schema.Type != "" {
		result["type"] = strings.ToLower(string(schema.Type))
	}

	// Description
	if schema.Description != "" {
		result["description"] = schema.Description
	}

	// Enum
	if len(schema.Enum) > 0 {
		result["enum"] = schema.Enum
	}

	// Format
	if schema.Format != "" {
		result["format"] = schema.Format
	}

	// Items (for arrays)
	if schema.Items != nil {
		result["items"] = SchemaToMap(schema.Items)
	}

	// Properties (for objects)
	if len(schema.Properties) > 0 {
		result["properties"] = schemaPropertiesToMap(schema.Properties)
	}

	// Required
	if len(schema.Required) > 0 {
		result["required"] = schema.Required
	}

	// Nullable
	if schema.Nullable != nil && *schema.Nullable {
		result["nullable"] = true
	}

	// Default
	if schema.Default != nil {
		result["default"] = schema.Default
	}

	// Min/Max constraints
	if schema.Minimum != nil {
		result["minimum"] = *schema.Minimum
	}
	if schema.Maximum != nil {
		result["maximum"] = *schema.Maximum
	}
	if schema.MinLength != nil {
		result["minLength"] = *schema.MinLength
	}
	if schema.MaxLength != nil {
		result["maxLength"] = *schema.MaxLength
	}
	if schema.MinItems != nil {
		result["minItems"] = *schema.MinItems
	}
	if schema.MaxItems != nil {
		result["maxItems"] = *schema.MaxItems
	}

	// Pattern
	if schema.Pattern != "" {
		result["pattern"] = schema.Pattern
	}

	// AnyOf
	if len(schema.AnyOf) > 0 {
		anyOf := make([]map[string]any, 0, len(schema.AnyOf))
		for _, s := range schema.AnyOf {
			if m := SchemaToMap(s); m != nil {
				anyOf = append(anyOf, m)
			}
		}
		if len(anyOf) > 0 {
			result["anyOf"] = anyOf
		}
	}

	return result
}

// toolChoiceKind represents the resolved tool choice type.
type toolChoiceKind int

const (
	toolChoiceNone toolChoiceKind = iota // omit tool_choice
	toolChoiceAuto
	toolChoiceAny
	toolChoiceTool
)

// resolvedToolChoice holds the result of resolving a ToolConfig into a tool choice decision.
type resolvedToolChoice struct {
	kind     toolChoiceKind
	toolName string // populated when kind == toolChoiceTool
}

// resolveToolChoice extracts the tool choice decision from a ToolConfig.
// Returns an error for unsupported configurations (multiple AllowedFunctionNames,
// unknown FunctionCallingConfig modes).
func resolveToolChoice(config *genai.ToolConfig) (resolvedToolChoice, error) {
	if config == nil || config.FunctionCallingConfig == nil {
		return resolvedToolChoice{kind: toolChoiceNone}, nil
	}

	fcc := config.FunctionCallingConfig

	if len(fcc.AllowedFunctionNames) > 1 {
		return resolvedToolChoice{}, fmt.Errorf(
			"Anthropic does not support multiple AllowedFunctionNames (got %d); use a single function name or remove the restriction",
			len(fcc.AllowedFunctionNames),
		)
	}

	switch fcc.Mode {
	case genai.FunctionCallingConfigModeNone:
		return resolvedToolChoice{kind: toolChoiceNone}, nil

	case genai.FunctionCallingConfigModeAuto:
		return resolvedToolChoice{kind: toolChoiceAuto}, nil

	case genai.FunctionCallingConfigModeAny:
		if len(fcc.AllowedFunctionNames) == 1 {
			return resolvedToolChoice{kind: toolChoiceTool, toolName: fcc.AllowedFunctionNames[0]}, nil
		}
		return resolvedToolChoice{kind: toolChoiceAny}, nil

	default:
		return resolvedToolChoice{}, fmt.Errorf(
			"unsupported FunctionCallingConfig mode %q; supported modes are: ModeNone, ModeAuto, ModeAny",
			fcc.Mode,
		)
	}
}

// ToolConfigToToolChoice converts a genai.ToolConfig to Anthropic's tool_choice parameter.
// Returns a zero-value union param when no tool_choice should be set (nil config, ModeNone),
// which is safe to assign unconditionally as the SDK omits it during serialization.
//
// Mapping:
//   - ModeNone -> zero value (omitted)
//   - ModeAuto -> "auto" (model decides whether to use tools)
//   - ModeAny -> "any" (model must use a tool)
//   - ModeAny + single AllowedFunctionNames -> "tool" with specific name
//
// Returns an error if AllowedFunctionNames contains more than one function name,
// or if the FunctionCallingConfig mode is not recognized.
func ToolConfigToToolChoice(config *genai.ToolConfig) (anthropic.ToolChoiceUnionParam, error) {
	resolved, err := resolveToolChoice(config)
	if err != nil {
		return anthropic.ToolChoiceUnionParam{}, err
	}

	switch resolved.kind {
	case toolChoiceNone:
		return anthropic.ToolChoiceUnionParam{}, nil
	case toolChoiceAuto:
		return anthropic.ToolChoiceUnionParam{
			OfAuto: &anthropic.ToolChoiceAutoParam{},
		}, nil
	case toolChoiceAny:
		return anthropic.ToolChoiceUnionParam{
			OfAny: &anthropic.ToolChoiceAnyParam{},
		}, nil
	case toolChoiceTool:
		return anthropic.ToolChoiceUnionParam{
			OfTool: &anthropic.ToolChoiceToolParam{
				Name: resolved.toolName,
			},
		}, nil
	default:
		return anthropic.ToolChoiceUnionParam{}, fmt.Errorf("unexpected tool choice kind: %d", resolved.kind)
	}
}

// IsForcedToolUse reports whether the given tool_choice forces the model to
// emit a tool_use block — i.e. tool_choice.type is "any" or "tool". Auto and
// the zero value (no tool_choice on the wire) are not forced.
//
// Anthropic rejects the combination of forced tool use and extended thinking
// (both manual and adaptive); callers should drop the thinking parameter when
// this returns true. See https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview#forced-tool-use-and-extended-thinking
func IsForcedToolUse(tc anthropic.ToolChoiceUnionParam) bool {
	return tc.OfAny != nil || tc.OfTool != nil
}
