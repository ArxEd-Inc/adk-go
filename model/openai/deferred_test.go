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

package openai

import (
	"encoding/json"
	"slices"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// deferring is a dispatch entry that defers its tool's declaration.
type deferring struct{}

func (deferring) DeferLoading() bool { return true }

const loadedToolsKey = "loaded_tools"

// catalogRequest is a conversation with a loader tool, which is declared, and
// two deferred tools, of which two loader responses both list loadedNames.
func catalogRequest(loadedNames any) *model.LLMRequest {
	declare := func(name string) *genai.FunctionDeclaration {
		return &genai.FunctionDeclaration{Name: name, Description: "Tool " + name + "."}
	}
	loaderResponse := func(id string) *genai.Part {
		return &genai.Part{FunctionResponse: &genai.FunctionResponse{
			ID: id, Name: "load_tools", Response: map[string]any{"status": "ok", loadedToolsKey: loadedNames},
		}}
	}
	loaderCall := func(id string) *genai.Part {
		return &genai.Part{FunctionCall: &genai.FunctionCall{ID: id, Name: "load_tools", Args: map[string]any{}}}
	}
	return &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("Cancel order 42.", genai.RoleUser),
			{Role: genai.RoleModel, Parts: []*genai.Part{loaderCall("call_1")}},
			{Role: genai.RoleUser, Parts: []*genai.Part{loaderResponse("call_1")}},
			{Role: genai.RoleModel, Parts: []*genai.Part{loaderCall("call_2")}},
			{Role: genai.RoleUser, Parts: []*genai.Part{loaderResponse("call_2")}},
		},
		Config: &genai.GenerateContentConfig{Tools: []*genai.Tool{{FunctionDeclarations: []*genai.FunctionDeclaration{
			declare("load_tools"), declare("cancel_order"), declare("refund_order"),
		}}}},
		Tools: map[string]any{"cancel_order": deferring{}, "refund_order": deferring{}},
	}
}

func toolNames(tools []any) []string {
	var names []string
	for _, tool := range tools {
		names = append(names, tool.(map[string]any)["name"].(string))
	}
	return names
}

func TestDeferredTools_LoadedAfterTheFirstResponseNamingThem(t *testing.T) {
	for name, loadedNames := range map[string]any{
		"names as a tool returns them":         []string{"cancel_order", "load_tools", "no_such_tool"},
		"names read back from a session store": []any{"cancel_order", "load_tools", "no_such_tool"},
	} {
		t.Run(name, func(t *testing.T) {
			body := sendAndCapture(t, Config{ToolReferencesResponseKey: loadedToolsKey}, catalogRequest(loadedNames))

			tools, _ := body["tools"].([]any)
			if got := toolNames(tools); !slices.Equal(got, []string{"load_tools"}) {
				t.Errorf("declared tools = %v, want the loader alone", got)
			}

			input, _ := body["input"].([]any)
			var shape []string
			for _, item := range input {
				m := item.(map[string]any)
				typ, _ := m["type"].(string)
				if typ == "" || typ == "message" {
					typ, _ = m["role"].(string)
				}
				shape = append(shape, typ)
				switch typ {
				case "additional_tools":
					loaded, _ := m["tools"].([]any)
					if got := toolNames(loaded); !slices.Equal(got, []string{"cancel_order"}) {
						t.Errorf("additional_tools loads %v, want cancel_order alone", got)
					}
				case "function_call_output":
					var output map[string]any
					if err := json.Unmarshal([]byte(m["output"].(string)), &output); err != nil {
						t.Fatalf("decode output: %v", err)
					}
					if _, ok := output[loadedToolsKey]; ok {
						t.Errorf("output %v still lists the loaded tools", output)
					}
					if output["status"] != "ok" {
						t.Errorf("output %v lost the rest of the response", output)
					}
				}
			}
			want := []string{"user", "function_call", "function_call_output", "additional_tools", "function_call", "function_call_output"}
			if !slices.Equal(shape, want) {
				t.Errorf("input = %v, want %v", shape, want)
			}
		})
	}
}

func TestDeferredTools_WithoutAKeyNothingLoads(t *testing.T) {
	body := sendAndCapture(t, Config{}, catalogRequest([]string{"cancel_order"}))
	input, _ := body["input"].([]any)
	for _, item := range input {
		m := item.(map[string]any)
		if m["type"] == "additional_tools" {
			t.Errorf("input carries %v, want no additional_tools", m)
		}
		if m["type"] == "function_call_output" {
			var output map[string]any
			_ = json.Unmarshal([]byte(m["output"].(string)), &output)
			if _, ok := output[loadedToolsKey]; !ok {
				t.Errorf("output %v dropped a key it was not configured to read", output)
			}
		}
	}
}

func TestIsToolLoadingDeferred(t *testing.T) {
	req := catalogRequest(nil)
	if !IsToolLoadingDeferred(req, "cancel_order") || IsToolLoadingDeferred(req, "load_tools") || IsToolLoadingDeferred(nil, "x") {
		t.Error("IsToolLoadingDeferred disagrees with the dispatch entries")
	}
}
