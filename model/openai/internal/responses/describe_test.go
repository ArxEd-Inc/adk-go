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

package responses

import (
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

func TestFunctionDescription_AppendsADocumentedResponseSchema(t *testing.T) {
	documented := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"status": map[string]any{"type": "string", "description": "Where the order is: placed, packed or shipped."},
		},
	}
	undocumented := map[string]any{"type": "object", "properties": map[string]any{"status": map[string]any{"type": "string"}}}

	tests := []struct {
		name string
		fn   *genai.FunctionDeclaration
		want string
	}{
		{
			name: "documented JSON schema",
			fn:   &genai.FunctionDeclaration{Name: "lookup_order", Description: "Looks an order up.", ResponseJsonSchema: documented},
			want: "Looks an order up.\n\n" + responseSchemaLeadIn +
				`{"properties":{"status":{"description":"Where the order is: placed, packed or shipped.","type":"string"}},"type":"object"}`,
		},
		{
			name: "documented genai schema",
			fn: &genai.FunctionDeclaration{Name: "lookup_order", Description: "Looks an order up.", Response: &genai.Schema{
				Type: genai.TypeObject, Properties: map[string]*genai.Schema{"sku": {Type: genai.TypeString, Description: "The SKU & bin."}},
			}},
			want: "Looks an order up.\n\n" + responseSchemaLeadIn +
				`{"properties":{"sku":{"description":"The SKU & bin.","type":"string"}},"type":"object"}`,
		},
		{
			name: "no description of its own",
			fn:   &genai.FunctionDeclaration{Name: "lookup_order", ResponseJsonSchema: documented},
			want: responseSchemaLeadIn +
				`{"properties":{"status":{"description":"Where the order is: placed, packed or shipped.","type":"string"}},"type":"object"}`,
		},
		{
			name: "undocumented schema adds nothing",
			fn:   &genai.FunctionDeclaration{Name: "lookup_order", Description: "Looks an order up.", ResponseJsonSchema: undocumented},
			want: "Looks an order up.",
		},
		{
			name: "no response schema",
			fn:   &genai.FunctionDeclaration{Name: "lookup_order", Description: "Looks an order up."},
			want: "Looks an order up.",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := functionDescription(tt.fn); got != tt.want {
				t.Errorf("functionDescription() =\n%s\nwant\n%s", got, tt.want)
			}
			param, err := convertFunctionDeclaration(tt.fn)
			if err != nil {
				t.Fatalf("convertFunctionDeclaration() err = %v", err)
			}
			if got := param.Description.Value; got != tt.want {
				t.Errorf("tool description = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestMediaResolution_SetsTheDetail(t *testing.T) {
	high := &genai.PartMediaResolution{Level: genai.PartMediaResolutionLevelMediaResolutionHigh}
	low := &genai.PartMediaResolution{Level: genai.PartMediaResolutionLevelMediaResolutionLow}
	tests := []struct {
		name    string
		request genai.MediaResolution
		part    *genai.PartMediaResolution
		want    string
	}{
		{name: "unset", want: "auto"},
		{name: "request high", request: genai.MediaResolutionHigh, want: "high"},
		{name: "request low", request: genai.MediaResolutionLow, want: "low"},
		{name: "request medium", request: genai.MediaResolutionMedium, want: "auto"},
		{name: "the part's own outranks the request's", request: genai.MediaResolutionHigh, part: low, want: "low"},
		{name: "part high", part: high, want: "high"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &model.LLMRequest{
				Contents: []*genai.Content{{Role: genai.RoleUser, Parts: []*genai.Part{
					{InlineData: &genai.Blob{MIMEType: "application/pdf", Data: []byte("%PDF-1.7 packing slip")}, MediaResolution: tt.part},
					{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("\x89PNG label")}, MediaResolution: tt.part},
					genai.NewPartFromText("Which SKUs are on this slip?"),
				}}},
				Config: &genai.GenerateContentConfig{MediaResolution: tt.request},
			}
			params, err := buildParamsWith("gpt-test", req, Options{})
			if err != nil {
				t.Fatalf("buildParamsWith() err = %v", err)
			}
			content := params.Input.OfInputItemList[0].OfMessage.Content.OfInputItemContentList
			if got := string(content[0].OfInputFile.Detail); got != tt.want {
				t.Errorf("file detail = %q, want %q", got, tt.want)
			}
			if got := string(content[1].OfInputImage.Detail); got != tt.want {
				t.Errorf("image detail = %q, want %q", got, tt.want)
			}
		})
	}
}
