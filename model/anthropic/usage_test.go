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

package anthropic

import (
	"testing"

	"google.golang.org/adk/v2/model"
)

// TestCacheCreationInputTokensFromResponse covers the accessor across the value shapes CustomMetadata can carry:
// the int64s the converter writes, the float64s they become after a JSON round-trip through a session store, and
// the absent/malformed cases that must report ok=false.
func TestCacheCreationInputTokensFromResponse(t *testing.T) {
	cases := []struct {
		name       string
		resp       *model.LLMResponse
		wantTokens CacheCreationInputTokens
		wantOK     bool
	}{
		{name: "nil response", resp: nil, wantOK: false},
		{name: "no custom metadata", resp: &model.LLMResponse{}, wantOK: false},
		{
			name: "int64 values as written by the converter",
			resp: &model.LLMResponse{CustomMetadata: map[string]any{
				CacheCreationInputTokensKey:            int64(300),
				CacheCreationEphemeral5mInputTokensKey: int64(200),
				CacheCreationEphemeral1hInputTokensKey: int64(100),
			}},
			wantTokens: CacheCreationInputTokens{TotalInputTokens: 300, Ephemeral5mInputTokens: 200, Ephemeral1hInputTokens: 100},
			wantOK:     true,
		},
		{
			name: "float64 values after a JSON round-trip",
			resp: &model.LLMResponse{CustomMetadata: map[string]any{
				CacheCreationInputTokensKey:            float64(300),
				CacheCreationEphemeral5mInputTokensKey: float64(200),
				CacheCreationEphemeral1hInputTokensKey: float64(100),
			}},
			wantTokens: CacheCreationInputTokens{TotalInputTokens: 300, Ephemeral5mInputTokens: 200, Ephemeral1hInputTokens: 100},
			wantOK:     true,
		},
		{
			name: "total without per-TTL breakdown",
			resp: &model.LLMResponse{CustomMetadata: map[string]any{
				CacheCreationInputTokensKey: int64(300),
			}},
			wantTokens: CacheCreationInputTokens{TotalInputTokens: 300},
			wantOK:     true,
		},
		{
			name:   "unrelated metadata only",
			resp:   &model.LLMResponse{CustomMetadata: map[string]any{"other": int64(1)}},
			wantOK: false,
		},
		{
			name: "non-numeric total",
			resp: &model.LLMResponse{CustomMetadata: map[string]any{
				CacheCreationInputTokensKey: "300",
			}},
			wantOK: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tokens, ok := CacheCreationInputTokensFromResponse(tc.resp)
			if ok != tc.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tc.wantOK)
			}
			if tokens != tc.wantTokens {
				t.Errorf("tokens = %+v, want %+v", tokens, tc.wantTokens)
			}
		})
	}
}
