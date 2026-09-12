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

package converters

import (
	"slices"
	"testing"

	"google.golang.org/genai"
)

// markedPart returns the given part with PartCacheBreakpointMetadataKey set.
func markedPart(part *genai.Part) *genai.Part {
	part.PartMetadata = map[string]any{PartCacheBreakpointMetadataKey: true}
	return part
}

// TestContentsToMessagesWithMarkedBlocks_OrdinalsIndexTheWireBlocks pins that
// a marked part's reported ordinal indexes the block it converts to in the
// flattened wire messages: parts that yield no block (nil, empty) are skipped,
// and a same-role merge moves a later content's blocks into the preceding
// message without renumbering them.
func TestContentsToMessagesWithMarkedBlocks_OrdinalsIndexTheWireBlocks(t *testing.T) {
	contents := []*genai.Content{
		{Role: "user", Parts: []*genai.Part{
			nil,
			{Text: "Field guide, pages 12-14:"},
			{}, // yields no block
			markedPart(&genai.Part{InlineData: &genai.Blob{MIMEType: "application/pdf", Data: []byte("%PDF-1.4")}}),
		}},
		// Same role as the previous content: merged into it on the wire.
		{Role: "user", Parts: []*genai.Part{
			markedPart(&genai.Part{Text: "Specimen notes for plot 7."}),
			{Text: "Which species is this?"},
		}},
		{Role: "model", Parts: []*genai.Part{{Text: "Looking at the plates."}}},
	}

	messages, ordinals, err := ContentsToMessagesWithMarkedBlocks(contents)
	if err != nil {
		t.Fatalf("ContentsToMessagesWithMarkedBlocks: %v", err)
	}
	if len(messages) != 2 {
		t.Fatalf("messages = %d, want 2 (the two user contents merged)", len(messages))
	}
	if got := len(messages[0].Content); got != 4 {
		t.Fatalf("merged user message blocks = %d, want 4", got)
	}
	if want := []int{1, 2}; !slices.Equal(ordinals, want) {
		t.Fatalf("marked block ordinals = %v, want %v", ordinals, want)
	}
	if messages[0].Content[1].OfDocument == nil {
		t.Errorf("ordinal 1 names block %+v, want the marked document block", messages[0].Content[1])
	}
	if text := messages[0].Content[2].OfText; text == nil || text.Text != "Specimen notes for plot 7." {
		t.Errorf("ordinal 2 names block %+v, want the marked notes text block", messages[0].Content[2])
	}
}

// TestContentsToMessagesWithMarkedBlocks_NoMarksReportsNil pins the nil
// ordinals of a request with no marked part, and that ContentsToMessages
// converts identically.
func TestContentsToMessagesWithMarkedBlocks_NoMarksReportsNil(t *testing.T) {
	contents := []*genai.Content{genai.NewContentFromText("Which species is this?", "user")}

	messages, ordinals, err := ContentsToMessagesWithMarkedBlocks(contents)
	if err != nil {
		t.Fatalf("ContentsToMessagesWithMarkedBlocks: %v", err)
	}
	if ordinals != nil {
		t.Errorf("marked block ordinals = %v, want nil", ordinals)
	}
	plainMessages, err := ContentsToMessages(contents)
	if err != nil {
		t.Fatalf("ContentsToMessages: %v", err)
	}
	if len(plainMessages) != len(messages) || len(plainMessages[0].Content) != len(messages[0].Content) {
		t.Errorf("ContentsToMessages = %+v, want the same messages as the marked-blocks variant %+v",
			plainMessages, messages)
	}
}

// TestIsCacheBreakpointMarked pins the marker predicate's nil safety and its
// insistence on a true boolean under the key.
func TestIsCacheBreakpointMarked(t *testing.T) {
	cases := []struct {
		name string
		part *genai.Part
		want bool
	}{
		{name: "nil part", part: nil},
		{name: "no metadata", part: &genai.Part{Text: "x"}},
		{name: "other key", part: &genai.Part{PartMetadata: map[string]any{"other": true}}},
		{name: "false", part: &genai.Part{PartMetadata: map[string]any{PartCacheBreakpointMetadataKey: false}}},
		{name: "non-boolean", part: &genai.Part{PartMetadata: map[string]any{PartCacheBreakpointMetadataKey: "true"}}},
		{name: "true", part: &genai.Part{PartMetadata: map[string]any{PartCacheBreakpointMetadataKey: true}}, want: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsCacheBreakpointMarked(tc.part); got != tc.want {
				t.Errorf("IsCacheBreakpointMarked = %v, want %v", got, tc.want)
			}
		})
	}
}
