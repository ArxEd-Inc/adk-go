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
	"bytes"
	"encoding/json"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
)

// repairAccumulatedToolInput mends a tool_use content block whose streamed input is not
// valid JSON, just before its content_block_stop event is folded into the accumulated
// message. Claude sometimes writes JavaScript's bare Infinity/-Infinity/NaN literals in
// tool-call JSON; [anthropicsdk.Message.Accumulate] concatenates input_json_delta
// fragments without validation and then rejects the block at content_block_stop, which
// kills the whole stream — the turn is lost before the tool call ever reaches tool
// dispatch, where malformed arguments would have produced a model-visible error the
// model can correct on its next turn. Rewriting the sentinels as their quoted JSON
// string forms restores that path: a tool that gives the sentinel strings meaning
// accepts them, and any other tool rejects them as an ordinary argument-validation
// error — either way recoverable, where today's failure is not. Input the rewrite
// cannot make valid is left untouched so the accumulator's original error still
// surfaces.
func repairAccumulatedToolInput(message *anthropicsdk.Message, event anthropicsdk.MessageStreamEventUnion) {
	stop, ok := event.AsAny().(anthropicsdk.ContentBlockStopEvent)
	if !ok {
		return
	}
	if stop.Index < 0 || stop.Index >= int64(len(message.Content)) {
		return
	}
	block := &message.Content[stop.Index]
	if block.Type != "tool_use" || len(block.Input) == 0 || json.Valid(block.Input) {
		return
	}
	if repaired, changed := quoteBareJSONSentinels(block.Input); changed && json.Valid(repaired) {
		block.Input = repaired
	}
}

// quoteBareJSONSentinels rewrites bare Infinity, -Infinity, and NaN value tokens in b
// to their quoted JSON-string forms, leaving string-literal contents untouched — a
// notes field whose text mentions Infinity must not change. It reports whether it
// rewrote anything. Token-boundary checks keep longer bare words (e.g. InfinityCap)
// intact; the caller's post-rewrite json.Valid check is the final guard against any
// rewrite this scan gets wrong.
func quoteBareJSONSentinels(b []byte) ([]byte, bool) {
	var out bytes.Buffer
	out.Grow(len(b) + 2*3)
	changed := false
	inString := false
	escaped := false
	for i := 0; i < len(b); i++ {
		c := b[i]
		if inString {
			out.WriteByte(c)
			switch {
			case escaped:
				escaped = false
			case c == '\\':
				escaped = true
			case c == '"':
				inString = false
			}
			continue
		}
		switch {
		case c == '"':
			inString = true
			out.WriteByte(c)
		case c == '-' && bareTokenAt(b, i+1, "Infinity"):
			out.WriteString(`"-Infinity"`)
			i += len("Infinity")
			changed = true
		case bareTokenAt(b, i, "Infinity"):
			out.WriteString(`"Infinity"`)
			i += len("Infinity") - 1
			changed = true
		case bareTokenAt(b, i, "NaN"):
			out.WriteString(`"NaN"`)
			i += len("NaN") - 1
			changed = true
		default:
			out.WriteByte(c)
		}
	}
	return out.Bytes(), changed
}

// bareTokenAt reports whether token occupies b[i:i+len(token)] with non-token bytes (or
// the buffer edge) on both sides.
func bareTokenAt(b []byte, i int, token string) bool {
	if i+len(token) > len(b) || string(b[i:i+len(token)]) != token {
		return false
	}
	if i > 0 && isTokenByte(b[i-1]) {
		return false
	}
	if end := i + len(token); end < len(b) && isTokenByte(b[end]) {
		return false
	}
	return true
}

func isTokenByte(c byte) bool {
	return c == '_' || ('0' <= c && c <= '9') || ('A' <= c && c <= 'Z') || ('a' <= c && c <= 'z')
}
