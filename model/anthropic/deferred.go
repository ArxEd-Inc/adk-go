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

import "google.golang.org/adk/v2/model"

// DeferredLoadingTool is implemented by a request's dispatch entry (a value of
// model.LLMRequest.Tools, keyed by the tool's name) to have the tool's
// declaration sent with defer_loading. Anthropic keeps a deferred definition
// out of the prompt, and out of the cache prefix the tools array forms, until a
// tool_reference block in the conversation names it (see
// Config.ToolReferencesResponseKey), so a conversation can declare a large
// catalog up front and make its tools callable later without changing the
// tools array. The signal rides on the dispatch entry rather than on the
// declaration, which is left as the tool built it. At least one tool of a
// request must stay non-deferred, and a deferred tool cannot carry a cache
// breakpoint; the Tools and ToolsStaticPrefixEnd breakpoints skip deferred
// tools.
type DeferredLoadingTool interface {
	// DeferLoading reports whether the tool's declaration is sent with
	// defer_loading.
	DeferLoading() bool
}

// IsToolLoadingDeferred reports whether the request's dispatch entry for the
// named tool implements DeferredLoadingTool and defers; nil-safe.
func IsToolLoadingDeferred(req *model.LLMRequest, name string) bool {
	if req == nil {
		return false
	}
	entry, ok := req.Tools[name].(DeferredLoadingTool)
	return ok && entry.DeferLoading()
}
