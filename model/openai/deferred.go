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
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/openai/internal/shared"
)

// DeferredLoadingTool is implemented by a request's dispatch entry (a value of
// model.LLMRequest.Tools, keyed by the tool's name) to keep the tool's
// declaration out of the request's tools until a function response names it
// under Config.ToolReferencesResponseKey. A conversation can then hold a large
// catalog of tools and make each callable when it is needed, without the tools
// array changing and the prompt cache's prefix with it. The signal rides on
// the dispatch entry rather than on the declaration, which is left as the tool
// built it.
type DeferredLoadingTool = shared.DeferredLoadingTool

// IsToolLoadingDeferred reports whether the request's dispatch entry for the
// named tool implements DeferredLoadingTool and defers; nil-safe.
func IsToolLoadingDeferred(req *model.LLMRequest, name string) bool {
	return shared.IsToolLoadingDeferred(req, name)
}
