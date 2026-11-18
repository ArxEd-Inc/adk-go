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

import "google.golang.org/adk/v2/model/openai/internal/responses"

// ThoughtSignaturePrefix begins the ThoughtSignature of every thought part this
// package produces. Another provider's model package can recognize such a
// part by it and leave the part out of its own requests, since neither the
// signature nor the reasoning behind it means anything to that provider.
const ThoughtSignaturePrefix = responses.ThoughtSignaturePrefix

// ReasoningSignaturePrefix begins the ThoughtSignature of a thought part that
// carries an OpenAI reasoning item for replay. It begins with
// ThoughtSignaturePrefix.
const ReasoningSignaturePrefix = responses.ReasoningSignaturePrefix
