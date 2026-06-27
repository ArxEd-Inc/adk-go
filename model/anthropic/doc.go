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

// Package anthropic implements the [model.LLM] interface for Anthropic Claude
// models via Google Cloud Vertex AI.
//
// It is vendored from github.com/Alcova-AI/adk-anthropic-go (v0.1.18, Apache
// 2.0) and adapted to the ADK's model/<provider> layout, then corrected for the
// latest Claude models and for Vertex AI: thinking is adaptive with a per-model
// effort for adaptive-capable models, or a budget_tokens form for models that
// reject adaptive thinking (selected per model via Config.ThinkingMode); tool-use
// IDs are sanitized to Anthropic's required shape; tool input schemas resolve a root
// $ref and alias over-long top-level property keys; non-streaming requests are
// issued as streaming internally (Vertex rejects large non-streaming calls); the
// refusal stop reason is surfaced; and redacted thinking is round-tripped
// faithfully (its data rides in the thought signature), while a thought that
// cannot be replayed is an error rather than silently dropped. Both the Vertex AI
// and direct Anthropic API backends are selectable via [Config] (Variant /
// ANTHROPIC_USE_VERTEX), but only the Vertex AI path is exercised.
//
// TODO(#225): replace this package with the upstream
// google.golang.org/adk/model/anthropic once google/adk-go merges Anthropic
// support (PR #598 / #233). https://github.com/google/adk-go/issues/225
package anthropic
