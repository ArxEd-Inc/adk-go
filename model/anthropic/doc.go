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
// latest Claude models: thinking is adaptive-only with a per-model effort (there
// is no legacy budget_tokens path), tool-use IDs are sanitized to Anthropic's
// required shape, and the refusal stop reason is surfaced. The direct Anthropic
// API backend is reserved on [Config] but not wired up; only Vertex AI is
// supported today.
//
// TODO(#225): replace this package with the upstream
// google.golang.org/adk/model/anthropic once google/adk-go merges Anthropic
// support (PR #598 / #233). https://github.com/google/adk-go/issues/225
package anthropic
