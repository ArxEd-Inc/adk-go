// Copyright 2025 Google LLC
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
	"context"
	"errors"
	"net/http"

	openaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/openai/internal/responses"
)

// Config configures a model talking to OpenAI's Responses API.
type Config struct {
	// APIKey is the OpenAI API key. It is required: unlike openai-go's own
	// default, this package never falls back to the OPENAI_API_KEY
	// environment variable, so a key meant for one project cannot be picked
	// up in place of another's.
	APIKey string

	// BaseURL points the model at another server implementing the Responses
	// API, such as a test fake. Empty uses openai-go's default.
	BaseURL string

	// HTTPClient, when set, carries every request.
	HTTPClient *http.Client

	// Options is an escape hatch for advanced openai-go request options,
	// appended after the options derived from the fields above.
	Options []option.RequestOption

	// DefaultMaxTokens caps the output of a request that sets no
	// MaxOutputTokens of its own. Zero leaves such a request uncapped, which
	// lets the model use its whole output limit.
	DefaultMaxTokens int

	// Effort, when set, is the reasoning effort of every request, in place of
	// the one a request's ThinkingConfig resolves to. genai's thinking levels
	// stop at High, so this is how a model reaches EffortXHigh or EffortMax.
	Effort Effort

	// PromptCaching places explicit prompt cache breakpoints on top of the
	// API's implicit one (see PromptCachingConfig). Nil places none.
	PromptCaching *PromptCachingConfig
}

// ErrAPIKeyRequired is returned by NewModel when Config.APIKey is empty.
var ErrAPIKeyRequired = errors.New("openai: an API key is required")

// FinishMessageKey is the [model.LLMResponse.CustomMetadata] key under which a
// turn that ended badly but still carries an answer reports the provider's own
// account of why — a content filter's "content_filter", an incomplete reason
// this package does not map, or a server error message. Its FinishReason says
// the turn was cut short; this says what the provider called it.
//
// A turn left with nothing to read reports the same wording in ErrorCode and
// ErrorMessage instead, which callers treat as a failed turn.
//
// It reaches in-process callers, session storage and A2A metadata, but not
// REST: server/adkrest maps events field by field and omits CustomMetadata, so
// an ADK Web consumer sees the FinishReason alone.
const FinishMessageKey = "openai_finish_message"

// NewModel constructs a model talking to the Responses API.
//
// Requests are never stored (store is false). openai-go's own retries are
// turned off, since a caller wrapping the model in a retry of its own would
// otherwise multiply the two.
//
// The context is unused but kept for signature parity with other model constructors (e.g., gemini.NewModel).
func NewModel(_ context.Context, modelName string, cfg *Config) (model.LLM, error) {
	if modelName == "" {
		return nil, ErrModelNameRequired
	}
	if cfg == nil || cfg.APIKey == "" {
		return nil, ErrAPIKeyRequired
	}
	opts := []option.RequestOption{option.WithAPIKey(cfg.APIKey), option.WithMaxRetries(0)}
	if cfg.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(cfg.BaseURL))
	}
	if cfg.HTTPClient != nil {
		opts = append(opts, option.WithHTTPClient(cfg.HTTPClient))
	}
	opts = append(opts, cfg.Options...)
	client := openaisdk.NewClient(opts...)
	return responses.New(&client, modelName, responses.Options{
		Effort:           cfg.Effort,
		DefaultMaxTokens: int64(cfg.DefaultMaxTokens),
		PromptCaching:    cfg.PromptCaching,
	}), nil
}
