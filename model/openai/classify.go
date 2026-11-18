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
	"errors"
	"net/http"

	openaisdk "github.com/openai/openai-go/v3"
)

// contextLengthExceededCode is the error code the API reports for a request
// whose input does not fit the model's context window.
const contextLengthExceededCode = "context_length_exceeded"

// IsContextWindowError reports whether err is the API refusing a request whose
// input does not fit the model's context window, whether it arrived as an HTTP
// error or as a failure reported inside a stream.
func IsContextWindowError(err error) bool {
	return errorCode(err) == contextLengthExceededCode
}

// IsRequestTooLargeError reports whether err is a request refused for its
// size before the model saw it: by this package, for carrying more inline
// media than FilesConfig.MaxInlineBytes allows, or by the API.
func IsRequestTooLargeError(err error) bool {
	if errors.Is(err, ErrRequestTooLarge) {
		return true
	}
	apiErr, ok := errors.AsType[*openaisdk.Error](err)
	return ok && apiErr.StatusCode == http.StatusRequestEntityTooLarge
}

// errorCode is the server's error code err carries, from an HTTP error or from
// a failure reported inside a stream, or "" when it carries none.
func errorCode(err error) string {
	if apiErr, ok := errors.AsType[*openaisdk.Error](err); ok {
		return apiErr.Code
	}
	if serverErr, ok := errors.AsType[*ServerError](err); ok {
		return serverErr.Code
	}
	return ""
}
