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

package shared

// ServerError is a failure the server reported after answering HTTP 200: an
// error event in a stream, or a response whose own status is failed. Code is
// the server's error code, such as "rate_limit_exceeded" or "server_error",
// which a caller deciding whether to retry needs and the rendered message
// alone would bury.
//
// Error and Unwrap defer to the rendered error, so its text is unchanged and a
// failed response still matches ErrResponseFailed with errors.Is.
type ServerError struct {
	// Code is the server's error code, or "" when it stated none.
	Code string
	// Message is the server's message, clipped as ClipServerText clips it.
	Message string

	rendered error
}

// NewServerError returns a ServerError with the given code and message that
// renders as rendered.
func NewServerError(code, message string, rendered error) *ServerError {
	return &ServerError{Code: ClipServerText(code), Message: ClipServerText(message), rendered: rendered}
}

func (e *ServerError) Error() string { return e.rendered.Error() }

func (e *ServerError) Unwrap() error { return e.rendered }
