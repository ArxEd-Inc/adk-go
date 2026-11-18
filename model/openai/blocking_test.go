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
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/openai/openai-go/v3/option"
)

// blockingAsStream lets a test server that answers with a blocking response
// body serve the streamed request this package always sends: a 200 JSON body
// is re-served as a stream of one "response.completed" event carrying it,
// which the stream path converts the way a blocking body converts.
var blockingAsStream = option.WithMiddleware(func(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
	resp, err := next(req)
	if err != nil || resp.StatusCode != http.StatusOK ||
		!strings.HasPrefix(resp.Header.Get("Content-Type"), "application/json") {
		return resp, err
	}
	body, err := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if err != nil {
		return nil, err
	}
	stream := fmt.Sprintf("data: {\"type\":\"response.completed\",\"response\":%s}\n\ndata: [DONE]\n\n", body)
	resp.Body = io.NopCloser(strings.NewReader(stream))
	resp.Header.Set("Content-Type", "text/event-stream")
	resp.ContentLength = int64(len(stream))
	return resp, nil
})
