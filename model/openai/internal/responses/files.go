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

package responses

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/openai/openai-go/v3"
	"golang.org/x/sync/singleflight"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/openai/internal/shared"
)

const (
	// defaultFilesExpiresIn is FilesConfig.ExpiresIn's default.
	defaultFilesExpiresIn = 24 * time.Hour

	// defaultFilesReuseMargin is FilesConfig.ReuseMargin's default.
	defaultFilesReuseMargin = time.Hour

	// minFilesExpiresIn and maxFilesExpiresIn bound FilesConfig.ExpiresIn to
	// the range the Files API accepts for expires_after.seconds.
	minFilesExpiresIn = time.Hour
	maxFilesExpiresIn = 30 * 24 * time.Hour
)

// filesConfigWithDefaults returns cfg with its zero durations replaced by
// their defaults, or an error when the result is out of range.
func filesConfigWithDefaults(cfg shared.FilesConfig) (shared.FilesConfig, error) {
	if cfg.ExpiresIn == 0 {
		cfg.ExpiresIn = defaultFilesExpiresIn
	}
	if cfg.ReuseMargin == 0 {
		cfg.ReuseMargin = defaultFilesReuseMargin
	}
	if cfg.ExpiresIn < minFilesExpiresIn || cfg.ExpiresIn > maxFilesExpiresIn {
		return shared.FilesConfig{}, fmt.Errorf("openai: Files.ExpiresIn %s is outside [%s, %s]", cfg.ExpiresIn, minFilesExpiresIn, maxFilesExpiresIn)
	}
	if cfg.ReuseMargin < 0 || cfg.ReuseMargin >= cfg.ExpiresIn {
		return shared.FilesConfig{}, fmt.Errorf("openai: Files.ReuseMargin %s must be non-negative and shorter than Files.ExpiresIn %s", cfg.ReuseMargin, cfg.ExpiresIn)
	}
	if cfg.MaxInlineBytes < 0 {
		return shared.FilesConfig{}, fmt.Errorf("openai: Files.MaxInlineBytes %d is negative", cfg.MaxInlineBytes)
	}
	return cfg, nil
}

// uploadedFile is one memo entry: a file the Files API holds for a document.
type uploadedFile struct {
	id        string
	expiresAt time.Time
}

// fileMemo maps a document key (see documentKey) to the file uploaded for it.
// It is shared by every model in the process, so models serving one
// conversation's agents upload a document once between them.
var fileMemo = struct {
	sync.Mutex
	byKey map[string]uploadedFile
}{byKey: map[string]uploadedFile{}}

// fileUploads collapses concurrent uploads of one document key into one.
var fileUploads singleflight.Group

// filesResolver resolves a request's inline documents to Files API file IDs
// for one model.
type filesResolver struct {
	cfg   shared.FilesConfig
	files *openai.FileService

	// identity distinguishes this model's API key and base URL in document
	// keys, so models of different projects never share a file ID.
	identity string

	// now is the clock; tests replace it.
	now func() time.Time
}

// newFilesResolver returns a resolver uploading through client, or an error
// when cfg is invalid.
func newFilesResolver(cfg shared.FilesConfig, client *openai.Client, identity string) (*filesResolver, error) {
	cfg, err := filesConfigWithDefaults(cfg)
	if err != nil {
		return nil, err
	}
	return &filesResolver{cfg: cfg, files: &client.Files, identity: identity, now: time.Now}, nil
}

// referenceableExtensions maps the MIME types a Files API reference can stand
// in for to the extension their uploads are named with.
var referenceableExtensions = map[string]string{
	"application/pdf": ".pdf",
	"image/jpeg":      ".jpg",
	"image/png":       ".png",
	"image/gif":       ".gif",
	"image/webp":      ".webp",
}

// resolve returns a file ID for each referenceable inline document in req,
// keyed by its blob, uploading those the memo cannot supply. A document whose
// upload fails is left out and stays inline. It fails, with an error wrapping
// ErrRequestTooLarge, only when the media left inline exceeds MaxInlineBytes.
func (r *filesResolver) resolve(ctx context.Context, req *model.LLMRequest) (map[*genai.Blob]string, error) {
	type pending struct {
		blob *genai.Blob
		key  string
	}
	var docs []pending
	var inlineBytes int
	for _, content := range req.Contents {
		if content == nil {
			continue
		}
		for _, part := range content.Parts {
			if part == nil || part.InlineData == nil {
				continue
			}
			mimeType := strings.ToLower(strings.TrimSpace(part.InlineData.MIMEType))
			if _, ok := referenceableExtensions[mimeType]; !ok {
				inlineBytes += len(part.InlineData.Data)
				continue
			}
			docs = append(docs, pending{blob: part.InlineData, key: r.documentKey(part.InlineData.Data, mimeType)})
		}
	}

	fileIDs := make(map[*genai.Blob]string, len(docs))
	uploadErrs := make([]error, len(docs))
	ids := make([]string, len(docs))
	var wg sync.WaitGroup
	for i, doc := range docs {
		wg.Go(func() {
			ids[i], uploadErrs[i] = r.fileIDFor(ctx, doc.key, doc.blob)
		})
	}
	wg.Wait()

	for i, doc := range docs {
		if uploadErrs[i] != nil {
			inlineBytes += len(doc.blob.Data)
			continue
		}
		fileIDs[doc.blob] = ids[i]
	}
	if r.cfg.MaxInlineBytes > 0 && inlineBytes > r.cfg.MaxInlineBytes {
		return nil, fmt.Errorf("%w: %d bytes of media would be sent inline, over the %d allowed: %w",
			shared.ErrRequestTooLarge, inlineBytes, r.cfg.MaxInlineBytes, errors.Join(uploadErrs...))
	}
	return fileIDs, nil
}

// documentKey returns the memo key for a document: this resolver's identity,
// the MIME type, and the SHA-256 of the bytes.
func (r *filesResolver) documentKey(data []byte, mimeType string) string {
	sum := sha256.Sum256(data)
	return r.identity + "/" + mimeType + "/" + hex.EncodeToString(sum[:])
}

// fileIDFor returns the ID of a file holding blob's bytes with at least
// ReuseMargin of life left, uploading one when the memo has none.
func (r *filesResolver) fileIDFor(ctx context.Context, key string, blob *genai.Blob) (string, error) {
	if id, ok := r.memoized(key); ok {
		return id, nil
	}
	result, err, _ := fileUploads.Do(key, func() (any, error) {
		// A concurrent caller may have finished the upload between the memo
		// check above and joining this flight.
		if id, ok := r.memoized(key); ok {
			return id, nil
		}
		return r.upload(ctx, key, blob)
	})
	if err != nil {
		return "", err
	}
	return result.(string), nil
}

// memoized returns the memo's file ID for key when its file has at least
// ReuseMargin of life left.
func (r *filesResolver) memoized(key string) (string, bool) {
	fileMemo.Lock()
	defer fileMemo.Unlock()
	file, ok := fileMemo.byKey[key]
	if !ok || file.expiresAt.Sub(r.now()) < r.cfg.ReuseMargin {
		return "", false
	}
	return file.id, true
}

// upload uploads blob's bytes as a user_data file expiring after ExpiresIn,
// records it in the memo under key, and returns its ID. The upload is named
// after the content hash, so it carries nothing of the document's own name.
func (r *filesResolver) upload(ctx context.Context, key string, blob *genai.Blob) (string, error) {
	mimeType := strings.ToLower(strings.TrimSpace(blob.MIMEType))
	sum := key[strings.LastIndex(key, "/")+1:]
	uploadedAt := r.now()
	file, err := r.files.New(ctx, openai.FileNewParams{
		File:         openai.File(bytes.NewReader(blob.Data), sum+referenceableExtensions[mimeType], mimeType),
		Purpose:      openai.FilePurposeUserData,
		ExpiresAfter: openai.FileNewParamsExpiresAfter{Seconds: int64(r.cfg.ExpiresIn / time.Second)},
	})
	if err != nil {
		return "", fmt.Errorf("openai: upload document: %w", err)
	}
	expiresAt := uploadedAt.Add(r.cfg.ExpiresIn)
	if file.ExpiresAt > 0 {
		expiresAt = time.Unix(file.ExpiresAt, 0)
	}

	fileMemo.Lock()
	defer fileMemo.Unlock()
	now := r.now()
	for memoKey, memoFile := range fileMemo.byKey {
		if !memoFile.expiresAt.After(now) {
			delete(fileMemo.byKey, memoKey)
		}
	}
	fileMemo.byKey[key] = uploadedFile{id: file.ID, expiresAt: expiresAt}
	return file.ID, nil
}

// forget drops the memo entries holding any of fileIDs, so the next
// resolution uploads those documents afresh.
func (r *filesResolver) forget(fileIDs map[*genai.Blob]string) {
	stale := make(map[string]bool, len(fileIDs))
	for _, id := range fileIDs {
		stale[id] = true
	}
	fileMemo.Lock()
	defer fileMemo.Unlock()
	for key, file := range fileMemo.byKey {
		if stale[file.id] {
			delete(fileMemo.byKey, key)
		}
	}
}

// isMissingFileError reports whether err is the API refusing a request that
// references a file it cannot find: deleted before it expired, say, or held by
// another project than the key now serving the request. The API states such a
// refusal as a 404, or as a 400 naming a file; a retry on a 400 that names a
// file for another reason costs one re-upload.
func isMissingFileError(err error) bool {
	apiErr, ok := errors.AsType[*openai.Error](err)
	if !ok {
		return false
	}
	switch apiErr.StatusCode {
	case http.StatusNotFound:
		return true
	case http.StatusBadRequest:
		return strings.Contains(apiErr.Param, "file") || strings.Contains(strings.ToLower(apiErr.Message), "file")
	default:
		return false
	}
}
