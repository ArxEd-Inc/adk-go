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

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"golang.org/x/sync/singleflight"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

const (
	// defaultFilesExpiresIn is FilesConfig.ExpiresIn's default.
	defaultFilesExpiresIn = 24 * time.Hour

	// defaultFilesReuseMargin is FilesConfig.ReuseMargin's default.
	defaultFilesReuseMargin = time.Hour

	// minFilesExpiresIn and maxFilesExpiresIn bound FilesConfig.ExpiresIn to
	// the range the Files API accepts for expires_in_seconds.
	minFilesExpiresIn = time.Hour
	maxFilesExpiresIn = 90 * 24 * time.Hour
)

// FilesConfig configures sending a request's inline PDFs and images as Files
// API references instead of base64. A reference carries a document of any size
// in a few dozen bytes, so a request's documents no longer count toward the
// Messages API's limit on request size, only toward the context window. The
// prompt cache keys a document on its content rather than on how it is sent,
// so a document's cache entry survives a switch between inline and reference
// and between two uploads of the same bytes.
//
// Each distinct document (by content and MIME type) is uploaded once per
// process and API key, and reused while its file has at least ReuseMargin of
// life left; concurrent requests needing the same document share one upload.
// Upload state lives in memory only: another process re-uploads, which costs
// the upload and nothing else, since the cache does not depend on the file ID.
// A document whose upload fails is sent inline, as it would be without this
// config. Files are only available on the direct Anthropic API.
type FilesConfig struct {
	// ExpiresIn is how long each uploaded file lives before the Files API
	// deletes it. Defaults to 24 hours when zero; must lie between one hour and
	// ninety days.
	ExpiresIn time.Duration

	// ReuseMargin is the least remaining life an uploaded file must have to be
	// referenced again, so a request never references a file that expires
	// before the request is served. Defaults to one hour when zero; must be
	// shorter than ExpiresIn.
	ReuseMargin time.Duration

	// MaxInlineBytes caps the raw bytes of documents a request may still carry
	// inline after resolution — the documents whose upload failed. A request
	// over the cap fails with an error a caller may retry, rather than being
	// sent with a body the Messages API would reject as too large. Zero means
	// no cap.
	MaxInlineBytes int
}

// withDefaults returns cfg with its zero durations replaced by their defaults,
// or an error when the result is out of range.
func (cfg FilesConfig) withDefaults() (FilesConfig, error) {
	if cfg.ExpiresIn == 0 {
		cfg.ExpiresIn = defaultFilesExpiresIn
	}
	if cfg.ReuseMargin == 0 {
		cfg.ReuseMargin = defaultFilesReuseMargin
	}
	if cfg.ExpiresIn < minFilesExpiresIn || cfg.ExpiresIn > maxFilesExpiresIn {
		return FilesConfig{}, fmt.Errorf("Files.ExpiresIn %s is outside [%s, %s]", cfg.ExpiresIn, minFilesExpiresIn, maxFilesExpiresIn)
	}
	if cfg.ReuseMargin < 0 || cfg.ReuseMargin >= cfg.ExpiresIn {
		return FilesConfig{}, fmt.Errorf("Files.ReuseMargin %s must be non-negative and shorter than Files.ExpiresIn %s", cfg.ReuseMargin, cfg.ExpiresIn)
	}
	if cfg.MaxInlineBytes < 0 {
		return FilesConfig{}, fmt.Errorf("Files.MaxInlineBytes %d is negative", cfg.MaxInlineBytes)
	}
	return cfg, nil
}

// uploadedFile is one memo entry: a file the Files API holds for a document.
type uploadedFile struct {
	id        string
	expiresAt time.Time
}

// fileMemo maps a document key (see documentKey) to the file uploaded for it.
// It is shared by every model in the process so that models serving one
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
	cfg   FilesConfig
	files *anthropicsdk.FileService

	// identity distinguishes this model's API key and base URL in document
	// keys, so models of different workspaces never share a file ID.
	identity string

	// now is the clock; tests replace it.
	now func() time.Time
}

// newFilesResolver returns a resolver uploading through client, or an error
// when cfg is invalid.
func newFilesResolver(cfg FilesConfig, client *anthropicsdk.Client, apiKey, baseURL string) (*filesResolver, error) {
	cfg, err := cfg.withDefaults()
	if err != nil {
		return nil, err
	}
	identity := sha256.Sum256([]byte(apiKey + "\x00" + baseURL))
	return &filesResolver{
		cfg:      cfg,
		files:    &client.Files,
		identity: hex.EncodeToString(identity[:8]),
		now:      time.Now,
	}, nil
}

// referenceableExtensions maps the MIME types a Files API reference can stand
// in for to the file extension their uploads are named with.
var referenceableExtensions = map[string]string{
	"application/pdf": ".pdf",
	"image/jpeg":      ".jpg",
	"image/png":       ".png",
	"image/gif":       ".gif",
	"image/webp":      ".webp",
}

// resolve returns a file ID for each referenceable inline document in req,
// keyed by its blob, uploading those the memo cannot supply. A document whose
// upload fails is left out and stays inline. It fails only when the documents
// left inline exceed MaxInlineBytes.
func (r *filesResolver) resolve(ctx context.Context, req *model.LLMRequest) (map[*genai.Blob]string, error) {
	type pending struct {
		blob *genai.Blob
		key  string
	}
	var docs []pending
	for _, content := range req.Contents {
		if content == nil {
			continue
		}
		for _, part := range content.Parts {
			if part == nil || part.InlineData == nil {
				continue
			}
			mimeType := strings.ToLower(part.InlineData.MIMEType)
			if _, ok := referenceableExtensions[mimeType]; !ok {
				continue
			}
			docs = append(docs, pending{blob: part.InlineData, key: r.documentKey(part.InlineData.Data, mimeType)})
		}
	}
	if len(docs) == 0 {
		return nil, nil
	}

	fileIDs := make(map[*genai.Blob]string, len(docs))
	uploadErrs := make([]error, len(docs))
	ids := make([]string, len(docs))
	var wg sync.WaitGroup
	for i, doc := range docs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ids[i], uploadErrs[i] = r.fileIDFor(ctx, doc.key, doc.blob)
		}()
	}
	wg.Wait()

	var inlineBytes int
	for i, doc := range docs {
		if uploadErrs[i] != nil {
			inlineBytes += len(doc.blob.Data)
			continue
		}
		fileIDs[doc.blob] = ids[i]
	}
	if r.cfg.MaxInlineBytes > 0 && inlineBytes > r.cfg.MaxInlineBytes {
		return nil, fmt.Errorf("%d bytes of documents could not be uploaded and exceed the %d bytes a request may carry inline: %w",
			inlineBytes, r.cfg.MaxInlineBytes, errors.Join(uploadErrs...))
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

// upload uploads blob's bytes, records the file in the memo under key, and
// returns its ID. The upload is named after the content hash, so it carries
// nothing of the document's own name.
func (r *filesResolver) upload(ctx context.Context, key string, blob *genai.Blob) (string, error) {
	mimeType := strings.ToLower(blob.MIMEType)
	sum := key[strings.LastIndex(key, "/")+1:]
	uploadedAt := r.now()
	metadata, err := r.files.Upload(ctx, anthropicsdk.FileUploadParams{
		File:             anthropicsdk.File(bytes.NewReader(blob.Data), sum+referenceableExtensions[mimeType], mimeType),
		ExpiresInSeconds: anthropicsdk.Int(int64(r.cfg.ExpiresIn / time.Second)),
	})
	if err != nil {
		return "", fmt.Errorf("failed to upload document: %w", err)
	}
	expiresAt := metadata.ExpiresAt
	if expiresAt.IsZero() {
		expiresAt = uploadedAt.Add(r.cfg.ExpiresIn)
	}

	fileMemo.Lock()
	defer fileMemo.Unlock()
	now := r.now()
	for memoKey, file := range fileMemo.byKey {
		if !file.expiresAt.After(now) {
			delete(fileMemo.byKey, memoKey)
		}
	}
	fileMemo.byKey[key] = uploadedFile{id: metadata.ID, expiresAt: expiresAt}
	return metadata.ID, nil
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

// isMissingFileError reports whether err is the Messages API's 404
// not_found_error, which a request referencing files draws when one of them
// no longer exists — deleted before it expired, say, or held by another
// workspace than the key now serving the request.
func isMissingFileError(err error) bool {
	var apiErr *anthropicsdk.Error
	return errors.As(err, &apiErr) && apiErr.StatusCode == http.StatusNotFound &&
		strings.Contains(apiErr.RawJSON(), "not_found_error")
}
