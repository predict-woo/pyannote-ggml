# Consolidate `whisper-cpp-node` into `pyannote-cpp-node`

## Summary

Create `pyannote-cpp-node` as the single public package and make it the only supported import path for both:
- the existing low-level Whisper/VAD API now provided by `whisper-cpp-node`
- the existing high-level pipeline API now provided by `pyannote-cpp-node`

Chosen defaults:
- Preserve the `whisper-cpp-node` API shape at the top level.
- Deprecate `whisper-cpp-node` immediately; do not keep a shim package.
- Preserve platform parity now for the low-level transcription/VAD surface.
- Keep the merged package ESM-only.
- Treat high-level pipeline support as macOS-only until the native pipeline itself becomes cross-platform.

## Target Architecture

### 1. Public package layout

`bindings/node/packages/pyannote-cpp-node` becomes the sole public facade.

It exports, from one package:
- Low-level whisper API: `WhisperContext`, `VadContext`, `createWhisperContext`, `createVadContext`, `transcribe`, `transcribeAsync`, `getGpuDevices`
- High-level pipeline API: `Pipeline`, `PipelineSession`
- Shared types for both surfaces

Recommended export shape:
- Keep all whisper-compatible APIs at the package top level for drop-in migration.
- Keep `Pipeline` and `PipelineSession` at the top level.
- Add optional clarity subpaths `pyannote-cpp-node/whisper` and `pyannote-cpp-node/pipeline`, but do not require them.

### 2. Native addon contract

Standardize one native export contract across all platform packages:

- `WhisperContext`
- `VadContext`
- `transcribe`
- `getGpuDevices`
- `PipelineModel`
- `PipelineSession`
- `getCapabilities(): { whisper: boolean; vad: boolean; gpuDiscovery: boolean; pipeline: boolean; diarization: boolean }`

Rule:
- Every platform addon exports the same symbol names.
- Unsupported features are reported through `getCapabilities()` and by throwing clear runtime errors when invoked.

This avoids JS-side per-platform export branching and keeps typing stable.

### 3. Native source organization

Do not keep shipping logic split between `whisper.cpp/examples/addon.node` and `bindings/node/packages/darwin-arm64/src`.

Instead:
- Extract the whisper addon implementation into maintained sources under `bindings/node/native/whisper` or equivalent.
- Extract shared addon helpers/workers/options parsing into `bindings/node/native/common`.
- Keep pipeline-specific sources under `bindings/node/native/pipeline`.
- Build per-platform addons from these shared sources.

This avoids long-term copy-paste drift and stops treating `examples/addon.node` as production source.

### 4. Platform behavior

macOS:
- Build a superset addon that exports both low-level whisper APIs and high-level pipeline APIs.
- Continue linking against `diarization-ggml/build-static` as today.
- Preserve current CoreML switching behavior.

Windows:
- Build a low-level addon that exports whisper/VAD/GPU-discovery parity.
- Export `PipelineModel` and `PipelineSession`, but make them fail fast with an explicit unsupported-platform error.
- `getCapabilities().pipeline` and `getCapabilities().diarization` must be `false`.

This gives immediate low-level parity without pretending the macOS-only pipeline is portable.

## Packaging And Release

### 5. Package topology

Adopt the `whisper-cpp-node` packaging model inside `bindings/node`:

- `pyannote-cpp-node` as the pure JS/TS facade
- platform-specific optional dependency packages:
  - `@pyannote-cpp-node/darwin-arm64`
  - `@pyannote-cpp-node/darwin-x64`
  - `@pyannote-cpp-node/win32-x64`
  - `@pyannote-cpp-node/win32-ia32` if you still need to preserve that legacy surface

End-user installs must not compile native code.
Platform packages should contain prebuilt binaries, not `install: cmake-js build`.

### 6. Loader strategy

Replace the current macOS-only loader in `bindings/node/packages/pyannote-cpp-node/src/binding.ts` with the resilient loader pattern from `whisper-cpp-node`:

- map `platform-arch` to optional dependency package names
- support workspace-development binary resolution
- emit Windows-specific missing-DLL / Electron `asarUnpack` hints
- validate the standardized native contract after load

### 7. Sentry and debug-symbol handling

Standardize native staging around symbolicated builds:

- Prefer `RelWithDebInfo` over `Release` when staging binaries.
- On Windows, stage `.pdb` files alongside the built addon for CI artifact upload to Sentry.
- On macOS, generate and preserve `.dSYM` bundles for CI artifact upload.
- Do not publish debug-symbol artifacts inside npm packages unless you explicitly want that; publish/upload them separately.
- Replace the current ad hoc copy steps with one staging script in `bindings/node/scripts/` that:
  - picks `RelWithDebInfo` first
  - copies the `.node` binary into the correct platform package
  - copies Windows runtime/OpenVINO DLLs when present
  - records the exact staged artifact paths for symbol upload

## Public API Changes

### 8. Additions to `pyannote-cpp-node`

Add these public exports:
- `WhisperContext`
- `VadContext`
- `createWhisperContext`
- `createVadContext`
- `transcribe`
- `transcribeAsync`
- `getGpuDevices`
- whisper-compatible option/result types
- `getCapabilities`

### 9. Compatibility rules

Preserve from `whisper-cpp-node`:
- function names
- option names
- callback shapes
- result shapes
- top-level named exports

Do not preserve:
- CommonJS support
- the old package name

Migration rule for consumers:
- replace `from 'whisper-cpp-node'` with `from 'pyannote-cpp-node'`
- keep the rest of the low-level call sites unchanged unless they use `require(...)`

### 10. Existing pyannote exports

Keep existing pipeline exports intact:
- `Pipeline`
- `PipelineSession`

Recommended cleanup:
- keep `getBinding` / `Native*` exports working temporarily if they are already public
- remove them from docs and treat them as internal compatibility only

## Build And CI

### 11. Development workflow

Update `bindings/node/rebuild.sh` so local development rebuilds:
- `diarization-ggml/build`
- `diarization-ggml/build-static`
- native addon(s)
- staged platform package binaries
- copied `node_modules` binary used by Vitest
- TypeScript output

Add a separate cross-platform staging script for release packaging; do not overload `rebuild.sh` with publish concerns.

### 12. CI matrix

Add CI jobs for:
- macOS ARM64: full low-level + pipeline tests
- macOS x64: binary load + low-level tests at minimum
- Windows x64: low-level tests + packaging smoke test + unsupported-pipeline assertions

## Tests And Acceptance Criteria

### 13. Test cases

Add or migrate tests so `bindings/node` covers:

- Whisper API parity:
  - context creation
  - file transcription
  - buffer transcription
  - streaming `on_new_segment`
  - VAD context behavior
  - GPU enumeration
- Loader behavior:
  - correct platform package resolution
  - helpful unsupported-platform message
  - helpful missing-DLL/Electron message on Windows
- Pipeline behavior:
  - existing macOS pipeline tests still pass
  - Windows `Pipeline.load()` fails with the expected unsupported-platform error
- Packaging behavior:
  - install `pyannote-cpp-node` alone and verify it loads the right optional binary
  - staged binary copy uses `RelWithDebInfo` when available

### 14. Validation gates

For implementation acceptance:
- Existing `bindings/node` Vitest suite passes with `--no-file-parallelism`.
- Migrated whisper low-level tests pass against `pyannote-cpp-node`.
- macOS pipeline behavior remains unchanged.
- If native C++ work affects inference output, also run the AGENTS-required numerical checks:
  - segmentation accuracy test
  - full DER RTTM comparison

## Rollout

### 15. Release steps

- Ship the expanded `pyannote-cpp-node`.
- Deprecate `whisper-cpp-node` on npm immediately with a message directing users to `pyannote-cpp-node`.
- Update README and migration docs with a one-line import replacement and note that the merged package is ESM-only.

## Assumptions And Defaults

- `pyannote-cpp-node` is the only future public package.
- Low-level whisper/VAD functionality must remain cross-platform.
- High-level pipeline remains macOS-only for now.
- ESM-only is acceptable even though `whisper-cpp-node` supported CommonJS.
- Prebuilt optional dependency binaries are the distribution model; no compile-on-install for consumers.
