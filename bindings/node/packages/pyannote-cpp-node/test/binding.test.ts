import { describe, expect, it } from 'vitest';

import { getNativePackageNameForPlatform, getPlatformKey } from '../src/binding.js';

describe('native package resolution', () => {
  it('builds platform keys', () => {
    expect(getPlatformKey('darwin', 'arm64')).toBe('darwin-arm64');
    expect(getPlatformKey('win32', 'x64')).toBe('win32-x64');
  });

  it('resolves supported packages', () => {
    expect(getNativePackageNameForPlatform('darwin', 'arm64')).toBe(
      '@pyannote-cpp-node/darwin-arm64',
    );
    expect(getNativePackageNameForPlatform('win32', 'x64')).toBe(
      '@pyannote-cpp-node/win32-x64',
    );
  });

  it('rejects unsupported platforms with a clear error', () => {
    expect(() => getNativePackageNameForPlatform('darwin', 'x64')).toThrow(
      'Unsupported platform: darwin-x64. pyannote-cpp-node supports macOS Apple Silicon and Windows x64 only.',
    );
    expect(() => getNativePackageNameForPlatform('win32', 'ia32')).toThrow(
      'Unsupported platform: win32-ia32. pyannote-cpp-node supports macOS Apple Silicon and Windows x64 only.',
    );
  });
});
