import { spawnSync } from 'node:child_process';
import os from 'node:os';

const [targetPlatform, targetArch, ...cmakeJsArgs] = process.argv.slice(2);

if (!targetPlatform || !targetArch || cmakeJsArgs.length === 0) {
  throw new Error('Usage: run-platform-build.js <platform> <arch> <cmake-js args...>');
}

if (os.platform() !== targetPlatform || os.arch() !== targetArch) {
  console.log(`Skipping native build for ${targetPlatform}-${targetArch} on ${os.platform()}-${os.arch()}`);
  process.exit(0);
}

const result = spawnSync('npx', ['cmake-js', ...cmakeJsArgs], {
  stdio: 'inherit',
  shell: process.platform === 'win32',
});

process.exit(result.status ?? 1);
