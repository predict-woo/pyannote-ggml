import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const bindingsRoot = path.resolve(scriptDir, '..');

function parseArgs(argv) {
  const args = new Map();
  for (let i = 0; i < argv.length; i += 2) {
    args.set(argv[i], argv[i + 1] ?? '');
  }
  return args;
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function copyIfExists(source, destination) {
  if (!fs.existsSync(source)) {
    return false;
  }

  ensureDir(path.dirname(destination));
  fs.copyFileSync(source, destination);
  return true;
}

function copyTreeContents(sourceDir, destinationDir) {
  if (!fs.existsSync(sourceDir)) {
    return;
  }

  ensureDir(destinationDir);

  for (const entry of fs.readdirSync(sourceDir, { withFileTypes: true })) {
    const sourcePath = path.join(sourceDir, entry.name);
    const destinationPath = path.join(destinationDir, entry.name);

    if (entry.isDirectory()) {
      fs.rmSync(destinationPath, { recursive: true, force: true });
      fs.cpSync(sourcePath, destinationPath, { recursive: true });
      continue;
    }

    copyIfExists(sourcePath, destinationPath);
  }
}

const args = parseArgs(process.argv.slice(2));
const platformKey = args.get('--platform') || `${os.platform()}-${os.arch()}`;
const packageDir = args.get('--package-dir') || path.join(bindingsRoot, 'packages', platformKey);
const buildDir = args.get('--build-dir') || path.join(packageDir, 'build');
const copyNodeModules = args.get('--copy-node-modules') === 'true';
const packageName = `@pyannote-cpp-node/${platformKey}`;

const configs = ['RelWithDebInfo', 'Release'];
const sourceConfig = configs.find((config) =>
  fs.existsSync(path.join(buildDir, config, 'pyannote-addon.node')),
);

if (!sourceConfig) {
  throw new Error(
    `Built addon not found for ${platformKey}. Looked in ${configs
      .map((config) => path.join(buildDir, config, 'pyannote-addon.node'))
      .join(', ')}`,
  );
}

const sourceDir = path.join(buildDir, sourceConfig);
const packageBuildDir = path.join(packageDir, 'build', 'Release');
const addonSource = path.join(sourceDir, 'pyannote-addon.node');
const addonDest = path.join(packageBuildDir, 'pyannote-addon.node');

copyIfExists(addonSource, addonDest);

const pdbSource = path.join(sourceDir, 'pyannote-addon.pdb');
const dSYMSource = `${addonSource}.dSYM`;
const copiedPdb = copyIfExists(pdbSource, path.join(packageBuildDir, 'pyannote-addon.pdb'));
const copiedDSYM = fs.existsSync(dSYMSource)
  ? (() => {
      const dSYMDest = path.join(packageBuildDir, 'pyannote-addon.node.dSYM');
      fs.rmSync(dSYMDest, { recursive: true, force: true });
      fs.cpSync(dSYMSource, dSYMDest, { recursive: true });
      return true;
    })()
  : false;

if (platformKey === 'win32-x64') {
  for (const entry of fs.readdirSync(sourceDir)) {
    if (entry.toLowerCase().endsWith('.dll')) {
      copyIfExists(path.join(sourceDir, entry), path.join(packageBuildDir, entry));
    }
  }
}

if (copyNodeModules) {
  const nodeModulesBuildDir = path.join(
    bindingsRoot,
    'node_modules',
    packageName,
    'build',
    'Release',
  );
  copyTreeContents(packageBuildDir, nodeModulesBuildDir);
}

console.log(`Staged ${platformKey} addon from ${sourceDir}`);
console.log(`Addon: ${addonDest}`);
if (copiedPdb) {
  console.log(`PDB: ${path.join(packageBuildDir, 'pyannote-addon.pdb')}`);
}
if (copiedDSYM) {
  console.log(`dSYM: ${path.join(packageBuildDir, 'pyannote-addon.node.dSYM')}`);
}
