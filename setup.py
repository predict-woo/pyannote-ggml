#!/usr/bin/env python3
"""
setup.py -- Automated setup for pyannote-ggml (diarization-ggml)

Sets up the full development environment from a fresh clone:
  1. Checks prerequisites (Python, Node.js, pnpm, CMake, platform toolchains)
  2. Installs Python dependencies (PyTorch CPU, pyannote.audio, etc.)
  3. Downloads Whisper models (ggml-base.en, ggml-large-v3-turbo-q5_0, silero VAD)
  4. Downloads and converts pyannote models (segmentation, embedding, PLDA)
  5. Installs Node.js dependencies (pnpm install)
  6. Builds native addon (cmake-js)
  7. Stages addon into node_modules
  8. Builds TypeScript
  9. Runs tests (vitest)

Usage:
    python setup.py                  # Full setup
    python setup.py --models-only    # Steps 1-4 only (download/convert models)
    python setup.py --build-only     # Steps 5-8 only (build addon + TypeScript)
    python setup.py --skip-pyannote  # Skip pyannote model download/conversion
    python setup.py --skip-tests     # Skip test verification
    python setup.py --force          # Re-download/rebuild even if outputs exist

Requirements:
    - Python 3.10+
    - Node.js 18+
    - pnpm
    - CMake 3.14+
    - Windows: Visual Studio Build Tools or VS2022
    - macOS: Xcode Command Line Tools

For pyannote model download (step 4), you need a HuggingFace account:
    1. Create account at https://huggingface.co
    2. Accept model terms at https://huggingface.co/pyannote/speaker-diarization-community-1
    3. Run: pip install huggingface_hub && huggingface-cli login
"""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

WHISPER_MODELS_DIR = PROJECT_ROOT / "whisper.cpp" / "models"
SEG_MODEL_PATH = PROJECT_ROOT / "models" / "segmentation-ggml" / "segmentation.gguf"
EMB_MODEL_PATH = PROJECT_ROOT / "models" / "embedding-ggml" / "embedding.gguf"
PLDA_MODEL_PATH = PROJECT_ROOT / "diarization-ggml" / "plda.gguf"

WHISPER_MODELS = {
    "ggml-base.en.bin": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
        "size_mb": 148,
    },
    "ggml-large-v3-turbo-q5_0.bin": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
        "size_mb": 574,
    },
    "ggml-silero-v6.2.0.bin": {
        "url": "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin",
        "size_mb": 1,
    },
}

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
PLATFORM_KEY = f"{sys.platform}-{'x64' if platform.machine() in ('x86_64', 'AMD64') else platform.machine()}"

NODE_BINDINGS_DIR = PROJECT_ROOT / "bindings" / "node"
PLATFORM_PKG_DIR = NODE_BINDINGS_DIR / "packages" / PLATFORM_KEY
TS_PKG_DIR = NODE_BINDINGS_DIR / "packages" / "pyannote-cpp-node"
ADDON_PATH = PLATFORM_PKG_DIR / "build" / "Release" / "pyannote-addon.node"

JFK_WAV = PROJECT_ROOT / "whisper.cpp" / "samples" / "jfk.wav"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class Colors:
    """ANSI color codes (disabled on Windows without VT support)."""

    _enabled = not IS_WINDOWS or os.environ.get("WT_SESSION") or os.environ.get("TERM")

    @staticmethod
    def _wrap(code: str, text: str) -> str:
        if Colors._enabled:
            return f"\033[{code}m{text}\033[0m"
        return text

    @staticmethod
    def green(text: str) -> str:
        return Colors._wrap("32", text)

    @staticmethod
    def red(text: str) -> str:
        return Colors._wrap("31", text)

    @staticmethod
    def yellow(text: str) -> str:
        return Colors._wrap("33", text)

    @staticmethod
    def cyan(text: str) -> str:
        return Colors._wrap("36", text)

    @staticmethod
    def bold(text: str) -> str:
        return Colors._wrap("1", text)


def ok(msg: str) -> None:
    print(f"  {Colors.green('[OK]')} {msg}")


def fail(msg: str) -> None:
    print(f"  {Colors.red('[FAIL]')} {msg}")


def skip(msg: str) -> None:
    print(f"  {Colors.yellow('[SKIP]')} {msg}")


def info(msg: str) -> None:
    print(f"  {Colors.cyan('[INFO]')} {msg}")


def step_header(num: int, total: int, title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Step {num}/{total}: {title}")
    print(f"{'=' * 60}")


def run_cmd(
    cmd: list,
    cwd: Path = PROJECT_ROOT,
    env: dict = None,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    """Run a subprocess command with proper error handling."""
    merged_env = None
    if env:
        merged_env = {**os.environ, **env}

    display_cmd = " ".join(str(c) for c in cmd)
    info(f"Running: {display_cmd}")

    kwargs = {
        "cwd": str(cwd),
        "env": merged_env,
        "check": check,
    }

    if IS_WINDOWS:
        kwargs["shell"] = True

    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
        kwargs["text"] = True

    try:
        return subprocess.run([str(c) for c in cmd], **kwargs)
    except subprocess.CalledProcessError as e:
        if capture and e.stderr:
            fail(f"Command failed:\n{e.stderr}")
        raise


def get_cmd_version(cmd: list, pattern: str = None) -> str:
    """Run a command and extract version string from output."""
    try:
        result = subprocess.run(
            [str(c) for c in cmd],
            capture_output=True,
            text=True,
            timeout=15,
            shell=IS_WINDOWS,
        )
        output = result.stdout.strip() or result.stderr.strip()
        if pattern:
            match = re.search(pattern, output)
            if match:
                return match.group(1)
        return output
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""


def parse_version(version_str: str) -> tuple:
    """Parse a version string like '3.10.4' into a tuple of ints."""
    parts = re.findall(r"\d+", version_str)
    return tuple(int(p) for p in parts[:3]) if parts else (0,)


def download_file(url: str, dest: Path, expected_mb: int = 0) -> None:
    """Download a file with progress reporting."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    info(f"Downloading: {dest.name}")
    if expected_mb > 0:
        info(f"Expected size: ~{expected_mb} MB")
    info(f"URL: {url}")

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "pyannote-ggml-setup/1.0")

        with urllib.request.urlopen(req, timeout=300) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1 MB
            start_time = time.time()
            last_pct_reported = -1

            with open(tmp, "wb") as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total > 0:
                        pct = downloaded * 100 // total
                        # Only update display every 5%
                        if pct >= last_pct_reported + 5 or downloaded >= total:
                            last_pct_reported = pct
                            elapsed = time.time() - start_time
                            speed = downloaded / (1024 * 1024 * max(elapsed, 0.01))
                            sys.stdout.write(
                                f"\r  ... {downloaded // (1024 * 1024)} / {total // (1024 * 1024)} MB"
                                f" ({pct}%) - {speed:.1f} MB/s"
                            )
                            sys.stdout.flush()

            print()  # newline after progress
        # On Windows, rename() fails if dest exists — remove first
        if dest.exists():
            dest.unlink()
        tmp.rename(dest)
        final_mb = dest.stat().st_size / (1024 * 1024)
        ok(f"Downloaded {dest.name} ({final_mb:.1f} MB)")

    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Download failed for {url}: {e}") from e


# ---------------------------------------------------------------------------
# Step 1: Check prerequisites
# ---------------------------------------------------------------------------


def check_prerequisites() -> bool:
    """Check that all required tools are installed and meet version requirements."""
    all_ok = True

    # Python 3.10+
    py_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    if sys.version_info >= (3, 10):
        ok(f"Python {py_version}")
    else:
        fail(f"Python {py_version} (need 3.10+)")
        all_ok = False

    # Node.js 18+
    node_ver = get_cmd_version(["node", "--version"], r"v?(\d+\.\d+\.\d+)")
    if node_ver and parse_version(node_ver) >= (18,):
        ok(f"Node.js {node_ver}")
    elif node_ver:
        fail(f"Node.js {node_ver} (need 18+)")
        all_ok = False
    else:
        fail("Node.js not found (need 18+)")
        all_ok = False

    # pnpm
    pnpm_ver = get_cmd_version(["pnpm", "--version"], r"(\d+\.\d+\.\d+)")
    if pnpm_ver:
        ok(f"pnpm {pnpm_ver}")
    else:
        fail("pnpm not found (install: npm install -g pnpm)")
        all_ok = False

    # CMake 3.14+
    cmake_ver = get_cmd_version(
        ["cmake", "--version"], r"cmake version (\d+\.\d+\.\d+)"
    )
    if cmake_ver and parse_version(cmake_ver) >= (3, 14):
        ok(f"CMake {cmake_ver}")
    elif cmake_ver:
        fail(f"CMake {cmake_ver} (need 3.14+)")
        all_ok = False
    else:
        fail("CMake not found (need 3.14+)")
        all_ok = False

    # Platform-specific toolchain
    if IS_WINDOWS:
        _check_windows_toolchain()
    elif IS_MACOS:
        _check_macos_toolchain()

    # Check for jfk.wav test sample
    if JFK_WAV.exists():
        ok(f"Test sample: {JFK_WAV.relative_to(PROJECT_ROOT)}")
    else:
        info(
            f"Note: {JFK_WAV.relative_to(PROJECT_ROOT)} not found (optional test sample)"
        )

    return all_ok


def _check_windows_toolchain() -> None:
    """Check for Visual Studio Build Tools on Windows."""
    # Check for cl.exe (MSVC compiler)
    cl_path = shutil.which("cl")
    if cl_path:
        ok(f"MSVC compiler: {cl_path}")
    else:
        # Check common VS installation paths
        vs_paths = [
            Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft Visual Studio",
            Path(os.environ.get("ProgramFiles", "")) / "Microsoft Visual Studio",
        ]
        found = False
        for vs_base in vs_paths:
            if vs_base.exists():
                for year in ["2022", "2019"]:
                    for edition in [
                        "BuildTools",
                        "Community",
                        "Professional",
                        "Enterprise",
                    ]:
                        vs_dir = vs_base / year / edition
                        if vs_dir.exists():
                            ok(f"Visual Studio {year} {edition} found")
                            info(
                                "  Run from 'Developer Command Prompt' or 'x64 Native Tools' for cl.exe"
                            )
                            found = True
                            break
                    if found:
                        break
            if found:
                break
        if not found:
            fail("Visual Studio Build Tools or VS2022 not found")
            info("  Download from: https://visualstudio.microsoft.com/downloads/")
            info("  Install 'Desktop development with C++' workload")

    # Check Vulkan SDK
    vulkan_sdk = os.environ.get("VULKAN_SDK")
    if vulkan_sdk and Path(vulkan_sdk).exists():
        ok(f"Vulkan SDK: {vulkan_sdk}")
    else:
        info(
            "Warning: VULKAN_SDK not set. Native addon will build without GPU support."
        )
        info("  Download from: https://vulkan.lunarg.com/sdk/home")


def _check_macos_toolchain() -> None:
    """Check for Xcode Command Line Tools on macOS."""
    result = subprocess.run(
        ["xcode-select", "-p"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        ok(f"Xcode CLT: {result.stdout.strip()}")
    else:
        fail("Xcode Command Line Tools not installed")
        info("  Install with: xcode-select --install")


# ---------------------------------------------------------------------------
# Step 2: Install Python dependencies
# ---------------------------------------------------------------------------


def install_python_deps(force: bool = False) -> None:
    """Install required Python packages."""
    # Check if torch is already installed
    if not force:
        try:
            import torch

            ok(f"PyTorch already installed ({torch.__version__})")
            torch_installed = True
        except ImportError:
            torch_installed = False
    else:
        torch_installed = False

    if not torch_installed:
        info("Installing PyTorch (CPU-only)...")
        run_cmd(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ]
        )
        ok("PyTorch (CPU) installed")

    # Install remaining packages
    packages = ["pyannote.audio", "huggingface_hub", "gguf", "scipy", "numpy"]

    if not force:
        # Check which packages need installing
        missing = []
        for pkg in packages:
            import_name = pkg.replace(".", "_").replace("-", "_")
            # Special import names
            import_map = {
                "pyannote_audio": "pyannote.audio",
                "huggingface_hub": "huggingface_hub",
            }
            try_import = import_map.get(import_name, import_name)
            try:
                __import__(try_import.split(".")[0])
            except ImportError:
                missing.append(pkg)

        if not missing:
            ok("All Python dependencies already installed")
            return
        packages = missing
        info(f"Installing missing packages: {', '.join(packages)}")

    run_cmd([sys.executable, "-m", "pip", "install"] + packages)
    ok("Python dependencies installed")


# ---------------------------------------------------------------------------
# Step 3: Download Whisper models
# ---------------------------------------------------------------------------


def download_whisper_models(force: bool = False) -> None:
    """Download Whisper GGML models."""
    WHISPER_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for filename, meta in WHISPER_MODELS.items():
        dest = WHISPER_MODELS_DIR / filename
        if dest.exists() and not force:
            size_mb = dest.stat().st_size / (1024 * 1024)
            ok(f"{filename} already exists ({size_mb:.1f} MB)")
            continue
        download_file(meta["url"], dest, meta["size_mb"])


# ---------------------------------------------------------------------------
# Step 4: Download and convert pyannote models
# ---------------------------------------------------------------------------


def check_hf_auth() -> bool:
    """Check if HuggingFace authentication is configured."""
    try:
        from huggingface_hub import get_token

        token = get_token()
        if token:
            ok("HuggingFace authentication configured")
            return True
    except ImportError:
        fail("huggingface_hub not installed (should have been installed in step 2)")
        return False

    fail("HuggingFace authentication required for pyannote models.")
    info("  1. Create account at https://huggingface.co")
    info(
        "  2. Accept model terms at https://huggingface.co/pyannote/speaker-diarization-community-1"
    )
    info("  3. Run: pip install huggingface_hub && huggingface-cli login")
    return False


def download_and_convert_pyannote(force: bool = False) -> None:
    """Download pyannote models from HuggingFace and convert to GGUF."""
    if not check_hf_auth():
        raise RuntimeError(
            "HuggingFace authentication is required. See instructions above."
        )

    # --- Embedding model ---
    if EMB_MODEL_PATH.exists() and not force:
        ok(
            f"Embedding model already exists: {EMB_MODEL_PATH.relative_to(PROJECT_ROOT)}"
        )
    else:
        info("Converting embedding model...")
        run_cmd(
            [sys.executable, "convert.py", "-o", str(EMB_MODEL_PATH)],
            cwd=PROJECT_ROOT / "models" / "embedding-ggml",
        )
        if EMB_MODEL_PATH.exists():
            size_mb = EMB_MODEL_PATH.stat().st_size / (1024 * 1024)
            ok(f"Embedding model created ({size_mb:.1f} MB)")
        else:
            raise RuntimeError(
                "Embedding model conversion failed: output file not created"
            )

    # --- PLDA model ---
    if PLDA_MODEL_PATH.exists() and not force:
        ok(f"PLDA model already exists: {PLDA_MODEL_PATH.relative_to(PROJECT_ROOT)}")
    else:
        info("Downloading PLDA source files from HuggingFace...")
        from huggingface_hub import hf_hub_download

        plda_npz_path = hf_hub_download(
            "pyannote/speaker-diarization-community-1",
            "plda/plda.npz",
        )
        xvec_npz_path = hf_hub_download(
            "pyannote/speaker-diarization-community-1",
            "plda/xvec_transform.npz",
        )
        ok("PLDA source files downloaded")

        # Copy to expected locations
        plda_dir = PROJECT_ROOT / "diarization-ggml"
        plda_src = plda_dir / "plda_source.npz"
        xvec_src = plda_dir / "xvec_transform_source.npz"

        shutil.copy2(plda_npz_path, str(plda_src))
        shutil.copy2(xvec_npz_path, str(xvec_src))

        info("Converting PLDA model...")
        run_cmd(
            [
                sys.executable,
                "convert_plda.py",
                "--transform-npz",
                str(xvec_src),
                "--plda-npz",
                str(plda_src),
                "-o",
                str(PLDA_MODEL_PATH),
            ],
            cwd=plda_dir,
        )
        if PLDA_MODEL_PATH.exists():
            size_kb = PLDA_MODEL_PATH.stat().st_size / 1024
            ok(f"PLDA model created ({size_kb:.1f} KB)")
        else:
            raise RuntimeError("PLDA model conversion failed: output file not created")

        # Clean up temporary npz files
        plda_src.unlink(missing_ok=True)
        xvec_src.unlink(missing_ok=True)

    # --- Segmentation model ---
    if SEG_MODEL_PATH.exists() and not force:
        ok(
            f"Segmentation model already exists: {SEG_MODEL_PATH.relative_to(PROJECT_ROOT)}"
        )
    else:
        info("Downloading segmentation model from HuggingFace...")
        from huggingface_hub import hf_hub_download

        seg_bin_path = hf_hub_download(
            "pyannote/speaker-diarization-community-1",
            "segmentation/pytorch_model.bin",
        )
        ok(f"Segmentation weights downloaded")

        info("Converting segmentation model...")
        run_cmd(
            [
                sys.executable,
                "convert.py",
                "--model-path",
                seg_bin_path,
                "--output",
                str(SEG_MODEL_PATH),
            ],
            cwd=PROJECT_ROOT / "models" / "segmentation-ggml",
        )
        if SEG_MODEL_PATH.exists():
            size_mb = SEG_MODEL_PATH.stat().st_size / (1024 * 1024)
            ok(f"Segmentation model created ({size_mb:.1f} MB)")
        else:
            raise RuntimeError(
                "Segmentation model conversion failed: output file not created"
            )


# ---------------------------------------------------------------------------
# Step 5: Install Node.js dependencies
# ---------------------------------------------------------------------------


def install_node_deps() -> None:
    """Run pnpm install in bindings/node."""
    info("Installing Node.js dependencies...")
    run_cmd(["pnpm", "install"], cwd=NODE_BINDINGS_DIR)
    ok("Node.js dependencies installed")


# ---------------------------------------------------------------------------
# Step 6: Build native addon
# ---------------------------------------------------------------------------


def build_native_addon(force: bool = False) -> None:
    """Build the platform-specific native addon using cmake-js."""
    if not PLATFORM_PKG_DIR.exists():
        raise RuntimeError(
            f"Platform package directory not found: {PLATFORM_PKG_DIR}\n"
            f"  This platform ({PLATFORM_KEY}) may not be supported."
        )

    if ADDON_PATH.exists() and not force:
        ok(f"Native addon already exists: {ADDON_PATH.relative_to(PROJECT_ROOT)}")
        return

    info(f"Building native addon for {PLATFORM_KEY}...")

    cmake_flags = []
    if IS_WINDOWS:
        cmake_flags = ["--CDADDON_VULKAN=ON", "--CDADDON_OPENVINO=ON"]
    elif IS_MACOS:
        cmake_flags = [
            "--CDEMBEDDING_COREML=ON",
            "--CDSEGMENTATION_COREML=ON",
            "--CDWHISPER_COREML=ON",
        ]

    run_cmd(
        ["npx", "cmake-js", "build"] + cmake_flags,
        cwd=PLATFORM_PKG_DIR,
    )

    # Verify build output exists
    # cmake-js may put output in Release or RelWithDebInfo
    build_dir = PLATFORM_PKG_DIR / "build"
    found = False
    for config in ["Release", "RelWithDebInfo"]:
        candidate = build_dir / config / "pyannote-addon.node"
        if candidate.exists():
            ok(f"Native addon built: {candidate.relative_to(PROJECT_ROOT)}")
            found = True
            break

    if not found:
        raise RuntimeError(
            f"Native addon build succeeded but output not found.\n"
            f"  Looked in: {build_dir / 'Release'}, {build_dir / 'RelWithDebInfo'}"
        )


# ---------------------------------------------------------------------------
# Step 7: Stage addon
# ---------------------------------------------------------------------------


def stage_addon() -> None:
    """Stage the native addon into the correct locations."""
    info(f"Staging native addon for {PLATFORM_KEY}...")
    run_cmd(
        [
            "node",
            "scripts/stage-native-addon.js",
            "--platform",
            PLATFORM_KEY,
            "--copy-node-modules",
            "true",
        ],
        cwd=NODE_BINDINGS_DIR,
    )
    ok("Native addon staged into packages and node_modules")


# ---------------------------------------------------------------------------
# Step 8: Build TypeScript
# ---------------------------------------------------------------------------


def build_typescript() -> None:
    """Build TypeScript sources."""
    info("Building TypeScript...")
    run_cmd(["npx", "tsc"], cwd=TS_PKG_DIR)
    ok("TypeScript compiled")


# ---------------------------------------------------------------------------
# Step 9: Verify
# ---------------------------------------------------------------------------


def run_tests() -> None:
    """Run the test suite to verify the setup."""
    info("Running tests (this may take a few minutes)...")
    run_cmd(
        ["pnpm", "test", "--", "--no-file-parallelism"],
        cwd=NODE_BINDINGS_DIR,
    )
    ok("All tests passed")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(steps_run: list, failed_step: str = None) -> None:
    """Print a summary of what was done."""
    print(f"\n{'=' * 60}")
    if failed_step:
        print(f"  {Colors.red('SETUP FAILED')}")
    else:
        print(f"  {Colors.green('SETUP COMPLETE')}")
    print(f"{'=' * 60}")

    for label, status in steps_run:
        if status == "ok":
            print(f"  {Colors.green('[OK]')}   {label}")
        elif status == "skip":
            print(f"  {Colors.yellow('[SKIP]')} {label}")
        elif status == "fail":
            print(f"  {Colors.red('[FAIL]')} {label}")

    if failed_step:
        print(f"\n  Setup failed at: {Colors.red(failed_step)}")
        print(f"  Fix the issue above and re-run: python setup.py")
    else:
        print(f"\n  Project root: {PROJECT_ROOT}")
        print(f"  Platform:     {PLATFORM_KEY}")

        # Print model locations
        models_found = []
        for label, path in [
            ("Whisper base.en", WHISPER_MODELS_DIR / "ggml-base.en.bin"),
            (
                "Whisper large-v3-turbo",
                WHISPER_MODELS_DIR / "ggml-large-v3-turbo-q5_0.bin",
            ),
            ("Silero VAD", WHISPER_MODELS_DIR / "ggml-silero-v6.2.0.bin"),
            ("Segmentation", SEG_MODEL_PATH),
            ("Embedding", EMB_MODEL_PATH),
            ("PLDA", PLDA_MODEL_PATH),
        ]:
            if path.exists():
                models_found.append(label)
        if models_found:
            print(f"  Models:       {', '.join(models_found)}")

        if ADDON_PATH.exists():
            print(f"  Addon:        {ADDON_PATH.relative_to(PROJECT_ROOT)}")

        print(
            f"\n  To run tests:  cd bindings/node && pnpm test -- --no-file-parallelism"
        )

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated setup for pyannote-ggml development environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Only download/convert models (steps 1-4)",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only build addon and TypeScript (steps 5-8)",
    )
    parser.add_argument(
        "--skip-pyannote",
        action="store_true",
        help="Skip pyannote model download/conversion (step 4)",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip test verification (step 9)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download/rebuild even if outputs already exist",
    )

    args = parser.parse_args()

    # Determine which steps to run
    do_prereqs = True
    do_python_deps = not args.build_only
    do_whisper = not args.build_only
    do_pyannote = not args.build_only and not args.skip_pyannote
    do_node_deps = not args.models_only
    do_build = not args.models_only
    do_stage = not args.models_only
    do_typescript = not args.models_only
    do_tests = not args.models_only and not args.skip_tests

    total_steps = sum(
        [
            do_prereqs,
            do_python_deps,
            do_whisper,
            do_pyannote,
            do_node_deps,
            do_build,
            do_stage,
            do_typescript,
            do_tests,
        ]
    )

    steps_run = []
    current_step = 0
    current_step_name = ""

    print(Colors.bold(f"\npyannote-ggml Setup"))
    print(f"  Platform: {PLATFORM_KEY}")
    print(f"  Root:     {PROJECT_ROOT}")
    if args.force:
        print(f"  Mode:     {Colors.yellow('FORCE (re-download/rebuild all)')}")

    try:
        # Step 1: Prerequisites
        if do_prereqs:
            current_step += 1
            current_step_name = "Check prerequisites"
            step_header(current_step, total_steps, current_step_name)
            prereqs_ok = check_prerequisites()
            if not prereqs_ok:
                steps_run.append((current_step_name, "fail"))
                print_summary(steps_run, current_step_name)
                sys.exit(1)
            steps_run.append((current_step_name, "ok"))

        # Step 2: Python dependencies
        if do_python_deps:
            current_step += 1
            current_step_name = "Install Python dependencies"
            step_header(current_step, total_steps, current_step_name)
            install_python_deps(force=args.force)
            steps_run.append((current_step_name, "ok"))

        # Step 3: Whisper models
        if do_whisper:
            current_step += 1
            current_step_name = "Download Whisper models"
            step_header(current_step, total_steps, current_step_name)
            download_whisper_models(force=args.force)
            steps_run.append((current_step_name, "ok"))

        # Step 4: pyannote models
        if do_pyannote:
            current_step += 1
            current_step_name = "Download/convert pyannote models"
            step_header(
                current_step, total_steps, "Download and convert pyannote models"
            )
            download_and_convert_pyannote(force=args.force)
            steps_run.append((current_step_name, "ok"))
        elif not args.build_only:
            steps_run.append(("Download/convert pyannote models", "skip"))

        # Step 5: Node.js dependencies
        if do_node_deps:
            current_step += 1
            current_step_name = "Install Node.js dependencies"
            step_header(current_step, total_steps, current_step_name)
            install_node_deps()
            steps_run.append((current_step_name, "ok"))

        # Step 6: Build native addon
        if do_build:
            current_step += 1
            current_step_name = "Build native addon"
            step_header(current_step, total_steps, current_step_name)
            build_native_addon(force=args.force)
            steps_run.append((current_step_name, "ok"))

        # Step 7: Stage addon
        if do_stage:
            current_step += 1
            current_step_name = "Stage addon"
            step_header(current_step, total_steps, current_step_name)
            stage_addon()
            steps_run.append((current_step_name, "ok"))

        # Step 8: Build TypeScript
        if do_typescript:
            current_step += 1
            current_step_name = "Build TypeScript"
            step_header(current_step, total_steps, current_step_name)
            build_typescript()
            steps_run.append((current_step_name, "ok"))

        # Step 9: Verify
        if do_tests:
            current_step += 1
            current_step_name = "Verify (run tests)"
            step_header(current_step, total_steps, current_step_name)
            run_tests()
            steps_run.append((current_step_name, "ok"))
        elif not args.models_only:
            steps_run.append(("Verify (run tests)", "skip"))

        print_summary(steps_run)

    except KeyboardInterrupt:
        print(f"\n\n{Colors.yellow('Setup interrupted by user.')}")
        steps_run.append(("(interrupted)", "fail"))
        print_summary(steps_run, "Interrupted by user")
        sys.exit(130)

    except subprocess.CalledProcessError as e:
        step_name = current_step_name or "Unknown step"
        steps_run.append((step_name, "fail"))
        fail(f"Command failed with exit code {e.returncode}")
        print_summary(steps_run, step_name)
        sys.exit(1)

    except RuntimeError as e:
        step_name = current_step_name or "Unknown step"
        steps_run.append((step_name, "fail"))
        fail(str(e))
        print_summary(steps_run, step_name)
        sys.exit(1)

    except Exception as e:
        step_name = current_step_name or "Unknown step"
        steps_run.append((step_name, "fail"))
        fail(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        print_summary(steps_run, step_name)
        sys.exit(1)


if __name__ == "__main__":
    main()
