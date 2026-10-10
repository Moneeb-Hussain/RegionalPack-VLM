"""Record GPU and library versions. Torch is imported only when a probe runs."""

from __future__ import annotations

import shutil
import subprocess
from typing import Any

from src.errors import GpuUnavailableError, PillowExtensionError
from src.versions import MINIMUM_TRANSFORMERS, version_at_least


def release_cached_pillow() -> None:
    """Drop Pillow modules imported before a pip upgrade in this process."""
    import sys

    for name in list(sys.modules):
        if name == "PIL" or name.startswith("PIL."):
            del sys.modules[name]


def assert_pillow_extension_matches() -> str:
    """Import Pillow and return its version when the compiled extension matches.

    Colab's preinstalled extension is Pillow 11.3.0. Replacing only the Python
    files with Pillow 12.x makes ``import PIL.Image`` raise. Call this after a
    force-reinstall and before PyTorch imports Pillow.
    """
    release_cached_pillow()
    try:
        import PIL
        from PIL import Image  # noqa: F401
        from PIL._typing import _Ink  # noqa: F401
    except ImportError as exc:
        raise PillowExtensionError(
            "Pillow's Python package and its compiled _imaging extension do not match, "
            "or PIL._typing._Ink is missing. Colab ships the 11.3.0 extension. "
            "This notebook force-reinstalls pillow>=12.1.0 before PyTorch starts. "
            "Use Runtime → Restart session, then run all cells from the top. "
            "One restart is enough. Do not keep running cells in this session, "
            "and do not switch models. "
            f"Original error: {exc}"
        ) from exc
    return str(PIL.__version__)


def pillow_process_status() -> dict[str, Any]:
    """Report the Pillow already imported in this process.

    Colab imports Pillow before the first notebook cell. A matching Pillow 12.1+
    build is usable and must be left in place. Unloading it and reinstalling is
    what splits the Python files from the compiled extension.
    """
    import sys

    loaded_before = any(name == "PIL" or name.startswith("PIL.") for name in sys.modules)
    try:
        import PIL
        from PIL import Image  # noqa: F401
        from PIL import _imaging as core
        from PIL._typing import _Ink  # noqa: F401
    except ImportError as exc:
        return {
            "loaded_before": loaded_before,
            "ok": False,
            "version": None,
            "core_version": None,
            "error": str(exc),
        }
    version = str(PIL.__version__)
    core_version = getattr(core, "PILLOW_VERSION", None)
    core_text = None if core_version is None else str(core_version)
    matches = core_text is None or core_text == version
    ok = matches and version_at_least(version, "12.1.0")
    error = None
    if not ok:
        error = f"core {core_text} vs package {version}"
    return {
        "loaded_before": loaded_before,
        "ok": ok,
        "version": version,
        "core_version": core_text,
        "error": error,
    }


def pillow_pin_for_loaded_core(core: str | None) -> str:
    """Choose a Pillow pin that matches the extension already loaded.

    Colab imports its extension before the notebook runs. Installing a newer
    Pillow then leaves Python files from 12.x on top of the 11.3.0 binary.
    """
    if core is None:
        return "pillow>=12.1.0,<13"
    return f"pillow=={core}"


def pillow_pin_for_installed_package(installed: str | None) -> str:
    """Pin Pillow to the build already on disk when this process has not loaded it.

    A fresh Colab kernel often has not imported Pillow yet. Upgrading that
    install to 12.x replaces the Python files and can leave the 11.3.0 extension.
    """
    if installed is None:
        return "pillow>=12.1.0,<13"
    return f"pillow=={installed}"


def _installed_pillow_version() -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("pillow")
    except PackageNotFoundError:
        return None


def _loaded_pillow_core() -> str | None:
    import sys

    core = sys.modules.get("PIL._imaging")
    if core is None:
        return None
    version = getattr(core, "PILLOW_VERSION", None)
    if version is None:
        return None
    if isinstance(version, bytes):
        text = version.decode()
    else:
        text = str(version)
    text = text.strip()
    return text or None


def _typing_path():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.find_spec("PIL")
    if spec is None or not spec.origin:
        raise PillowExtensionError("Pillow is not installed. Do not switch models.")
    return Path(spec.origin).resolve().parent / "_typing.py"


def _ensure_ink_alias() -> None:
    """Define ``PIL._typing._Ink`` when this Pillow build does not ship it."""
    import importlib
    import sys

    path = _typing_path()
    text = path.read_text(encoding="utf-8")
    if "_Ink" not in text:
        if text and not text.endswith("\n"):
            text += "\n"
        path.write_text(text + "\n_Ink = float | tuple[int, ...] | str\n", encoding="utf-8")
    loaded = sys.modules.get("PIL._typing")
    if loaded is not None:
        importlib.reload(loaded)


def _pillow_imports() -> str:
    import PIL
    from PIL import Image  # noqa: F401
    from PIL import _imaging as core
    from PIL._typing import _Ink  # noqa: F401

    version = str(PIL.__version__)
    core_version = getattr(core, "PILLOW_VERSION", None)
    core_text = None if core_version is None else str(core_version).strip()
    if core_text and core_text != version:
        raise PillowExtensionError(
            f"Pillow package {version} does not match the loaded _imaging extension {core_text}. "
            "Do not switch models."
        )
    return version


def _install_requirements(requirements: Path, pin: str, *, force_pillow: bool) -> None:
    import sys
    import tempfile
    from pathlib import Path as PathType

    requirements = PathType(requirements)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        handle.write(pin + "\n")
        constraint = handle.name
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements), "-c", constraint],
        check=False,
    )
    if completed.returncode != 0:
        raise PillowExtensionError(
            "pip install failed. Pillow stays pinned to the extension already loaded "
            f"({pin}). Do not replace Colab's CUDA torch, and do not switch models."
        )
    if not force_pillow:
        return
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--force-reinstall",
            "--no-cache-dir",
            "--no-deps",
            pin,
        ],
        check=False,
    )
    if completed.returncode != 0:
        raise PillowExtensionError(f"Could not install {pin}. Do not switch models.")


def ensure_pillow(requirements: Path) -> str:
    """Install project requirements and leave this process able to import Pillow.

    When an extension is already loaded, the package version is pinned to it.
    Pillow 11.3.0 has no ``_Ink`` symbol; that alias is added in place.
    """
    from pathlib import Path as PathType

    requirements = PathType(requirements)
    core = _loaded_pillow_core()
    if core is None:
        installed = _installed_pillow_version()
        if installed is None:
            _install_requirements(
                requirements, pillow_pin_for_installed_package(None), force_pillow=True
            )
            _ensure_ink_alias()
            return _pillow_imports()
        try:
            _ensure_ink_alias()
            version = _pillow_imports()
        except ImportError:
            version = None
        if version is not None:
            _install_requirements(requirements, f"pillow=={version}", force_pillow=False)
            _ensure_ink_alias()
            return _pillow_imports()
        core = _loaded_pillow_core()
        if core is None:
            raise PillowExtensionError(
                "Pillow could not be imported, and its compiled extension version could not be read. "
                "Do not switch models."
            )

    pin = pillow_pin_for_loaded_core(core)
    try:
        _ensure_ink_alias()
        version = _pillow_imports()
    except ImportError:
        version = None
    if version is not None:
        _install_requirements(requirements, pin, force_pillow=False)
        _ensure_ink_alias()
        return _pillow_imports()

    _install_requirements(requirements, pin, force_pillow=True)
    release_cached_pillow()
    _ensure_ink_alias()
    return _pillow_imports()


def verify_pillow_install() -> str:
    """Import Pillow in a fresh interpreter and return its version.

    A fresh process reads the files on disk. The current process can still be
    holding Colab's old ``_imaging`` extension after a pip upgrade.
    """
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import PIL; from PIL import Image; from PIL._typing import _Ink; print(PIL.__version__)",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise PillowExtensionError(
            "Pillow on disk is still inconsistent after reinstall. "
            "The Python package and the compiled _imaging extension must be the same build. "
            "Do not switch models. "
            f"Checker output: {detail}"
        )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise PillowExtensionError(
            "Pillow reinstall finished, but the version check printed nothing. "
            "Restart the runtime and run the notebook from the top."
        )
    return lines[-1]


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise GpuUnavailableError(
            "No CUDA GPU is visible. In Colab, choose Runtime > Change runtime type > T4 GPU, "
            "then rerun this notebook from the top. Phase 0 does not run Qwen3-VL on CPU."
        )


def cuda_memory_bytes() -> dict[str, int] | None:
    """Return CUDA memory counters. These are measurements, not estimates."""
    import torch

    if not torch.cuda.is_available():
        return None
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def reset_peak_memory() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def classify_bf16_support(
    *,
    compute_capability: tuple[int, int],
    pytorch_including_emulation: bool,
) -> dict[str, Any]:
    """Separate hardware bfloat16 from PyTorch's emulation flag.

    Ampere and newer (compute capability 8.0+) have hardware bfloat16.
    Tesla T4 is Turing, capability 7.5, and does not. PyTorch's
    ``is_bf16_supported()`` defaults to ``including_emulation=True``, so it
    can still return true on a T4. That flag is not hardware support.
    """
    hardware = tuple(compute_capability) >= (8, 0)
    emulation_flag = bool(pytorch_including_emulation)
    return {
        "bf16_hardware_supported": hardware,
        "bf16_pytorch_including_emulation": emulation_flag,
        "bf16_emulation_only": (not hardware) and emulation_flag,
        "bf16_supported": hardware,
    }


def probe_gpu() -> dict[str, Any]:
    """Read the visible GPU. Raises if CUDA is missing."""
    require_cuda()
    import torch

    properties = torch.cuda.get_device_properties(0)
    gpu_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    bf16 = classify_bf16_support(
        compute_capability=capability,
        pytorch_including_emulation=bool(torch.cuda.is_bf16_supported()),
    )
    return {
        "gpu_name": gpu_name,
        "gpu_index": 0,
        "t4_name_confirmed": "T4" in gpu_name,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        **bf16,
        "total_vram_bytes": int(properties.total_memory),
        "nvidia_smi": _nvidia_smi(),
    }


def probe_libraries() -> dict[str, Any]:
    """Read installed library versions after the notebook has installed them."""
    import torch

    info: dict[str, Any] = {
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    try:
        import transformers

        info["transformers_version"] = transformers.__version__
    except ImportError as exc:
        raise GpuUnavailableError(
            "transformers is not installed. Run the install cell before loading the model."
        ) from exc
    for module_name, key in (
        ("accelerate", "accelerate_version"),
        ("PIL", "pillow_version"),
        ("huggingface_hub", "huggingface_hub_version"),
    ):
        try:
            module = __import__(module_name)
        except ImportError:
            info[key] = None
            continue
        info[key] = getattr(module, "__version__", "unknown")
    version = info["transformers_version"]
    info["transformers_satisfies_minimum"] = version_at_least(version, MINIMUM_TRANSFORMERS)
    if not info["transformers_satisfies_minimum"]:
        from src.errors import TransformersVersionError

        raise TransformersVersionError(
            f"transformers {version} is older than {MINIMUM_TRANSFORMERS}, which is the first "
            "stable release with Qwen3-VL. Install it with pip install -U "
            f'"transformers>={MINIMUM_TRANSFORMERS}". Building transformers from source is not '
            "the default path."
        )
    return info


def _nvidia_smi() -> str:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return "nvidia-smi was not on PATH"
    try:
        completed = subprocess.run(
            [
                binary,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"nvidia-smi failed: {exc}"
    text = (completed.stdout or completed.stderr or "").strip()
    if completed.returncode != 0:
        return f"nvidia-smi exited {completed.returncode}: {text}"
    return text
