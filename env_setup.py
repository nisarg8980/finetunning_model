import os
from pathlib import Path


def load_env(path=None) -> None:
    env_path = Path(path) if path else Path(__file__).with_name(".env")

    # Prefer python-dotenv when available (handles quoting/multiline edge cases).
    try:
        from dotenv import load_dotenv
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)  # does not override existing env vars
        return
    except ImportError:
        pass

    # Dependency-free fallback: parse simple KEY=VALUE lines ourselves.
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Real environment wins; only fill in what isn't already set.
        if key and key not in os.environ:
            os.environ[key] = value
