import torch
import importlib.util
import sys
import pathlib
import os
from typing import Any, Dict

"""Example
 mio.save_model_state(model=model, 
                      path=f"{result_dir}/net_params_setting{model.params_setting_number}.pth", 
                      model_args=model.param)

net = load_model_from_results(
        result_dir,
        pth_name="net_params_setting1.pth",
        # py_name=None       # 让函数自动寻找首个 *.py
        device="cuda"
      )
"""

def _import_from_file(py_file: str):
    """Dynamically import a .py file and return the loaded module object.

    Parameters
    ----------
    py_file : str
        Absolute or relative path to a Python script that defines the model
        class named ``model``. The file does **not** need to be in
        ``PYTHONPATH`` and the directory does **not** need to be a package.

    Returns
    -------
    module
        A live Python module with all definitions executed.
    """
    # Generate a unique name to avoid clashing with real packages
    unique_name = f"_restored_{pathlib.Path(py_file).stem}"
    spec = importlib.util.spec_from_file_location(unique_name, py_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load spec from {py_file}")

    module = importlib.util.module_from_spec(spec)
    # Execute the module in its own namespace
    spec.loader.exec_module(module)  # type: ignore

    # Register the module so it can be found via sys.modules later if needed
    sys.modules[unique_name] = module
    return module


def save_model_state(model: torch.nn.Module, path: str, model_args: Dict[str, Any] | None = None):
    """Save *only* the state_dict along with construction arguments.

    This keeps checkpoint files small and avoids pickling issues. The
    ``model`` class itself is **not** serialized, so you must keep a copy of
    the defining ``.py`` file next to the checkpoint when using
    :func:`load_model_from_results`.
    """
    ckpt = {
        "model_args": model_args,
        "state_dict": model.state_dict(),
    }
    torch.save(ckpt, path)
    print(f"✅ state_dict saved to {path}")



def load_model_from_results(
    result_dir: str,
    pth_name: str = "net_params_setting1.pth",
    py_name: str | None = None,
    device: str | torch.device = "cpu",
):
    """Load a trained model given only the *results folder*.

    Parameters
    ----------
    result_dir : str
        Path to the experiment folder that contains both the ``.pth``
        checkpoint and the copied ``.py`` file that defines ``class model``.
    pth_name : str, default ``"net_params_setting1.pth"``
        File name of the checkpoint to load.
    py_name : str | None, default ``None``
        If you know the exact script name that defines the model, pass it
        here.  Otherwise the function will auto‑detect the **first** Python
        script in *result_dir*.
    device : str | torch.device, default ``"cpu"``
        Device to map tensors to when loading.

    Returns
    -------
    torch.nn.Module
        The reconstructed model in ``eval`` mode.
    """
    result_dir = os.path.abspath(result_dir)
    ckpt_path = os.path.join(result_dir, pth_name)

    # ── 1. locate the python file that defines the model ────────────────────
    if py_name is None:
        py_files = list(pathlib.Path(result_dir).glob("*.py"))
        if not py_files:
            raise FileNotFoundError(
                f"No .py file found in {result_dir}. "
                "Make sure you copied the training script here."
            )
        if len(py_files) > 1:
            raise RuntimeError(
                "Multiple .py files detected: \n" + "\n".join(map(str, py_files)) +
                "\nSpecify which one via the py_name argument."
            )
        py_file = str(py_files[0])
    else:
        py_file = os.path.join(result_dir, py_name)
        if not os.path.exists(py_file):
            raise FileNotFoundError(f"Specified model file {py_file} does not exist.")

    # ── 2. dynamically import that file ────────────────────────────────────
    module = _import_from_file(py_file)
    if not hasattr(module, "model"):
        raise AttributeError(
            f"{py_file} does not define a 'model' class. Update py_name or "
            "ensure the file contains `class model(...)`."
        )
    ModelClass = module.model  # type: ignore[attr-defined]

    # ── 3. load checkpoint ─────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device)
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        model_args = ckpt.get("model_args", {})
    else:
        # fallback for full‑model checkpoints
        state_dict = ckpt
        model_args = {}

    # ── 4. rebuild model and load weights ──────────────────────────────────
    model = ModelClass(model_args).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("⚠️  Missing keys:", missing)
    if unexpected:
        print("⚠️  Unexpected keys:", unexpected)
    model.eval()

    print(
        "✓ Model restored from:", ckpt_path,
        "\n  using definition :", py_file,
    )
    return model
