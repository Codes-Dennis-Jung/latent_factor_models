"""Execute every example notebook end to end (slow; skipped without nbclient/ipykernel)."""

import json
import os
import sys
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")
pytest.importorskip("matplotlib")
pytest.importorskip("pandas")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
NOTEBOOKS = sorted(EXAMPLES.glob("*.ipynb"))

# pyzmq warns on Windows' default Proactor event loop; the project turns RuntimeWarnings
# into errors, which would kill the kernel client's I/O thread and hang the test.
pytestmark = pytest.mark.filterwarnings("ignore:Proactor event loop:RuntimeWarning")
if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@pytest.fixture(scope="module", autouse=True)
def kernel_is_this_interpreter(tmp_path_factory):
    """Run notebooks with the interpreter executing the tests (not whatever 'python' is on PATH)."""
    root = tmp_path_factory.mktemp("jupyter")
    spec_dir = root / "kernels" / "python3"
    spec_dir.mkdir(parents=True)
    spec = {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Python 3",
        "language": "python",
    }
    (spec_dir / "kernel.json").write_text(json.dumps(spec))
    old = os.environ.get("JUPYTER_PATH")
    os.environ["JUPYTER_PATH"] = str(root) + (os.pathsep + old if old else "")
    os.environ.setdefault("MPLBACKEND", "Agg")
    yield
    if old is None:
        os.environ.pop("JUPYTER_PATH", None)
    else:
        os.environ["JUPYTER_PATH"] = old


def test_all_notebooks_present():
    assert len(NOTEBOOKS) == 10


@pytest.mark.slow
@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_notebook_executes(path):
    nb = nbformat.read(path, as_version=4)
    nbclient.NotebookClient(
        nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(EXAMPLES)}}
    ).execute()
    errors = [
        o for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"
    ]
    assert not errors
