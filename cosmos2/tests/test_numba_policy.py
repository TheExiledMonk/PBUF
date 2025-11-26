from pathlib import Path


def test_numba_imports_limited_to_kernels():
    """
    Enforce that only numeric kernel modules rely on Numba.
    High-level orchestration (api/models/threads/utils/...) must stay pure Python.
    """

    root = Path(__file__).resolve().parents[1]
    forbidden = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root)
        if str(rel).startswith("kernels/"):
            continue
        if str(rel).startswith("tests/"):
            continue
        text = path.read_text(encoding="utf-8")
        if "import numba" in text or "from numba" in text:
            forbidden.append(rel)
    assert not forbidden, f"Unexpected numba imports outside kernels: {forbidden}"


def test_no_numpy_allocations_inside_numba_loops():
    """
    Guardrail to avoid allocations inside Numba-jitted loops: flag np.empty/zeros/ones/array/asarray created inside a loop body.
    Heuristic parser tracks indentation-based loops across kernel modules.
    """

    kernel_dir = Path(__file__).resolve().parents[1] / "kernels"
    offenders: list[str] = []
    alloc_tokens = ("np.empty", "np.zeros", "np.ones", "np.array(", "np.asarray(", "np.linspace(")

    for path in kernel_dir.rglob("*.py"):
        lines = path.read_text(encoding="utf-8").splitlines()
        loop_stack: list[int] = []
        for idx, line in enumerate(lines, start=1):
            stripped = line.lstrip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(line) - len(stripped)
            while loop_stack and indent < loop_stack[-1]:
                loop_stack.pop()
            if stripped.startswith("for ") and stripped.endswith(":"):
                # Expect loop body to be indented beyond current indent.
                loop_stack.append(indent + 4)
                continue
            if loop_stack and any(token in stripped for token in alloc_tokens) and indent >= loop_stack[-1]:
                offenders.append(f"{path.relative_to(kernel_dir)}:{idx}")

    assert not offenders, f"Found NumPy allocations inside Numba loops (preallocate outside): {offenders}"
