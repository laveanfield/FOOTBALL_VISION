# Contributing

## Workflow

* `main` always runs. Work on branches named `feat/<phase>-<topic>` (for example
  `feat/p3-dlt`) and open a pull request; another member reviews it.
* Commit messages: `[p3] add Hartley normalization`. Commit your own work, since the
  commit history documents each member's technical contribution.
* Tag milestones: `m1`, `m2`, `m3`, `v1.0`.
* A task is **done** when: the code runs from the CLI/config, a figure shows the result,
  the metric is logged in `reports/experiments.md`, the docstring is complete, a reviewer
  approved it, and the end-to-end pipeline still works.

## Implementing a stub

1. Find open work: `grep -rn "TODO(" src`. `TODO(A|B|C|D)` names the owner.
2. Read the docstring: it describes the algorithm, arguments and result.
3. Implement the function; keep the signature unless the team agrees to change it (then
   update callers in `pipeline/stages.py` and the tests).
4. Remove the `xfail` marker of its contract test in `tests/test_contracts.py` (if any), add
   more tests, and run `make check`.
5. Replace the `Raises: NotImplementedError` entry of the docstring with the exceptions the
   function really raises (or drop the section).

## Code style

* Python >= 3.10, type hints everywhere, `from __future__ import annotations`.
* Names: `snake_case` functions and variables, `CapWords` classes, `UPPER_CASE` constants.
  Write the homography as `homography` (or `h_previous`), not `H`, in code.
* Line length 88; `ruff format` formats the code, `ruff check` lints it.
* Comments and identifiers are English. TODO format: `# TODO(owner): what to do`.

## Docstrings: Google style

We follow the [Google Python Style Guide, section 3.8](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
It is enforced by `ruff` (pydocstyle rules, `convention = "google"`), by `pydoclint`
(sections must match the signature) and by `tests/test_docstrings.py` (dataclass fields).

* Every module, public class, public function and method has a docstring. A module
  docstring starts the file.
* The summary is one line, ends with a period, and is **descriptive** ("Computes the IoU
  matrix."), not imperative. A blank line separates it from the rest.
* Do not repeat types that the annotations already give; describe meaning, units and shape.
* Sections: `Args:`, `Returns:`, `Yields:` (generators), `Raises:`, plus `Attributes:` in
  class docstrings (public attributes and dataclass fields). Constructor arguments go in
  the `__init__` docstring.

```python
def iou_matrix(boxes_a: npt.ArrayLike, boxes_b: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Computes pairwise intersection-over-union between two box sets.

    Args:
        boxes_a: Array of shape ``(N, 4)`` in ``(x1, y1, x2, y2)`` format.
        boxes_b: Array of shape ``(M, 4)`` in the same format.

    Returns:
        Array of shape ``(N, M)`` where entry ``(i, j)`` is the IoU of
        ``boxes_a[i]`` and ``boxes_b[j]``.

    Raises:
        ValueError: If a box array does not have four columns.
    """
```

Run `make check` before every push.
