import ast
from pathlib import Path


EXPERIMENTS = Path("src/experiments")
RUNNER_PREFIXES = ("stage_", "analyze_stage_")


def _module_name(path: Path) -> str:
    relative = path.with_suffix("").as_posix().replace("/", ".")
    return relative.removesuffix(".__init__")


def _resolved_from_module(path: Path, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""

    package = _module_name(path).split(".")[:-1]
    keep = len(package) - (node.level - 1)
    base = package[:keep]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def _runner_name(module: str) -> str | None:
    prefix = "src.experiments."
    if not module.startswith(prefix):
        return None
    remainder = module.removeprefix(prefix)
    if "." in remainder or not remainder.startswith(RUNNER_PREFIXES):
        return None
    return remainder


def _imports(path: Path):
    return [
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]


def test_shared_modules_never_import_runner_modules():
    violations = []
    for path in sorted((EXPERIMENTS / "shared").glob("*.py")):
        for node in _imports(path):
            if isinstance(node, ast.Import):
                imported_runners = [
                    alias.name for alias in node.names if _runner_name(alias.name)
                ]
            else:
                module = _resolved_from_module(path, node)
                imported_runners = [module] if _runner_name(module) else []
                if module == "src.experiments":
                    imported_runners.extend(
                        f"{module}.{alias.name}"
                        for alias in node.names
                        if alias.name.startswith(RUNNER_PREFIXES)
                    )
            for module in imported_runners:
                violations.append(f"{path}:{node.lineno} imports {module}")

    assert violations == []


def test_runners_never_import_private_names_from_other_runners():
    violations = []
    for path in sorted(EXPERIMENTS.glob("*.py")):
        if not path.stem.startswith(RUNNER_PREFIXES):
            continue
        for node in _imports(path):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = _resolved_from_module(path, node)
            private = [alias.name for alias in node.names if alias.name.startswith("_")]
            if private and _runner_name(module):
                violations.append(
                    f"{path}:{node.lineno} imports {', '.join(private)} from {module}"
                )

    assert violations == []
