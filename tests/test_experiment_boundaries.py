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
    remainder = module.removeprefix(prefix) if module.startswith(prefix) else module
    if "." in remainder or not remainder.startswith(RUNNER_PREFIXES):
        return None
    return remainder


def _imports(path: Path):
    return [
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]


def _argument_names(arguments: ast.arguments) -> set[str]:
    positional = [*arguments.posonlyargs, *arguments.args]
    names = {argument.arg for argument in [*positional, *arguments.kwonlyargs]}
    if arguments.vararg:
        names.add(arguments.vararg.arg)
    if arguments.kwarg:
        names.add(arguments.kwarg.arg)
    return names


class _ScopeBindings(ast.NodeVisitor):
    """Collect names and runner-module aliases bound directly in one lexical scope."""

    def __init__(self, path: Path):
        self.path = path
        self.names = set()
        self.runners = {}

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            local_name = alias.asname or alias.name.split(".")[0]
            self.names.add(local_name)
            if _runner_name(alias.name):
                access_path = alias.asname or alias.name
                self.runners[tuple(access_path.split("."))] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = _resolved_from_module(self.path, node)
        for alias in node.names:
            local_name = alias.asname or alias.name
            self.names.add(local_name)
            if module in ("src.experiments", "experiments") and _runner_name(alias.name):
                runner_module = f"src.experiments.{alias.name}"
                self.runners[(local_name,)] = runner_module

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef):
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda):
        return

    def visit_ListComp(self, node: ast.ListComp):
        return

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp


def _scope_bindings(path: Path, body) -> _ScopeBindings:
    bindings = _ScopeBindings(path)
    for statement in body:
        bindings.visit(statement)
    return bindings


def _attribute_parts(node: ast.Attribute) -> tuple[str, ...] | None:
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    return (current.id, *reversed(parts))


class _PrivateRunnerAttributeVisitor(ast.NodeVisitor):
    """Find private attributes accessed through runner-module imports, respecting scopes."""

    def __init__(self, path: Path):
        self.path = path
        self.bindings = {}
        self.violations = set()

    def scan(self, tree: ast.Module) -> list[str]:
        self._visit_scope(tree.body, {})
        return sorted(self.violations)

    def _visit_scope(self, body, inherited, argument_names=()):
        local = _scope_bindings(self.path, body)
        local_names = local.names | set(argument_names)
        effective = {
            access_path: module
            for access_path, module in inherited.items()
            if access_path[0] not in local_names
        }
        effective.update(local.runners)
        previous = self.bindings
        self.bindings = effective
        for statement in body:
            self.visit(statement)
        self.bindings = previous

    def visit_Attribute(self, node: ast.Attribute):
        parts = _attribute_parts(node)
        if parts:
            for access_path, module in self.bindings.items():
                if (
                    len(parts) > len(access_path)
                    and parts[:len(access_path)] == access_path
                    and parts[len(access_path)].startswith("_")
                ):
                    private_name = parts[len(access_path)]
                    self.violations.add(
                        f"{self.path}:{node.lineno} accesses {private_name} on {module}"
                    )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        self._visit_scope(node.body, self.bindings, _argument_names(node.args))

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef):
        for expression in [*node.decorator_list, *node.bases, *node.keywords]:
            self.visit(expression)
        self._visit_scope(node.body, self.bindings)

    def visit_Lambda(self, node: ast.Lambda):
        previous = self.bindings
        argument_names = _argument_names(node.args)
        self.bindings = {
            access_path: module
            for access_path, module in previous.items()
            if access_path[0] not in argument_names
        }
        self.visit(node.body)
        self.bindings = previous


def _private_runner_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = _resolved_from_module(path, node)
        private = [alias.name for alias in node.names if alias.name.startswith("_")]
        if private and _runner_name(module):
            violations.append(
                f"{path}:{node.lineno} imports {', '.join(private)} from {module}"
            )
    violations.extend(_PrivateRunnerAttributeVisitor(path).scan(tree))
    return sorted(violations)


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
        violations.extend(_private_runner_violations(path))

    assert violations == []


def test_private_runner_access_through_relative_and_absolute_module_aliases_is_rejected(
    tmp_path, monkeypatch
):
    experiments = tmp_path / "src" / "experiments"
    experiments.mkdir(parents=True)
    (experiments / "analyze_stage_f.py").write_text("PUBLIC_API = object()\n")
    (experiments / "stage_f_fixture.py").write_text(
        "from .analyze_stage_f import _direct as direct_alias\n"
        "from . import analyze_stage_f as relative_analyzer\n"
        "relative_analyzer._relative_private()\n"
        "\n"
        "def absolute_function():\n"
        "    import src.experiments.analyze_stage_f as absolute_analyzer\n"
        "    absolute_analyzer._absolute_private()\n"
        "\n"
        "def relative_function():\n"
        "    from . import analyze_stage_f as local_analyzer\n"
        "    local_analyzer._function_private()\n"
    )
    monkeypatch.chdir(tmp_path)
    fixture = Path("src/experiments/stage_f_fixture.py")

    assert set(_private_runner_violations(fixture)) == {
        f"{fixture}:1 imports _direct from src.experiments.analyze_stage_f",
        f"{fixture}:3 accesses _relative_private on src.experiments.analyze_stage_f",
        f"{fixture}:7 accesses _absolute_private on src.experiments.analyze_stage_f",
        f"{fixture}:11 accesses _function_private on src.experiments.analyze_stage_f",
    }


def test_runner_module_aliases_allow_public_and_unrelated_private_attributes(
    tmp_path, monkeypatch
):
    experiments = tmp_path / "src" / "experiments"
    experiments.mkdir(parents=True)
    (experiments / "analyze_stage_f.py").write_text("PUBLIC_API = object()\n")
    (experiments / "stage_f_fixture.py").write_text(
        "from . import analyze_stage_f as relative_analyzer\n"
        "import src.experiments.analyze_stage_f as absolute_analyzer\n"
        "import unrelated_package as unrelated\n"
        "relative_analyzer.PUBLIC_API\n"
        "absolute_analyzer.public_api\n"
        "unrelated._private_api\n"
        "\n"
        "def shadowed_aliases(relative_analyzer, absolute_analyzer):\n"
        "    relative_analyzer._local_private\n"
        "    absolute_analyzer._local_private\n"
    )
    monkeypatch.chdir(tmp_path)
    fixture = Path("src/experiments/stage_f_fixture.py")

    assert _private_runner_violations(fixture) == []
