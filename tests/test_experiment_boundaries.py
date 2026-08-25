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


class _ScopeLocals(ast.NodeVisitor):
    """Collect names Python treats as local without descending into nested scopes."""

    def __init__(self):
        self.names = set()
        self.declared_outer = set()

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            local_name = alias.asname or alias.name.split(".")[0]
            self.names.add(local_name)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        for alias in node.names:
            local_name = alias.asname or alias.name
            self.names.add(local_name)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef):
        self.names.add(node.name)

    def visit_Global(self, node: ast.Global):
        self.declared_outer.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal):
        self.declared_outer.update(node.names)

    def visit_Lambda(self, node: ast.Lambda):
        return

    def visit_ListComp(self, node: ast.ListComp):
        return

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp


def _scope_local_names(body) -> set[str]:
    bindings = _ScopeLocals()
    for statement in body:
        bindings.visit(statement)
    return bindings.names - bindings.declared_outer


def _tracked_module(module: str) -> str | None:
    if module in ("src.experiments", "experiments"):
        return "src.experiments"
    runner = _runner_name(module)
    return f"src.experiments.{runner}" if runner else None


def _stored_names(node) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        return set().union(*(_stored_names(element) for element in node.elts))
    if isinstance(node, ast.Starred):
        return _stored_names(node.value)
    return set()


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
    """Track imports and rebindings in statement order, then reject private runner access."""

    def __init__(self, path: Path):
        self.path = path
        self.bindings = {}
        self.violations = set()

    def scan(self, tree: ast.Module) -> list[str]:
        self._visit_scope(tree.body, {})
        return sorted(self.violations)

    def _visit_scope(self, body, inherited):
        previous = self.bindings
        self.bindings = dict(inherited)
        for statement in body:
            self.visit(statement)
        self.bindings = previous

    def _unbind(self, names):
        names = set(names)
        self.bindings = {
            access_path: module
            for access_path, module in self.bindings.items()
            if access_path[0] not in names
        }

    def _private_access(self, parts, access_path, module):
        if parts[:len(access_path)] != access_path:
            return None
        remainder = parts[len(access_path):]
        if module == "src.experiments":
            if len(remainder) < 2 or not _runner_name(remainder[0]):
                return None
            return (
                f"src.experiments.{remainder[0]}", remainder[1]
            ) if remainder[1].startswith("_") else None
        if remainder and remainder[0].startswith("_"):
            return module, remainder[0]
        return None

    def visit_Attribute(self, node: ast.Attribute):
        parts = _attribute_parts(node)
        if parts:
            for access_path, module in self.bindings.items():
                private_access = self._private_access(parts, access_path, module)
                if private_access:
                    runner_module, private_name = private_access
                    self.violations.add(
                        f"{self.path}:{node.lineno} accesses {private_name} on {runner_module}"
                    )
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            local_name = alias.asname or alias.name.split(".")[0]
            self._unbind({local_name})
            module = _tracked_module(alias.name)
            if module:
                access_path = alias.asname or alias.name
                self.bindings[tuple(access_path.split("."))] = module

    def visit_ImportFrom(self, node: ast.ImportFrom):
        source_module = _resolved_from_module(self.path, node)
        for alias in node.names:
            local_name = alias.asname or alias.name
            self._unbind({local_name})
            module = _tracked_module(f"{source_module}.{alias.name}".strip("."))
            if module:
                self.bindings[(local_name,)] = module

    def visit_Assign(self, node: ast.Assign):
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)
        self._unbind(set().union(*(_stored_names(target) for target in node.targets)))

    def visit_AnnAssign(self, node: ast.AnnAssign):
        self.visit(node.annotation)
        if node.value:
            self.visit(node.value)
        self.visit(node.target)
        self._unbind(_stored_names(node.target))

    def visit_AugAssign(self, node: ast.AugAssign):
        self.visit(node.target)
        self.visit(node.value)
        self._unbind(_stored_names(node.target))

    def visit_NamedExpr(self, node: ast.NamedExpr):
        self.visit(node.value)
        self.visit(node.target)
        self._unbind(_stored_names(node.target))

    def visit_For(self, node: ast.For):
        self.visit(node.iter)
        self.visit(node.target)
        self._unbind(_stored_names(node.target))
        for statement in [*node.body, *node.orelse]:
            self.visit(statement)

    visit_AsyncFor = visit_For

    def visit_With(self, node: ast.With):
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars:
                self.visit(item.optional_vars)
                self._unbind(_stored_names(item.optional_vars))
        for statement in node.body:
            self.visit(statement)

    visit_AsyncWith = visit_With

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        if node.type:
            self.visit(node.type)
        if node.name:
            self._unbind({node.name})
        for statement in node.body:
            self.visit(statement)

    def visit_Delete(self, node: ast.Delete):
        for target in node.targets:
            self.visit(target)
        self._unbind(set().union(*(_stored_names(target) for target in node.targets)))

    def visit_FunctionDef(self, node: ast.FunctionDef):
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        local_names = _scope_local_names(node.body) | _argument_names(node.args)
        inherited = {
            access_path: module
            for access_path, module in self.bindings.items()
            if access_path[0] not in local_names
        }
        self._visit_scope(node.body, inherited)
        self._unbind({node.name})

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef):
        for expression in [*node.decorator_list, *node.bases, *node.keywords]:
            self.visit(expression)
        inherited = {
            access_path: module
            for access_path, module in self.bindings.items()
            if access_path[0] not in _scope_local_names(node.body)
        }
        self._visit_scope(node.body, inherited)
        self._unbind({node.name})

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


def test_private_runner_access_through_parent_package_aliases_is_rejected(
    tmp_path, monkeypatch
):
    experiments = tmp_path / "src" / "experiments"
    experiments.mkdir(parents=True)
    (experiments / "stage_f_fixture.py").write_text(
        "import src.experiments as experiments_alias\n"
        "experiments_alias.analyze_stage_f._module_private()\n"
        "\n"
        "def absolute_function():\n"
        "    import src.experiments as local_experiments\n"
        "    local_experiments.stage_f_qwen._absolute_private()\n"
        "\n"
        "def from_function():\n"
        "    from src import experiments as imported_experiments\n"
        "    imported_experiments.analyze_stage_f._from_private()\n"
    )
    monkeypatch.chdir(tmp_path)
    fixture = Path("src/experiments/stage_f_fixture.py")

    assert set(_private_runner_violations(fixture)) == {
        f"{fixture}:2 accesses _module_private on src.experiments.analyze_stage_f",
        f"{fixture}:6 accesses _absolute_private on src.experiments.stage_f_qwen",
        f"{fixture}:10 accesses _from_private on src.experiments.analyze_stage_f",
    }


def test_runner_alias_reassignment_stops_private_access_detection(tmp_path, monkeypatch):
    experiments = tmp_path / "src" / "experiments"
    experiments.mkdir(parents=True)
    (experiments / "stage_f_fixture.py").write_text(
        "from . import analyze_stage_f as runner\n"
        "runner._module_violation()\n"
        "runner = object()\n"
        "runner._module_local_only\n"
        "\n"
        "def function_scope():\n"
        "    import src.experiments.analyze_stage_f as local_runner\n"
        "    local_runner._function_violation()\n"
        "    local_runner = object()\n"
        "    local_runner._function_local_only\n"
    )
    monkeypatch.chdir(tmp_path)
    fixture = Path("src/experiments/stage_f_fixture.py")

    assert set(_private_runner_violations(fixture)) == {
        f"{fixture}:2 accesses _module_violation on src.experiments.analyze_stage_f",
        f"{fixture}:8 accesses _function_violation on src.experiments.analyze_stage_f",
    }


def test_global_and_nonlocal_declarations_preserve_alias_until_assignment(
    tmp_path, monkeypatch
):
    experiments = tmp_path / "src" / "experiments"
    experiments.mkdir(parents=True)
    (experiments / "stage_f_fixture.py").write_text(
        "from . import analyze_stage_f as global_runner\n"
        "\n"
        "def global_scope():\n"
        "    global global_runner\n"
        "    global_runner._global_violation()\n"
        "    global_runner = object()\n"
        "    global_runner._global_local_only\n"
        "\n"
        "def outer_scope():\n"
        "    from . import analyze_stage_f as nonlocal_runner\n"
        "\n"
        "    def inner_scope():\n"
        "        nonlocal nonlocal_runner\n"
        "        nonlocal_runner._nonlocal_violation()\n"
        "        nonlocal_runner = object()\n"
        "        nonlocal_runner._nonlocal_local_only\n"
    )
    monkeypatch.chdir(tmp_path)
    fixture = Path("src/experiments/stage_f_fixture.py")

    assert set(_private_runner_violations(fixture)) == {
        f"{fixture}:5 accesses _global_violation on src.experiments.analyze_stage_f",
        f"{fixture}:14 accesses _nonlocal_violation on src.experiments.analyze_stage_f",
    }


def test_for_and_async_for_targets_expire_runner_aliases(tmp_path, monkeypatch):
    experiments = tmp_path / "src" / "experiments"
    experiments.mkdir(parents=True)
    (experiments / "stage_f_fixture.py").write_text(
        "from . import analyze_stage_f as runner\n"
        "runner._for_violation()\n"
        "for runner in rows:\n"
        "    runner._for_local_only\n"
        "\n"
        "async def async_scope():\n"
        "    import src.experiments.analyze_stage_f as async_runner\n"
        "    async_runner._async_for_violation()\n"
        "    async for async_runner in rows:\n"
        "        async_runner._async_for_local_only\n"
    )
    monkeypatch.chdir(tmp_path)
    fixture = Path("src/experiments/stage_f_fixture.py")

    assert set(_private_runner_violations(fixture)) == {
        f"{fixture}:2 accesses _for_violation on src.experiments.analyze_stage_f",
        f"{fixture}:8 accesses _async_for_violation on src.experiments.analyze_stage_f",
    }


def test_with_and_async_with_targets_expire_runner_aliases(tmp_path, monkeypatch):
    experiments = tmp_path / "src" / "experiments"
    experiments.mkdir(parents=True)
    (experiments / "stage_f_fixture.py").write_text(
        "from . import analyze_stage_f as runner\n"
        "runner._with_violation()\n"
        "with manager() as runner:\n"
        "    runner._with_local_only\n"
        "\n"
        "async def async_scope():\n"
        "    import src.experiments.analyze_stage_f as async_runner\n"
        "    async_runner._async_with_violation()\n"
        "    async with manager() as async_runner:\n"
        "        async_runner._async_with_local_only\n"
    )
    monkeypatch.chdir(tmp_path)
    fixture = Path("src/experiments/stage_f_fixture.py")

    assert set(_private_runner_violations(fixture)) == {
        f"{fixture}:2 accesses _with_violation on src.experiments.analyze_stage_f",
        f"{fixture}:8 accesses _async_with_violation on src.experiments.analyze_stage_f",
    }


def test_except_target_and_del_expire_runner_aliases(tmp_path, monkeypatch):
    experiments = tmp_path / "src" / "experiments"
    experiments.mkdir(parents=True)
    (experiments / "stage_f_fixture.py").write_text(
        "from . import analyze_stage_f as runner\n"
        "runner._except_violation()\n"
        "try:\n"
        "    pass\n"
        "except Exception as runner:\n"
        "    runner._except_local_only\n"
        "\n"
        "import src.experiments.analyze_stage_f as deleted_runner\n"
        "deleted_runner._del_violation()\n"
        "del deleted_runner\n"
        "deleted_runner._del_local_only\n"
    )
    monkeypatch.chdir(tmp_path)
    fixture = Path("src/experiments/stage_f_fixture.py")

    assert set(_private_runner_violations(fixture)) == {
        f"{fixture}:2 accesses _except_violation on src.experiments.analyze_stage_f",
        f"{fixture}:9 accesses _del_violation on src.experiments.analyze_stage_f",
    }
