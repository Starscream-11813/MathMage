"""Conservative syntax checks for computation-based evidence.

These checks reject obvious constant-output pseudo-verifiers. They cannot certify
that a generated program models the mathematical problem faithfully.
"""
from __future__ import annotations

import ast


_COMPUTING_CALLS = {"sum", "prod", "factorial", "comb", "perm", "gcd", "lcm", "sqrt", "isqrt",
                    "Fraction", "Decimal", "pow", "min", "max", "abs", "all", "any", "len",
                    "sin", "cos", "tan", "exp", "log", "log2", "log10", "ceil", "floor"}


def computational_evidence(code: str, purpose: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return False, "program could not be parsed"
    names = set()
    functions = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and any(isinstance(child, (ast.BinOp, ast.Compare, ast.For, ast.While, ast.comprehension))
                         for child in ast.walk(node))}

    def computed(node):
        if isinstance(node, (ast.BinOp, ast.Compare, ast.comprehension)):
            return True
        if isinstance(node, ast.Name):
            return node.id in names
        if isinstance(node, ast.Call):
            function = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            if function in _COMPUTING_CALLS or function in functions:
                return True
        return any(computed(child) for child in ast.iter_child_nodes(node))

    # Fixed point supports assignments whose computed dependency appears later
    # in a loop body or function definition; no model code executes here.
    for _ in range(5):
        previous = set(names)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and computed(node.value):
                for target in node.targets:
                    names.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
            elif isinstance(node, ast.AugAssign):
                names.update(n.id for n in ast.walk(node.target) if isinstance(n, ast.Name))
        if names == previous:
            break
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    prints = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Name) and node.func.id == "print"]

    def contains_marker(node, markers):
        return any(isinstance(child, ast.Constant) and isinstance(child.value, str)
                   and any(marker in child.value for marker in markers) for child in ast.walk(node))

    def guarded_by_computation(node):
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.If, ast.IfExp, ast.While)) and computed(node.test):
                return True
        return False

    def explicit_abstention(node):
        # A literal failure branch is allowed alongside a computed answer. Only
        # recognize the exact printed sentinel, never a dynamic expression or a
        # line that also prints a proposed answer.
        def literal_text(value):
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return value.value
            if isinstance(value, ast.JoinedStr) and all(isinstance(part, ast.Constant)
                                                       and isinstance(part.value, str) for part in value.values):
                return "".join(part.value for part in value.values)
            return None

        parts = [literal_text(arg) for arg in node.args]
        if not parts or any(part is None for part in parts):
            return False
        separator, ending = " ", "\n"
        for keyword in node.keywords:
            value = literal_text(keyword.value)
            if value is None or keyword.arg not in {"sep", "end"}:
                return False
            if keyword.arg == "sep":
                separator = value
            else:
                ending = value
        return (separator.join(parts) + ending).strip().upper() == "ANSWER: NONE"

    if purpose == "independent_forward":
        marked = [node for node in prints if contains_marker(node, ("ANSWER:",))]
        outputs = [node for node in (marked or prints) if not explicit_abstention(node)]
        if not outputs or not all(any(computed(arg) for arg in node.args) for node in outputs):
            return False, "final output has no visible computational dependency"
    else:
        if not any(isinstance(node, ast.Compare) for node in ast.walk(tree)):
            return False, "claim checker contains no explicit comparison"
        marked = [node for node in prints if contains_marker(node, ("CLAIM_FALSE", "CLAIM_TRUE", "PASS:", "FAIL:"))]
        if marked and not all(any(computed(arg) for arg in node.args) or guarded_by_computation(node)
                              for node in marked):
            return False, "claim output has no visible computational dependency"
    return True, "syntax indicates computation; semantic faithfulness remains unverified"
