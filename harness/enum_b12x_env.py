import pathlib
import re

root = pathlib.Path("/usr/local/lib/python3.12/dist-packages/b12x")
keys = {}
pat = re.compile(
    r"""os\.(?:environ\.get|getenv)\(\s*['\"](B12X_[A-Z0-9_]+)['\"](?:\s*,\s*['\"]([^'\"]*)['\"])?"""
)
for p in root.rglob("*.py"):
    t = p.read_text(errors="ignore")
    for m in pat.finditer(t):
        keys.setdefault(m.group(1), (m.group(2), p.name))
for k in sorted(keys):
    d, f = keys[k]
    print(f"{k} default={d!r} {f}")
