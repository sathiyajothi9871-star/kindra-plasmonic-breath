"""
Citation renumbering.

References are authored with stable identifiers and renumbered at build time in
order of first appearance, so the printed list satisfies the sequential
numbering convention without the author having to maintain it by hand.  Adjacent
runs of three or more are contracted to a range.
"""
from __future__ import annotations
import re

TOKEN = re.compile(r"\[\[(\d+(?:\s*,\s*\d+)*)\]\]")


class Renumber:
    def __init__(self):
        self.order = []
        self.map = {}

    def _num(self, k):
        if k not in self.map:
            self.order.append(k)
            self.map[k] = len(self.order)
        return self.map[k]

    def scan_and_replace(self, text: str) -> str:
        def rep(m):
            keys = [int(x) for x in m.group(1).split(",")]
            nums = sorted(self._num(k) for k in keys)
            return "[" + _format(nums) + "]"
        return TOKEN.sub(rep, text)

    def final_list(self, authored):
        return [authored[k - 1] for k in self.order]


def _format(nums):
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        if j - i >= 2:
            out.append(f"{nums[i]}]-[{nums[j]}")
        else:
            out.extend(str(n) for n in nums[i:j + 1])
        i = j + 1
    return "], [".join(out)
