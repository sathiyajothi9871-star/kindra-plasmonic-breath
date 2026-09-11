"""
Conversion of a restricted LaTeX syntax into Office MathML.

Every mathematical object in the manuscript - display equations, inline symbols
in running text, symbols inside table cells and captions, and the variables used
in the algorithm listings - is emitted through this module, so the resulting
document contains native Word equation objects throughout and no images of
mathematics.
"""
from __future__ import annotations
import re

M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
RPR = ('<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/>'
       '<w:sz w:val="20"/></w:rPr>')
CTRL = f"<m:ctrlPr>{RPR}</m:ctrlPr>"

GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ϵ", "varepsilon": "ε", "zeta": "ζ", "eta": "η",
    "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ",
    "upsilon": "υ", "phi": "ϕ", "varphi": "φ", "chi": "χ",
    "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ",
    "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ",
    "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}
SYMS = {
    "cdot": "⋅", "times": "×", "odot": "⊙", "otimes": "⊗",
    "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥",
    "neq": "≠", "approx": "≈", "equiv": "≡", "sim": "∼",
    "propto": "∝", "in": "∈", "notin": "∉", "subset": "⊂",
    "to": "→", "rightarrow": "→", "mapsto": "↦",
    "leftarrow": "←", "Rightarrow": "⇒", "partial": "∂",
    "nabla": "∇", "infty": "∞", "pm": "±", "mp": "∓",
    "ll": "≪", "gg": "≫", "forall": "∀", "exists": "∃",
    "ldots": "…", "cdots": "⋯", "dots": "…", "perp": "⊥",
    "circ": "∘", "star": "⋆", "prime": "′", "angle": "∠",
    "top": "⊤", "setminus": "∖", "cup": "∪", "cap": "∩",
    "emptyset": "∅", "Re": "ℜ", "Im": "ℑ", "hbar": "ℏ",
    "ast": "∗", "bullet": "∙", "oplus": "⊕", "sqrtsym": "√",
}
BB = {"R": "ℝ", "N": "ℕ", "Z": "ℤ", "C": "ℂ", "E": "\U0001D53C",
      "P": "ℙ", "Q": "ℚ"}
FUNCS = ("softplus", "sigmoid", "exp", "log", "ln", "sin", "cos", "tan", "max",
         "min", "argmax", "argmin", "diag", "tr", "det", "softmax", "sgn",
         "erf", "median", "clip", "vec", "sup", "inf", "lim", "Var", "Cov")
ACCENTS = {"hat": "̂", "tilde": "̃", "bar": "̄", "dot": "̇",
           "ddot": "̈", "vec": "⃗", "check": "̌"}
BIGOPS = {"sum": "∑", "prod": "∏", "int": "∫", "oint": "∮",
          "bigcup": "⋃", "bigcap": "⋂", "coprod": "∐"}
OPEN = {"(": "(", "[": "[", "\\{": "{", "|": "|", "\\|": "‖",
        "\\langle": "⟨", "\\lceil": "⌈", "\\lfloor": "⌊"}
CLOSE = {")": ")", "]": "]", "\\}": "}", "|": "|", "\\|": "‖",
         "\\rangle": "⟩", "\\rceil": "⌉", "\\rfloor": "⌋"}


def esc(t: str) -> str:
    return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def run(text: str, nor=False, bold=False, script=None) -> str:
    if text == "":
        return ""
    pr = ""
    inner = ""
    if nor:
        inner += "<m:nor/>"
    if bold:
        inner += '<m:sty m:val="b"/>'
    if script:
        inner += f'<m:scr m:val="{script}"/><m:sty m:val="p"/>'
    if inner:
        pr = f"<m:rPr>{inner}</m:rPr>"
    return f"<m:r>{pr}{RPR}<m:t xml:space=\"preserve\">{esc(text)}</m:t></m:r>"


# ----------------------------------------------------------------------
class Tok:
    def __init__(self, s):
        self.s = s
        self.i = 0

    def eof(self):
        return self.i >= len(self.s)

    def peek(self):
        return self.s[self.i] if not self.eof() else ""

    def next(self):
        c = self.s[self.i]; self.i += 1
        return c

    def command(self):
        """Read a backslash command name."""
        j = self.i
        while j < len(self.s) and (self.s[j].isalpha()):
            j += 1
        if j == self.i:
            j = self.i + 1
        name = self.s[self.i:j]
        self.i = j
        return name

    def group(self):
        """Read a braced group or a single token, returning its source."""
        while not self.eof() and self.peek() == " ":
            self.next()
        if self.eof():
            return ""
        if self.peek() == "{":
            depth = 0
            j = self.i
            while j < len(self.s):
                if self.s[j] == "{":
                    depth += 1
                elif self.s[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            out = self.s[self.i + 1:j]
            self.i = j + 1
            return out
        if self.peek() == "\\":
            self.next()
            return "\\" + self.command()
        return self.next()


def _split_top(src, sep):
    """Split on a separator that is not inside braces."""
    out, buf, depth, i = [], "", 0, 0
    while i < len(src):
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        if depth == 0 and src.startswith(sep, i):
            out.append(buf); buf = ""; i += len(sep); continue
        buf += c; i += 1
    out.append(buf)
    return out


def convert(src: str) -> str:
    """LaTeX subset -> OMML fragment (no wrapper element)."""
    t = Tok(src)
    return _seq(t)


def _seq(t: Tok, stop=None) -> str:
    out = []
    buf = ""

    def flush():
        nonlocal buf
        if buf:
            out.append(run(buf)); buf = ""

    while not t.eof():
        c = t.peek()
        if stop and t.s.startswith(stop, t.i):
            break
        if c == "\\":
            t.next()
            name = t.command()
            flush()
            out.append(_command(t, name))
        elif c == "_" or c == "^":
            t.next()
            base = out.pop() if (out and not buf) else run(buf[-1] if buf else "")
            if buf:
                base = run(buf[-1]); buf = buf[:-1]
                flush()
                out.append(_script(t, base, c))
            else:
                out.append(_script(t, base, c))
        elif c == "{":
            g = t.group()
            flush()
            out.append(_seq(Tok(g)))
        elif c == "}":
            t.next()
        elif c in "([":
            t.next()
            flush()
            out.append(_auto_delim(t, c, {"(": ")", "[": "]"}[c]))
        elif c == " ":
            t.next()
            buf += " "
        else:
            buf += t.next()
    flush()
    return "".join(out)


def _auto_delim(t: Tok, op, cl):
    """Balanced ( ) or [ ] group rendered as a delimiter object."""
    depth = 1
    j = t.i
    while j < len(t.s):
        if t.s[j] == op:
            depth += 1
        elif t.s[j] == cl:
            depth -= 1
            if depth == 0:
                break
        j += 1
    if j >= len(t.s):
        return run(op)
    inner = t.s[t.i:j]
    t.i = j + 1
    body = _seq(Tok(inner))
    return _delim(op, cl, [body])


def _delim(op, cl, parts):
    beg = f'm:begChr m:val="{esc(op)}"' if op else 'm:begChr m:val=""'
    end = f'm:endChr m:val="{esc(cl)}"' if cl else 'm:endChr m:val=""'
    es = "".join(f"<m:e>{p}</m:e>" for p in parts)
    sep = '<m:sepChr m:val="|"/>' if len(parts) > 1 else ""
    return (f"<m:d><m:dPr><{beg}/>{sep}<{end}/>{CTRL}</m:dPr>{es}</m:d>")


def _script(t: Tok, base, kind):
    g = t.group()
    body = _seq(Tok(g))
    other = None
    save = t.i
    while not t.eof() and t.peek() == " ":
        t.next()
    if not t.eof() and t.peek() in "_^" and t.peek() != kind:
        k2 = t.next()
        other = _seq(Tok(t.group()))
        if kind == "_":
            return (f"<m:sSubSup><m:sSubSupPr>{CTRL}</m:sSubSupPr><m:e>{base}</m:e>"
                    f"<m:sub>{body}</m:sub><m:sup>{other}</m:sup></m:sSubSup>")
        return (f"<m:sSubSup><m:sSubSupPr>{CTRL}</m:sSubSupPr><m:e>{base}</m:e>"
                f"<m:sub>{other}</m:sub><m:sup>{body}</m:sup></m:sSubSup>")
    t.i = save
    tag = "sSub" if kind == "_" else "sSup"
    slot = "sub" if kind == "_" else "sup"
    return (f"<m:{tag}><m:{tag}Pr>{CTRL}</m:{tag}Pr><m:e>{base}</m:e>"
            f"<m:{slot}>{body}</m:{slot}></m:{tag}>")


def _bigop(t: Tok, glyph):
    sub = sup = ""
    while True:
        save = t.i
        while not t.eof() and t.peek() == " ":
            t.next()
        if not t.eof() and t.peek() == "_":
            t.next(); sub = _seq(Tok(t.group()))
        elif not t.eof() and t.peek() == "^":
            t.next(); sup = _seq(Tok(t.group()))
        else:
            t.i = save; break
    body = _seq(t)
    pr = f'<m:naryPr><m:chr m:val="{glyph}"/><m:limLoc m:val="undOvr"/>'
    if not sub:
        pr += '<m:subHide m:val="1"/>'
    if not sup:
        pr += '<m:supHide m:val="1"/>'
    pr += f"{CTRL}</m:naryPr>"
    return (f"<m:nary>{pr}<m:sub>{sub}</m:sub><m:sup>{sup}</m:sup>"
            f"<m:e>{body}</m:e></m:nary>")


def _matrix(rows, op="[", cl="]"):
    mr = ""
    for r in rows:
        mr += "<m:mr>" + "".join(f"<m:e>{c}</m:e>" for c in r) + "</m:mr>"
    ncol = max(len(r) for r in rows)
    mm = (f'<m:m><m:mPr><m:mcs><m:mc><m:mcPr><m:count m:val="{ncol}"/>'
          f'<m:mcJc m:val="center"/></m:mcPr></m:mc></m:mcs>{CTRL}</m:mPr>{mr}</m:m>')
    return _delim(op, cl, [mm]) if op else mm


def _cases(rows):
    mr = ""
    for r in rows:
        mr += "<m:mr>" + "".join(f"<m:e>{c}</m:e>" for c in r) + "</m:mr>"
    mm = (f'<m:m><m:mPr><m:mcs><m:mc><m:mcPr><m:count m:val="2"/>'
          f'<m:mcJc m:val="left"/></m:mcPr></m:mc></m:mcs>{CTRL}</m:mPr>{mr}</m:m>')
    return _delim("{", "", [mm])


def _command(t: Tok, name):
    if name in GREEK:
        return run(GREEK[name])
    if name in SYMS:
        return run(SYMS[name])
    if name in BIGOPS:
        return _bigop(t, BIGOPS[name])
    if name in ACCENTS:
        g = _seq(Tok(t.group()))
        return (f'<m:acc><m:accPr><m:chr m:val="{ACCENTS[name]}"/>{CTRL}</m:accPr>'
                f"<m:e>{g}</m:e></m:acc>")
    if name == "frac" or name == "dfrac" or name == "tfrac":
        a = _seq(Tok(t.group())); b = _seq(Tok(t.group()))
        return (f"<m:f><m:fPr>{CTRL}</m:fPr><m:num>{a}</m:num>"
                f"<m:den>{b}</m:den></m:f>")
    if name == "sqrt":
        deg = ""
        while not t.eof() and t.peek() == " ":
            t.next()
        if not t.eof() and t.peek() == "[":
            j = t.s.index("]", t.i)
            deg = _seq(Tok(t.s[t.i + 1:j])); t.i = j + 1
        g = _seq(Tok(t.group()))
        pr = "<m:radPr>" + ("" if deg else '<m:degHide m:val="1"/>') + CTRL + "</m:radPr>"
        return f"<m:rad>{pr}<m:deg>{deg}</m:deg><m:e>{g}</m:e></m:rad>"
    if name in ("mathbf", "bm", "boldsymbol"):
        g = t.group()
        return _bold(_seq(Tok(g)))
    if name in ("mathrm", "text", "textrm", "mbox", "operatorname"):
        return run(t.group(), nor=True)
    if name == "mathbb":
        g = t.group()
        return run(BB.get(g, g), script="double-struck")
    if name in ("mathcal", "mathscr"):
        return run(t.group(), script="script")
    if name in FUNCS:
        return run(name, nor=True)
    if name == "left":
        op = t.next()
        if op == "\\":
            op = "\\" + t.command()
        body_src, close = _until_right(t)
        body = _seq(Tok(body_src))
        return _delim(OPEN.get(op, op), CLOSE.get(close, close), [body])
    if name == "right":
        return ""
    if name == "begin":
        env = t.group()
        return _environment(t, env)
    if name == "end":
        t.group(); return ""
    if name in (",", ";", ":", "!", " ", "quad", "qquad", "thinspace"):
        return run(" " if name not in ("quad", "qquad") else "  ")
    if name == "\\":
        return ""
    if name == "%":
        return run("%")
    return run(name, nor=True)


def _until_right(t: Tok):
    depth = 0
    j = t.i
    while j < len(t.s):
        if t.s.startswith("\\left", j):
            depth += 1; j += 5; continue
        if t.s.startswith("\\right", j):
            if depth == 0:
                body = t.s[t.i:j]
                k = j + 6
                if k < len(t.s) and t.s[k] == "\\":
                    k += 1
                    m = re.match(r"[A-Za-z]+|.", t.s[k:])
                    close = "\\" + m.group(0); k += len(m.group(0))
                else:
                    close = t.s[k] if k < len(t.s) else ""
                    k += 1
                t.i = k
                return body, close
            depth -= 1; j += 6; continue
        j += 1
    body = t.s[t.i:]; t.i = len(t.s)
    return body, ""


def _bold(frag):
    return frag.replace("<m:r>" + RPR, '<m:r><m:rPr><m:sty m:val="b"/></m:rPr>' + RPR)


def _environment(t: Tok, env):
    end = "\\end{" + env + "}"
    j = t.s.find(end, t.i)
    body = t.s[t.i:j] if j >= 0 else t.s[t.i:]
    t.i = (j + len(end)) if j >= 0 else len(t.s)
    rows = [_split_top(r, "&") for r in _split_top(body, "\\\\")]
    rows = [[_seq(Tok(c.strip())) for c in r] for r in rows if any(c.strip() for c in r)]
    if env in ("bmatrix", "matrix", "pmatrix", "vmatrix"):
        op, cl = {"bmatrix": ("[", "]"), "pmatrix": ("(", ")"),
                  "vmatrix": ("|", "|"), "matrix": ("", "")}[env]
        return _matrix(rows, op, cl)
    if env == "cases":
        return _cases(rows)
    if env in ("aligned", "align", "array", "gathered"):
        return _matrix(rows, "", "")
    return _matrix(rows, "", "")


# ----------------------------------------------------------------------
def inline(latex: str) -> str:
    """Inline equation object for use inside a run of text."""
    return f"<m:oMath>{convert(latex)}</m:oMath>"


def display(latex: str, number=None) -> str:
    """
    Centred display equation with a right-hand number, using the four-column
    hidden-matrix layout of the master template.
    """
    body = convert(latex)
    num = run(f"({number})", nor=True) if number is not None else ""
    mm = ('<m:m><m:mPr><m:plcHide m:val="1"/><m:mcs><m:mc><m:mcPr>'
          '<m:count m:val="4"/><m:mcJc m:val="center"/></m:mcPr></m:mc></m:mcs>'
          f'{CTRL}</m:mPr><m:mr><m:e/><m:e>{body}</m:e><m:e/>'
          f'<m:e>{num}</m:e></m:mr></m:m>')
    return f"<m:oMathPara><m:oMath>{mm}</m:oMath></m:oMathPara>"
