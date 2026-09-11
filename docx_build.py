"""
Assembly of the manuscript as an Office Open XML document.

The master template supplies the styles, page geometry and footers; this module
rebuilds the document body inside that shell so that every heading, paragraph,
table, caption and equation carries the template's own formatting.  Mathematics
is written as native Word equation objects through the omml module, both for
display equations and for every symbol that appears inside running text, table
cells, algorithm listings and captions.
"""
from __future__ import annotations
import os, re, shutil, zipfile
import omml

NS = ('xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
      'xmlns:cx="http://schemas.microsoft.com/office/drawing/2014/chartex" '
      'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
      'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
      'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
      'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
      'xmlns:w10="urn:schemas-microsoft-com:office:word" '
      'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
      'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
      'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
      'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
      'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
      'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
      'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture" '
      'mc:Ignorable="w14 wp14"')

PGSZ = '<w:pgSz w:w="12240" w:h="15840"/>'
PGMAR = ('<w:pgMar w:top="1008" w:right="936" w:bottom="1080" w:left="936" '
         'w:header="720" w:footer="720" w:gutter="0"/>')
EMU_PER_IN = 914400


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _runs(text, size=None, bold=False, italic=False, smallcaps=False):
    """
    Split a string on $...$ and emit ordinary runs and inline equation objects.

    Every symbol written between dollar signs becomes a real Word equation, so
    the manuscript contains no mathematics typed as plain characters.
    """
    out = []
    for i, part in enumerate(re.split(r"\$([^$]*)\$", text)):
        if i % 2 == 1:
            out.append(omml.inline(part))
        elif part:
            pr = ""
            inner = ""
            if bold:
                inner += "<w:b/>"
            if italic:
                inner += "<w:i/>"
            if smallcaps:
                inner += "<w:smallCaps/>"
            if size:
                inner += f'<w:sz w:val="{size}"/>'
            if inner:
                pr = f"<w:rPr>{inner}</w:rPr>"
            out.append(f'<w:r>{pr}<w:t xml:space="preserve">{esc(part)}</w:t></w:r>')
    return "".join(out)


class Doc:
    def __init__(self, template_path):
        self.template = template_path
        self.blocks = []          # (columns, xml)
        self.images = []          # (rel_id, filename, path)
        self._img_n = 0

    # -- low level ------------------------------------------------------
    def _add(self, xml, cols=2):
        self.blocks.append((cols, xml))

    def para(self, text, style=None, jc="both", indent=True, cols=2,
             space_before=None, keep=False):
        pr = "<w:pPr>"
        if style:
            pr += f'<w:pStyle w:val="{style}"/>'
        if space_before is not None:
            pr += f'<w:spacing w:before="{space_before}" w:after="60"/>'
        if jc:
            pr += f'<w:jc w:val="{jc}"/>'
        if indent and not style:
            pr += '<w:ind w:firstLine="216"/>'
        if keep:
            pr += "<w:keepNext/>"
        pr += "</w:pPr>"
        self._add(f"<w:p>{pr}{_runs(text)}</w:p>", cols)

    def section(self, title, cols=2):
        self._add(f'<w:p><w:pPr><w:pStyle w:val="IEEESection"/><w:keepNext/></w:pPr>'
                  f'{_runs(title)}</w:p>', cols)

    def subsection(self, title, cols=2):
        self._add(f'<w:p><w:pPr><w:pStyle w:val="IEEESubsection"/><w:keepNext/></w:pPr>'
                  f'{_runs(title)}</w:p>', cols)

    def subsubsection(self, title, cols=2):
        self._add('<w:p><w:pPr><w:spacing w:after="40"/><w:ind w:firstLine="216"/>'
                  '<w:keepNext/></w:pPr>'
                  f'<w:r><w:rPr><w:i/></w:rPr><w:t xml:space="preserve">'
                  f'{esc(title)}</w:t></w:r></w:p>', cols)

    def equation(self, latex, number, cols=2):
        self._add('<w:p><w:pPr><w:jc w:val="both"/><w:spacing w:before="20" '
                  f'w:after="20" w:line="200" w:lineRule="auto"/></w:pPr>'
                  f'{omml.display(latex, number)}</w:p>', cols)

    def blank(self, cols=2):
        self._add('<w:p><w:pPr><w:spacing w:after="0"/></w:pPr></w:p>', cols)

    def caption(self, text, style="FigureCaption", cols=2, keep=False):
        pr = f'<w:pPr><w:pStyle w:val="{style}"/>'
        if keep:
            pr += "<w:keepNext/>"
        pr += "</w:pPr>"
        self._add(f"<w:p>{pr}{_runs(text)}</w:p>", cols)

    # -- images ---------------------------------------------------------
    def image(self, path, width_in, cols=2):
        self._img_n += 1
        rid = f"rIdImg{self._img_n}"
        name = f"image{self._img_n}{os.path.splitext(path)[1]}"
        self.images.append((rid, name, path))
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
        cx = int(width_in * EMU_PER_IN)
        cy = int(cx * h / w)
        xml = (
            '<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="40" w:after="20"/>'
            '<w:keepNext/></w:pPr><w:r><w:drawing>'
            f'<wp:inline distT="0" distB="0" distL="0" distR="0">'
            f'<wp:extent cx="{cx}" cy="{cy}"/>'
            '<wp:effectExtent l="0" t="0" r="0" b="0"/>'
            f'<wp:docPr id="{self._img_n}" name="Figure {self._img_n}"/>'
            '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/>'
            '</wp:cNvGraphicFramePr><a:graphic><a:graphicData '
            'uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:pic><pic:nvPicPr><pic:cNvPr id="{self._img_n}" name="{name}"/>'
            '<pic:cNvPicPr/></pic:nvPicPr><pic:blipFill>'
            f'<a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch>'
            f'</pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/>'
            f'<a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
            '</a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')
        self._add(xml, cols)

    # -- tables ---------------------------------------------------------
    def table(self, rows, widths=None, cols=2, size=16, header_rows=1,
              align=None, total_width=None):
        ncol = max(len(r) for r in rows)
        total = total_width or (3350 if cols == 2 else 10200)
        widths = widths or [total // ncol] * ncol
        grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
        body = ""
        for ri, r in enumerate(rows):
            hdr = ri < header_rows
            cells = ""
            for ci, c in enumerate(r):
                jc = "center" if (hdr or align is None or align[ci] == "c") else \
                     ("left" if align[ci] == "l" else "right")
                borders = ('<w:tcBorders>'
                           '<w:top w:val="single" w:sz="6" w:color="000000"/>'
                           '<w:bottom w:val="single" w:sz="6" w:color="000000"/>'
                           '</w:tcBorders>') if (hdr or ri == len(rows) - 1) else ""
                cells += (f'<w:tc><w:tcPr><w:tcW w:w="{widths[ci]}" w:type="dxa"/>'
                          f'{borders}<w:vAlign w:val="center"/></w:tcPr>'
                          f'<w:p><w:pPr><w:jc w:val="{jc}"/>'
                          f'<w:spacing w:after="0" w:line="200" w:lineRule="auto"/>'
                          f'<w:rPr><w:sz w:val="{size}"/></w:rPr></w:pPr>'
                          f'{_runs(c, size=size, bold=hdr)}</w:p></w:tc>')
            body += f'<w:tr><w:trPr>{"<w:tblHeader/>" if hdr else ""}</w:trPr>{cells}</w:tr>'
        tbl = ('<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/>'
               f'<w:tblW w:w="{sum(widths)}" w:type="dxa"/>'
               '<w:jc w:val="center"/>'
               '<w:tblBorders>'
               '<w:top w:val="single" w:sz="6" w:color="000000"/>'
               '<w:bottom w:val="single" w:sz="6" w:color="000000"/>'
               '</w:tblBorders>'
               '<w:tblCellMar><w:top w:w="20" w:type="dxa"/>'
               '<w:left w:w="60" w:type="dxa"/><w:bottom w:w="20" w:type="dxa"/>'
               '<w:right w:w="60" w:type="dxa"/></w:tblCellMar></w:tblPr>'
               f'<w:tblGrid>{grid}</w:tblGrid>{body}</w:tbl>')
        self._add(tbl, cols)

    def algorithm(self, title, lines, cols=2):
        """Framed pseudocode listing."""
        rows = [[title]] + [[l] for l in lines]
        ncol = 1
        total = 3350 if cols == 2 else 10200
        body = ""
        for ri, r in enumerate(rows):
            hdr = ri == 0
            ind = 0
            txt = r[0]
            while txt.startswith("  "):
                ind += 108
                txt = txt[2:]
            borders = ('<w:tcBorders><w:top w:val="single" w:sz="6" w:color="000000"/>'
                       '<w:bottom w:val="single" w:sz="6" w:color="000000"/></w:tcBorders>'
                       ) if (hdr or ri == len(rows) - 1) else ""
            body += (f'<w:tr><w:tc><w:tcPr><w:tcW w:w="{total}" w:type="dxa"/>{borders}'
                     f'</w:tcPr><w:p><w:pPr><w:spacing w:after="0" w:line="200" '
                     f'w:lineRule="auto"/><w:ind w:left="{ind}"/>'
                     f'<w:rPr><w:sz w:val="16"/></w:rPr></w:pPr>'
                     f'{_runs(txt, size=16, bold=hdr)}</w:p></w:tc></w:tr>')
        tbl = ('<w:tbl><w:tblPr>'
               f'<w:tblW w:w="{total}" w:type="dxa"/><w:jc w:val="center"/>'
               '<w:tblBorders><w:top w:val="single" w:sz="6" w:color="000000"/>'
               '<w:bottom w:val="single" w:sz="6" w:color="000000"/></w:tblBorders>'
               '<w:tblCellMar><w:top w:w="10" w:type="dxa"/><w:left w:w="60" w:type="dxa"/>'
               '<w:bottom w:w="10" w:type="dxa"/><w:right w:w="60" w:type="dxa"/>'
               '</w:tblCellMar></w:tblPr>'
               f'<w:tblGrid><w:gridCol w:w="{total}"/></w:tblGrid>{body}</w:tbl>')
        self._add(tbl, cols)

    # -- assembly -------------------------------------------------------
    def _sectpr(self, cols, first=False):
        """
        Section properties.  Every break is continuous so that the single-column
        title block, the two-column body and any full-width figure or table flow
        on the same page exactly as the master template lays them out.
        """
        c = ('<w:cols w:space="288"/>' if cols == 1
             else '<w:cols w:num="2" w:space="288" w:equalWidth="1"/>')
        return (f'<w:sectPr><w:type w:val="continuous"/>{PGSZ}{PGMAR}{c}'
                f'<w:docGrid w:linePitch="360"/></w:sectPr>')

    def _body(self):
        out = []
        i = 0
        n = len(self.blocks)
        while i < n:
            cols = self.blocks[i][0]
            j = i
            while j < n and self.blocks[j][0] == cols:
                j += 1
            chunk = [b[1] for b in self.blocks[i:j]]
            if j < n:
                # close this section on a paragraph carrying its properties
                chunk.append(f'<w:p><w:pPr><w:spacing w:after="0"/>'
                             f'{self._sectpr(cols, first=(i == 0))}</w:pPr></w:p>')
            else:
                self._final_cols = cols
                self._final_first = (i == 0)
            out.extend(chunk)
            i = j
        out.append(self._sectpr(getattr(self, "_final_cols", 2),
                                first=getattr(self, "_final_first", False)))
        return "".join(out)

    def save(self, out_path):
        tmp = out_path + ".tmp"
        shutil.copy(self.template, tmp)
        zin = zipfile.ZipFile(tmp, "r")
        names = zin.namelist()
        doc = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
               f"<w:document {NS}><w:body>{self._body()}</w:body></w:document>")
        rels = zin.read("word/_rels/document.xml.rels").decode("utf8")
        extra = "".join(
            f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/'
            f'officeDocument/2006/relationships/image" Target="media/{name}"/>'
            for rid, name, _ in self.images)
        rels = rels.replace("</Relationships>", extra + "</Relationships>")
        ct = zin.read("[Content_Types].xml").decode("utf8")
        for ext, mime in (("png", "image/png"), ("jpeg", "image/jpeg"),
                          ("jpg", "image/jpeg")):
            if f'Extension="{ext}"' not in ct:
                ct = ct.replace("<Types ", "<Types ", 1)
                ct = ct.replace("</Types>",
                                f'<Default Extension="{ext}" ContentType="{mime}"/>'
                                "</Types>")
        zout = zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED)
        for item in names:
            if item in ("word/document.xml", "word/_rels/document.xml.rels",
                        "[Content_Types].xml"):
                continue
            zout.writestr(item, zin.read(item))
        zout.writestr("word/document.xml", doc)
        zout.writestr("word/_rels/document.xml.rels", rels)
        zout.writestr("[Content_Types].xml", ct)
        for rid, name, path in self.images:
            zout.writestr(f"word/media/{name}", open(path, "rb").read())
        zout.close(); zin.close(); os.remove(tmp)
        return out_path
