"""Word exports from allowlisted local research artifacts."""
import csv
import io
import json
from pathlib import Path
import tempfile
import pypandoc
from docx import Document
from docx.shared import Inches, Pt, RGBColor

MIME='application/vnd.openxmlformats-officedocument.wordprocessingml.document'

def structured(value, level=2):
    if isinstance(value,dict):
        return '\n\n'.join('#'*min(level,6)+' '+str(k)+'\n\n'+structured(v,level+1) for k,v in value.items())
    if isinstance(value,list):
        return '\n\n'.join(structured(v,level) for v in value)
    return str(value)

def export_docx(path:Path):
    if path.suffix=='.md':
        markdown=path.read_text(encoding='utf-8')
    elif path.suffix=='.csv':
        rows=list(csv.reader(io.StringIO(path.read_text(encoding='utf-8-sig'))))
        # Wide machine tables become readable record sections without dropping columns.
        markdown='# Таблица результатов исследования\n\nИсточник '+path.name+'\n\n'
        for i,row in enumerate(rows[1:],1):
            markdown+='## Запись '+str(i)+'\n\n'+'\n\n'.join('**'+k+'**: '+v for k,v in zip(rows[0],row))+'\n\n'
    elif path.suffix=='.json':
        markdown='# Данные исследовательского отчёта\n\nИсточник '+path.name+'\n\n'+structured(json.loads(path.read_text(encoding='utf-8')))
    elif path.suffix=='.png':
        markdown='# График результатов исследования\n\n![]('+path.as_posix()+'){width=6in}\n'
    else:
        markdown='# Журнал выполнения эксперимента\n\n'+path.read_text(encoding='utf-8',errors='replace')
    return markdown_docx(markdown)

def markdown_docx(markdown):
    with tempfile.TemporaryDirectory(prefix='edurisk-word-') as folder:
        output=Path(folder)/'document.docx'
        pypandoc.convert_text(markdown,'docx',format='markdown+tex_math_dollars',outputfile=str(output),extra_args=['--standalone'])
        doc=Document(output)
        for section in doc.sections:
            section.page_width=Inches(8.5);section.page_height=Inches(11)
            section.top_margin=section.bottom_margin=Inches(.8)
            section.left_margin=section.right_margin=Inches(.8)
        for name in ('Normal','Body Text'):
            doc.styles[name].font.name='Calibri';doc.styles[name].font.size=Pt(11)
        if doc.paragraphs and doc.paragraphs[0].style.name=='Heading 1':
            doc.paragraphs[0].style=doc.styles['Title']
        doc.styles['Title'].font.color.rgb=RGBColor(0,0,0)
        for table in doc.tables:
            table.autofit=True
            for row in table.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        for run in p.runs:run.font.size=Pt(10)
        stream=io.BytesIO();doc.save(stream);return stream.getvalue()
