from io import BytesIO
from zipfile import ZipFile
from fastapi.testclient import TestClient
from edurisk.api import app,MATERIAL_FILES
client=TestClient(app)

def test_all_materials_export_word():
 for key in MATERIAL_FILES:
  r=client.get('/api/materials/'+key)
  assert r.status_code==200,key
  assert 'wordprocessingml' in r.headers['content-type']
  with ZipFile(BytesIO(r.content)) as z: assert 'word/document.xml' in z.namelist()

def test_plan_word_contains_notes_and_blocks_image_injection():
 r=client.post('/api/word/plan',json={'months':{'2026-11':{'status':'active','note':'Проверка ![x](file:///C:/secret.png)'}}})
 assert r.status_code==200
 with ZipFile(BytesIO(r.content)) as z:
  xml=z.read('word/document.xml').decode()
  assert 'Проверка' in xml
  assert not any(n.startswith('word/media/') for n in z.namelist())

def test_word_allowlist():
 assert client.get('/api/materials/unknown?format=docx').status_code==404
 assert client.get('/api/materials/literature?format=exe').status_code==422
