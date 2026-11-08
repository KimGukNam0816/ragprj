from fastapi.testclient import TestClient
from backend.app.main import app

client=TestClient(app)

def test_health_discloses_no_keys():
    r=client.get('/health')
    assert r.status_code==200 and r.json()['local_only'] is True
    assert 'api_key' not in r.text and 'secret_key' not in r.text

def test_invalid_question_rejected_before_inference():
    assert client.post('/query',json={'question':'  '}).status_code==422
    assert client.post('/query',json={'question':'hello','mode':'unknown'}).status_code==422
