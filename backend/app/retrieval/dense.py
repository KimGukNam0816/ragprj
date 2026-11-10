import httpx
from ..config import Settings
from ..schemas import ServiceError


class QdrantStore:
    def __init__(self, s: Settings):
        self.s = s
        self.client = httpx.Client(base_url=s.qdrant_url.rstrip('/'), headers={'api-key': s.qdrant_api_key.get_secret_value()} if s.qdrant_api_key.get_secret_value() else {}, timeout=60)

    def close(self):
        self.client.close()

    def request(self, method, path, **kwargs):
        try:
            r = self.client.request(method, path, **kwargs)
            r.raise_for_status()
            return r.json()['result']
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise ServiceError('qdrant_error', 'Qdrant request failed: ' + type(e).__name__) from None

    def ensure(self, collection, dimensions):
        r = self.client.get('/collections/' + collection)
        if r.status_code == 404:
            self.request('PUT', '/collections/' + collection, json={'vectors': {'size': dimensions, 'distance': 'Cosine'}, 'on_disk_payload': True})
        else:
            r.raise_for_status()
            config = r.json()['result']['config']['params']['vectors']
            if config.get('size') != dimensions or config.get('distance') != 'Cosine':
                raise ValueError('Existing collection vector configuration mismatch')

    def count(self, collection):
        return self.request('POST', f'/collections/{collection}/points/count', json={'exact': True})['count']

    def upsert(self, collection, points):
        self.request('PUT', f'/collections/{collection}/points', params={'wait': 'true'}, json={'points': points})

    def search(self, collection, vector, k):
        points = self.request('POST', f'/collections/{collection}/points/query', json={'query': vector, 'limit': k, 'with_payload': True})['points']
        return [{**p['payload'], 'score': p['score']} for p in points]


