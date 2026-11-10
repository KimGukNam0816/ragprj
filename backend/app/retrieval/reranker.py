import threading
from functools import lru_cache

from ..config import Settings


class Reranker:
    """Official Qwen yes/no next-token probability scoring; no text generation."""
    def __init__(self, model_name, device, batch_size, max_length):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if self.device.startswith('cuda') else torch.float32, attn_implementation='sdpa').to(self.device).eval()
        self.batch_size, self.max_length = batch_size, max_length
        self.yes = self.tokenizer.convert_tokens_to_ids('yes')
        self.no = self.tokenizer.convert_tokens_to_ids('no')
        self.prefix = self.tokenizer.encode('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n', add_special_tokens=False)
        self.suffix = self.tokenizer.encode('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', add_special_tokens=False)
        self.lock = threading.Lock()

    def score(self, query, documents):
        scores = []
        with self.lock, self.torch.inference_mode():
            for offset in range(0, len(documents), self.batch_size):
                pairs = ['<Instruct>: Given a Korean shipping question, retrieve relevant passages that answer the question\n<Query>: ' + query + '\n<Document>: ' + doc for doc in documents[offset:offset+self.batch_size]]
                inputs = self.tokenizer(pairs, padding=False, truncation=True, max_length=self.max_length-len(self.prefix)-len(self.suffix), add_special_tokens=False)
                inputs['input_ids'] = [self.prefix + ids + self.suffix for ids in inputs['input_ids']]
                inputs.pop('attention_mask', None)
                encoded = self.tokenizer.pad(inputs, padding=True, return_tensors='pt').to(self.device)
                logits = self.model(**encoded, use_cache=False, logits_to_keep=1).logits[:, -1, :]
                score = self.torch.softmax(self.torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float(), dim=1)[:, 1]
                scores.extend(score.cpu().tolist())
        return scores

    def rank(self, query, rows):
        scores = self.score(query, [r['text'] for r in rows])
        return sorted([{**r, 'rerank_score': score} for r, score in zip(rows, scores)], key=lambda r: (-r['rerank_score'], r['chunk_id']))


@lru_cache(maxsize=1)
def _cached(model, device, batch, length):
    return Reranker(model, device, batch, length)


def get_reranker(s: Settings):
    return _cached(s.reranker_model, s.reranker_device, s.reranker_batch_size, s.reranker_max_length)
