from backend.app.observability.usage import generation_metadata, usage_details


def test_usage_does_not_turn_unknown_into_zero_or_duplicate_content():
    generated = {'content':'답변', 'input_tokens':None, 'output_tokens':0, 'attempts':[]}
    assert usage_details(generated) == {'output':0}
    meta = generation_metadata(generated, provider='openai', request_id='r1')
    assert meta['request_id'] == 'r1' and 'content' not in meta
