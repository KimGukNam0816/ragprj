"""관측 값 변환. 미확인 usage는 0으로 만들지 않는다."""


def usage_details(generated):
    return {key: value for key, value in {
        'input': generated.get('input_tokens'), 'output': generated.get('output_tokens')
    }.items() if value is not None}


def generation_metadata(generated, *, provider, request_id):
    # 원문은 output으로 분리하며 API 인증 헤더를 받지 않는다.
    return {'provider': provider, 'request_id': request_id,
            **{k: v for k, v in generated.items() if k != 'content'}}
