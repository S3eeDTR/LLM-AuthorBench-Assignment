"""OpenRouter requests. Keys stay in request memory and are never written to disk."""
import json
import urllib.request
import urllib.error
MODELS = {
    'anthropic/claude-3.5-haiku': 'Claude 3.5 Haiku',
    'deepseek/deepseek-chat': 'DeepSeek Chat',
    'google/gemini-2.5-flash-preview-05-20': 'Gemini 2.5 Flash preview 05-20',
    'google/gemini-2.5-flash': 'Gemini 2.5 Flash (different from training preview)',
    'openai/gpt-4.1': 'GPT-4.1',
    'meta-llama/llama-3.3-70b-instruct': 'Llama 3.3 70B Instruct',
}

def available_models():
    with urllib.request.urlopen('https://openrouter.ai/api/v1/models', timeout=25) as response:
        entries = {item['id']: item for item in json.load(response)['data']}
    return {'models': [{'id': key, 'name': name, 'available': key in entries,
                        'canonical_slug': entries.get(key, {}).get('canonical_slug'),
                        'pricing': entries.get(key, {}).get('pricing', {})}
                       for key, name in MODELS.items()]}

def generate(data):
    key = data.get('api_key', '')
    prompt = data.get('prompt', '')
    model = data.get('model')
    if not isinstance(key, str) or not 10 <= len(key.strip()) <= 500:
        raise ValueError('Enter a valid OpenRouter API key.')
    if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 12000:
        raise ValueError('Enter a prompt of at most 12,000 characters.')
    if model not in MODELS: raise ValueError('Choose a supported generator ID.')
    payload = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
               'temperature': 0.7, 'max_tokens': 768,
               'provider': {'allow_fallbacks': False}}
    request = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',
        data=json.dumps(payload).encode(), headers={'Authorization': 'Bearer ' + key.strip(),
        'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=120) as response: output = json.load(response)
    except urllib.error.HTTPError as error:
        # Do not echo upstream bodies or credentials into logs or the page.
        messages = {401:'API key rejected.',402:'Insufficient OpenRouter credits.',
                    404:'No provider is available for this model.',429:'Rate limit reached. Try later.'}
        return {'error': messages.get(error.code, f'OpenRouter returned HTTP {error.code}.'), 'requested_model': model}
    except (OSError, TimeoutError):
        return {'error': 'Could not reach OpenRouter. Check your connection and try later.', 'requested_model': model}
    choices = output.get('choices') or []
    if output.get('error') or not choices:
        return {'error': 'OpenRouter did not return a completion.', 'requested_model': model}
    content = choices[0].get('message', {}).get('content')
    if not isinstance(content, str) or not content.strip():
        return {'error': 'The model returned no text.', 'requested_model': model}
    return {'requested_model': model, 'returned_model': output.get('model'),
            'provider': output.get('provider'), 'response_id': output.get('id'),
            'finish_reason': choices[0].get('finish_reason'), 'usage': output.get('usage'),
            'code': content, 'prompt': prompt, 'temperature': .7, 'max_tokens': 768}
