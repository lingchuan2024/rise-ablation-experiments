#!/usr/bin/env python3
"""Check the serving context window; optional short generation smoke test."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rise.decompose import make_client
from rise.local_vllm import apply_budget, is_local_qwen
from rise.api_retry import chat_with_retry, extract_reasoning


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen3.5-35B-A3B')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not is_local_qwen(args.model):
        parser.error('Set RISE_LOCAL_VLLM_BASE_URL to the existing vLLM /v1 endpoint')
    client = make_client(args.model)
    request = dict(model=args.model, messages=[{'role':'user','content':'What is 2 + 2? Answer briefly.'}],
                   max_tokens=2048, extra_body={'chat_template_kwargs':{'enable_thinking':True}})
    try:
        print(json.dumps(apply_budget(client, request), indent=2))
        if args.smoke:
            response = chat_with_retry(client, max_retries=1, **request)
            msg = response.choices[0].message
            if not (msg.content or '').strip() or response.choices[0].finish_reason != 'stop':
                raise RuntimeError('Smoke test did not return a complete answer')
            if not extract_reasoning(msg):
                raise RuntimeError('Thinking was enabled but no reasoning field was received; inspect diagnostics')
            print(json.dumps({'answer':msg.content,'reasoning_chars':len(extract_reasoning(msg))},ensure_ascii=False))
    finally:
        client.close()


if __name__ == '__main__':
    main()
