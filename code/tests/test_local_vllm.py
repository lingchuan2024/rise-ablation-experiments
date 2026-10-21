"""Exercise the real OpenAI SDK against a deterministic local HTTP transport."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rise.api_retry import chat_with_retry, RequestDeadlineExceeded, extract_reasoning
from rise.local_vllm import ContextBudgetError, settings
from rise.agent import Agent
from rise.turn_checkpoint import TurnCheckpoint

MODEL = 'Qwen3.5-35B-A3B'


def reply(content=None, reasoning=None, tools=None, finish='stop'):
    return dict(id='test', object='chat.completion', created=1, model=MODEL,
                choices=[dict(index=0, finish_reason=finish, message=dict(
                    role='assistant', content=content, reasoning=reasoning, tool_calls=tools))],
                usage=dict(prompt_tokens=50, completion_tokens=10, total_tokens=60))


class LocalVllmTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = patch.dict(os.environ, {'RISE_LOCAL_VLLM_BASE_URL':'http://test/v1',
            'RISE_LOCAL_VLLM_CONTEXT_LENGTH':'0', 'RISE_LOCAL_VLLM_TOKEN_MARGIN':'256',
            'RISE_LOCAL_VLLM_MIN_OUTPUT_TOKENS':'1024',
            'RISE_LOCAL_VLLM_TIMEOUT':'1200', 'RISE_QWEN35_PROFILE':'legacy',
            'RISE_LOCAL_VLLM_SERVICE_SHA256':'test-service-sha',
            'RISE_API_DIAGNOSTICS_DIR':self.tmp.name})
        self.env.start(); self.addCleanup(self.env.stop)
        self.requests=[]
        self.request_timeouts=[]
        self.on_request=None

    def client(self, responses, count=80000, limit=100000):
        def handle(request):
            body=json.loads(request.content);self.requests.append((request.url.path,body))
            self.request_timeouts.append(dict(request.extensions.get('timeout', {})))
            if self.on_request is not None:self.on_request(request.url.path, body)
            if request.url.path=='/tokenize':
                return httpx.Response(200,json={'count':count,'max_model_len':limit,'tokens':[]})
            self.assertEqual(request.url.path,'/v1/chat/completions')
            item=responses.pop(0)
            if isinstance(item,tuple):return httpx.Response(item[0],json=item[1])
            return httpx.Response(200,json=item)
        client=OpenAI(api_key='do-not-log-secret',base_url='http://test/v1',max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        self.addCleanup(client.close);return client

    def call(self,client,**kw):
        return chat_with_retry(client,model=MODEL,messages=[{'role':'user','content':'Question'}],
                               max_tokens=32000,max_retries=1,**kw)

    def test_80k_input_uses_remaining_window_without_truncating_history(self):
        r=self.call(self.client([reply('answer','thinking')]),tools=[{'type':'function','function':{'name':'search','parameters':{}}}])
        self.assertEqual(extract_reasoning(r.choices[0].message),'thinking')
        tokenize,completion=self.requests
        self.assertEqual(completion[1]['max_tokens'],19744)
        self.assertEqual(tokenize[1]['messages'],completion[1]['messages'])
        self.assertEqual(tokenize[1]['tools'],completion[1]['tools'])
        self.assertTrue(tokenize[1]['chat_template_kwargs']['enable_thinking'])
        audit=json.loads(next(Path(self.tmp.name).glob('*.jsonl')).read_text())
        self.assertEqual(audit['response']['choices'][0]['message']['reasoning'],'thinking')
        self.assertNotIn('do-not-log-secret',json.dumps(audit))

    def test_262k_server_is_not_capped_at_100k(self):
        self.call(self.client([reply('answer')],count=150000,limit=262144))
        self.assertEqual(self.requests[-1][1]['max_tokens'],32000)

    def test_exhausted_window_sends_no_completion(self):
        with self.assertRaises(ContextBudgetError):self.call(self.client([],count=99500))
        self.assertEqual(len(self.requests),1)

    def test_server_configuration_mismatch_fails_before_generation(self):
        with patch.dict(os.environ,{'RISE_LOCAL_VLLM_CONTEXT_LENGTH':'262144'}):
            with self.assertRaisesRegex(ValueError,'mismatch'):self.call(self.client([]))
        self.assertEqual(len(self.requests),1)

    def test_reasoning_only_saved_and_never_scored_as_answer(self):
        client=self.client([reply(reasoning='Exact Answer: unfinished draft')])
        run=Agent(client=client,model=MODEL,tools={},tool_specs=[],max_turns=3,max_retries=3).run(system_prompt='s',user_prompt='u')
        self.assertEqual(run.terminated_reason,'incomplete_response')
        self.assertEqual(run.final_answer,'')
        self.assertEqual((run.total_prompt_tokens,run.total_completion_tokens),(50,10))
        self.assertEqual(run.turns[-1].reasoning_content,'Exact Answer: unfinished draft')
        self.assertEqual(sum(p=='/v1/chat/completions' for p,_ in self.requests),1)

    def test_missing_choices_preserves_usage_without_index_error(self):
        value=reply();value['choices']=[]
        run=Agent(client=self.client([value]),model=MODEL,tools={},tool_specs=[],max_turns=3,max_retries=3).run(system_prompt='s',user_prompt='u')
        self.assertEqual(run.terminated_reason,'incomplete_response')
        self.assertEqual(run.turns[-1].response_category,'missing_choices')
        self.assertEqual(run.total_completion_tokens,10)

    def test_reasoning_is_preserved_in_next_tool_turn_and_trace(self):
        call={'id':'c1','type':'function','function':{'name':'search','arguments':'{}'}}
        client=self.client([reply(reasoning='plan search',tools=[call],finish='tool_calls'),reply('answer','conclusion')],count=50)
        trace=Path(self.tmp.name)/'trace.json'
        run=Agent(client=client,model=MODEL,tools={'search':lambda _: 'evidence'},tool_specs=[],max_turns=3,max_retries=1).run(system_prompt='system',user_prompt='question',trace_path=trace)
        self.assertEqual(run.terminated_reason,'done')
        self.assertEqual(run.turns[0].reasoning_content,'plan search')
        second=[b for p,b in self.requests if p=='/v1/chat/completions'][1]
        self.assertEqual(second['messages'][2]['reasoning'],'plan search')
        self.assertEqual(json.loads(trace.read_text())['turns'][1]['reasoning_content'],'conclusion')

    def test_local_truncation_returns_once_and_keeps_failed_usage(self):
        client=self.client([reply(reasoning='unfinished',finish='length')])
        run=Agent(client=client,model=MODEL,tools={},tool_specs=[],max_turns=3,max_retries=3).run(system_prompt='s',user_prompt='u')
        self.assertEqual(run.terminated_reason,'output_truncated')
        self.assertEqual(run.final_answer,'')
        self.assertEqual(run.total_completion_tokens,10)
        self.assertEqual(sum(p=='/v1/chat/completions' for p,_ in self.requests),1)

    def test_official_profile_actual_sdk_payload_and_budget_match(self):
        with patch.dict(os.environ,{'RISE_QWEN35_PROFILE':'official-thinking-v1'}):
            self.call(self.client([reply('answer')]))
        count,completion=self.requests
        body=completion[1]
        self.assertEqual(body['temperature'],1.0)
        self.assertEqual(body['top_p'],.95)
        self.assertEqual(body['top_k'],20)
        self.assertEqual(body['presence_penalty'],1.5)
        self.assertEqual(body['repetition_penalty'],1.0)
        self.assertEqual(body['min_p'],0.0)
        self.assertEqual(count[1]['chat_template_kwargs'],body['chat_template_kwargs'])

    def test_partial_text_and_partial_tool_calls_do_not_finish_or_execute(self):
        cases=[reply('partial',finish='length'),reply('partial',finish=None),
               reply('blocked',finish='content_filter'),
               reply(tools=[{'id':'c','type':'function','function':{'name':'bash','arguments':'{'}}]),
               reply(tools=[{'id':'same','type':'function','function':{'name':'bash','arguments':'{}'}}]*2)]
        for response in cases:
            with self.subTest(response=response):
                run=Agent(client=self.client([response]),model=MODEL,tools={'bash':lambda _:self.fail('partial tool executed')},tool_specs=[],max_turns=3).run(system_prompt='s',user_prompt='u')
                self.assertNotEqual(run.terminated_reason,'done')
                self.assertEqual(run.final_answer,'')
                self.assertEqual(run.total_completion_tokens,10)

    def test_invalid_message_keeps_failed_usage_without_regeneration(self):
        value=reply();value['choices'][0]['message']=None
        run=Agent(client=self.client([value]),model=MODEL,tools={},tool_specs=[],max_turns=3,max_retries=3).run(system_prompt='s',user_prompt='u')
        self.assertEqual(run.terminated_reason,'incomplete_response')
        self.assertEqual(run.turns[0].response_category,'invalid_response')
        self.assertEqual((run.total_prompt_tokens,run.total_completion_tokens),(50,10))
        self.assertEqual(sum(p=='/v1/chat/completions' for p,_ in self.requests),1)

    def test_expired_deadline_sends_no_preflight_or_generation(self):
        with self.assertRaises(RequestDeadlineExceeded):
            self.call(self.client([]),deadline=time.monotonic()-1)
        self.assertEqual(self.requests,[])

    def test_tokenize_and_generation_timeouts_use_remaining_deadline(self):
        self.call(self.client([reply('answer')]),deadline=time.monotonic()+10)
        self.assertEqual(len(self.request_timeouts),2)
        for timeout in self.request_timeouts:
            self.assertGreater(timeout['read'],0)
            self.assertLessEqual(timeout['read'],10)

    def test_retry_backoff_does_not_outlive_deadline_or_hide_sdk_retries(self):
        client=self.client([(503,{'error':{'message':'temporary','type':'server_error'}})]).with_options(max_retries=4)
        with patch('rise.api_retry.time.sleep') as sleep,self.assertRaisesRegex(RequestDeadlineExceeded,'backoff'):
            chat_with_retry(client,model=MODEL,messages=[{'role':'user','content':'Q'}],max_tokens=32000,
                            max_retries=3,deadline=time.monotonic()+10,initial_backoff=100,jitter=0)
        sleep.assert_not_called()
        self.assertEqual(sum(p=='/v1/chat/completions' for p,_ in self.requests),1)

    def test_preflight_returning_after_deadline_sends_no_completion(self):
        clock=[0.0]
        self.on_request=lambda route,_body:clock.__setitem__(0,20.0) if route=='/tokenize' else None
        with patch('rise.local_vllm.time.monotonic',side_effect=lambda:clock[0]),self.assertRaisesRegex(RequestDeadlineExceeded,'preflight'):
            self.call(self.client([]),deadline=10.0)
        self.assertEqual([p for p,_ in self.requests],['/tokenize'])

    def test_preflight_timeout_at_deadline_is_terminal_deadline_error(self):
        clock=[0.0]
        def timeout(route,_body):
            if route=='/tokenize':
                clock[0]=20.0
                raise httpx.ReadTimeout('deterministic preflight timeout')
        self.on_request=timeout
        with patch('rise.local_vllm.time.monotonic',side_effect=lambda:clock[0]),self.assertRaises(RequestDeadlineExceeded):
            self.call(self.client([]),deadline=10.0)
        self.assertEqual([p for p,_ in self.requests],['/tokenize'])

    def test_checkpoint_loading_consumes_the_current_request_deadline(self):
        clock=[0.0]
        class Checkpoint:
            def load(self):
                clock[0]=5.0
                return None
            def save(self,*args):
                pass
        with patch('rise.local_vllm.time.monotonic',side_effect=lambda:clock[0]):
            run=Agent(client=self.client([reply('answer')]),model=MODEL,tools={},tool_specs=[],
                      max_turns=3,wall_clock_timeout_sec=10).run(system_prompt='s',user_prompt='u',turn_checkpoint=Checkpoint())
        self.assertEqual(run.terminated_reason,'done')
        self.assertEqual(len(self.request_timeouts),2)
        self.assertTrue(all(0<timeout['read']<=5 for timeout in self.request_timeouts))

    def test_late_completion_saves_usage_as_timeout_and_does_not_coerce(self):
        clock=[0.0]
        self.on_request=lambda route,_body:clock.__setitem__(0,20.0) if route=='/v1/chat/completions' else None
        with patch('rise.local_vllm.time.monotonic',side_effect=lambda:clock[0]):
            run=Agent(client=self.client([reply('late answer')]),model=MODEL,tools={},tool_specs=[],
                      max_turns=3,wall_clock_timeout_sec=10,coerce_final_on_max_turns=True).run(system_prompt='s',user_prompt='u')
        self.assertEqual(run.terminated_reason,'wall_clock_timeout')
        self.assertEqual(run.final_answer,'')
        self.assertEqual((run.total_prompt_tokens,run.total_completion_tokens),(50,10))
        self.assertTrue(run.turns[0].api_metadata['deadline_exceeded'])
        self.assertEqual(sum(p=='/v1/chat/completions' for p,_ in self.requests),1)

    def test_tool_deadline_stops_remaining_tools_and_keeps_raw_calls(self):
        clock=[0.0]
        executed=[]
        calls=[{'id':f'c{i}','type':'function','function':{'name':'lookup','arguments':json.dumps({'i':i})}} for i in [1,2]]
        def lookup(args):
            executed.append(args['i']);clock[0]=20.0
            return 'evidence'
        with patch('rise.local_vllm.time.monotonic',side_effect=lambda:clock[0]):
            run=Agent(client=self.client([reply(tools=calls,finish='tool_calls')]),model=MODEL,
                      tools={'lookup':lookup},tool_specs=[],max_turns=3,wall_clock_timeout_sec=10).run(system_prompt='s',user_prompt='u')
        self.assertEqual(executed,[1])
        self.assertEqual(run.terminated_reason,'wall_clock_timeout')
        self.assertEqual(len(run.turns[0].raw_tool_calls),2)
        self.assertEqual(run.tool_call_count,1)

    def test_terminal_failures_and_forced_answers_resume_without_new_requests(self):
        for name,value,max_turns,timeout in [
            ('reasoning',reply(reasoning='unfinished'),3,60),
            ('filter',reply('blocked',finish='content_filter'),3,60),
            ('coerced_tools',reply(tools=[{'id':'c','type':'function','function':{'name':'lookup','arguments':'{}'}}],finish='tool_calls'),0,60),
            ('coerced_missing',dict(reply(),choices=[]),0,60),
            ('deadline',reply('unused'),3,0),
        ]:
            with self.subTest(name=name):
                root=Path(self.tmp.name)/name
                corpus,workspace=root/'corpus',root/'workspace'
                corpus.mkdir(parents=True);workspace.mkdir()
                checkpoint=TurnCheckpoint(root/'checkpoint.json',identity={'model':MODEL,'query_id':'1','run_config_hash':'test'},
                                          working_dir=workspace,corpus_root=corpus,relpath_to_docid={})
                agent=Agent(client=self.client([value]),model=MODEL,tools={'lookup':lambda _:self.fail('forced tool executed')},
                            tool_specs=[],max_turns=max_turns,wall_clock_timeout_sec=timeout,coerce_final_on_max_turns=True)
                run=agent.run(system_prompt='s',user_prompt='u',turn_checkpoint=checkpoint)
                self.assertIn(run.terminated_reason,{'incomplete_response','content_filter','wall_clock_timeout'})
                before=len(self.requests)
                agent.client=self.client([])
                resumed=agent.run(system_prompt='s',user_prompt='u',turn_checkpoint=checkpoint)
                self.assertEqual(resumed.terminated_reason,run.terminated_reason)
                self.assertEqual(resumed.total_completion_tokens,run.total_completion_tokens)
                self.assertEqual(len(self.requests),before)

    def test_versioned_profile_and_service_identity_are_recorded(self):
        with patch.dict(os.environ,{'RISE_QWEN35_PROFILE':'official-thinking-v1'}):
            config=settings()
        self.assertEqual(config['adapter'],'qwen-vllm-v4-native-history')
        self.assertEqual(config['profile'],'official-thinking-v1')
        self.assertEqual(config['service_sha256'],'test-service-sha')

    def test_explicit_non_thinking_mode_matches_tokenize_and_completion(self):
        self.call(self.client([reply('answer')]),extra_body={'chat_template_kwargs':{'enable_thinking':False}})
        self.assertFalse(self.requests[0][1]['chat_template_kwargs']['enable_thinking'])
        self.assertFalse(self.requests[1][1]['chat_template_kwargs']['enable_thinking'])

    def test_native_wire_history_is_shared_by_tokenize_and_generation_without_mutation(self):
        from rise.local_vllm import prepare_request
        messages=[{'role':'user','content':'Q'},{'role':'assistant','content':'','reasoning_content':'Retain this plan'}]
        original=json.loads(json.dumps(messages))
        wire=prepare_request({'model':MODEL,'messages':messages})
        self.assertEqual(messages,original)
        self.assertEqual(wire['messages'][1]['reasoning'],'Retain this plan')
        self.assertNotIn('reasoning_content',wire['messages'][1])
        client=self.client([reply('answer')])
        chat_with_retry(client,model=MODEL,messages=messages,max_tokens=32000,max_retries=1)
        self.assertEqual(self.requests[0][1]['messages'],self.requests[1][1]['messages'])
        self.assertEqual(self.requests[1][1]['messages'][1]['reasoning'],'Retain this plan')
        self.assertEqual(messages,original)

    def test_legacy_field_takes_precedence_without_concatenation(self):
        self.assertEqual(extract_reasoning({'reasoning_content':'old','reasoning':'new'}),'old')
        self.assertEqual(extract_reasoning({'reasoning_content':'','reasoning':'new'}),'new')

if __name__=='__main__': unittest.main()
