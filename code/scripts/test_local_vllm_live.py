#!/usr/bin/env python3
"""Bounded serial acceptance tests for the deployed local model; no benchmark rewrite."""
import os,json,time,sys,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rise.decompose import make_client
from rise.local_vllm import apply_budget,extract_reasoning
from rise.api_retry import chat_with_retry
from rise.agent import Agent
from rise.tools import bm25_search_tool_spec,bash_tool_spec,read_tool_spec

MODEL='Qwen3.5-35B-A3B'
OUT=Path('logs/live-adaptation-20260927');OUT.mkdir(parents=True,exist_ok=True)
os.environ['RISE_LOCAL_VLLM_BASE_URL']='http://127.0.0.1:8000/v1'
os.environ['RISE_LOCAL_VLLM_TIMEOUT']='600'
os.environ['RISE_API_DIAGNOSTICS_DIR']=str(OUT/'api')
results=[]

def stage(name,fn):
 t=time.time()
 try:r={'name':name,'ok':True,**fn()}
 except Exception as e:r={'name':name,'ok':False,'error':str(e),'traceback':traceback.format_exc()}
 r['seconds']=round(time.time()-t,2);results.append(r)
 (OUT/'summary.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
 print(json.dumps(r,ensure_ascii=False),flush=True)

client=make_client(MODEL)
def smoke():
 r=chat_with_retry(client,model=MODEL,messages=[{'role':'user','content':'What is 2 + 2? Answer briefly.'}],max_tokens=2048,max_retries=1,diagnostic_path=OUT/'smoke.api.jsonl')
 m=r.choices[0].message
 assert '4' in (m.content or '') and extract_reasoning(m) and r.choices[0].finish_reason=='stop'
 return {'answer':m.content,'reasoning_chars':len(extract_reasoning(m)),'usage':r.usage.model_dump()}

def tool_cycle():
 seen=[]
 def lookup(args):seen.append(args);return 'The verification code is COPPER-6289.'
 tool={'type':'function','function':{'name':'lookup','description':'Return the verification code.','parameters':{'type':'object','properties':{},'additionalProperties':False}}}
 run=Agent(client=client,model=MODEL,tools={'lookup':lookup},tool_specs=[tool],max_turns=4,max_retries=1,per_call_max_tokens=4096).run(system_prompt='Call lookup once to obtain the code, then answer with the code. Do not invent it.',user_prompt='What is the verification code?',trace_path=OUT/'tool_cycle.json')
 assert run.terminated_reason=='done' and seen and 'COPPER-6289' in run.final_answer
 assert any(t.reasoning_content for t in run.turns)
 events=[json.loads(l) for l in (OUT/'tool_cycle.api.jsonl').read_text().splitlines()]
 assert any(m.get('reasoning_content') for e in events[1:] for m in e['request']['messages'] if m['role']=='assistant')
 return {'turns':len(run.turns),'tool_calls':len(seen),'answer':run.final_answer,'reasoning_chars':sum(len(t.reasoning_content) for t in run.turns)}

def replay():
 p=Path('runs/rise_qwen35_35ba3b_100/_checkpoints/qid_707.json');d=json.loads(p.read_text())
 r=chat_with_retry(client,model=MODEL,messages=d['messages'],tools=[bm25_search_tool_spec(),bash_tool_spec(),read_tool_spec()],tool_choice='auto',max_tokens=32000,temperature=0.7,max_retries=1,diagnostic_path=OUT/'replay707.api.jsonl')
 m=r.choices[0].message
 return {'finish_reason':r.choices[0].finish_reason,'content_chars':len(m.content or ''),'reasoning_chars':len(extract_reasoning(m)),'tool_calls':len(m.tool_calls or []),'note':'one next-turn replay; not a completed benchmark question'}

def long_input(target):
 n=target//6
 for _ in range(4):
  content=('Neutral padding record.\n'*n)+'\nThe verification code is ORANGE-5821. Reply only with this code.'
  req=dict(model=MODEL,messages=[{'role':'user','content':content}],max_tokens=32000,temperature=0,extra_body={'chat_template_kwargs':{'enable_thinking':False}})
  b=apply_budget(client,req)
  if abs(b['input_tokens']-target)<128:break
  n=max(1,int(n*target/b['input_tokens']))
 r=chat_with_retry(client,**req,max_retries=1,diagnostic_path=OUT/f'long{target}.api.jsonl')
 assert r.choices[0].finish_reason=='stop' and 'ORANGE-5821' in (r.choices[0].message.content or '')
 return {**b,'answer':r.choices[0].message.content,'actual_prompt_tokens':r.usage.prompt_tokens,'finish_reason':r.choices[0].finish_reason,'note':'synthetic capacity check, not long-context research accuracy'}

stage('thinking_smoke',smoke)
stage('tool_reasoning_roundtrip',tool_cycle)
stage('failed_checkpoint707_next_turn',replay)
for size in (80000,120000,240000):stage(f'context_{size}',lambda size=size:long_input(size))
client.close()
print('TESTS_FINISHED',flush=True)
