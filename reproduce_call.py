from pathlib import Path
import argparse, datetime, hashlib, json, os, shutil, subprocess, time, uuid
parser=argparse.ArgumentParser()
parser.add_argument('--run-name',required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parent
if Path(args.run_name).name!=args.run_name:
    raise SystemExit('Use a simple new directory name')
out=root/args.run_name
out.mkdir(exist_ok=False)
cli=shutil.which('bohr') or shutil.which('bohr.exe')
if not cli:
    raise SystemExit('Install and log into the official Bohrium CLI first')
prompt='Independent run '+str(uuid.uuid4())+'\n'+(root/'task.txt').read_text(encoding='utf8')+'\n'+(root/'numbered.txt').read_text(encoding='utf8')
(out/'request.txt').write_text(prompt,encoding='utf8')
command=[cli,'chat',prompt,'--model','gpt-5-mini','--payment-source','photon','--output','json']
pre=subprocess.run(command+['--dry-run'],capture_output=True,text=True,encoding='utf8',timeout=30,check=True)
planned=json.loads(pre.stdout)
assert planned['data']['operations'][0]['body']['message']==prompt
(out/'preflight.json').write_text(pre.stdout,encoding='utf8')
started=time.monotonic()
result=subprocess.run(command,capture_output=True,text=True,encoding='utf8',timeout=180)
response=json.loads(result.stdout)
billing=response.get('notice',{}).get('billing',{})
if 'balance' in billing:
    billing['balance']={'redacted':'private balance'}
(out/'response.json').write_text(json.dumps(response,ensure_ascii=False,indent=2),encoding='utf8')
(out/'stderr.txt').write_text(result.stderr,encoding='utf8')
data=response['data']
(out/'output.json').write_text(data['choices'][0]['message']['content'],encoding='utf8')
metadata={'model':data['model'],'completion_id':data['id'],'usage':data['usage'],'seconds':time.monotonic()-started,'cost':None,'request_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
(out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
print('Saved new call:',out)
