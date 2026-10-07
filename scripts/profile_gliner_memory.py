#!/usr/bin/env python3
"""Run isolated GLiNER memory comparisons after other model jobs finish."""
import argparse
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = {'stenograf-llm-probe', 'kwseval', 'speech', 'audio', 'speech-server', 'xctest'}
# Compilers and other benchmark-style executables also perturb timing and memory.
BUSY_NAMES = MODEL_NAMES | {'swift-build', 'swift-test', 'swift-frontend', 'swift-driver'}
BUSY_SUFFIXES = ('-bench', '-gate', '-probe', 'Tests')


def active_models(ignore_pid=None):
    rows = subprocess.check_output(['ps', '-Ao', 'pid,comm'], text=True).splitlines()[1:]
    busy = []
    for row in rows:
        parts = row.strip().split(None, 1)
        if len(parts) != 2: continue
        pid, command = int(parts[0]), parts[1]
        name = Path(command).name
        if pid != ignore_pid and (name in BUSY_NAMES or name.endswith(BUSY_SUFFIXES)):
            busy.append((pid, name))
    return busy


def wait_idle(status_path=None):
    while True:
        busy = active_models()
        if status_path:
            status_path.write_text(json.dumps({'state':'waiting','blockingJobs':busy,'updatedUnix':time.time()},indent=2))
        if not busy:
            time.sleep(2)
            if not active_models(): return
        else:
            print('Waiting for model jobs: '+', '.join(f'{name} ({pid})' for pid, name in busy), flush=True)
            time.sleep(30)


def check_reference(result, references, tolerance):
    issues = []
    max_delta = 0.0
    if len(result['rows']) != len(references): raise ValueError('Fixture count changed')
    for row, gold in zip(result['rows'], references):
        if (row['text'], row['task']) != (gold['text'], gold['task']):
            raise ValueError('Fixture order changed')
        if row['task'] == 'routing':
            winner = max(row['choices'], key=lambda x:x['probability'])
            target = gold['output']['action']
            if winner['label'] != target['label']: issues.append(row['text']+': label mismatch')
            max_delta = max(max_delta, abs(winner['probability']-target['confidence']))
        else:
            for label in ['person', 'time']:
                got, expected = row['entities'][label], gold['output']['entities'][label]
                if sorted(x['text'] for x in got) != sorted(x['text'] for x in expected):
                    issues.append(row['text']+': spans mismatch'); continue
                for span in got:
                    target = next(x for x in expected if x['text']==span['text'])
                    if span['start'] != target['start'] or span['end'] != target['end']:
                        issues.append(row['text']+': offsets mismatch')
                    max_delta = max(max_delta,abs(span['score']-target['confidence']))
    if max_delta > tolerance: issues.append(f'Max confidence delta {max_delta:.6f} > {tolerance}')
    return {'passed':not issues, 'maxConfidenceDelta':max_delta, 'issues':issues}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fp32', type=Path, required=True)
    parser.add_argument('--fp16', type=Path, required=True)
    parser.add_argument('--int8', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = ROOT/'scripts/tests/fixtures/gliner/cases.json'
    references = json.loads((cases.parent/'reference-fixtures.json').read_text())
    variants = [('fp32-lazy-default',args.fp32,False,None),
                ('fp32-lazy-cache64',args.fp32,False,64),
                ('fp32-layerwise-cache64',args.fp32,True,64),
                ('fp16-lazy-default',args.fp16,False,None),
                ('fp16-lazy-cache64',args.fp16,False,64),
                ('fp16-layerwise-cache64',args.fp16,True,64)]
    if args.int8:
        variants += [('int8-lazy-default',args.int8,False,None),
                     ('int8-lazy-cache64',args.int8,False,64),
                     ('int8-layerwise-cache64',args.int8,True,64)]
    summaries = []
    for name, model, layers, cache in variants:
        while True:
            wait_idle(args.output/"STATUS.json")
            (args.output/"STATUS.json").write_text(json.dumps({"state":"running","variant":name,"updatedUnix":time.time()},indent=2))
            print('Profiling '+name, flush=True)
            output = args.output/(name+'.json')
            command = [str(ROOT/'.build/release/gliner-bench'),'--model',str(model),'--cases',str(cases),
                       '--output',str(output),'--evaluate-layers',str(layers).lower()]
            if cache is not None: command += ['--cache-limit-mb',str(cache)]
            contended = False
            with (args.output/(name+'.log')).open('w') as log:
                process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                while process.poll() is None:
                    if active_models(process.pid):
                        process.terminate()
                        try: process.wait(timeout=10)
                        except subprocess.TimeoutExpired: process.kill(); process.wait()
                        contended = True; break
                    time.sleep(.5)
            if contended:
                output.unlink(missing_ok=True)
                print('Other model job started; discarded run and waiting again.',flush=True)
                continue
            if process.returncode:
                (args.output/'STATUS.json').write_text(json.dumps({'state':'failed','variant':name,'reason':'Benchmark process failed'},indent=2))
                raise RuntimeError(f'{name} failed; see its log')
            break
        result = json.loads(output.read_text())
        tolerance = .02 if name.startswith('int8') else (.005 if name.startswith('fp16') else .001)
        reference = check_reference(result,references,tolerance)
        summary = {'variant':name,'physicalPeakBytes':result['sampledPhysicalFootprintBytes'],
                   'rssPeakBytes':result['processPeakRSSBytes'],'memoryProfile':result['memoryProfile'],
                   'referenceValidation':reference}
        for task in ['routing','extraction']:
            rows = [x for x in result['rows'] if x['task']==task]
            times = sorted(t for row in rows for t in row['milliseconds'])
            summary[task]={'p50':statistics.median(times),'p95':times[math.ceil(.95*len(times))-1],
                           'matches':sum(x['correct'] for x in rows),'cases':len(rows)}
        summaries.append(summary)
        (args.output/'summary.json').write_text(json.dumps(summaries,indent=2))
        print(name+f": peak physical {summary['physicalPeakBytes']/1e9:.3f} GB; parity {reference['passed']}",flush=True)
        if not reference['passed'] and name.startswith('fp32'):
            (args.output/'STATUS.json').write_text(json.dumps({'state':'failed','variant':name,'issues':reference['issues']},indent=2))
            raise RuntimeError('FP32 reference regression: '+str(reference['issues']))
    lines = ['# GLiNER memory profile — 2026-09-26','',
             f'{len(variants)} separate processes, same source checkpoint and handwritten cases. The runner waits for other known model jobs and discards a run if one starts. Process physical footprint is sampled every 20 ms; MLX peak active memory is allocator high-water. Cache snapshots are reusable buffers, not live model weights. OS and framework overhead are included in process memory.','',
             '| Variant | Peak physical GB | Peak RSS GB | Routing median ms | Extraction median ms | Reference parity |',
             '| --- | ---: | ---: | ---: | ---: | --- |']
    for s in summaries:
        lines.append(f"| {s['variant']} | {s['physicalPeakBytes']/1e9:.3f} | {s['rssPeakBytes']/1e9:.3f} | {s['routing']['p50']:.2f} | {s['extraction']['p50']:.2f} | {s['referenceValidation']['passed']} |")
    lines += ['', 'FP32 confidence tolerance: 0.001; FP16: 0.005; INT8: 0.02. INT8 is weight quantization of encoder matrices and token embeddings; heads and activations are FP16. No calibration data or fine-tuning is used. Exact winning labels, entity text and offsets are required. This checks fidelity to the original model, not broad model accuracy. A candidate qualifies only if it preserves the saved decisions/spans and passes its score-drift gate. Human expectation counts are reported separately; the reference model itself matches 12/16 routing and 7/8 span expectations. See raw outputs and summary.json for phase-by-phase live allocations and cache.','']
    (args.output/'REPORT.md').write_text('\n'.join(lines))
    failures = [s['variant'] for s in summaries if not s['referenceValidation']['passed']]
    (args.output/'STATUS.json').write_text(json.dumps({'state':'complete','candidateValidationFailures':failures,'updatedUnix':time.time()},indent=2))
    print('Completed: '+str(args.output/'REPORT.md'),flush=True)

if __name__ == '__main__': main()
