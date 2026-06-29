#!/usr/bin/env python3
import csv, json, math, re, sys
from datetime import datetime
from pathlib import Path

WORKERS = [0, 2, 4, 8, 16]
LOG_FREQ = 10
WARMUP_STEPS = 20
TOTAL_STEPS = 120
DATASET_FRAMES = 273465
BATCH_SIZE = 24
STEPS_PER_EPOCH = math.ceil(DATASET_FRAMES / BATCH_SIZE)

METRIC_RE = re.compile(
    r"INFO (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?"
    r"epch:(?P<epoch>[0-9.]+)\s+loss:(?P<loss>[0-9.]+)\s+grdn:(?P<grad_norm>[0-9.]+)\s+"
    r"lr:(?P<lr>[0-9.eE+-]+)\s+updt_s:(?P<update_s>[0-9.]+)\s+data_s:(?P<data_s>[0-9.]+)"
)

def mean(xs):
    xs=[x for x in xs if x is not None and not (isinstance(x,float) and math.isnan(x))]
    return sum(xs)/len(xs) if xs else None

def pct(xs,q):
    xs=sorted(x for x in xs if x is not None and not (isinstance(x,float) and math.isnan(x)))
    if not xs: return None
    if len(xs)==1: return xs[0]
    pos=(len(xs)-1)*q; lo=math.floor(pos); hi=math.ceil(pos)
    return xs[lo] if lo==hi else xs[lo]+(xs[hi]-xs[lo])*(pos-lo)

def fnum(v):
    if v is None: return None
    t=str(v).strip().replace(' MiB','').replace(' W','').replace('%','')
    if t in ('','N/A','[Not Supported]'): return None
    try: return float(t)
    except ValueError: return None

def dt(v):
    if not v: return None
    t=v.strip()
    for fmt in ('%Y/%m/%d %H:%M:%S.%f','%Y/%m/%d %H:%M:%S','%Y-%m-%d %H:%M:%S'):
        try: return datetime.strptime(t,fmt)
        except ValueError: pass
    return None

def read_meta(d):
    p=d/'run_meta.json'
    if not p.exists(): return {}
    try: return json.loads(p.read_text())
    except Exception: return {}

def parse_log(d):
    p=d/'train.log'
    rows=[]
    if not p.exists(): return rows
    idx=0
    for line in p.read_text(errors='ignore').replace('\r','\n').splitlines():
        m=METRIC_RE.search(line)
        if not m: continue
        idx += 1
        step = idx * LOG_FREQ
        data_s=float(m.group('data_s')); update_s=float(m.group('update_s')); step_s=data_s+update_s
        rows.append({
            'timestamp':m.group('ts'), 'exact_step':step, 'measurement':step>WARMUP_STEPS,
            'epoch':float(m.group('epoch')), 'loss':float(m.group('loss')),
            'grad_norm':float(m.group('grad_norm')), 'learning_rate':float(m.group('lr')),
            'update_s':update_s, 'data_s':data_s, 'step_s':step_s,
            'samples_s':BATCH_SIZE/step_s if step_s else None,
        })
    if rows:
        with (d/'metrics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return rows

def parse_gpu(d,start=None,end=None):
    p=d/'gpu.csv'
    rows=[]
    if not p.exists(): return rows
    with p.open(newline='',errors='ignore') as f:
        reader=csv.DictReader(f)
        for raw in reader:
            r={k.strip():v for k,v in raw.items() if k is not None}
            tk=next((k for k in r if k.startswith('timestamp')),None)
            uk=next((k for k in r if 'utilization.gpu' in k),None)
            muk=next((k for k in r if 'memory.used' in k),None)
            mtk=next((k for k in r if 'memory.total' in k),None)
            pk=next((k for k in r if 'power.draw' in k),None)
            ts=dt(r.get(tk)) if tk else None
            if start and ts and ts<start: continue
            if end and ts and ts>end: continue
            rows.append({'timestamp':r.get(tk,''),'gpu_util':fnum(r.get(uk)) if uk else None,
                         'memory_used_mib':fnum(r.get(muk)) if muk else None,
                         'memory_total_mib':fnum(r.get(mtk)) if mtk else None,
                         'power_w':fnum(r.get(pk)) if pk else None})
    return rows

def main():
    exp=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
    out=[]
    for nw in WORKERS:
        d=exp/f'nw{nw}'
        metrics=parse_log(d)
        meas=[r for r in metrics if r['measurement']]
        start=dt(meas[0]['timestamp']) if meas else None
        end=dt(meas[-1]['timestamp']) if meas else None
        gpu=parse_gpu(d,start,end)
        meta=read_meta(d)
        status=meta.get('status','missing')
        if status=='failed' and (d/'train.log').exists() and 'out of memory' in (d/'train.log').read_text(errors='ignore').lower():
            status='oom'
        mean_step=mean(r['step_s'] for r in meas)
        log_sps=mean(r['samples_s'] for r in meas)
        wall=meta.get('wall_seconds')
        wall_sps=(BATCH_SIZE*TOTAL_STEPS/wall) if status=='completed' and isinstance(wall,(int,float)) and wall>0 else None
        out.append({
            'num_workers':nw, 'status':status, 'return_code':meta.get('return_code',''),
            'batch_size':BATCH_SIZE, 'total_steps':TOTAL_STEPS, 'warmup_steps':WARMUP_STEPS,
            'measurement_steps':TOTAL_STEPS-WARMUP_STEPS, 'metric_points':len(metrics), 'measurement_points':len(meas),
            'wall_seconds':wall if wall is not None else '', 'wall_samples_s':wall_sps if wall_sps is not None else '',
            'mean_step_s':mean_step if mean_step is not None else '',
            'estimated_epoch_time_s':(mean_step*STEPS_PER_EPOCH) if mean_step is not None else '',
            'estimated_epoch_time_min':(mean_step*STEPS_PER_EPOCH/60) if mean_step is not None else '',
            'mean_data_s':mean(r['data_s'] for r in meas) or '', 'mean_update_s':mean(r['update_s'] for r in meas) or '',
            'log_samples_s':log_sps if log_sps is not None else '',
            'final_loss':metrics[-1]['loss'] if metrics else '', 'final_grad_norm':metrics[-1]['grad_norm'] if metrics else '',
            'mean_gpu_util':mean(r['gpu_util'] for r in gpu) or '', 'p95_gpu_util':pct([r['gpu_util'] for r in gpu],0.95) or '',
            'peak_memory_mib':max([r['memory_used_mib'] for r in gpu if r['memory_used_mib'] is not None], default=''),
            'mean_power_w':mean(r['power_w'] for r in gpu) or '', 'gpu_samples':len(gpu),
        })
    fields=list(out[0].keys())
    for name in ['workers_summary.csv','workers_summary_clean.csv']:
        with (exp/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(out)
    print(exp/'workers_summary_clean.csv')

if __name__=='__main__': main()
