"""Compare vehicle candidates on identical audited source groups, without activation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import time

from train_vehicle import NAMES


def iou(a, b):
    intersection = max(0, min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union else 0


def score(predictions, truth):
    counts = {name: Counter(tp=0, fp=0, fn=0) for name in NAMES}
    used = set()
    for cls, confidence, box in sorted(predictions, key=lambda p: -p[1]):
        matches = [(iou(box, other), i) for i, (label, other) in enumerate(truth) if i not in used and label == cls]
        overlap, index = max(matches, default=(0, -1))
        if overlap >= .5:
            used.add(index)
            counts[cls]['tp'] += 1
        else:
            counts[cls]['fp'] += 1
    for i, (cls, _) in enumerate(truth):
        if i not in used:
            counts[cls]['fn'] += 1
    return counts


def evaluate(weights, data, split):
    from ultralytics import YOLO
    import cv2
    model = YOLO(str(weights))
    names = model.names
    selected = [i for i,n in names.items() if n in NAMES]
    entries = [e for e in json.loads((data/'manifest.json').read_text()) if e['split']==split]
    totals = {name: Counter(tp=0,fp=0,fn=0) for name in NAMES}
    images = []
    times = []
    for entry in entries:
        image = cv2.imread(str(data/'images'/split/Path(entry['image']).name))
        h,w = image.shape[:2]
        if not times:
            model.predict(image, imgsz=640, conf=.35, classes=selected, agnostic_nms=True, verbose=False)
        start = time.perf_counter()
        result = model.predict(image, imgsz=640, conf=.35, classes=selected, agnostic_nms=True, verbose=False)[0]
        predictions = [(names[int(b.cls[0])],float(b.conf[0]),b.xyxy[0].cpu().tolist()) for b in result.boxes]
        times.append(time.perf_counter()-start)
        truth = [(NAMES[c],[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]) for c,x,y,bw,bh in entry['boxes']]
        counts = score(predictions, truth)
        for name in NAMES:
            totals[name].update(counts[name])
        images.append(dict(image=entry['image'], predictions=predictions, counts=counts))
    metrics = {}
    for name,c in totals.items():
        tp,fp,fn=c['tp'],c['fp'],c['fn']
        metrics[name] = dict(c, precision=tp/max(1,tp+fp), recall=tp/max(1,tp+fn), f1=2*tp/max(1,2*tp+fp+fn))
    return dict(weights=str(weights), split=split, images=images, metrics=metrics,
                supported_macro_f1=statistics.mean(metrics[n]['f1'] for n in NAMES[:4]),
                warm_mean_seconds=statistics.mean(times), threshold=.35, iou=.5)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', nargs='+', required=True)
    parser.add_argument('--data', type=Path, default=Path('ml/data/vehicle_sample'))
    parser.add_argument('--split', choices=['val','test'], default='val')
    parser.add_argument('--out', type=Path, required=True)
    args=parser.parse_args()
    reports=[]
    for weights in args.weights:
        report=evaluate(Path(weights),args.data,args.split)
        reports.append(report)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(reports,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k!='images'},indent=2),flush=True)
