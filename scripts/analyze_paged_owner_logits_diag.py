"""CPU summary of forced-AA model diagnostic; never relaxes qualification."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    path=Path(a.input);data=json.loads(path.read_text());result={'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'status':data['status'],'layer_output_gate':data['qualification_layer_output_gate'],'comparisons':{},'performance_evidence':False}
    for phase,comp in data['qualification_same_teacher_path_logits'].items():
        rows=comp['checks'];norms=[x for x in data['layer_norm_differences_from_e0'] if x['phase']==phase]
        changed=[x for x in norms if x['query_norm_abs_difference'] or x['output_norm_abs_difference']]
        perlayer=[]
        for layer in range(48):
            selected=[x for x in norms if x['layer']==layer]
            perlayer.append({'layer':layer,'max_query_norm_difference':max(x['query_norm_abs_difference'] for x in selected),
                'max_output_norm_difference':max(x['output_norm_abs_difference'] for x in selected)})
        result['comparisons'][phase]={'passed':comp['passed'],'vectors':len(rows),'finite':all(x['finite'] for x in rows),
            'cosine_threshold':comp['cosine_threshold'],'below_threshold':sum(x['cosine']<comp['cosine_threshold'] for x in rows),
            'minimum_cosine':min(x['cosine'] for x in rows),'max_logit_absolute_difference':max(x['max_abs'] for x in rows),
            'top1_equal':sum(x['top1_equal'] for x in rows),'layer_norm_rows':len(norms),'nonzero_layer_norm_rows':len(changed),
            'first_changed_layer_records':changed[:4],'layer_max_norm_differences':perlayer}
    result['interpretation']='E0 AA stability is checked separately. Norm divergence localizes where output changes enter the model, but does not identify MoE routing or a specific arithmetic cause. Passing local attention shadows is distinct from passing the frozen final-logit gate.'
    out=Path(a.output);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='comparisons'}))
if __name__=='__main__':main()
