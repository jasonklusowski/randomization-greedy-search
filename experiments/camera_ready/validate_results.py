"""Independent numerical validation of revised Figure 2 and its saved replicates."""
from pathlib import Path
import importlib.util, csv, json, math, hashlib, datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'experiments/camera_ready'
OUT=BASE/'results'
def module(name):
 spec=importlib.util.spec_from_file_location(name,BASE/f'{name}.py'); obj=importlib.util.module_from_spec(spec); spec.loader.exec_module(obj); return obj
c=module('correlated')
report={'status':'passed','audit_timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'auditor':'independent manuscript_audit agent','execution_command':'python experiments/camera_ready/validate_results.py','source_sha256':{name:hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in ['correlated.py','gram_fs.cpp','correlated_plot.py','validate_results.py']}}
rng=np.random.default_rng(9131)
X=rng.normal(size=(70,16)); X[:,1:]+=0.45*X[:,:1]; X*=np.geomspace(.1,10,16); X+=rng.normal(size=16)*3
y=4+X[:,[2,5,11]]@np.array([1.,-.4,.7])+rng.normal(size=70)
G,g,xm,ym=c.centered_stats(X,y); xc=X-xm; yc=y-ym
errs=[]; steps=0
for size in [1,3,8,16]:
 coefs,sel,cand=c.fit_gram(G,g,14,size,11,89100+size,True); direct=np.zeros_like(coefs)
 for b in range(len(sel)):
  active=[]
  for k in range(14):
   choices=cand[b,k]; choices=choices[choices>=0]
   assert len(choices)==min(size,16-k) and len(set(choices))==len(choices) and set(choices).isdisjoint(active)
   scores=[]
   for j in choices:
    cols=active+[int(j)]; bhat=np.linalg.lstsq(xc[:,cols],yc,rcond=None)[0]; scores.append(np.sum((yc-xc[:,cols]@bhat)**2))
   winner=int(choices[np.argmin(scores)]); assert winner==sel[b,k]
   active.append(winner); bhat=np.linalg.lstsq(xc[:,active],yc,rcond=None)[0]; direct[k+1,active]+=bhat/len(sel); steps+=1
 errs.append(float(np.max(np.abs(direct-coefs)))); np.testing.assert_allclose(coefs,direct,atol=1e-10,rtol=1e-10)
 np.testing.assert_allclose((X@coefs.T +ym-coefs@xm).mean(axis=0),ym,atol=1e-11)
report['correlated_kernel_independent_direct_ols']={'status':'passed','seed':9131,'n':70,'p':16,'kmax':14,'ensemble_size':11,'m_values':[1,3,8,16],'selected_step_checks':steps,'nonzero_means':True,'correlated_unequally_scaled_features':True,'candidate_subset_validity':'passed','intercept_mean_identity':'passed','maximum_coefficient_error':max(errs)}
# Independently invert the Student-t CDF using composite Simpson quadrature.
nu=199; factor=math.exp(math.lgamma((nu+1)/2)-math.lgamma(nu/2))/math.sqrt(nu*math.pi)
def tcdf(x):
 xs=np.linspace(0,x,4097); ys=factor*(1+xs*xs/nu)**(-(nu+1)/2);return .5+x/4096/3*(ys[0]+ys[-1]+4*ys[1:-1:2].sum()+2*ys[2:-1:2].sum())
lo,hi=0.,4.
for _ in range(55):
 mid=(lo+hi)/2
 if tcdf(mid)<.975:lo=mid
 else:hi=mid
tcrit=(lo+hi)/2
paired=list(csv.DictReader((OUT/'correlated_paired_summary.csv').open())); marginal=list(csv.DictReader((OUT/'correlated_marginal_summary.csv').open()));maxerr=0.;maxid=0.;seeds=set();nraw=0
for snr in c.CONFIG['snrs']:
 vals=[]
 for rep in range(200):
  with np.load(OUT/'replicates'/f'correlated_snr{snr:g}_rep{rep:04d}.npz') as z:
   assert json.loads(str(z['config_json']))==c.CONFIG;vals.append(z['metrics']);seed=int(z['data_seed']);assert seed not in seeds;seeds.add(seed)
   cv=z['cv_losses'];bm=np.argmin(cv,axis=0);bk=np.argmin(cv[bm,np.arange(c.CONFIG['kmax'])])+1
   assert int(z['selected_k'])==bk and int(z['selected_m'])==c.CONFIG['m_grid'][bm[bk-1]]
   if int(z['selected_m'])==100:np.testing.assert_array_equal(z['metrics'][0],z['metrics'][1])
   sigma2=float(z['sigma2'])
 vals=np.stack(vals);nraw+=len(vals);d=vals[:,1]-vals[:,0]
 maxid=max(maxid,float(np.max(np.abs(d[:,:,1]-d[:,:,0]-2*sigma2/1000*d[:,:,3]))))
 for table,ispaired in [(paired,True),(marginal,False)]:
  for row in table:
   if float(row['snr'])!=snr:continue
   ki=int(row['k'])-1;mi=c.METRICS.index(row['metric']);a=d[:,ki,mi] if ispaired else vals[:,['FS','EFS'].index(row['method']),ki,mi]
   mean=float(a.mean());sd=float(a.std(ddof=1));se=sd/math.sqrt(len(a))
   for col,value in [('mean',mean),('sd',sd),('mcse',se),('ci_low',mean-tcrit*se),('ci_high',mean+tcrit*se)]:maxerr=max(maxerr,abs(float(row[col])-value))
assert maxerr<1e-9 and maxid<1e-9
report['correlated_raw_summary_recomputation']={'status':'passed','independent_datasets':nraw,'unique_data_seeds':len(seeds),'paired_summary_rows':len(paired),'marginal_summary_rows':len(marginal),'student_t_df':199,'independently_computed_t975':tcrit,'checked_fields':['mean','sd','mcse','ci_low','ci_high'],'maximum_absolute_discrepancy':maxerr,'paired_risk_training_df_identity_maximum_error':maxid,'joint_cv_selection_and_m_equals_p_endpoint':'passed'}
claims={}
for snr in c.CONFIG['snrs']:
 claims[str(snr)]={}
 for metric in ['training_mse','df_covariance','prediction_risk']:
  sub=[r for r in paired if float(r['snr'])==snr and r['metric']==metric]
  claims[str(snr)][metric]={'negative_pointwise_interval_k':[int(r['k']) for r in sub if float(r['ci_high'])<0],'positive_pointwise_interval_k':[int(r['k']) for r in sub if float(r['ci_low'])>0],'interval_contains_zero_k':[int(r['k']) for r in sub if float(r['ci_low'])<=0<=float(r['ci_high'])]}
report['numerical_claims_evidence']=claims
report['scope_notes']=['This validation covers revised Figure 2 and its correlated-design study.','Monte Carlo intervals are pointwise, not simultaneous.','FS and EFS use the same dataset within each replicate.','Paired summary CSVs are retained for numerical checks; only the three marginal Figure 2 panels are generated.']
path=OUT/'independent_validation.json';path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'written':str(path),'status':report['status'],'direct_ols_max_error':max(errs),'correlated_raw_summary_max_error':maxerr,'t975':tcrit}))
