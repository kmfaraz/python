import json,re,math
from collections import defaultdict,deque
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier,export_text
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,RandomForestRegressor,GradientBoostingRegressor,IsolationForest
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score,log_loss,brier_score_loss,roc_auc_score,mean_absolute_error,mean_squared_error,r2_score,silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.inspection import permutation_importance
S=2025;R=np.random.default_rng(S);C='83733587353df8a41f2fd4f516147d5aa83f5a8d';U=f'https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/{C}/atp/atp_matches_2025.csv';O=Path(__file__).parent/'results';O.mkdir(exist_ok=True)
def js(x):
 if isinstance(x,(np.integer,)):return int(x)
 if isinstance(x,(np.floating,)):return None if np.isnan(x) else float(x)
 if isinstance(x,(pd.Timestamp,)):return str(x)
 raise TypeError
def rec(df,n=30): return json.loads(df.head(n).to_json(orient='records',date_format='iso'))
def met(y,p,name):
 p=np.clip(np.asarray(p),1e-6,1-1e-6);y=np.asarray(y)
 return dict(model=name,accuracy=accuracy_score(y,p>=.5),log_loss=log_loss(y,np.c_[1-p,p],labels=[0,1]),brier=brier_score_loss(y,p),auc=roc_auc_score(y,p))
def prep(n,c,scale=False):
 ns=[('i',SimpleImputer(strategy='median'))]+([('s',StandardScaler())] if scale else [])
 return ColumnTransformer([('n',Pipeline(ns),n),('c',Pipeline([('i',SimpleImputer(strategy='most_frequent')),('o',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),c)])
def split(d):
 q=np.sort(d.date.dropna().unique());cut=pd.Timestamp(q[int(.8*len(q))-1]);return d.date<=cut,d.date>cut,cut
def scoreparse(s):
 if not isinstance(s,str) or 'W/O' in s.upper() or 'DEF' in s.upper():return (np.nan,)*4
 wg=lg=ws=ls=0;ok=0
 for t in s.split():
  m=re.match(r'^(\d+)-(\d+)',t.replace('RET',''))
  if m:
   a,b=map(int,m.groups());wg+=a;lg+=b;ws+=a>b;ls+=b>a;ok=1
 return (wg,lg,ws,ls) if ok else (np.nan,)*4
def dyn(raw,k=32):
 e=defaultdict(lambda:1500.);se=defaultdict(lambda:1500.);r=defaultdict(lambda:deque(maxlen=10));sr=defaultdict(lambda:defaultdict(lambda:deque(maxlen=10)));op=defaultdict(lambda:deque(maxlen=10));last={};hh=defaultdict(lambda:[0,0]);rows=[];traj=[]
 E=lambda a,b:1/(1+10**((b-a)/400))
 for ix,x in raw.sort_values(['date','tourney_id','match_num'],kind='stable').iterrows():
  if pd.isna(x.winner_id) or pd.isna(x.loser_id):continue
  w,l=int(x.winner_id),int(x.loser_id);sf=str(x.surface);ew,el=e[w],e[l];sw,sl=se[w,sf],se[l,sf];key=tuple(sorted((w,l)));h=hh[key];hw,hl=(h if w==key[0] else h[::-1]);pw,ps=E(ew,el),E(sw,sl)
  rows.append(dict(row_id=ix,we=ew,le=el,wse=sw,lse=sl,wr=np.mean(r[w]) if r[w] else .5,lr=np.mean(r[l]) if r[l] else .5,wsr=np.mean(sr[w][sf]) if sr[w][sf] else .5,lsr=np.mean(sr[l][sf]) if sr[l][sf] else .5,wrest=(x.date-last[w]).days if w in last else np.nan,lrest=(x.date-last[l]).days if l in last else np.nan,wh=hw,lh=hl,wop=np.mean(op[w]) if op[w] else 1500.,lop=np.mean(op[l]) if op[l] else 1500.,pw=pw,ps=ps))
  de=k*(1-pw);ds=k*(1-ps);e[w]+=de;e[l]-=de;se[w,sf]+=ds;se[l,sf]-=ds;r[w].append(1);r[l].append(0);sr[w][sf].append(1);sr[l][sf].append(0);op[w].append(el);op[l].append(ew);last[w]=x.date;last[l]=x.date;h[0 if w==key[0] else 1]+=1
  traj+= [dict(date=x.date,id=w,name=x.winner_name,elo=e[w],surface=sf,selo=se[w,sf]),dict(date=x.date,id=l,name=x.loser_name,elo=e[l],surface=sf,selo=se[l,sf])]
 return pd.DataFrame(rows),pd.DataFrame(traj),dict(e),dict(se)
def orient(raw,d):
 z=raw.merge(d,left_index=True,right_on='row_id').sort_values(['date','tourney_id','match_num'],kind='stable').reset_index();f=np.random.default_rng(S).integers(0,2,len(z)).astype(bool);o=pd.DataFrame(dict(date=z.date,tourney=z.tourney_name,surface=z.surface,level=z.tourney_level,round=z['round'],best_of=pd.to_numeric(z.best_of,errors='coerce'),y=(~f).astype(int),score=z.score,minutes=pd.to_numeric(z.minutes,errors='coerce')))
 A={'name':'winner_name','rank':'winner_rank','points':'winner_rank_points','age':'winner_age','ht':'winner_ht','hand':'winner_hand','elo':'we','selo':'wse','r10':'wr','sr10':'wsr','rest':'wrest','h2h':'wh','op':'wop'};B={'name':'loser_name','rank':'loser_rank','points':'loser_rank_points','age':'loser_age','ht':'loser_ht','hand':'loser_hand','elo':'le','selo':'lse','r10':'lr','sr10':'lsr','rest':'lrest','h2h':'lh','op':'lop'}
 for k in A:o['p1_'+k]=np.where(f,z[B[k]],z[A[k]]);o['p2_'+k]=np.where(f,z[A[k]],z[B[k]])
 N=lambda x:pd.to_numeric(x,errors='coerce');o['rank_adv']=N(o.p2_rank)-N(o.p1_rank);o['points_adv']=np.log1p(N(o.p1_points))-np.log1p(N(o.p2_points));o['age_diff']=N(o.p1_age)-N(o.p2_age);o['ht_diff']=N(o.p1_ht)-N(o.p2_ht);o['elo_adv']=N(o.p1_elo)-N(o.p2_elo);o['selo_adv']=N(o.p1_selo)-N(o.p2_selo);o['r10_adv']=N(o.p1_r10)-N(o.p2_r10);o['sr10_adv']=N(o.p1_sr10)-N(o.p2_sr10);o['rest_adv']=N(o.p1_rest)-N(o.p2_rest);o['h2h_adv']=N(o.p1_h2h)-N(o.p2_h2h);o['op_adv']=N(o.p1_op)-N(o.p2_op);o['hand']=o.p1_hand.fillna('U').astype(str)+'-'+o.p2_hand.fillna('U').astype(str);o['lefty_adv']=(o.p1_hand=='L').astype(int)-(o.p2_hand=='L').astype(int)
 for sf in ['Hard','Clay','Grass']:o['lefty_'+sf]=o.lefty_adv*(o.surface==sf)
 for q in ['ace','df','svpt','1stIn','1stWon','2ndWon','bpSaved','bpFaced']:
  a=pd.to_numeric(z['w_'+q],errors='coerce');b=pd.to_numeric(z['l_'+q],errors='coerce');o[q+'_diff']=np.where(f,b-a,a-b)
 sc=z.score.map(scoreparse);wg=sc.map(lambda x:x[0]);lg=sc.map(lambda x:x[1]);ws=sc.map(lambda x:x[2]);ls=sc.map(lambda x:x[3]);o['games_diff']=np.where(f,lg-wg,wg-lg);o['sets_diff']=np.where(f,ls-ws,ws-ls)
 return o
def cvpick(X,y,dates,makers):
 order=np.argsort(np.asarray(dates));X=X.iloc[order].reset_index(drop=True);y=np.asarray(y)[order];cv=TimeSeriesSplit(5);res=[]
 for label,mk in makers:
  v=[]
  for a,b in cv.split(X):m=mk();m.fit(X.iloc[a],y[a]);p=m.predict_proba(X.iloc[b])[:,1];v.append(log_loss(y[b],np.c_[1-p,p],labels=[0,1]))
  res.append((np.mean(v),label,mk))
 return min(res,key=lambda x:x[0]),pd.DataFrame([{'candidate':b,'cv_log_loss':a} for a,b,_ in res]).sort_values('cv_log_loss')
def main():
 raw=pd.read_csv(U);raw['date']=pd.to_datetime(raw.tourney_date.astype(str),format='%Y%m%d',errors='coerce');raw=raw.dropna(subset=['date','winner_name','loser_name']);q=np.sort(raw.date.unique());cut0=pd.Timestamp(q[int(.8*len(q))-1]);kt=[]
 for k in [12,20,28,36,48,64]:
  d,*_=dyn(raw,k);z=raw[['date']].merge(d,left_index=True,right_on='row_id');p=np.clip(z[z.date<=cut0].pw,1e-6,1);kt.append((float(-np.log(p).mean()),k))
 bk=min(kt)[1];d,traj,elo,selo=dyn(raw,bk);x=orient(raw,d);tr,te,cut=split(x);train,test=x[tr],x[te];ytr,yte=train.y,test.y;sn=['rank_adv','points_adv','age_diff','ht_diff','best_of'];sc=['surface','level','round','hand'];rn=sn+['elo_adv','selo_adv','r10_adv','sr10_adv','rest_adv','h2h_adv','op_adv'];res={};models={};probs={}
 (best,lrcv)=cvpick(train[sn+sc],ytr,train.date,[(f'C={c}',lambda c=c:Pipeline([('p',prep(sn,sc,1)),('m',LogisticRegression(C=c,max_iter=3000))])) for c in [.03,.1,.3,1,3]]);lr=best[2]().fit(train[sn+sc],ytr);models['logistic']=lr;probs['Logistic']=lr.predict_proba(test[sn+sc])[:,1]
 P=prep(sn,sc);Xt=P.fit_transform(train[sn+sc]);Xe=P.transform(test[sn+sc]);base=DecisionTreeClassifier(min_samples_leaf=10,random_state=S).fit(Xt,ytr);aa=np.unique(base.cost_complexity_pruning_path(Xt,ytr).ccp_alphas);aa=np.unique(np.quantile(aa,np.linspace(0,1,min(20,len(aa)))));order=np.argsort(train.date.values);cv=TimeSeriesSplit(5);ar=[]
 for a in aa:
  vv=[]
  for i,j in cv.split(Xt[order]):m=DecisionTreeClassifier(min_samples_leaf=10,ccp_alpha=a,random_state=S).fit(Xt[order][i],ytr.values[order][i]);p=m.predict_proba(Xt[order][j])[:,1];vv.append(log_loss(ytr.values[order][j],np.c_[1-p,p],labels=[0,1]))
  m=DecisionTreeClassifier(min_samples_leaf=10,ccp_alpha=a,random_state=S).fit(Xt,ytr);ar.append((np.mean(vv),a,m.get_n_leaves(),m.get_depth()))
 ba=min(ar)[1];tree=DecisionTreeClassifier(min_samples_leaf=10,ccp_alpha=ba,random_state=S).fit(Xt,ytr);probs['Tree']=tree.predict_proba(Xe)[:,1]
 (best,rfcv)=cvpick(train[sn+sc],ytr,train.date,[(str(z),lambda z=z:Pipeline([('p',prep(sn,sc)),('m',RandomForestClassifier(n_estimators=300,max_depth=z[0],min_samples_leaf=z[1],max_features=z[2],n_jobs=-1,random_state=S))])) for z in [(None,5,'sqrt'),(10,5,'sqrt'),(8,10,.7),(6,10,'sqrt')]]);rf=best[2]().fit(train[sn+sc],ytr);models['rf']=rf;probs['Random forest']=rf.predict_proba(test[sn+sc])[:,1]
 (best,gbcv)=cvpick(train[sn+sc],ytr,train.date,[(str(z),lambda z=z:Pipeline([('p',prep(sn,sc)),('m',GradientBoostingClassifier(n_estimators=z[0],learning_rate=z[1],max_depth=z[2],random_state=S))])) for z in [(120,.03,1),(150,.03,2),(200,.05,2),(120,.05,3)]]);gb=best[2]().fit(train[sn+sc],ytr);models['gb']=gb;probs['Gradient boosting']=gb.predict_proba(test[sn+sc])[:,1]
 (best,rollcv)=cvpick(train[rn+sc],ytr,train.date,[(str(z),lambda z=z:Pipeline([('p',prep(rn,sc)),('m',GradientBoostingClassifier(n_estimators=z[0],learning_rate=z[1],max_depth=z[2],random_state=S))])) for z in [(120,.03,1),(150,.03,2),(200,.04,2),(120,.05,3)]]);roll=best[2]().fit(train[rn+sc],ytr);probs['Rolling boosting']=roll.predict_proba(test[rn+sc])[:,1];probs['Elo blend']=1/(1+10**(-(.65*test.elo_adv.fillna(0)+.35*test.selo_adv.fillna(0))/400));show=pd.DataFrame([met(yte,p,k) for k,p in probs.items()]).sort_values('log_loss')
 ex=test[['date','tourney','p1_name','p2_name','y']].copy();ex['p']=probs['Logistic'];ex['actual']=np.where(ex.y==1,ex.p1_name,ex.p2_name);ex['pred']=np.where(ex.p>=.5,ex.p1_name,ex.p2_name);ex['conf']=np.maximum(ex.p,1-ex.p)
 res['01_logistic']={'metrics':met(yte,probs['Logistic'],'Logistic'),'cv':rec(lrcv),'confident_correct':rec(ex[ex.actual==ex.pred].sort_values('conf',ascending=False),12),'confident_wrong':rec(ex[ex.actual!=ex.pred].sort_values('conf',ascending=False),12)}
 res['02_tree']={'metrics':met(yte,probs['Tree'],'Tree'),'alpha_cv':rec(pd.DataFrame(ar,columns=['cv_log_loss','alpha','leaves','depth']).sort_values('cv_log_loss'),15),'leaves':tree.get_n_leaves(),'depth':tree.get_depth(),'rules':export_text(tree,feature_names=list(P.get_feature_names_out()),max_depth=6)}
 res['03_random_forest']={'metrics':met(yte,probs['Random forest'],'Random forest'),'cv':rec(rfcv)};res['04_gradient_boosting']={'metrics':met(yte,probs['Gradient boosting'],'Gradient boosting'),'cv':rec(gbcv)}
 latest={}
 for _,a in raw.sort_values('date').iterrows():
  for side in ['winner','loser']:
   if pd.notna(a[side+'_id']):latest[int(a[side+'_id'])]=(a[side+'_name'],a[side+'_rank'])
 er=pd.DataFrame([{'name':latest[i][0],'elo':v,'rank':latest[i][1]} for i,v in elo.items() if i in latest]).sort_values('elo',ascending=False);rho=spearmanr(er.dropna().elo,-pd.to_numeric(er.dropna()['rank'])).statistic
 res['05_elo']={'K_tuning':[{'K':k,'log_loss':ll} for ll,k in sorted(kt)],'chosen_K':bk,'metrics':met(yte,probs['Elo blend'],'Elo blend'),'top25':rec(er,25),'rank_spearman':rho};res['06_showdown']=rec(show,20)
 cal=[]
 for nm,p in probs.items():
  b=pd.cut(p,np.linspace(0,1,11),include_lowest=True);g=pd.DataFrame({'b':b,'p':p,'y':yte.values}).groupby('b',observed=False).agg(n=('y','size'),mean_pred=('p','mean'),actual=('y','mean')).reset_index();g['model']=nm;cal+=rec(g,20)
 res['07_calibration']={'metrics':rec(show[['model','log_loss','brier']]),'bins':cal}
 rg=raw.copy();rg['minutes']=pd.to_numeric(rg.minutes,errors='coerce');rg=rg[rg.minutes.between(20,300)&~rg.score.fillna('').str.upper().str.contains('RET|W/O|DEF')];rg['rank_gap']=(pd.to_numeric(rg.winner_rank,errors='coerce')-pd.to_numeric(rg.loser_rank,errors='coerce')).abs();rg['mean_rank']=(pd.to_numeric(rg.winner_rank,errors='coerce')+pd.to_numeric(rg.loser_rank,errors='coerce'))/2;rg['age_gap']=(pd.to_numeric(rg.winner_age,errors='coerce')-pd.to_numeric(rg.loser_age,errors='coerce')).abs();rg['ht_gap']=(pd.to_numeric(rg.winner_ht,errors='coerce')-pd.to_numeric(rg.loser_ht,errors='coerce')).abs();n=['rank_gap','mean_rank','age_gap','ht_gap','best_of'];c=['surface','tourney_level','round'];a,b,_=split(rg);rr=[]
 for nm,M in [('RF',RandomForestRegressor(n_estimators=350,min_samples_leaf=5,n_jobs=-1,random_state=S)),('GB',GradientBoostingRegressor(n_estimators=180,learning_rate=.03,max_depth=2,random_state=S))]:
  m=Pipeline([('p',prep(n,c)),('m',M)]).fit(rg.loc[a,n+c],rg.loc[a,'minutes']);p=m.predict(rg.loc[b,n+c]);rr.append({'model':nm,'MAE':mean_absolute_error(rg.loc[b,'minutes'],p),'RMSE':mean_squared_error(rg.loc[b,'minutes'],p)**.5,'R2':r2_score(rg.loc[b,'minutes'],p)})
 res['08_duration_regression']={'usable_matches':len(rg),'metrics':rr}
 F=[]
 for side,w in [('w',1),('l',0)]:
  pr='winner' if side=='w' else 'loser';F.append(pd.DataFrame({'id':raw[pr+'_id'],'name':raw[pr+'_name'],'ht':pd.to_numeric(raw[pr+'_ht'],errors='coerce'),'win':w,'ace':pd.to_numeric(raw[side+'_ace'],errors='coerce'),'df':pd.to_numeric(raw[side+'_df'],errors='coerce'),'sv':pd.to_numeric(raw[side+'_svpt'],errors='coerce'),'fi':pd.to_numeric(raw[side+'_1stIn'],errors='coerce'),'fw':pd.to_numeric(raw[side+'_1stWon'],errors='coerce'),'sw':pd.to_numeric(raw[side+'_2ndWon'],errors='coerce'),'bs':pd.to_numeric(raw[side+'_bpSaved'],errors='coerce'),'bf':pd.to_numeric(raw[side+'_bpFaced'],errors='coerce')}))
 p=pd.concat(F).groupby(['id','name']).agg(matches=('win','size'),wins=('win','sum'),ht=('ht','median'),ace=('ace','sum'),df=('df','sum'),sv=('sv','sum'),fi=('fi','sum'),fw=('fw','sum'),sw=('sw','sum'),bs=('bs','sum'),bf=('bf','sum')).reset_index();p['winpct']=p.wins/p.matches;p['ace100']=100*p.ace/p.sv;p['df100']=100*p.df/p.sv;p['firstin']=p.fi/p.sv;p['firstwon']=p.fw/p.fi;p['secondwon']=p.sw/(p.sv-p.fi);p['bpsave']=p.bs/p.bf;ff=['ht','winpct','ace100','df100','firstin','firstwon','secondwon','bpsave'];p=p[p.matches>=12].replace([np.inf,-np.inf],np.nan).dropna(subset=ff).reset_index(drop=True);Z=StandardScaler().fit_transform(p[ff]);sil=[]
 for k in range(2,7):lab=KMeans(k,n_init=50,random_state=S).fit_predict(Z);sil.append((silhouette_score(Z,lab),k))
 kb=max(sil)[1];km=KMeans(kb,n_init=100,random_state=S).fit(Z);p['cluster']=km.labels_;res['09_clustering']={'k_scores':[{'k':k,'silhouette':s} for s,k in sil],'chosen_k':kb,'profiles':rec(p.groupby('cluster')[ff+['matches']].mean().reset_index(),10)}
 pc=PCA(2).fit(Z);Q=pc.transform(Z);ld=pd.DataFrame({'feature':ff,'PC1':pc.components_[0],'PC2':pc.components_[1]});pp=p[['name','matches']].copy();pp['PC1']=Q[:,0];pp['PC2']=Q[:,1];res['10_pca']={'explained':pc.explained_variance_ratio_.tolist(),'loadings':rec(ld),'extremes':rec(pd.concat([pp.nsmallest(7,'PC1').assign(which='low PC1'),pp.nlargest(7,'PC1').assign(which='high PC1'),pp.nsmallest(7,'PC2').assign(which='low PC2'),pp.nlargest(7,'PC2').assign(which='high PC2')]),40)}
 nn=NearestNeighbors(n_neighbors=min(6,len(p))).fit(Z);ds,ii=nn.kneighbors(Z);pairs=[]
 for i in range(len(p)):
  for j in range(1,ii.shape[1]):pairs.append({'player':p.loc[i,'name'],'neighbor_rank':j,'neighbor':p.loc[ii[i,j],'name'],'distance':ds[i,j]})
 pair=pd.DataFrame(pairs);names=['Jannik Sinner','Carlos Alcaraz','Novak Djokovic','Alexander Zverev','Taylor Fritz','Jack Draper','Ben Shelton','Alex De Minaur'];res['11_neighbors']=rec(pair[pair.player.isin(names)],50)
 an=raw.copy();sp=an.score.map(scoreparse);an['games']=sp.map(lambda x:x[0]+x[1] if pd.notna(x[0]) else np.nan);an['rankgap']=(pd.to_numeric(an.winner_rank,errors='coerce')-pd.to_numeric(an.loser_rank,errors='coerce')).abs();an['aces']=pd.to_numeric(an.w_ace,errors='coerce')+pd.to_numeric(an.l_ace,errors='coerce');an['dfs']=pd.to_numeric(an.w_df,errors='coerce')+pd.to_numeric(an.l_df,errors='coerce');an['sv']=pd.to_numeric(an.w_svpt,errors='coerce')+pd.to_numeric(an.l_svpt,errors='coerce');af=['minutes','games','rankgap','aces','dfs','sv'];an=an[~an.score.fillna('').str.upper().str.contains('RET|W/O|DEF')].dropna(subset=af);iz=StandardScaler().fit_transform(an[af]);iso=IsolationForest(n_estimators=400,contamination=.02,random_state=S).fit(iz);an['score_anom']=-iso.score_samples(iz);res['12_anomalies']=rec(an.nlargest(25,'score_anom')[['date','tourney_name','surface','winner_name','loser_name','score','minutes','rankgap','aces','dfs','score_anom']],25)
 pi=permutation_importance(rf,test[sn+sc],yte,n_repeats=10,random_state=S,scoring='neg_log_loss',n_jobs=-1);res['13_importance']=rec(pd.DataFrame({'feature':sn+sc,'damage':pi.importances_mean,'sd':pi.importances_std}).sort_values('damage',ascending=False))
 le=['ace_diff','df_diff','svpt_diff','1stIn_diff','1stWon_diff','2ndWon_diff','bpSaved_diff','bpFaced_diff'];lm=Pipeline([('p',prep(sn+le,sc)),('m',GradientBoostingClassifier(n_estimators=180,learning_rate=.04,max_depth=2,random_state=S))]).fit(train[sn+le+sc],ytr);pl=lm.predict_proba(test[sn+le+sc])[:,1];sb=test.dropna(subset=['games_diff','sets_diff']);sm=LogisticRegression(C=1000).fit(sb[['games_diff','sets_diff']],sb.y);ps=sm.predict_proba(sb[['games_diff','sets_diff']])[:,1];res['14_leakage']=[met(yte,probs['Gradient boosting'],'honest'),met(yte,pl,'in-match stats'),met(sb.y,ps,'final-score features')]
 t=traj.sort_values('date');chg=[]
 for _,g in t.groupby('id'):
  if len(g)>=15:chg.append({'name':g.iloc[-1]['name'],'matches':len(g),'change':g.iloc[-1].elo-g.iloc[0].elo,'start':g.iloc[0].elo,'end':g.iloc[-1].elo})
 ch=pd.DataFrame(chg);res['15_time_series']={'rises':rec(ch.nlargest(12,'change'),12),'falls':rec(ch.nsmallest(12,'change'),12)};res['16_rolling_form']={'cv':rec(rollcv),'static':met(yte,probs['Gradient boosting'],'static GB'),'rolling':met(yte,probs['Rolling boosting'],'rolling GB')}
 ls=[]
 for i,(nm,rk) in latest.items():
  if pd.notna(rk) and i in elo:ls.append({'id':i,'name':nm,'rank':float(rk),'elo':elo[i],'hard':selo.get((i,'Hard'),1500.)})
 seeds=pd.DataFrame(ls).sort_values('rank').head(16).reset_index(drop=True);bo=[1,16,8,9,4,13,5,12,2,15,7,10,3,14,6,11];pl=[seeds.iloc[i-1].to_dict() for i in bo];cnt=defaultdict(lambda:defaultdict(int));rr=np.random.default_rng(S)
 def wp(a,b):return 1/(1+10**(-(.65*(a['elo']-b['elo'])+.35*(a['hard']-b['hard']))/400))
 N=50000
 for _ in range(N):
  cur=pl
  for lab in ['QF','SF','F']:
   nx=[]
   for j in range(0,len(cur),2):a,b=cur[j],cur[j+1];w=a if rr.random()<wp(a,b) else b;nx.append(w);cnt[w['name']][lab]+=1
   cur=nx
  a,b=cur;w=a if rr.random()<wp(a,b) else b;cnt[w['name']]['W']+=1
 sim=pd.DataFrame([{'player':a['name'],'rank':a['rank'],'QF%':100*cnt[a['name']]['QF']/N,'SF%':100*cnt[a['name']]['SF']/N,'F%':100*cnt[a['name']]['F']/N,'Title%':100*cnt[a['name']]['W']/N} for a in seeds.to_dict('records')]).sort_values('Title%',ascending=False);res['17_simulation']=rec(sim,16)
 up=raw.dropna(subset=['winner_rank','loser_rank']).copy();up['wr']=pd.to_numeric(up.winner_rank);up['lr']=pd.to_numeric(up.loser_rank);up['upset']=(up.wr>up.lr).astype(int);up['gap']=up.wr-up.lr;res['18_upsets']={'rate':up.upset.mean(),'surface':rec(up.groupby('surface').agg(matches=('upset','size'),rate=('upset','mean')).reset_index()),'biggest':rec(up[up.upset==1].nlargest(25,'gap')[['date','tourney_name','surface','winner_name','wr','loser_name','lr','score','gap']],25)}
 z=raw.merge(d,left_index=True,right_on='row_id');sr=[]
 for _,a in z.iterrows():pw=1/(1+10**((a['le']-a['we'])/400));sr+=[{'player':a.winner_name,'surface':a.surface,'a':1,'e':pw},{'player':a.loser_name,'surface':a.surface,'a':0,'e':1-pw}]
 ss=pd.DataFrame(sr).groupby(['player','surface']).agg(matches=('a','size'),actual=('a','sum'),expected=('e','sum')).reset_index();ss['boost']=(ss.actual-ss.expected)/ss.matches;ss=ss[ss.matches>=8];spec=[]
 for sf,g in ss.groupby('surface'):spec+=rec(g.nlargest(10,'boost').assign(group=sf+' over'),10)+rec(g.nsmallest(10,'boost').assign(group=sf+' under'),10)
 res['19_surface_specialists']=spec
 mc=['rank_adv','points_adv','age_diff','ht_diff','lefty_adv','lefty_Hard','lefty_Clay','lefty_Grass'];mm=x.dropna(subset=mc[:4]);a,b,_=split(mm);M=Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler()),('m',LogisticRegression(C=.3,max_iter=3000))]).fit(mm.loc[a,mc],mm.loc[a,'y']);co=M.named_steps['m'].coef_[0];close=mm[mm.rank_adv.abs()<=20];hv=close[((close.p1_hand=='L')&(close.p2_hand=='R'))|((close.p1_hand=='R')&(close.p2_hand=='L'))].copy();hv['lw']=np.where(hv.p1_hand=='L',hv.y,1-hv.y);res['20_matchups']={'coefficients':[{'feature':f,'coef':v,'odds_ratio_1sd':math.exp(v)} for f,v in zip(mc,co)],'close_rank_lefty_vs_righty':rec(hv.groupby('surface').agg(matches=('lw','size'),lefty_win_rate=('lw','mean')).reset_index())}
 meta={'source_commit':C,'matches':len(raw),'train':int(tr.sum()),'test':int(te.sum()),'cutoff':str(cut.date()),'best_elo_K':bk};json.dump({'meta':meta,'results':res},open(O/'all_results.json','w'),indent=2,default=js);pd.DataFrame(show).to_csv(O/'model_showdown.csv',index=False);print(meta);print('done',len(res))
if __name__=='__main__':main()
