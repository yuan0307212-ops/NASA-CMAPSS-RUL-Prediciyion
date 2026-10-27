import os, json, numpy as np, pandas as pd
from functools import lru_cache
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

BASE=os.path.dirname(__file__)
FEATURES=['cycle']+[f'setting_{i}' for i in range(1,4)]+[f'sensor_{i}' for i in range(1,22)]
COLS=['unit']+FEATURES

@lru_cache(maxsize=8)
def load(subset, kind):
    if subset not in ('FD001','FD002','FD003','FD004'): raise ValueError('不支援的資料集')
    if kind not in ('train','test'): raise ValueError('不支援的資料種類')
    p=os.path.join(BASE,f'{kind}_{subset}.txt')
    df=pd.read_csv(p,sep=r'\s+',header=None)
    if df.shape[1]!=26: raise ValueError(f'資料欄數錯誤: {df.shape[1]}')
    df.columns=COLS
    return df

@lru_cache(maxsize=4)
def rul_truth(subset):
    return pd.read_csv(os.path.join(BASE,f'RUL_{subset}.txt'),header=None,sep=r'\s+').iloc[:,0].to_numpy()

def train_labels(df):
    out=df.copy()
    out['RUL']=out.groupby('unit')['cycle'].transform('max')-out['cycle']
    return out

def metrics(y,p):
    return {'MAE':round(float(mean_absolute_error(y,p)),3),'MSE':round(float(mean_squared_error(y,p)),3),'RMSE':round(float(np.sqrt(mean_squared_error(y,p))),3),'R2':round(float(r2_score(y,p)),4)}

def prepare(subset):
    train=train_labels(load(subset,'train'))
    test=load(subset,'test')
    last=test.sort_values('cycle').groupby('unit',sort=True).tail(1).sort_values('unit')
    truth=rul_truth(subset)
    if len(last)!=len(truth): raise ValueError('測試引擎數與 RUL 答案筆數不一致')
    return train,last,truth

def summarize(subset):
    tr,te,y=prepare(subset)
    std=tr[FEATURES].std()
    constants=std[std<1e-10].index.tolist()
    q=tr['RUL'].quantile([0,.25,.5,.75,1]).round(1).tolist()
    corr=tr[FEATURES+['RUL']].corr(numeric_only=True)['RUL'].drop('RUL').replace([np.inf,-np.inf],np.nan).fillna(0).sort_values(key=abs,ascending=False)
    return {'subset':subset,'rows':len(tr),'test_rows':len(load(subset,'test')),'train_engines':int(tr.unit.nunique()),'test_engines':int(te.unit.nunique()),'raw_columns':26,'candidate_features':25,'nonconstant_features':25-len(constants),'constants':constants,'missing':int(tr.isna().sum().sum()),'rul_quantiles':q,'correlations':[{'name':k,'value':round(float(v),4)} for k,v in corr.items()],'sample':tr.head(8).round(4).to_dict(orient='records'),'test_truth_preview':y[:10].tolist()}

def fit(subset='FD002',selection='kbest',k=16,model='linear',seed=42):
    tr,te,truth=prepare(subset)
    candidates=[c for c in FEATURES if tr[c].std()>1e-10]
    k=max(1,min(int(k),len(candidates)))
    splitter=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed)
    a,b=next(splitter.split(tr,groups=tr['unit']))
    train=tr.iloc[a]; val=tr.iloc[b]
    X=train[candidates]; y=train.RUL
    if selection=='kbest':
        selector=SelectKBest(f_regression,k=k).fit(X,y)
        chosen=np.array(candidates)[selector.get_support()].tolist()
    elif selection=='rfe':
        # RFE on a reproducible subsample to keep the UI responsive
        sample=train.sample(n=min(len(train),9000),random_state=seed)
        sc=StandardScaler().fit(sample[candidates]); xs=sc.transform(sample[candidates])
        selector=RFE(LinearRegression(),n_features_to_select=k,step=2).fit(xs,sample.RUL)
        chosen=np.array(candidates)[selector.support_].tolist()
    elif selection=='lasso':
        pipe=Pipeline([('scale',StandardScaler()),('model',Lasso(alpha=1.0,max_iter=5000,random_state=seed))])
        pipe.fit(X,y)
        weights=np.abs(pipe.named_steps['model'].coef_)
        chosen=np.array(candidates)[np.argsort(weights)[-k:]].tolist()
    elif selection=='all': chosen=candidates
    else: raise ValueError('未知特徵選擇方法')
    models={'linear':LinearRegression(),'ridge':Ridge(alpha=100.0),'lasso':Lasso(alpha=1.0,max_iter=5000,random_state=seed)}
    if model not in models: raise ValueError('未知回歸模型')
    def pipe(): return Pipeline([('scale',StandardScaler()),('model',models[model])])
    p=pipe().fit(train[chosen],train.RUL)
    pv=p.predict(val[chosen]); baseline=np.full(len(val),float(train.RUL.mean()))
    final=pipe().fit(tr[chosen],tr.RUL)
    pt=final.predict(te[chosen]); pbase=np.full(len(te),float(tr.RUL.mean()))
    resid=val.RUL.to_numpy()-pv
    # Empirical absolute-error quantile on engine-disjoint validation; heuristic interval, not formal coverage guarantee.
    margin=float(np.quantile(np.abs(resid),.90))
    coef=final.named_steps['model'].coef_
    order=np.argsort(np.abs(coef))[::-1]
    data={'subset':subset,'selection':selection,'k':len(chosen),'model':model,'features':chosen,'validation':metrics(val.RUL,pv),'test':metrics(truth,pt),'validation_baseline':metrics(val.RUL,baseline),'test_baseline':metrics(truth,pbase),'interval_margin':round(margin,2),'coefficients':[{'name':chosen[i],'value':round(float(coef[i]),3)} for i in order], 'actual_vs_pred':[{'actual':round(float(a),1),'pred':round(float(b),1),'unit':int(u)} for a,b,u in zip(truth,pt,te.unit)],'residuals':[round(float(v),2) for v in resid[:1200]],'prediction_examples':[{'unit':int(u),'cycle':int(c),'actual':int(a),'pred':round(float(p),1),'lower':round(max(0,float(p)-margin),1),'upper':round(float(p)+margin,1)} for u,c,a,p in zip(te.unit,te.cycle,truth,pt)][:120], 'train_units':int(train.unit.nunique()),'validation_units':int(val.unit.nunique()),'n_train':len(train),'n_val':len(val),'notes':'標準化後係數；區間為驗證集絕對殘差90百分位的經驗區間，非正式信賴區間。'}
    return data,final

if __name__=='__main__':
    os.makedirs(os.path.join(BASE,'outputs'),exist_ok=True)
    for subset in ['FD001','FD002','FD003','FD004']:
        print('summary',subset,flush=True)
        with open(os.path.join(BASE,'outputs',f'summary_{subset}.json'),'w') as f:json.dump(summarize(subset),f,ensure_ascii=False)
    for subset in ['FD002','FD004']:
        print('model',subset,flush=True)
        result,_=fit(subset,'kbest',16,'linear')
        with open(os.path.join(BASE,'outputs',f'model_{subset}.json'),'w') as f:json.dump(result,f,ensure_ascii=False)
