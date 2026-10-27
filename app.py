from flask import Flask,render_template,jsonify,request,send_from_directory
from engine import load,prepare,summarize,fit,FEATURES
from functools import lru_cache
import os,json,numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
BASE=os.path.dirname(__file__)
app=Flask(__name__, template_folder=BASE, static_folder=BASE, static_url_path='/static')
app.config['MAX_CONTENT_LENGTH']=12*1024*1024
@app.route('/')
def home():return render_template('index.html')
@app.route('/api/summary/<subset>')
def summary(subset):
    if subset not in ('FD001','FD002','FD003','FD004'):return jsonify(error='無效資料集'),400
    path=os.path.join(BASE,f'summary_{subset}.json')
    if os.path.exists(path):return send_from_directory(BASE,f'summary_{subset}.json')
    return jsonify(summarize(subset))
@app.route('/api/model/<subset>')
def model(subset):
    if subset not in ('FD001','FD002','FD003','FD004'):return jsonify(error='無效資料集'),400
    path=os.path.join(BASE,f'model_{subset}.json')
    if os.path.exists(path):return send_from_directory(BASE,f'model_{subset}.json')
    return jsonify(fit(subset)[0])
@app.route('/api/train',methods=['POST'])
def train():
    try:
        d=request.get_json(force=True)
        sub=d.get('subset','FD002');select=d.get('selection','kbest');k=int(d.get('k',16));mod=d.get('model','linear')
        if sub not in ('FD001','FD002','FD003','FD004') or select not in ('all','kbest','rfe','lasso') or mod not in ('linear','ridge','lasso') or not 1<=k<=25:raise ValueError('參數無效')
        result,_=fit(sub,select,k,mod)
        return jsonify(result)
    except Exception as e:return jsonify(error=str(e)),400
@app.route('/api/engine/<subset>/<int:unit>')
def engine_trace(subset,unit):
    try:
        if subset not in ('FD001','FD002','FD003','FD004'):raise ValueError('無效資料集')
        df=load(subset,'train');one=df[df.unit==unit]
        if one.empty:raise ValueError('找不到引擎')
        stride=max(1,len(one)//250)
        result=one.iloc[::stride]
        return jsonify({'unit':unit,'total_cycles':int(one.cycle.max()),'series':[{'cycle':int(row.cycle),'RUL':int(one.cycle.max()-row.cycle),'sensor_2':float(row.sensor_2),'sensor_4':float(row.sensor_4),'sensor_11':float(row.sensor_11),'sensor_15':float(row.sensor_15)} for row in result.itertuples()]})
    except Exception as e:return jsonify(error=str(e)),400
@app.route('/api/custom',methods=['POST'])
def custom():
    try:
        f=request.files.get('file'); target=request.form.get('target','').strip()
        if not f or not f.filename.lower().endswith('.csv'):raise ValueError('請上傳 CSV 檔')
        df=pd.read_csv(f)
        if len(df)<30 or len(df)>200000:raise ValueError('資料筆數需介於 30 至 200,000')
        if target not in df.columns:raise ValueError('請指定存在的預測目標欄位')
        y=pd.to_numeric(df[target],errors='coerce');x=df.drop(columns=[target]).select_dtypes(include='number').replace([np.inf,-np.inf],np.nan)
        valid=y.notna();x=x.loc[valid];y=y.loc[valid];x=x.loc[:,x.nunique(dropna=True)>1]
        if x.empty:raise ValueError('沒有可用數值特徵')
        x=x.fillna(x.median()).fillna(0)
        cols=x.columns.tolist();n=len(cols)
        xa,xb,ya,yb=train_test_split(x,y,test_size=.2,random_state=42)
        k=min(20,n)
        select=SelectKBest(f_regression,k=k).fit(xa,ya);chosen=x.columns[select.get_support()].tolist()
        pipe=Pipeline([('scale',StandardScaler()),('model',LinearRegression())]).fit(xa[chosen],ya)
        pred=pipe.predict(xb[chosen]);mse=float(mean_squared_error(yb,pred))
        return jsonify({'rows':len(x),'candidate_features':n,'selected_features':chosen,'MAE':round(float(mean_absolute_error(yb,pred)),3),'MSE':round(mse,3),'RMSE':round(mse**.5,3),'R2':round(float(r2_score(yb,pred)),4),'note':'此匯入工具只處理數值欄位；隨機切分僅適用於獨立同分布資料。時間序列／同個體重複量測請另做分組或時間切分。'})
    except Exception as e:return jsonify(error=str(e)),400
@app.route('/download/<filename>')
def download(filename):
    if filename not in [f'{kind}_{sub}.txt' for kind in ('train','test','RUL') for sub in ('FD001','FD002','FD003','FD004')]+['readme.txt']:return 'Not found',404
    return send_from_directory(BASE,filename,as_attachment=True)
@app.route('/health')
def health():return jsonify(status='ok')
if __name__=='__main__':app.run(host='0.0.0.0',port=int(os.environ.get('PORT',5000)),debug=False)
