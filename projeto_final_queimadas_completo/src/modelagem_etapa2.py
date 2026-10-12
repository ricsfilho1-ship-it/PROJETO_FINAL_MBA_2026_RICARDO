from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from lightgbm import LGBMClassifier


def get_project_root():
    here = Path(__file__).resolve()
    return here.parent.parent


def preparar_features(df):
    mb_raw = [c for c in df.columns if c.startswith('mb_area_') and c != 'mb_area_id0_sem_legenda']
    df = df.copy()
    df['mb_total_area'] = df[mb_raw].sum(axis=1, min_count=1)
    for c in mb_raw:
        df[c.replace('mb_area_', 'mb_share_')] = np.where(df['mb_total_area'] > 0, df[c] / df['mb_total_area'], np.nan)
    df['mb_log_total_area'] = np.log1p(df['mb_total_area'])

    weather = ['precipitacao_diaria','pressao_media','radiacao_global_diaria','temperatura_media','temperatura_maxima','temperatura_minima','umidade_media','rajada_maxima','vento_medio']
    mb_share = [c.replace('mb_area_', 'mb_share_') for c in mb_raw]
    numeric = weather + ['latitude','longitude','altitude','mb_log_total_area','mes','dia_do_ano','dia_semana'] + mb_share
    categorical = ['sigla_uf','regiao']
    return df, numeric, categorical


def criar_modelos(numeric, categorical):
    num_pre = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    cat_pre = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    pre = ColumnTransformer([('num', num_pre, numeric),('cat', cat_pre, categorical)])
    pre_scaled = Pipeline([('preprocessor', pre),('scaler', StandardScaler())])

    return {
        'Regressão Logística': Pipeline([('prep', clone(pre_scaled)),('model', LogisticRegression(max_iter=3000,class_weight='balanced',random_state=42))]),
        'Random Forest': Pipeline([('prep', clone(pre)),('model', RandomForestClassifier(n_estimators=400,min_samples_leaf=2,max_features='sqrt',class_weight='balanced_subsample',random_state=42,n_jobs=-1))]),
        'LightGBM': Pipeline([('prep', clone(pre)),('model', LGBMClassifier(n_estimators=400,learning_rate=0.03,num_leaves=31,class_weight='balanced',random_state=42,n_jobs=-1,verbosity=-1))])
    }


def avaliar(model, X, y, threshold=0.5):
    prob = model.predict_proba(X)[:,1]
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        'acuracia': accuracy_score(y,pred),
        'precisao': precision_score(y,pred,zero_division=0),
        'recall': recall_score(y,pred,zero_division=0),
        'f1_score': f1_score(y,pred,zero_division=0),
        'roc_auc': roc_auc_score(y,prob),
        'false_negative_rate': fn/(fn+tp),
        'TN': int(tn),'FP': int(fp),'FN': int(fn),'TP': int(tp)
    }


def main():
    root = get_project_root()
    df = pd.read_csv(root/'data'/'processed'/'dataset_ml.csv', parse_dates=['data'])
    df, numeric, categorical = preparar_features(df)
    features = numeric + categorical
    dates = np.array(sorted(df['data'].unique()))
    train_dates, valid_dates, test_dates = dates[:28], dates[28:35], dates[35:]
    train = df[df['data'].isin(train_dates)]
    valid = df[df['data'].isin(valid_dates)]
    test = df[df['data'].isin(test_dates)]
    models = criar_modelos(numeric, categorical)
    rows=[]
    for name, model in models.items():
        model.fit(train[features], train['fire_occurrence'])
        rows.append({'modelo':name, **avaliar(model,test[features],test['fire_occurrence'],0.5)})
        if name == 'LightGBM':
            joblib.dump(model, root/'models'/'lightgbm_stage2.joblib')
    out = pd.DataFrame(rows)
    out.to_csv(root/'reports'/'etapa2'/'metricas_teste_reexecucao.csv', index=False)
    print(out)

if __name__ == '__main__':
    main()
