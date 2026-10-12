from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import optuna
import shap
from lightgbm import LGBMClassifier

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42


def project_root():
    return Path(__file__).resolve().parent.parent


def preparar_features(df):
    mb_raw = [
        c for c in df.columns
        if c.startswith('mb_area_') and c != 'mb_area_id0_sem_legenda'
    ]

    df = df.copy()
    df['mb_total_area'] = df[mb_raw].sum(axis=1, min_count=1)

    for col in mb_raw:
        share_col = col.replace('mb_area_', 'mb_share_')
        df[share_col] = np.where(
            df['mb_total_area'] > 0,
            df[col] / df['mb_total_area'],
            np.nan
        )

    df['mb_log_total_area'] = np.log1p(df['mb_total_area'])

    weather = [
        'precipitacao_diaria', 'pressao_media', 'radiacao_global_diaria',
        'temperatura_media', 'temperatura_maxima', 'temperatura_minima',
        'umidade_media', 'rajada_maxima', 'vento_medio'
    ]
    mb_share = [c.replace('mb_area_', 'mb_share_') for c in mb_raw]

    numeric = weather + [
        'latitude', 'longitude', 'altitude', 'mb_log_total_area',
        'mes', 'dia_do_ano', 'dia_semana'
    ] + mb_share

    categorical = ['sigla_uf', 'regiao']
    return df, numeric, categorical


def criar_preprocessador(numeric, categorical):
    num_pre = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])

    cat_pre = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    return ColumnTransformer([
        ('num', num_pre, numeric),
        ('cat', cat_pre, categorical)
    ])


def avaliar_probs(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()

    return {
        'acuracia': accuracy_score(y_true, pred),
        'precisao': precision_score(y_true, pred, zero_division=0),
        'recall': recall_score(y_true, pred, zero_division=0),
        'f1_score': f1_score(y_true, pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, prob),
        'false_negative_rate': fn / (fn + tp) if (fn + tp) else np.nan,
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)
    }


def metricas_grupo(group, threshold):
    y = group['y_true'].to_numpy()
    p = group['prob'].to_numpy()
    pred = (p >= threshold).astype(int)

    positives = int(y.sum())
    negatives = int((1-y).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())

    return pd.Series({
        'n': len(group),
        'positivos_reais': positives,
        'taxa_real_positiva': positives / len(group) if len(group) else np.nan,
        'taxa_alerta_modelo': pred.mean() if len(group) else np.nan,
        'precisao': precision_score(y, pred, zero_division=0),
        'recall': recall_score(y, pred, zero_division=0),
        'f1_score': f1_score(y, pred, zero_division=0),
        'roc_auc': roc_auc_score(y, p) if positives > 0 and negatives > 0 else np.nan,
        'false_negative_rate': fn / positives if positives else np.nan,
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn
    })


def nome_original_feature(feature_name):
    name = feature_name.replace('num__', '').replace('cat__', '')
    if name.startswith('sigla_uf_'):
        return 'sigla_uf'
    if name.startswith('regiao_'):
        return 'regiao'
    return name


def main(n_trials=6):
    root = project_root()
    out = root / 'reports' / 'etapa3'
    fig_dir = root / 'reports' / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(root / 'data' / 'processed' / 'dataset_ml.csv', parse_dates=['data'])
    df, numeric, categorical = preparar_features(df)
    features = numeric + categorical

    dates = np.array(sorted(df['data'].unique()))
    train_dates = dates[:28]
    valid_dates = dates[28:35]
    test_dates = dates[35:]

    train = df[df['data'].isin(train_dates)].copy()
    valid = df[df['data'].isin(valid_dates)].copy()
    test = df[df['data'].isin(test_dates)].copy()

    # ------------------------------------------------------------------
    # 1. Optuna: CV temporal somente dentro do período de treinamento
    # ------------------------------------------------------------------
    rolling_windows = [(14,18), (18,23), (23,28)]
    cv_data = []

    for end_train, end_valid in rolling_windows:
        tr = df[df['data'].isin(train_dates[:end_train])]
        va = df[df['data'].isin(train_dates[end_train:end_valid])]

        prep = criar_preprocessador(numeric, categorical)
        X_tr = prep.fit_transform(tr[features])
        X_va = prep.transform(va[features])

        cv_data.append((
            X_tr, tr['fire_occurrence'].to_numpy(),
            X_va, va['fire_occurrence'].to_numpy()
        ))

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.08, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 55),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_samples': trial.suggest_int('min_child_samples', 15, 70, step=5),
            'subsample': trial.suggest_float('subsample', 0.75, 1.0),
            'subsample_freq': 1,
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.75, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 2.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 2.0, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.0),
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'verbosity': -1
        }

        aucs = []
        for X_tr, y_tr, X_va, y_va in cv_data:
            model = LGBMClassifier(**params)
            model.fit(X_tr, y_tr)
            prob = model.predict_proba(X_va)[:, 1]
            aucs.append(roc_auc_score(y_va, prob))

        return float(np.mean(aucs))

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trials = study.trials_dataframe()
    trials.to_csv(out / 'optuna_trials.csv', index=False)

    best_params = dict(study.best_params)
    best_params.update({
        'subsample_freq': 1,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbosity': -1
    })

    with open(out / 'optuna_best_params.json', 'w', encoding='utf-8') as f:
        json.dump({
            'best_cv_roc_auc': study.best_value,
            'n_trials': n_trials,
            'params': study.best_params
        }, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 2. Modelo otimizado: fit em treino, threshold escolhido em validação
    # ------------------------------------------------------------------
    prep = criar_preprocessador(numeric, categorical)
    X_train_t = prep.fit_transform(train[features])
    X_valid_t = prep.transform(valid[features])
    X_test_t = prep.transform(test[features])

    model = LGBMClassifier(**best_params)
    model.fit(X_train_t, train['fire_occurrence'])

    valid_prob = model.predict_proba(X_valid_t)[:, 1]
    threshold_rows = []

    for threshold in np.round(np.arange(0.10, 0.81, 0.01), 2):
        m = avaliar_probs(valid['fire_occurrence'], valid_prob, threshold)
        threshold_rows.append({'threshold': threshold, **m})

    threshold_df = pd.DataFrame(threshold_rows)
    elegiveis = threshold_df[
        (threshold_df['recall'] >= 0.70) &
        (threshold_df['precisao'] >= 0.30)
    ]

    if len(elegiveis):
        selected = elegiveis.sort_values(
            ['f1_score', 'recall'], ascending=False
        ).iloc[0]
    else:
        selected = threshold_df.sort_values('f1_score', ascending=False).iloc[0]

    operational_threshold = float(selected['threshold'])
    threshold_df.to_csv(out / 'selecao_threshold_validacao.csv', index=False)

    test_prob = model.predict_proba(X_test_t)[:, 1]
    optimized_metrics = avaliar_probs(
        test['fire_occurrence'], test_prob, operational_threshold
    )

    # Comparação justa com o modelo escolhido na Etapa 2
    stage2_meta_path = root / 'models' / 'lightgbm_stage2_metadata.json'
    with open(stage2_meta_path, encoding='utf-8') as f:
        stage2_meta = json.load(f)
    stage2_metrics = stage2_meta['test_metrics_operational_threshold']

    compare = pd.DataFrame([
        {'modelo': 'LightGBM Etapa 2', 'threshold': stage2_meta['operational_threshold'], **stage2_metrics},
        {'modelo': 'LightGBM + Optuna Etapa 3', 'threshold': operational_threshold, **optimized_metrics}
    ])
    compare.to_csv(out / 'comparacao_etapa2_etapa3.csv', index=False)

    # pipeline de avaliação salvo
    evaluation_pipeline = Pipeline([
        ('prep', prep),
        ('model', model)
    ])
    joblib.dump(evaluation_pipeline, root / 'models' / 'lightgbm_stage3_evaluation.joblib')

    # ------------------------------------------------------------------
    # 3. SHAP — explicabilidade
    # ------------------------------------------------------------------
    feature_names = prep.get_feature_names_out()
    sample_n = min(1200, len(test))
    sample = test.sample(sample_n, random_state=RANDOM_STATE)
    X_sample_t = prep.transform(sample[features])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample_t)

    if isinstance(shap_values, list):
        shap_arr = np.asarray(shap_values[-1])
    else:
        shap_arr = np.asarray(shap_values)
        if shap_arr.ndim == 3:
            shap_arr = shap_arr[:, :, -1]

    transformed_importance = pd.DataFrame({
        'feature_transformada': feature_names,
        'mean_abs_shap': np.abs(shap_arr).mean(axis=0)
    })
    transformed_importance['variavel_original'] = transformed_importance['feature_transformada'].map(nome_original_feature)

    shap_agg = (
        transformed_importance.groupby('variavel_original', as_index=False)['mean_abs_shap']
        .sum()
        .sort_values('mean_abs_shap', ascending=False)
    )
    shap_agg['participacao_pct'] = 100 * shap_agg['mean_abs_shap'] / shap_agg['mean_abs_shap'].sum()
    shap_agg.to_csv(out / 'shap_importancia_variaveis.csv', index=False)
    transformed_importance.sort_values('mean_abs_shap', ascending=False).to_csv(
        out / 'shap_importancia_features_transformadas.csv', index=False
    )

    top15 = shap_agg.head(15).sort_values('mean_abs_shap')
    plt.figure(figsize=(9, 6))
    plt.barh(top15['variavel_original'], top15['mean_abs_shap'])
    plt.xlabel('Média de |SHAP|')
    plt.title('Etapa 3 — importância global das variáveis (SHAP)')
    plt.tight_layout()
    plt.savefig(fig_dir / 'etapa3_shap_top15.png', dpi=160, bbox_inches='tight')
    plt.close()

    # ------------------------------------------------------------------
    # 4. Viés / desempenho desigual por região e UF
    # ------------------------------------------------------------------
    bias_base = test[['sigla_uf', 'regiao']].copy()
    bias_base['y_true'] = test['fire_occurrence'].to_numpy()
    bias_base['prob'] = test_prob

    bias_regiao = (
        bias_base.groupby('regiao', dropna=False)
        .apply(lambda g: metricas_grupo(g, operational_threshold), include_groups=False)
        .reset_index()
        .sort_values('recall', ascending=False)
    )
    bias_regiao.to_csv(out / 'analise_vies_regiao.csv', index=False)

    bias_uf = (
        bias_base.groupby('sigla_uf', dropna=False)
        .apply(lambda g: metricas_grupo(g, operational_threshold), include_groups=False)
        .reset_index()
    )
    # relatório principal por UF: grupos com base mínima para reduzir leituras instáveis
    bias_uf['elegivel_comparacao'] = (bias_uf['n'] >= 50) & (bias_uf['positivos_reais'] >= 10)
    bias_uf.sort_values(['elegivel_comparacao', 'recall'], ascending=[False, False]).to_csv(
        out / 'analise_vies_uf.csv', index=False
    )

    reg_valid = bias_regiao[bias_regiao['positivos_reais'] >= 10].dropna(subset=['recall'])
    uf_valid = bias_uf[bias_uf['elegivel_comparacao']].dropna(subset=['recall'])

    bias_summary = {
        'threshold': operational_threshold,
        'recall_gap_regioes': float(reg_valid['recall'].max() - reg_valid['recall'].min()) if len(reg_valid) else None,
        'melhor_recall_regiao': str(reg_valid.sort_values('recall', ascending=False).iloc[0]['regiao']) if len(reg_valid) else None,
        'pior_recall_regiao': str(reg_valid.sort_values('recall').iloc[0]['regiao']) if len(reg_valid) else None,
        'recall_gap_ufs_elegiveis': float(uf_valid['recall'].max() - uf_valid['recall'].min()) if len(uf_valid) else None,
        'n_ufs_elegiveis': int(len(uf_valid)),
        'nota': 'A base não contém gênero, raça ou outros atributos demográficos; portanto a análise de viés foi limitada a geografia (região e UF).'
    }
    with open(out / 'resumo_vies.json', 'w', encoding='utf-8') as f:
        json.dump(bias_summary, f, indent=2, ensure_ascii=False)

    plot_reg = bias_regiao.sort_values('recall')
    plt.figure(figsize=(8, 5))
    plt.barh(plot_reg['regiao'], plot_reg['recall'])
    plt.axvline(0.70, linestyle='--', linewidth=1)
    plt.xlim(0, 1)
    plt.xlabel('Recall')
    plt.title('Etapa 3 — Recall por região')
    plt.tight_layout()
    plt.savefig(fig_dir / 'etapa3_recall_regiao.png', dpi=160, bbox_inches='tight')
    plt.close()

    uf_plot = bias_uf[bias_uf['elegivel_comparacao']].sort_values('recall')
    if len(uf_plot):
        plt.figure(figsize=(9, max(5, len(uf_plot)*0.3)))
        plt.barh(uf_plot['sigla_uf'], uf_plot['recall'])
        plt.axvline(0.70, linestyle='--', linewidth=1)
        plt.xlim(0, 1)
        plt.xlabel('Recall')
        plt.title('Etapa 3 — Recall por UF (n ≥ 50 e ≥ 10 positivos)')
        plt.tight_layout()
        plt.savefig(fig_dir / 'etapa3_recall_uf.png', dpi=160, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------
    # 5. Gráficos de otimização e comparação
    # ------------------------------------------------------------------
    trials_ok = trials[trials['state'] == 'COMPLETE'].copy()
    plt.figure(figsize=(8, 4.5))
    plt.plot(trials_ok['number'], trials_ok['value'], marker='o')
    plt.axhline(study.best_value, linestyle='--', linewidth=1)
    plt.xlabel('Trial')
    plt.ylabel('ROC-AUC médio da CV temporal')
    plt.title('Etapa 3 — histórico de otimização Optuna')
    plt.tight_layout()
    plt.savefig(fig_dir / 'etapa3_optuna_historico.png', dpi=160, bbox_inches='tight')
    plt.close()

    metrics_for_plot = compare.set_index('modelo')[['precisao', 'recall', 'f1_score', 'roc_auc']]
    ax = metrics_for_plot.plot(kind='bar', figsize=(9,5))
    ax.set_ylim(0,1)
    ax.set_ylabel('Métrica')
    ax.set_title('Etapa 2 x Etapa 3 — LightGBM')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(fig_dir / 'etapa3_comparacao_modelo.png', dpi=160, bbox_inches='tight')
    plt.close()

    # ------------------------------------------------------------------
    # 6. Modelo final para uso futuro: treino + validação, teste continua intocado
    # ------------------------------------------------------------------
    train_valid = df[df['data'].isin(np.concatenate([train_dates, valid_dates]))].copy()
    final_prep = criar_preprocessador(numeric, categorical)
    X_tv = final_prep.fit_transform(train_valid[features])
    final_model = LGBMClassifier(**best_params)
    final_model.fit(X_tv, train_valid['fire_occurrence'])

    final_pipeline = Pipeline([
        ('prep', final_prep),
        ('model', final_model)
    ])
    joblib.dump(final_pipeline, root / 'models' / 'lightgbm_stage3_final.joblib')

    metadata = {
        'model': 'LightGBM otimizado com Optuna',
        'n_optuna_trials': n_trials,
        'best_cv_roc_auc': study.best_value,
        'best_params': study.best_params,
        'operational_threshold': operational_threshold,
        'training_period_evaluation_model': [str(pd.Timestamp(train_dates[0]).date()), str(pd.Timestamp(train_dates[-1]).date())],
        'validation_period_threshold': [str(pd.Timestamp(valid_dates[0]).date()), str(pd.Timestamp(valid_dates[-1]).date())],
        'test_period': [str(pd.Timestamp(test_dates[0]).date()), str(pd.Timestamp(test_dates[-1]).date())],
        'test_metrics_evaluation_model': optimized_metrics,
        'top3_shap': shap_agg.head(3).to_dict('records'),
        'bias_summary': bias_summary,
        'final_model_note': 'lightgbm_stage3_final.joblib foi reajustado em treino+validação após a avaliação final; o teste não foi usado no fit.'
    }
    with open(root / 'models' / 'lightgbm_stage3_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 7. Relatório executivo
    # ------------------------------------------------------------------
    top3 = shap_agg.head(3)
    top3_text = ', '.join(top3['variavel_original'].tolist())
    stage2_f1 = float(stage2_metrics['f1_score'])
    stage2_auc = float(stage2_metrics['roc_auc'])
    stage2_recall = float(stage2_metrics['recall'])

    report = f"""# Relatório executivo — Etapa 3\n\n## Otimização\n\nO LightGBM foi otimizado com **Optuna** em {n_trials} trials. A função objetivo foi o ROC-AUC médio de três janelas de validação temporal construídas somente dentro do período de treinamento. O conjunto de teste não participou da escolha de hiperparâmetros nem do threshold.\n\n- melhor ROC-AUC médio na CV temporal: **{study.best_value:.3f}**;\n- threshold operacional escolhido na validação: **{operational_threshold:.2f}**.\n\n## Resultado final no teste\n\n- Accuracy: **{optimized_metrics['acuracia']:.3f}**\n- Precision: **{optimized_metrics['precisao']:.3f}**\n- Recall: **{optimized_metrics['recall']:.3f}**\n- F1-score: **{optimized_metrics['f1_score']:.3f}**\n- ROC-AUC: **{optimized_metrics['roc_auc']:.3f}**\n- False Negative Rate: **{optimized_metrics['false_negative_rate']:.3f}**\n\nComparado à Etapa 2, o F1 passou de **{stage2_f1:.3f}** para **{optimized_metrics['f1_score']:.3f}**, o ROC-AUC de **{stage2_auc:.3f}** para **{optimized_metrics['roc_auc']:.3f}** e o recall de **{stage2_recall:.3f}** para **{optimized_metrics['recall']:.3f}**.\n\n## Explicabilidade SHAP\n\nAs três variáveis com maior importância global agregada pelo valor absoluto médio de SHAP foram: **{top3_text}**. A importância SHAP mede contribuição para a predição do modelo; ela não deve ser interpretada como causalidade.\n\n## Análise de viés\n\nComo a base não possui gênero, raça ou outros atributos demográficos, a análise de viés possível neste projeto foi geográfica. O desempenho foi comparado por região e por UF, com um filtro mínimo para evitar conclusões baseadas em grupos muito pequenos.\n\n- gap de recall entre regiões elegíveis: **{bias_summary['recall_gap_regioes']:.3f}**;\n- região com maior recall: **{bias_summary['melhor_recall_regiao']}**;\n- região com menor recall: **{bias_summary['pior_recall_regiao']}**;\n- UFs elegíveis para comparação: **{bias_summary['n_ufs_elegiveis']}**.\n\nDiferenças entre regiões não provam viés injusto por si só: também podem refletir composição climática, frequência de eventos e tamanho da amostra. Antes de uso operacional, recomenda-se monitorar recall/FNR por região ao longo do tempo e ampliar a janela histórica.\n\n## Recomendações\n\n1. revalidar o modelo com pelo menos um ciclo anual completo e, idealmente, anos adicionais;\n2. monitorar drift e métricas por região/UF;\n3. manter o threshold como parâmetro de negócio, revisado conforme o custo real de falsos negativos e falsos positivos;\n4. usar SHAP como apoio de explicação, sem tratá-lo como evidência causal;\n5. somente considerar produção após teste A/B/controlado e definição de governança dos alertas.\n"""
    (out / 'relatorio_executivo_etapa3.md').write_text(report, encoding='utf-8')

    # resumo compactado para notebook/readme
    summary = {
        'best_cv_roc_auc': study.best_value,
        'operational_threshold': operational_threshold,
        'test_metrics': optimized_metrics,
        'top3_shap': top3.to_dict('records'),
        'bias_summary': bias_summary
    }
    with open(out / 'resumo_etapa3.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print('Etapa 3 concluída.')
    print('Melhor ROC-AUC CV:', round(study.best_value, 4))
    print('Threshold:', operational_threshold)
    print('Métricas de teste:', {k: round(v,4) if isinstance(v,(float,np.floating)) else v for k,v in optimized_metrics.items()})
    print('Top 3 SHAP:', top3[['variavel_original','mean_abs_shap']].to_dict('records'))
    print('Resumo de viés:', bias_summary)


if __name__ == '__main__':
    import os
    main(n_trials=int(os.getenv('OPTUNA_TRIALS', '6')))
