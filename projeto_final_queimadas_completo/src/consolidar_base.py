from pathlib import Path
import argparse
import zipfile
import pandas as pd
import numpy as np

START = pd.Timestamp('2025-01-01')
END = pd.Timestamp('2025-02-12')

WEATHER_COLS = [
    'precipitacao_diaria', 'pressao_media', 'radiacao_global_diaria',
    'temperatura_media', 'temperatura_maxima', 'temperatura_minima',
    'umidade_media', 'rajada_maxima', 'vento_medio'
]

MAPBIOMAS_CLASSES = {
    0: ('id0_sem_legenda', 'ID 0 presente no arquivo fornecido; não consta na legenda oficial da Coleção 7'),
    3: ('formacao_florestal', 'Formação Florestal'),
    4: ('formacao_savanica', 'Formação Savânica'),
    5: ('mangue', 'Mangue'),
    9: ('silvicultura', 'Silvicultura / Forest Plantation'),
    11: ('campo_alagado_area_pantanosa', 'Campo Alagado e Área Pantanosa'),
    12: ('formacao_campestre', 'Formação Campestre'),
    13: ('outras_formacoes_nao_florestais', 'Outras Formações não Florestais'),
    15: ('pastagem', 'Pastagem'),
    20: ('cana', 'Cana'),
    21: ('mosaico_de_usos', 'Mosaico de Usos'),
    23: ('praia_duna_areal', 'Praia, Duna e Areal'),
    24: ('area_urbanizada', 'Área Urbanizada'),
    25: ('outras_areas_nao_vegetadas', 'Outras Áreas não Vegetadas'),
    29: ('afloramento_rochoso', 'Afloramento Rochoso'),
    30: ('mineracao', 'Mineração'),
    31: ('aquicultura', 'Aquicultura'),
    32: ('apicum', 'Apicum'),
    33: ('rio_lago_oceano', 'Rio, Lago e Oceano'),
    39: ('soja', 'Soja'),
    40: ('arroz_beta', 'Arroz (beta)'),
    41: ('outras_lavouras_temporarias', 'Outras Lavouras Temporárias'),
    46: ('cafe', 'Café'),
    47: ('citrus', 'Citrus'),
    48: ('outras_lavouras_perenes', 'Outras Lavouras Perenes'),
    49: ('restinga_arborizada', 'Restinga Arborizada'),
    50: ('restinga_herbacea', 'Restinga Herbácea'),
    62: ('algodao_beta', 'Algodão (beta)'),
}

REGION_BY_UF = {
    'AC': 'Norte', 'AP': 'Norte', 'AM': 'Norte', 'PA': 'Norte', 'RO': 'Norte', 'RR': 'Norte', 'TO': 'Norte',
    'AL': 'Nordeste', 'BA': 'Nordeste', 'CE': 'Nordeste', 'MA': 'Nordeste', 'PB': 'Nordeste', 'PE': 'Nordeste',
    'PI': 'Nordeste', 'RN': 'Nordeste', 'SE': 'Nordeste',
    'DF': 'Centro-Oeste', 'GO': 'Centro-Oeste', 'MT': 'Centro-Oeste', 'MS': 'Centro-Oeste',
    'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste',
    'PR': 'Sul', 'RS': 'Sul', 'SC': 'Sul'
}


def extraterrestrial_radiation_mj(latitude_deg, dates):
    """FAO-56: radiação extraterrestre diária (MJ m-2 dia-1)."""
    latitude = np.deg2rad(np.asarray(latitude_deg, dtype=float))
    day = pd.to_datetime(dates).dt.dayofyear.to_numpy()
    dr = 1 + 0.033 * np.cos(2 * np.pi * day / 365)
    delta = 0.409 * np.sin(2 * np.pi * day / 365 - 1.39)
    arg = np.clip(-np.tan(latitude) * np.tan(delta), -1, 1)
    ws = np.arccos(arg)
    gsc = 0.0820
    return (24 * 60 / np.pi) * gsc * dr * (
        ws * np.sin(latitude) * np.sin(delta)
        + np.cos(latitude) * np.cos(delta) * np.sin(ws)
    )


def consolidar_base(raw_dir, output_path, quality_dir):
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    quality_dir = Path(quality_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)

    meteo_path = raw_dir / 'BDMEP-2025.csv'
    station_path = raw_dir / 'BDMEP-estacao.csv'
    map_path = raw_dir / 'MapBiomas.csv'
    fire_zip = raw_dir / 'queimadas_2024_2025.zip'

    required = [meteo_path, station_path, map_path, fire_zip]
    missing_files = [str(p) for p in required if not p.exists()]
    if missing_files:
        raise FileNotFoundError('Arquivos brutos ausentes: ' + ', '.join(missing_files))

    # 1) Meteorologia — somente a janela comum com queimadas.
    meteo_cols = [
        'data', 'hora', 'id_estacao', 'precipitacao_total', 'pressao_atm_hora',
        'radiacao_global', 'temperatura_bulbo_hora', 'temperatura_max',
        'temperatura_min', 'umidade_rel_hora', 'vento_rajada_max', 'vento_velocidade'
    ]
    partes = []
    for chunk in pd.read_csv(meteo_path, usecols=meteo_cols, chunksize=500_000, parse_dates=['data']):
        x = chunk[(chunk['data'] >= START) & (chunk['data'] <= END)].copy()
        if not x.empty:
            numeric_cols = [c for c in meteo_cols if c not in ['data', 'hora', 'id_estacao']]
            # O INMET informa que falhas podem vir como 9999/Null/em branco.
            x[numeric_cols] = x[numeric_cols].replace([9999, 9999.0, -9999, -9999.0], np.nan)
            partes.append(x)
    meteo = pd.concat(partes, ignore_index=True)
    meteo = meteo.drop_duplicates()

    # Consolida conflitos remanescentes no mesmo estação/data/hora.
    hourly = (
        meteo.groupby(['id_estacao', 'data', 'hora'], as_index=False)
        .agg({
            'precipitacao_total': 'mean', 'pressao_atm_hora': 'mean',
            'radiacao_global': 'mean', 'temperatura_bulbo_hora': 'mean',
            'temperatura_max': 'mean', 'temperatura_min': 'mean',
            'umidade_rel_hora': 'mean', 'vento_rajada_max': 'mean',
            'vento_velocidade': 'mean'
        })
    )

    g = hourly.groupby(['id_estacao', 'data'])
    daily_station = pd.DataFrame({
        'precipitacao_diaria': g['precipitacao_total'].sum(min_count=1),
        'pressao_media': g['pressao_atm_hora'].mean(),
        'radiacao_global_diaria': g['radiacao_global'].sum(min_count=1),
        'temperatura_media': g['temperatura_bulbo_hora'].mean(),
        'temperatura_maxima': g['temperatura_max'].max(),
        'temperatura_minima': g['temperatura_min'].min(),
        'umidade_media': g['umidade_rel_hora'].mean(),
        'rajada_maxima': g['vento_rajada_max'].max(),
        'vento_medio': g['vento_velocidade'].mean(),
    }).reset_index()

    stations = pd.read_csv(station_path)
    stations['id_municipio'] = stations['id_municipio'].astype('Int64')
    daily_station = daily_station.merge(
        stations[['id_estacao', 'id_municipio', 'id_municipio_nome', 'latitude', 'longitude', 'altitude']],
        on='id_estacao', how='left'
    ).dropna(subset=['id_municipio'])

    # 2) Controle de qualidade físico básico antes do IQR.
    quality_rows = []

    def flag_and_nan(col, mask, rule):
        mask = mask.fillna(False)
        quality_rows.append({
            'variavel': col, 'regra': rule, 'n_invalidos': int(mask.sum()),
            'acao': 'convertido para NaN'
        })
        daily_station.loc[mask, col] = np.nan

    flag_and_nan('precipitacao_diaria', daily_station['precipitacao_diaria'] < 0, 'precipitação < 0 mm')
    flag_and_nan('pressao_media', daily_station['pressao_media'] <= 0, 'pressão <= 0 hPa')
    flag_and_nan(
        'umidade_media',
        (~daily_station['umidade_media'].between(0, 100)) & daily_station['umidade_media'].notna(),
        'umidade fora de 0–100%'
    )
    flag_and_nan('vento_medio', daily_station['vento_medio'] < 0, 'vento médio < 0 m/s')
    flag_and_nan('rajada_maxima', daily_station['rajada_maxima'] < 0, 'rajada < 0 m/s')

    inv_temp = (
        daily_station['temperatura_maxima'].notna()
        & daily_station['temperatura_minima'].notna()
        & (daily_station['temperatura_maxima'] < daily_station['temperatura_minima'])
    )
    quality_rows.append({
        'variavel': 'temperatura_maxima/minima',
        'regra': 'temperatura máxima < temperatura mínima',
        'n_invalidos': int(inv_temp.sum()), 'acao': 'ambas convertidas para NaN'
    })
    daily_station.loc[inv_temp, ['temperatura_maxima', 'temperatura_minima']] = np.nan

    # Radiação: o INMET mede a radiação global horária em kJ/m².
    # A soma diária à superfície não deve exceder a radiação extraterrestre diária.
    # Usamos tolerância de 5% para ruído de medição.
    daily_station['radiacao_extraterrestre_mj_m2_dia'] = extraterrestrial_radiation_mj(
        daily_station['latitude'], daily_station['data']
    )
    daily_station['radiacao_global_mj_m2_dia_original'] = daily_station['radiacao_global_diaria'] / 1000.0
    rad_zero = daily_station['radiacao_global_diaria'].notna() & (daily_station['radiacao_global_diaria'] <= 0)
    rad_high = (
        daily_station['radiacao_global_diaria'].notna()
        & (daily_station['radiacao_global_mj_m2_dia_original']
           > 1.05 * daily_station['radiacao_extraterrestre_mj_m2_dia'])
    )
    rad_invalid = rad_zero | rad_high
    rad_audit = daily_station.loc[rad_invalid, [
        'id_estacao', 'id_municipio', 'id_municipio_nome', 'data', 'latitude', 'longitude',
        'radiacao_global_diaria', 'radiacao_global_mj_m2_dia_original',
        'radiacao_extraterrestre_mj_m2_dia'
    ]].copy()
    rad_audit['motivo'] = np.where(
        rad_zero.loc[rad_audit.index], 'radiação diária <= 0',
        'radiação diária > 105% da radiação extraterrestre diária (FAO-56)'
    )
    rad_audit.to_csv(quality_dir / 'radiacao_invalida.csv', index=False)
    quality_rows.append({
        'variavel': 'radiacao_global_diaria',
        'regra': '<=0 ou >105% da radiação extraterrestre diária',
        'n_invalidos': int(rad_invalid.sum()), 'acao': 'convertido para NaN'
    })
    daily_station.loc[rad_invalid, 'radiacao_global_diaria'] = np.nan
    pd.DataFrame(quality_rows).to_csv(quality_dir / 'regras_qualidade.csv', index=False)

    # 3) Município-dia.
    muni_daily = daily_station.groupby(
        ['id_municipio', 'id_municipio_nome', 'data'], as_index=False
    ).agg(
        precipitacao_diaria=('precipitacao_diaria', 'mean'),
        pressao_media=('pressao_media', 'mean'),
        radiacao_global_diaria=('radiacao_global_diaria', 'mean'),
        temperatura_media=('temperatura_media', 'mean'),
        temperatura_maxima=('temperatura_maxima', 'mean'),
        temperatura_minima=('temperatura_minima', 'mean'),
        umidade_media=('umidade_media', 'mean'),
        rajada_maxima=('rajada_maxima', 'mean'),
        vento_medio=('vento_medio', 'mean'),
        latitude=('latitude', 'mean'), longitude=('longitude', 'mean'),
        altitude=('altitude', 'mean'), n_estacoes=('id_estacao', 'nunique')
    )

    # Auditoria da ausência simultânea de todas as variáveis climáticas.
    muni_daily['sem_nenhum_dado_climatico'] = muni_daily[WEATHER_COLS].isna().all(axis=1)
    coverage = (
        muni_daily.groupby(['id_municipio', 'id_municipio_nome'], as_index=False)
        .agg(
            dias_no_periodo=('data', 'nunique'),
            dias_sem_nenhum_clima=('sem_nenhum_dado_climatico', 'sum')
        )
    )
    coverage['dias_com_algum_clima'] = coverage['dias_no_periodo'] - coverage['dias_sem_nenhum_clima']
    coverage['pct_dias_com_algum_clima'] = 100 * coverage['dias_com_algum_clima'] / coverage['dias_no_periodo']
    coverage['municipio_sem_clima_total'] = coverage['dias_com_algum_clima'].eq(0)
    coverage.to_csv(quality_dir / 'cobertura_climatica_municipios.csv', index=False)
    coverage.loc[coverage['municipio_sem_clima_total']].to_csv(
        quality_dir / 'municipios_excluidos_sem_clima.csv', index=False
    )

    # Linhas sem qualquer observação de clima são removidas da base de ML.
    muni_daily = muni_daily.loc[~muni_daily['sem_nenhum_dado_climatico']].copy()
    muni_daily.drop(columns='sem_nenhum_dado_climatico', inplace=True)

    # 4) Queimadas e alvo.
    fire_parts = []
    with zipfile.ZipFile(fire_zip) as zf:
        with zf.open('queimadas_2024_2025.csv') as f:
            for chunk in pd.read_csv(
                f, sep=';', chunksize=500_000, encoding='utf-8-sig', parse_dates=['data_hora']
            ):
                x = chunk[
                    (chunk['data_hora'] >= START)
                    & (chunk['data_hora'] < END + pd.Timedelta(days=1))
                ]
                if not x.empty:
                    fire_parts.append(x[['data_hora', 'id_municipio']])
    fire = pd.concat(fire_parts, ignore_index=True)
    fire['data'] = fire['data_hora'].dt.normalize()
    fire_daily = fire.groupby(['id_municipio', 'data']).size().rename('focos_queimada').reset_index()

    base = muni_daily.merge(fire_daily, on=['id_municipio', 'data'], how='left')
    base['focos_queimada'] = base['focos_queimada'].fillna(0).astype(int)
    base['fire_occurrence'] = (base['focos_queimada'] > 0).astype(int)

    # 5) MapBiomas 2021. O arquivo fornecido cobre 1985–2021, consistente com a Coleção 7.
    target_munis = set(base['id_municipio'].astype(int))
    mb_parts = []
    for chunk in pd.read_csv(map_path, chunksize=500_000):
        x = chunk[(chunk['ano'] == 2021) & (chunk['id_municipio'].isin(target_munis))]
        if not x.empty:
            mb_parts.append(x)
    mb = pd.concat(mb_parts, ignore_index=True)
    uf_map = mb[['id_municipio', 'sigla_uf', 'sigla_uf_nome']].drop_duplicates('id_municipio')
    mb_pivot = mb.pivot_table(
        index='id_municipio', columns='id_classe', values='area', aggfunc='sum', fill_value=0
    )
    mb_pivot = mb_pivot.rename(columns={
        c: 'mb_area_' + MAPBIOMAS_CLASSES.get(int(c), (f'classe_{int(c)}', ''))[0]
        for c in mb_pivot.columns
    }).reset_index()

    base = base.merge(uf_map, on='id_municipio', how='left')
    base['regiao'] = base['sigla_uf'].map(REGION_BY_UF)
    base = base.merge(mb_pivot, on='id_municipio', how='left')

    # 6) Calendário.
    base['mes'] = base['data'].dt.month
    base['dia_do_ano'] = base['data'].dt.dayofyear
    base['dia_semana'] = base['data'].dt.dayofweek

    id_cols = ['data', 'id_municipio', 'id_municipio_nome', 'sigla_uf', 'sigla_uf_nome', 'regiao']
    geo_cols = ['latitude', 'longitude', 'altitude', 'n_estacoes']
    map_cols = [c for c in base.columns if c.startswith('mb_area_')]
    base = base[
        id_cols + WEATHER_COLS + geo_cols + map_cols
        + ['mes', 'dia_do_ano', 'dia_semana', 'focos_queimada', 'fire_occurrence']
    ].sort_values(['data', 'id_municipio']).reset_index(drop=True)

    base.to_csv(output_path, index=False)
    return base


def main():
    parser = argparse.ArgumentParser(description='Consolida dados para o projeto FireRisk AI.')
    parser.add_argument('--raw-dir', default='data/raw', help='Pasta com os quatro arquivos brutos.')
    parser.add_argument('--output', default='data/processed/dataset_ml.csv', help='CSV final.')
    parser.add_argument('--quality-dir', default='reports/qualidade', help='Pasta dos relatórios de qualidade.')
    args = parser.parse_args()
    df = consolidar_base(args.raw_dir, args.output, args.quality_dir)
    print('Base salva:', args.output)
    print('Dimensão:', df.shape)
    print('Municípios:', df['id_municipio'].nunique())
    print('Taxa positiva:', round(df['fire_occurrence'].mean(), 4))


if __name__ == '__main__':
    main()
