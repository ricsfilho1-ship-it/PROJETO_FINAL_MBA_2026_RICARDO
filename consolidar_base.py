from pathlib import Path
import zipfile
import pandas as pd

START = pd.Timestamp('2025-01-01')
END = pd.Timestamp('2025-02-12')


def consolidar_base(raw_dir: str | Path, output_path: str | Path):
    """Consolida clima + estações + queimadas + MapBiomas em município-dia.

    A janela temporal é limitada ao período comum entre o BDMEP-2025 e a base
    de queimadas de 2025: 01/01/2025 a 12/02/2025.
    """
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)

    meteo_path = raw_dir / 'BDMEP-2025.csv'
    station_path = raw_dir / 'BDMEP-estacao.csv'
    map_path = raw_dir / 'MapBiomas.csv'
    fire_zip = raw_dir / 'queimadas_2024_2025.zip'

    # 1) Meteorologia: carrega apenas a janela comum e as colunas necessárias.
    meteo_cols = [
        'data', 'hora', 'id_estacao', 'precipitacao_total', 'pressao_atm_hora',
        'radiacao_global', 'temperatura_bulbo_hora', 'temperatura_max',
        'temperatura_min', 'umidade_rel_hora', 'vento_rajada_max',
        'vento_velocidade'
    ]
    partes = []
    for chunk in pd.read_csv(
        meteo_path, usecols=meteo_cols, chunksize=500_000, parse_dates=['data']
    ):
        filtro = chunk[(chunk['data'] >= START) & (chunk['data'] <= END)]
        if not filtro.empty:
            partes.append(filtro)
    meteo = pd.concat(partes, ignore_index=True)

    # Há duplicidade quase integral por estação/data/hora na fonte. Primeiro
    # removemos duplicatas exatas; conflitos remanescentes no mesmo horário são
    # consolidados pela média, garantindo 24 registros horários por estação-dia.
    meteo = meteo.drop_duplicates()
    hourly = (
        meteo.groupby(['id_estacao', 'data', 'hora'], as_index=False)
        .agg({
            'precipitacao_total': 'mean',
            'pressao_atm_hora': 'mean',
            'radiacao_global': 'mean',
            'temperatura_bulbo_hora': 'mean',
            'temperatura_max': 'mean',
            'temperatura_min': 'mean',
            'umidade_rel_hora': 'mean',
            'vento_rajada_max': 'mean',
            'vento_velocidade': 'mean',
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

    # 2) Associação estação -> município.
    stations = pd.read_csv(station_path)
    stations['id_municipio'] = stations['id_municipio'].astype('Int64')
    daily_station = daily_station.merge(
        stations[[
            'id_estacao', 'id_municipio', 'id_municipio_nome',
            'latitude', 'longitude', 'altitude'
        ]],
        on='id_estacao', how='left'
    ).dropna(subset=['id_municipio'])

    # Se o município tiver mais de uma estação, usamos a média das estações.
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
        latitude=('latitude', 'mean'),
        longitude=('longitude', 'mean'),
        altitude=('altitude', 'mean'),
        n_estacoes=('id_estacao', 'nunique'),
    )

    # 3) Queimadas: cria quantidade de focos e alvo binário município-dia.
    fire_parts = []
    with zipfile.ZipFile(fire_zip) as zf:
        with zf.open('queimadas_2024_2025.csv') as f:
            for chunk in pd.read_csv(
                f, sep=';', chunksize=500_000, encoding='utf-8-sig',
                parse_dates=['data_hora']
            ):
                filtro = chunk[
                    (chunk['data_hora'] >= START)
                    & (chunk['data_hora'] < END + pd.Timedelta(days=1))
                ]
                if not filtro.empty:
                    fire_parts.append(filtro[['data_hora', 'id_municipio']])
    fire = pd.concat(fire_parts, ignore_index=True)
    fire['data'] = fire['data_hora'].dt.normalize()
    fire_daily = (
        fire.groupby(['id_municipio', 'data'])
        .size().rename('focos_queimada').reset_index()
    )

    base = muni_daily.merge(fire_daily, on=['id_municipio', 'data'], how='left')
    base['focos_queimada'] = base['focos_queimada'].fillna(0).astype(int)
    base['fire_occurrence'] = (base['focos_queimada'] > 0).astype(int)

    # 4) MapBiomas: usa 2021, último ano presente no arquivo fornecido.
    target_munis = set(base['id_municipio'].astype(int))
    mb_parts = []
    for chunk in pd.read_csv(map_path, chunksize=500_000):
        filtro = chunk[
            (chunk['ano'] == 2021)
            & (chunk['id_municipio'].isin(target_munis))
        ]
        if not filtro.empty:
            mb_parts.append(filtro)
    mb = pd.concat(mb_parts, ignore_index=True)

    uf_map = mb[['id_municipio', 'sigla_uf', 'sigla_uf_nome']].drop_duplicates('id_municipio')
    mb_pivot = mb.pivot_table(
        index='id_municipio', columns='id_classe', values='area',
        aggfunc='sum', fill_value=0
    )
    mb_pivot.columns = [f'mb_area_class_{int(c)}' for c in mb_pivot.columns]
    mb_pivot = mb_pivot.reset_index()

    base = base.merge(uf_map, on='id_municipio', how='left')
    base = base.merge(mb_pivot, on='id_municipio', how='left')

    # 5) Variáveis de calendário.
    base['mes'] = base['data'].dt.month
    base['dia_do_ano'] = base['data'].dt.dayofyear
    base['dia_semana'] = base['data'].dt.dayofweek

    # Mantemos valores ausentes meteorológicos documentados. A imputação deve
    # ser ajustada apenas no conjunto de treino na Etapa 2 para evitar leakage.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(output_path, index=False)
    return base


if __name__ == '__main__':
    # Ajuste RAW_DIR ao local em que os quatro arquivos originais estiverem.
    RAW_DIR = Path('/mnt/data')
    ROOT = Path(__file__).resolve().parents[1]
    OUT = ROOT / 'data' / 'processed' / 'dataset_ml.csv'
    df = consolidar_base(RAW_DIR, OUT)
    print(df.shape)
    print(df['fire_occurrence'].value_counts(normalize=True))
