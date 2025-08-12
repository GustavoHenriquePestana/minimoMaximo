# -*- coding: utf-8 -*-
# Para executar esta aplicação:
# 1. Crie um arquivo 'requirements.txt' com o conteúdo abaixo.
# 2. Instale as bibliotecas: pip install -r requirements.txt
# 3. Execute no terminal: streamlit run app.py

# Conteúdo para requirements.txt:
# streamlit
# requests
# pandas
# pytz
# c8y-api
# Pillow
# plotly
# scipy

import streamlit as st
import requests
import json
import time
from datetime import datetime, timezone, timedelta
import pytz
import os
import pandas as pd
from threading import Thread, Event
from queue import Queue
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import re
from scipy import stats  # Importado para regressão linear

from c8y_api import CumulocityApi
from c8y_api.model import Alarm, Event as C8yEvent

# --- Configuração da Página ---
st.set_page_config(
    page_title="Analisador de Mínimos e Máximos",
    page_icon="📊",
    layout="wide"
)

# --- Estilo CSS Personalizado ---
st.markdown("""
<style>
    .log-container {
        background-color: #1a1a1a;
        color: #fafafa;
        padding: 1rem;
        border-radius: 0.5rem;
        height: 300px;
        overflow-y: auto;
        font-family: monospace;
        font-size: 0.875rem;
        border: 1px solid #2D3748;
    }
    .log-entry { margin-bottom: 0.25rem; }
    .log-error { color: #ff4b4b; }
    .log-warning { color: #ffc400; }
    .log-debug { color: #808080; }
    .log-success { color: #28a745; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background-color: #262730;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a1a1a;
    }
</style>
""", unsafe_allow_html=True)


# --- Funções Auxiliares ---
def format_timestamp_to_brasilia(dt_obj):
    if not dt_obj or pd.isna(dt_obj): return ""
    try:
        if isinstance(dt_obj, str):
            dt_obj = datetime.fromisoformat(dt_obj.replace("Z", "+00:00"))
        brasilia_tz = pytz.timezone("America/Sao_Paulo")
        if dt_obj.tzinfo is None:
            dt_obj = pytz.utc.localize(dt_obj)
        return dt_obj.astimezone(brasilia_tz).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(dt_obj)


def format_uptime(total_seconds):
    if pd.isna(total_seconds) or total_seconds < 0:
        return "N/A"

    total_seconds = int(total_seconds)
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")

    if not parts and seconds >= 0:
        return f"{seconds}s"

    return " ".join(parts) if parts else "0s"


def extract_measurement_value(measurement, measurement_type):
    if measurement_type not in measurement:
        for key in measurement.keys():
            if key.startswith(measurement_type):
                try:
                    fragment = measurement[key]
                    first_series = next(iter(fragment.values()))
                    return float(first_series['value'])
                except (StopIteration, KeyError, ValueError, TypeError):
                    continue
        return None
    try:
        fragment = measurement[measurement_type]
        first_series = next(iter(fragment.values()))
        return float(first_series['value'])
    except (StopIteration, KeyError, ValueError, TypeError):
        return None


def calculate_health_index(kpis):
    if not kpis or kpis.get('is_mkpred'):
        return 0

    availability_score = kpis.get('availability', 100)
    number_of_faults = kpis.get('number_of_faults', 0)
    stability_score = max(0, 100 - (number_of_faults * 10))
    ratio = kpis.get('mean_compression_ratio', 0)
    optimal_ratio_min, optimal_ratio_max = 3.0, 4.5
    performance_score = 0
    if ratio > 0:
        if optimal_ratio_min <= ratio <= optimal_ratio_max:
            performance_score = 100
        else:
            distance = min(abs(ratio - optimal_ratio_min), abs(ratio - optimal_ratio_max))
            performance_score = max(0, 100 - (distance * 50))

    health_index = (availability_score * 0.5) + (stability_score * 0.3) + (performance_score * 0.2)
    return health_index


def is_likely_mkpred(series_list):
    """Heurística para verificar se um dispositivo é para análise de vibração (MKPRED)."""
    vibration_pattern = re.compile(r'^S\d+_(AC|VEL)_\d+$')
    legacy_vibration_measurements = {'v_rms', 'a_rms', 'a_peak', 'kurtosis', 'crest_factor', 'temperature'}
    motor_measurements = {'MA_01', 'MA_02'}

    has_vibration = any(
        vibration_pattern.match(s.split('.')[0]) or s.split('.')[0] in legacy_vibration_measurements
        for s in series_list
    )
    has_motor = any(s.split('.')[0] in motor_measurements for s in series_list)

    return has_vibration and not has_motor


# --- Funções de API (Thread-safe) ---
@st.cache_data(ttl=300)
def fetch_devices(tenant, user, password):
    try:
        c8y = CumulocityApi(base_url=tenant, tenant_id=tenant.split('.')[0].split('//')[1], username=user,
                            password=password)
        all_devices = c8y.inventory.select(query="$filter=has(c8y_IsDevice)")

        devices_structured_list = []
        for device in all_devices:
            name = device.name or "Dispositivo sem nome"
            serial = device.get('c8y_Hardware.serialNumber', 'N/A')
            device_id = device.id
            display_name = f"{name} (S/N: {serial})"
            devices_structured_list.append({
                'display': display_name,
                'name': name,
                'serial': serial,
                'id': device_id
            })
        return sorted(devices_structured_list, key=lambda d: d['display'])
    except Exception as e:
        st.error(f"Erro ao buscar dispositivos: {e}")
        return []


@st.cache_data(ttl=300)
def fetch_supported_series(tenant, user, password, device_id):
    try:
        c8y = CumulocityApi(base_url=tenant, tenant_id=tenant.split('.')[0].split('//')[1], username=user,
                            password=password)
        endpoint = f'/inventory/managedObjects/{device_id}/supportedSeries'
        response_json = c8y.get(endpoint)
        return response_json.get('c8y_SupportedSeries', [])
    except Exception as e:
        st.error(f"Erro ao buscar medições suportadas: {e}")
        return []


# --- Lógica de Análise (Backend) ---

def _fetch_all_raw_data(c8y, device_id, measurements_to_fetch, date_from, date_to):
    """Busca todas as medições brutas necessárias para a análise de um dispositivo."""
    raw_data = {}
    api_call_counter = 0
    for measurement_name in measurements_to_fetch:
        measurements = list(
            c8y.measurements.select(source=device_id, type=measurement_name, date_from=date_from,
                                    date_to=date_to))
        api_call_counter += 1
        points = [(datetime.fromisoformat(m.time.replace("Z", "+00:00")),
                   extract_measurement_value(m, measurement_name)) for m in measurements if
                  extract_measurement_value(m, measurement_name) is not None]
        raw_data[measurement_name] = sorted(points, key=lambda x: x[0])
    return raw_data, api_call_counter


def _processar_ciclos_operacionais(raw_data, job_params, log_queue, device_display_name, job_label):
    """Identifica ciclos operacionais com base nas medições de gatilho e calcula KPIs básicos."""
    log_queue.put({'type': 'log',
                   'data': f"[{device_display_name} | {job_label}] Buscando dados dos gatilhos: {job_params['load_measurement_names']}"})

    all_points = []
    for trigger_name in job_params['load_measurement_names']:
        if trigger_name in raw_data:
            for ts, val in raw_data[trigger_name]:
                all_points.append({'time': ts, 'value': val, 'type': trigger_name})
    all_points.sort(key=lambda p: p['time'])

    summed_trigger_measurements = []
    last_known_values = {name: 0.0 for name in job_params['load_measurement_names']}
    for point in all_points:
        last_known_values[point['type']] = point['value']
        current_sum = sum(last_known_values.values())
        summed_trigger_measurements.append((point['time'], current_sum))

    operational_cycles = []
    operational_kpis = {}
    if summed_trigger_measurements:
        cycle_start_time = None
        for ts, summed_value in summed_trigger_measurements:
            is_on = summed_value > job_params["operating_current"]
            if is_on and cycle_start_time is None:
                cycle_start_time = ts
            elif not is_on and cycle_start_time is not None:
                operational_cycles.append({"start": cycle_start_time, "end": ts})
                cycle_start_time = None
        if cycle_start_time is not None:
            last_ts = summed_trigger_measurements[-1][0]
            operational_cycles.append({"start": cycle_start_time, "end": last_ts})

    log_queue.put({'type': 'log',
                   'data': f"[{device_display_name} | {job_label}] Mapeamento concluído. {len(operational_cycles)} ciclos encontrados."})

    operational_kpis['num_cycles'] = len(operational_cycles)
    total_uptime_seconds = sum((c['end'] - c['start']).total_seconds() for c in operational_cycles)
    operational_kpis['total_uptime'] = total_uptime_seconds

    if operational_kpis['num_cycles'] > 0:
        operational_kpis['mean_cycle_duration'] = total_uptime_seconds / operational_kpis['num_cycles']

    if operational_kpis['num_cycles'] > 1:
        total_off_time_seconds = sum(
            (operational_cycles[i + 1]['start'] - operational_cycles[i]['end']).total_seconds() for i in
            range(len(operational_cycles) - 1))
        operational_kpis['mean_time_between_cycles'] = total_off_time_seconds / (
                operational_kpis['num_cycles'] - 1)

    date_from_obj = datetime.strptime(job_params['date_from'], '%Y-%m-%d')
    date_to_obj = datetime.strptime(job_params['date_to'], '%Y-%m-%d') + timedelta(days=1)
    total_analysis_duration_seconds = (date_to_obj - date_from_obj).total_seconds()

    if total_analysis_duration_seconds > 0:
        operational_kpis['duty_cycle'] = (total_uptime_seconds / total_analysis_duration_seconds) * 100

    return operational_cycles, operational_kpis, total_analysis_duration_seconds


def _calcular_kpis_de_confiabilidade(operational_cycles, alarms, total_analysis_duration_seconds, log_queue,
                                     device_display_name, job_label):
    """Calcula KPIs de disponibilidade e falhas com base nos ciclos e alarmes."""
    kpis = {}
    if not operational_cycles or not alarms:
        kpis['availability'] = 100.0
        kpis['downtime_due_to_fault'] = 0
        kpis['number_of_faults'] = 0
        return kpis

    log_queue.put(
        {'type': 'log', 'data': f"[{device_display_name} | {job_label}] Calculando disponibilidade..."})
    alarm_timestamps = sorted([pd.to_datetime(a['time']) for a in alarms])

    total_downtime_due_to_fault_seconds = 0
    number_of_faults = 0

    for i in range(len(operational_cycles) - 1):
        cycle_end_time = operational_cycles[i]['end']
        next_cycle_start_time = operational_cycles[i + 1]['start']

        fault_window_start = cycle_end_time - timedelta(minutes=1)
        is_fault_stop = any(
            fault_window_start <= alarm_time <= cycle_end_time for alarm_time in alarm_timestamps)

        if is_fault_stop:
            downtime_duration = (next_cycle_start_time - cycle_end_time).total_seconds()
            total_downtime_due_to_fault_seconds += downtime_duration
            number_of_faults += 1

    if total_analysis_duration_seconds > 0:
        availability = ((
                                total_analysis_duration_seconds - total_downtime_due_to_fault_seconds) / total_analysis_duration_seconds) * 100
        kpis['availability'] = availability

    kpis['downtime_due_to_fault'] = total_downtime_due_to_fault_seconds
    kpis['number_of_faults'] = number_of_faults
    return kpis


def _analisar_dados_nos_ciclos(raw_data, operational_cycles, job_params):
    """Analisa as medições alvo dentro dos períodos de estabilização de cada ciclo."""
    results_data = {
        target: {"min": None, "max": None, "count_valid": 0, "min_time": None, "max_time": None, "all_values": []}
        for target in job_params["target_measurements_list"]}

    for cycle in operational_cycles:
        analysis_start = cycle['start'] + timedelta(seconds=job_params['stabilization_delay'])
        analysis_end = cycle['end'] - timedelta(seconds=job_params['shutdown_delay'])
        if analysis_start >= analysis_end: continue

        for target_name in job_params['target_measurements_list']:
            for time_obj, value in raw_data.get(target_name, []):
                if analysis_start <= time_obj <= analysis_end:
                    res = results_data[target_name]
                    if res["min"] is None or value < res["min"]: res["min"], res["min_time"] = value, time_obj
                    if res["max"] is None or value > res["max"]: res["max"], res["max_time"] = value, time_obj
                    res["count_valid"] += 1
                    res["all_values"].append(value)
    return results_data


def _analisar_assinatura_de_partida(raw_data, operational_cycles, job_params, log_queue, device_display_name,
                                    job_label):
    """Processa e analisa as curvas de partida do motor."""
    startup_analysis = {}
    motor_measurements = [m for m in job_params['load_measurement_names'] if m.startswith('MA_')]

    for motor_measurement in motor_measurements:
        if motor_measurement in raw_data and raw_data[motor_measurement]:
            log_queue.put({'type': 'log',
                           'data': f"[{device_display_name} | {job_label}] Analisando partidas de {motor_measurement}..."})
            startup_curves = []
            df_ma = pd.DataFrame(raw_data[motor_measurement], columns=['time', motor_measurement]).set_index(
                'time')

            for cycle in operational_cycles:
                startup_window_end = cycle['start'] + timedelta(seconds=job_params['startup_duration'])
                curve_df = df_ma[(df_ma.index >= cycle['start']) & (df_ma.index <= startup_window_end)]

                if not curve_df.empty:
                    curve_df = curve_df.copy()
                    curve_df['relative_time'] = (curve_df.index - cycle['start']).total_seconds()
                    startup_curves.append(curve_df.set_index('relative_time')[motor_measurement])

            if startup_curves:
                try:
                    resample_index = pd.to_timedelta(np.arange(0, job_params['startup_duration'], 0.1),
                                                     unit='s')
                    resampled_curves = [
                        s.reindex(pd.to_timedelta(s.index, unit='s').union(resample_index)).interpolate(
                            method='time').reindex(resample_index) for s in startup_curves]

                    combined_df = pd.concat(resampled_curves, axis=1)
                    if not combined_df.empty:
                        combined_df.index = combined_df.index.total_seconds()
                        startup_analysis[motor_measurement] = {
                            'mean': combined_df.mean(axis=1).to_dict(),
                            'std': combined_df.std(axis=1).to_dict(),
                            'curves': [s.dropna().to_dict() for s in startup_curves]
                        }
                except Exception as e:
                    log_queue.put({'type': 'log',
                                   'data': f"AVISO: Falha ao processar curvas de partida para {motor_measurement}. Causa: {e}",
                                   'color': 'warning'})
    return startup_analysis


def _calcular_relacao_compressao(raw_data, results_data, operational_kpis, log_queue, device_display_name, job_label):
    """Calcula a relação de compressão se os dados estiverem disponíveis."""
    try:
        required_measurements = ['DP_01', 'SP_01']
        if all(m in raw_data and raw_data[m] for m in required_measurements):
            log_queue.put({'type': 'log',
                           'data': f"[{device_display_name} | {job_label}] Calculando Relação de Compressão..."})
            df_dp = pd.DataFrame(raw_data['DP_01'], columns=['time', 'DP_01']).set_index('time')
            df_sp = pd.DataFrame(raw_data['SP_01'], columns=['time', 'SP_01']).set_index('time')
            df_aligned = pd.concat([df_dp, df_sp], axis=1).interpolate(method='time').dropna()
            if not df_aligned.empty:
                df_aligned['ratio'] = (df_aligned['DP_01'] + 1.013) / (df_aligned['SP_01'] + 1.013)
                ratio_series = df_aligned['ratio']
                if not ratio_series.empty:
                    results_data['Relação de Compressão'] = {"min": ratio_series.min(),
                                                             "max": ratio_series.max(),
                                                             "count_valid": len(ratio_series),
                                                             "min_time": ratio_series.idxmin(),
                                                             "max_time": ratio_series.idxmax(),
                                                             "all_values": ratio_series.tolist()}
                    operational_kpis['mean_compression_ratio'] = ratio_series.mean()
                    raw_data['Relação de Compressão'] = list(ratio_series.reset_index().to_records(index=False))
    except Exception as e:
        log_queue.put({'type': 'log',
                       'data': f"AVISO: Não foi possível calcular a Relação de Compressão para {device_display_name}. Causa: {e}",
                       'color': 'warning'})
    return results_data, operational_kpis, raw_data


def _analisar_alarmes_recorrentes(alarms_and_events, log_queue, device_display_name, job_label):
    """Analisa a frequência e o MTBA dos alarmes."""
    alarm_analysis = {}
    if not alarms_and_events['alarms']:
        return alarm_analysis

    log_queue.put(
        {'type': 'log', 'data': f"[{device_display_name} | {job_label}] Analisando alarmes recorrentes..."})
    df_alarms = pd.DataFrame(alarms_and_events['alarms'])
    df_alarms['time'] = pd.to_datetime(df_alarms['time'])
    df_alarms['base_text'] = df_alarms['text'].apply(lambda x: x.split(' - "')[0] if ' - "' in x else x)
    alarm_counts = df_alarms['base_text'].value_counts().reset_index()
    alarm_counts.columns = ['Alarme', 'Ocorrências']
    df_alarms = df_alarms.sort_values('time')
    mtba_list = []
    for alarm_text, group in df_alarms.groupby('base_text'):
        if len(group) > 1:
            diffs = group['time'].diff().dt.total_seconds()
            mtba_seconds = diffs.mean()
            mtba_list.append({'Alarme': alarm_text, 'MTBA_seconds': mtba_seconds})

    if mtba_list:
        df_mtba = pd.DataFrame(mtba_list)
        final_alarm_df = pd.merge(alarm_counts, df_mtba, on='Alarme', how='left')
    else:
        final_alarm_df = alarm_counts
        final_alarm_df['MTBA_seconds'] = pd.NA

    final_alarm_df['MTBA'] = final_alarm_df['MTBA_seconds'].apply(format_uptime)
    alarm_analysis['ranking'] = final_alarm_df.to_dict('records')
    type_counts = df_alarms['type'].value_counts().reset_index()
    type_counts.columns = ['Tipo', 'Ocorrências']
    alarm_analysis['by_type'] = type_counts.to_dict('records')
    return alarm_analysis


def _calculate_trend_indicators(points):
    """
    Calcula indicadores de tendência para uma série de dados de medição (MKPRED).
    """
    if not points or len(points) < 2:
        return {
            'std_dev': 0, 'slope': 0, 'intercept': 0,
            'r_squared': 0, 'rate_of_change_day': 0, 'mean': 0
        }

    timestamps, values = zip(*points)
    series = pd.Series(values)

    numeric_time = [(t - timestamps[0]).total_seconds() for t in timestamps]

    std_dev = series.std()
    mean_value = series.mean()

    slope, intercept, r_value, p_value, std_err = stats.linregress(numeric_time, values)
    r_squared = r_value ** 2

    change_per_day = slope * 86400

    if abs(mean_value) > 1e-6:
        rate_of_change_day = (change_per_day / mean_value) * 100
    else:
        rate_of_change_day = float('inf') if change_per_day > 0 else 0

    return {
        'std_dev': std_dev,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'rate_of_change_day': rate_of_change_day,
        'mean': mean_value
    }


def _calculate_predictive_health_index(indicators, measurement_name, current_value):
    """
    Calcula o Índice de Saúde Preditivo v2.1 para um dispositivo MKPRED.
    """
    if not indicators or current_value is None:
        return 0

    limits = {
        'v_rms': 5.0,
        'a_rms': 2.5,
        'a_peak': 2.5 * 1.414
    }

    limit = None
    if 'VEL' in measurement_name:
        limit = limits['v_rms']
    elif 'AC' in measurement_name:
        limit = limits['a_rms']

    if limit is None: return 50

    percentage_of_limit = (current_value / limit) * 100
    severity_score = 100 - percentage_of_limit
    severity_score = max(0, min(100, severity_score))

    rate_of_change_day = indicators.get('rate_of_change_day', 0)

    if rate_of_change_day <= 0.5:
        base_degradation_score = 100
    elif rate_of_change_day > 5:
        base_degradation_score = 0
    else:
        base_degradation_score = 100 - (((rate_of_change_day - 0.5) / (5 - 0.5)) * 100)

    r_squared = indicators.get('r_squared', 0)
    penalty = (100 - base_degradation_score) * r_squared
    degradation_score = 100 - penalty
    degradation_score = max(0, min(100, degradation_score))

    final_score = (severity_score * 0.4) + (degradation_score * 0.6)

    return max(0, min(100, final_score))


def get_trend_status(health_index):
    """Retorna uma string formatada para o status da tendência com base no índice de saúde."""
    if health_index >= 80:
        return "🟢 Estável"
    elif 60 <= health_index < 80:
        return "🟡 Atenção"
    else:
        return "🔴 Alerta Crítico"


def _analisar_assinatura_de_ciclo_completo(raw_data, operational_cycles, job_params, log_queue, device_display_name,
                                           job_label):
    """
    Processa e analisa a assinatura do ciclo de operação completo para medições de carga.
    Normaliza a duração de cada ciclo para uma escala de 0 a 100 e calcula uma
    "assinatura" média e o desvio padrão.
    """
    cycle_signature_analysis = {}
    # Pega a primeira medição de carga (ex: 'MA_01') como alvo para a assinatura
    motor_measurement = next((m for m in job_params.get('load_measurement_names', []) if m.startswith('MA_')), None)

    if not motor_measurement or not operational_cycles:
        return cycle_signature_analysis

    if motor_measurement in raw_data and raw_data[motor_measurement]:
        log_queue.put({'type': 'log',
                       'data': f"[{device_display_name} | {job_label}] Analisando Assinatura de Ciclo Completo para {motor_measurement}..."})

        all_cycle_curves = []
        df_motor = pd.DataFrame(raw_data[motor_measurement], columns=['time', motor_measurement]).set_index('time')

        for i, cycle in enumerate(operational_cycles):
            # Filtra os dados do motor para o ciclo atual
            cycle_df = df_motor[(df_motor.index >= cycle['start']) & (df_motor.index <= cycle['end'])]

            if not cycle_df.empty:
                cycle_df = cycle_df.copy()
                cycle_duration = (cycle['end'] - cycle['start']).total_seconds()
                if cycle_duration == 0: continue

                # Normaliza o tempo do ciclo para uma escala de 0 a 100
                cycle_df['normalized_time'] = ((cycle_df.index - cycle['start']).total_seconds() / cycle_duration) * 100

                # Guarda a curva original e o ID do ciclo para referência
                original_curve = cycle_df.set_index('normalized_time')[motor_measurement]
                all_cycle_curves.append({'id': i, 'curve': original_curve})

        if all_cycle_curves:
            try:
                # Cria um índice de tempo normalizado comum (de 0 a 100, com 101 pontos)
                resample_index = np.linspace(0, 100, 101)

                resampled_curves_dict = {}
                for cycle_data in all_cycle_curves:
                    # Interpola cada curva para o novo índice de tempo
                    # Isso garante que todas as curvas tenham o mesmo "comprimento"
                    s = cycle_data['curve']
                    resampled_s = np.interp(resample_index, s.index, s.values)
                    resampled_curves_dict[f"ciclo_{cycle_data['id']}"] = resampled_s

                # Combina todas as curvas reamostradas em um DataFrame
                combined_df = pd.DataFrame(resampled_curves_dict, index=resample_index)

                if not combined_df.empty:
                    # Calcula a média (assinatura dourada) e o desvio padrão
                    mean_curve = combined_df.mean(axis=1)
                    std_curve = combined_df.std(axis=1)

                    cycle_signature_analysis[motor_measurement] = {
                        'mean': mean_curve.to_dict(),
                        'std': std_curve.to_dict(),
                        'upper_bound': (mean_curve + std_curve).to_dict(),
                        'lower_bound': (mean_curve - std_curve).to_dict(),
                        'curves': {f"ciclo_{cd['id']}": cd['curve'].to_dict() for cd in all_cycle_curves}
                    }
            except Exception as e:
                log_queue.put({'type': 'log',
                               'data': f"AVISO: Falha ao processar assinatura de ciclo completo para {motor_measurement}. Causa: {e}",
                               'color': 'warning'})

    return cycle_signature_analysis


# --- FUNÇÃO PRINCIPAL REATORADA ---
def analyze_single_device(job_params, log_queue):
    job_label = job_params['job_label']
    device_id = job_params['device_id']
    device_display_name = job_params['device_display_name']
    api_call_counter = 0

    try:
        c8y = CumulocityApi(base_url=job_params["tenant_url"],
                            tenant_id=job_params["tenant_url"].split('.')[0].split('//')[1],
                            username=job_params["username"], password=job_params["password"])

        log_queue.put({'type': 'log', 'data': f"[{device_display_name} | {job_label}] Iniciando análise..."})

        # --- 1. Coleta de Dados Brutos ---
        all_measurements_to_fetch = set(job_params["target_measurements_list"])
        if not job_params.get('is_mkpred', False):
            all_measurements_to_fetch.update(job_params['load_measurement_names'])

        raw_data, api_calls = _fetch_all_raw_data(c8y, device_id, all_measurements_to_fetch, job_params['date_from'],
                                                  job_params['date_to'])
        api_call_counter += api_calls

        alarms_and_events = {'alarms': [], 'events': []}
        if job_params.get('fetch_alarms'):
            alarms = c8y.alarms.select(source=device_id, date_from=job_params['date_from'],
                                       date_to=job_params['date_to'])
            api_call_counter += 1
            for a in alarms:
                alarms_and_events['alarms'].append(
                    {'time': a.time, 'text': a.text, 'type': a.type, 'severity': a.severity})

        # --- 2. Lógica de Análise Condicional ---
        results_data = {}
        operational_kpis = {}
        startup_analysis = {}
        trend_analysis = {}
        cycle_signature_analysis = {}  # Inicializa o novo dicionário

        if job_params.get('is_mkpred', False):
            log_queue.put({'type': 'log',
                           'data': f"[{device_display_name} | {job_label}] Modo MKPRED: analisando período completo."})
            results_data = {
                target: {"min": None, "max": None, "count_valid": 0, "min_time": None, "max_time": None,
                         "all_values": []}
                for target in job_params["target_measurements_list"]}

            for target_name in job_params['target_measurements_list']:
                points = raw_data.get(target_name, [])
                if points:
                    timestamps, values = zip(*points)
                    series = pd.Series(values)
                    results_data[target_name].update({
                        "min": series.min(), "max": series.max(), "count_valid": len(series),
                        "min_time": timestamps[series.idxmin()], "max_time": timestamps[series.idxmax()],
                        "all_values": values
                    })
                    trend_indicators = _calculate_trend_indicators(points)

                    current_value = series.mean()
                    health_index = _calculate_predictive_health_index(trend_indicators, target_name, current_value)
                    trend_indicators['health_index'] = health_index
                    trend_indicators['status'] = get_trend_status(health_index)
                    trend_analysis[target_name] = trend_indicators

            operational_kpis = {'is_mkpred': True}

        else:
            operational_cycles, operational_kpis, total_duration = _processar_ciclos_operacionais(raw_data, job_params,
                                                                                                  log_queue,
                                                                                                  device_display_name,
                                                                                                  job_label)

            if not operational_cycles:
                log_queue.put({'type': 'log',
                               'data': f"[{device_display_name} | {job_label}] Nenhum ciclo operacional encontrado. Encerrando análise para este dispositivo.",
                               'color': 'warning'})
            else:
                results_data = _analisar_dados_nos_ciclos(raw_data, operational_cycles, job_params)
                startup_analysis = _analisar_assinatura_de_partida(raw_data, operational_cycles, job_params, log_queue,
                                                                   device_display_name, job_label)
                # Chama a nova função de análise de assinatura de ciclo
                cycle_signature_analysis = _analisar_assinatura_de_ciclo_completo(raw_data, operational_cycles,
                                                                                  job_params, log_queue,
                                                                                  device_display_name, job_label)

                kpis_confiabilidade = _calcular_kpis_de_confiabilidade(operational_cycles, alarms_and_events['alarms'],
                                                                       total_duration, log_queue, device_display_name,
                                                                       job_label)
                operational_kpis.update(kpis_confiabilidade)
                results_data, operational_kpis, raw_data = _calcular_relacao_compressao(raw_data, results_data,
                                                                                        operational_kpis, log_queue,
                                                                                        device_display_name, job_label)

        alarm_analysis = _analisar_alarmes_recorrentes(alarms_and_events, log_queue, device_display_name, job_label)

        # --- 3. Finalização ---
        for target_name, data in results_data.items():
            if data.get('all_values'):
                series = pd.Series(data['all_values'])
                data['mean'] = series.mean()
                data['median'] = series.median()
                data['std_dev'] = series.std()
                data['range'] = data['max'] - data['min'] if data['max'] is not None and data['min'] is not None else 0
                data['p95'] = series.quantile(0.95)
            if 'all_values' in data:
                del data['all_values']

        operational_kpis['health_index'] = calculate_health_index(operational_kpis)

        # Retorna o novo dicionário com os resultados
        return job_label, device_display_name, results_data, raw_data, api_call_counter, operational_kpis, alarms_and_events, alarm_analysis, startup_analysis, trend_analysis, cycle_signature_analysis

    except Exception as e:
        import traceback
        log_queue.put(
            {'type': 'log',
             'data': f"ERRO FATAL ao analisar {device_display_name} ({job_label}): {e}\n{traceback.format_exc()}",
             'color': 'error'})
        return job_label, device_display_name, {}, {}, api_call_counter, {}, {}, {}, {}, {}, {}


def perform_analysis_master_thread(stop_event, log_queue, jobs_to_run):
    total_api_calls = 0
    final_results = {}
    final_raw_data = {}
    final_kpis = {}
    final_alarms_events = {}
    final_alarm_analysis = {}
    final_startup_analysis = {}
    final_trend_analysis = {}
    final_cycle_signature_analysis = {}  # Adiciona o novo dicionário

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_job = {executor.submit(analyze_single_device, job, log_queue): job for job in jobs_to_run}

        for i, future in enumerate(as_completed(future_to_job)):
            if stop_event.is_set():
                log_queue.put({'type': 'log', 'data': "Cancelamento solicitado pelo usuário.", 'color': 'warning'})
                break

            # Desempacota o novo valor retornado
            job_label, device_name, results, raw, api_calls, kpis, alarms_events, alarm_analysis, startup_analysis, trend_analysis, cycle_signature = future.result()

            final_results.setdefault(job_label, {})[device_name] = results
            final_raw_data.setdefault(job_label, {})[device_name] = raw
            final_kpis.setdefault(job_label, {})[device_name] = kpis
            final_alarms_events.setdefault(job_label, {})[device_name] = alarms_events
            final_alarm_analysis.setdefault(job_label, {})[device_name] = alarm_analysis
            final_startup_analysis.setdefault(job_label, {})[device_name] = startup_analysis
            final_trend_analysis.setdefault(job_label, {})[device_name] = trend_analysis
            final_cycle_signature_analysis.setdefault(job_label, {})[
                device_name] = cycle_signature  # Armazena o novo resultado

            total_api_calls += api_calls
            log_queue.put(
                {'type': 'status', 'data': f"Análise concluída para {i + 1}/{len(jobs_to_run)} jobs.",
                 'progress': (i + 1) / len(jobs_to_run)})

    log_queue.put({'type': 'log', 'data': f"Análise concluída. Total de Chamadas à API: {total_api_calls}."})
    log_queue.put({'type': 'finished',
                   'data': {'results': final_results, 'raw': final_raw_data, 'api_calls': total_api_calls,
                            'kpis': final_kpis, 'alarms_events': final_alarms_events,
                            'alarm_analysis': final_alarm_analysis,
                            'startup_analysis': final_startup_analysis,
                            'trend_analysis': final_trend_analysis,
                            'cycle_signature_analysis': final_cycle_signature_analysis  # Inclui no resultado final
                            }})


# --- Funções para Salvar/Carregar Configurações ---
def save_settings():
    settings = {
        'tenant': st.session_state.get('tenant'),
        'username': st.session_state.get('username'),
        'saved_devices_display': st.session_state.get('selected_devices_display', [])
    }
    with open("analyzer_settings.json", "w") as f:
        json.dump(settings, f, indent=4)
    st.toast("Configurações salvas!", icon="💾")


def load_settings():
    if os.path.exists("analyzer_settings.json"):
        with open("analyzer_settings.json", "r") as f:
            settings = json.load(f)
        st.session_state.tenant = settings.get('tenant')
        st.session_state.username = settings.get('username')
        st.session_state.selected_devices_display = settings.get('saved_devices_display', [])
        st.toast("Configurações carregadas!", icon="✅")


# --- Inicialização do Estado da Sessão ---
if 'is_running' not in st.session_state: st.session_state.is_running = False
if 'log_messages' not in st.session_state: st.session_state.log_messages = []
if 'status_text' not in st.session_state: st.session_state.status_text = "Aguardando início..."
if 'progress_value' not in st.session_state: st.session_state.progress_value = 0.0
if 'results_df' not in st.session_state: st.session_state.results_df = None
if 'raw_data' not in st.session_state: st.session_state.raw_data = {}
if 'log_queue' not in st.session_state: st.session_state.log_queue = Queue()
if 'api_call_count' not in st.session_state: st.session_state.api_call_count = 0
if 'kpis' not in st.session_state: st.session_state.kpis = {}
if 'alarms_events' not in st.session_state: st.session_state.alarms_events = {}
if 'alarm_analysis' not in st.session_state: st.session_state.alarm_analysis = {}
if 'startup_analysis' not in st.session_state: st.session_state.startup_analysis = {}
if 'trend_analysis' not in st.session_state: st.session_state.trend_analysis = {}
if 'cycle_signature_analysis' not in st.session_state: st.session_state.cycle_signature_analysis = {}  # Inicializa o novo estado
if 'analysis_mode' not in st.session_state: st.session_state.analysis_mode = "Análise Detalhada"

# --- Interface Gráfica ---
st.title("📊 Analisador de Mínimos e Máximos")

with st.sidebar:
    st.header("⚙️ Ações")
    st.button("Salvar Configurações", on_click=save_settings, use_container_width=True)
    st.button("Carregar Configurações", on_click=load_settings, use_container_width=True)

if st.session_state.is_running:
    # --- Tela de Execução ---
    if 'analysis_thread' not in st.session_state or not st.session_state.analysis_thread.is_alive():
        stop_event = Event()
        st.session_state.stop_event = stop_event
        st.session_state.analysis_thread = Thread(target=perform_analysis_master_thread, args=(
            stop_event, st.session_state.log_queue, st.session_state.jobs))
        st.session_state.analysis_thread.start()

    while not st.session_state.log_queue.empty():
        msg = st.session_state.log_queue.get()
        if msg['type'] == 'log':
            st.session_state.log_messages.append(msg)
        elif msg['type'] == 'status':
            st.session_state.status_text = msg['data']
            if 'progress' in msg: st.session_state.progress_value = msg['progress']
        elif msg['type'] == 'finished':
            st.session_state.is_running = False
            results_by_job = msg['data']['results']
            st.session_state.api_call_count = msg['data']['api_calls']
            st.session_state.kpis = msg['data']['kpis']
            st.session_state.alarms_events = msg['data']['alarms_events']
            st.session_state.alarm_analysis = msg['data']['alarm_analysis']
            st.session_state.startup_analysis = msg['data']['startup_analysis']
            st.session_state.trend_analysis = msg['data']['trend_analysis']
            st.session_state.cycle_signature_analysis = msg['data']['cycle_signature_analysis']  # Recebe os novos dados

            df_data = []
            for job_label, devices in results_by_job.items():
                for device_name, results in devices.items():
                    for name, data in results.items():
                        df_data.append({
                            "Período/Job": job_label, "Dispositivo": device_name, "Medição": name,
                            "Mínimo": data.get('min'), "Máximo": data.get('max'), "Amplitude": data.get('range'),
                            "Média": data.get('mean'), "Mediana": data.get('median'),
                            "Desvio Padrão": data.get('std_dev'),
                            "P95": data.get('p95'),
                            "Timestamp Mínimo": format_timestamp_to_brasilia(data.get('min_time')),
                            "Timestamp Máximo": format_timestamp_to_brasilia(data.get('max_time')),
                            "Ocorrências": data.get('count_valid')
                        })

            st.session_state.results_df = pd.DataFrame(df_data) if df_data else pd.DataFrame()
            st.session_state.raw_data = msg['data']['raw']
            st.rerun()

    st.info(st.session_state.status_text)
    st.progress(st.session_state.progress_value)

    st.markdown("### Log de Execução")
    log_html = "".join([f'<div class="log-entry log-{msg.get("color", "")}">{msg["data"]}</div>' for msg in
                        st.session_state.log_messages])
    st.markdown(f'<div class="log-container">{log_html}</div>', unsafe_allow_html=True)

    if st.button("Cancelar Análise", type="primary"):
        st.session_state.stop_event.set()
        st.info("Cancelamento solicitado. Aguardando a finalização do ciclo atual...")

    time.sleep(1)
    st.rerun()

else:
    # --- Tela de Configuração e Resultados ---
    if st.session_state.results_df is not None:
        if st.session_state.results_df.empty:
            st.warning(
                "Nenhum dado encontrado para os parâmetros selecionados. Verifique os ciclos de operação e o período de análise.")
        else:
            st.success("Análise Concluída!")
            st.metric("Chamadas à API", st.session_state.api_call_count)

            analysis_mode = st.session_state.params.get('analysis_mode', 'Análise Detalhada')

            if analysis_mode == "Comparar Dispositivos":
                st.header("📈 Resumo Comparativo de KPIs")
                kpi_summary_data = []
                main_job_kpis = next(iter(st.session_state.kpis.values()), {})
                for device, kpis in main_job_kpis.items():
                    kpi_summary_data.append({
                        "Dispositivo": device,
                        "Índice de Saúde": kpis.get('health_index', 0),
                        "Disponibilidade (%)": kpis.get('availability', 100),
                        "Nº de Paragens por Falha": kpis.get('number_of_faults', 0),
                        "Fator de Carga (%)": kpis.get('duty_cycle', 0),
                        "Relação de Compressão Média": kpis.get('mean_compression_ratio', 0),
                    })

                if kpi_summary_data:
                    kpi_df = pd.DataFrame(kpi_summary_data).set_index("Dispositivo")
                    st.dataframe(kpi_df.style.format(precision=2), use_container_width=True)
                st.divider()

            elif analysis_mode == "Comparar Períodos":
                st.header("🆚 Comparação de Períodos por Dispositivo")

                analyzed_devices = st.session_state.results_df['Dispositivo'].unique()

                if len(analyzed_devices) > 0:
                    device_tabs = st.tabs(list(analyzed_devices))

                    for i, tab in enumerate(device_tabs):
                        with tab:
                            current_device = analyzed_devices[i]

                            device_df = st.session_state.results_df[
                                st.session_state.results_df['Dispositivo'] == current_device]
                            df_a = device_df[device_df['Período/Job'] == 'Período A']
                            df_b = device_df[device_df['Período/Job'] == 'Período B']
                            kpis_a = st.session_state.kpis.get('Período A', {}).get(current_device, {})
                            kpis_b = st.session_state.kpis.get('Período B', {}).get(current_device, {})

                            col_a, col_b = st.columns(2)

                            with col_a:
                                st.subheader(f"Período A")
                                if kpis_a:
                                    st.metric("Índice de Saúde", f"{kpis_a.get('health_index', 0):.1f}")
                                    st.metric("Disponibilidade", f"{kpis_a.get('availability', 100):.2f}%")
                                    st.metric("Nº de Paragens por Falha", kpis_a.get('number_of_faults', 0))
                                st.dataframe(df_a.drop(columns=['Período/Job', 'Dispositivo']),
                                             use_container_width=True)

                            with col_b:
                                st.subheader(f"Período B")
                                if kpis_b:
                                    st.metric("Índice de Saúde", f"{kpis_b.get('health_index', 0):.1f}")
                                    st.metric("Disponibilidade", f"{kpis_b.get('availability', 100):.2f}%")
                                    st.metric("Nº de Paragens por Falha", kpis_b.get('number_of_faults', 0))
                                st.dataframe(df_b.drop(columns=['Período/Job', 'Dispositivo']),
                                             use_container_width=True)
                else:
                    st.warning("Nenhum dispositivo encontrado nos resultados da análise.")
                st.divider()

            if analysis_mode in ["Análise Detalhada", "Comparar Dispositivos"]:
                st.header("🔍 Análise Detalhada por Dispositivo")
                main_job_label = next(iter(st.session_state.kpis.keys()), None)
                if main_job_label:
                    analyzed_devices = list(st.session_state.kpis.get(main_job_label, {}).keys())

                    if analyzed_devices:
                        device_tabs = st.tabs(analyzed_devices)
                        for i, tab in enumerate(device_tabs):
                            with tab:
                                current_device = analyzed_devices[i]
                                device_df = st.session_state.results_df[
                                    st.session_state.results_df['Dispositivo'] == current_device]
                                kpis = st.session_state.kpis.get(main_job_label, {}).get(current_device, {})

                                if kpis.get('is_mkpred'):
                                    st.info("Resultados da Análise de Tendência (MKPRED)")

                                    trend_data = st.session_state.trend_analysis.get(main_job_label, {}).get(
                                        current_device, {})

                                    if trend_data:
                                        health_indexes = [v['health_index'] for v in trend_data.values() if
                                                          'health_index' in v]
                                        if health_indexes:
                                            critical_health_score = np.min(health_indexes)

                                            color = "green"
                                            if critical_health_score < 60:
                                                color = "red"
                                            elif critical_health_score < 80:
                                                color = "orange"

                                            fig_gauge = go.Figure(go.Indicator(
                                                mode="gauge+number", value=critical_health_score,
                                                title={'text': "Índice de Saúde Preditivo"},
                                                gauge={'axis': {'range': [None, 100]}, 'bar': {'color': color},
                                                       'steps': [
                                                           {'range': [0, 60], 'color': "#C70039"},
                                                           {'range': [60, 80], 'color': "#FFC300"},
                                                           {'range': [80, 100], 'color': "#28a745"}],
                                                       }))
                                            fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=80, b=10))
                                            # CORREÇÃO: Adicionada key única
                                            st.plotly_chart(fig_gauge, use_container_width=True,
                                                            key=f"gauge_mkpred_{current_device}")

                                        st.subheader("Análise de Tendência")
                                        trend_df_data = []
                                        for measurement, indicators in trend_data.items():
                                            trend_df_data.append({
                                                "Medição": measurement,
                                                "Status da Tendência": indicators.get('status'),
                                                "Índice de Saúde Preditivo": indicators.get('health_index'),
                                                "Desvio Padrão": indicators.get('std_dev'),
                                                "Inclinação da Tendência": indicators.get('slope'),
                                                "Qualidade da Tendência (R²)": indicators.get('r_squared'),
                                                "Crescimento (% ao dia)": indicators.get('rate_of_change_day')
                                            })

                                        trend_df = pd.DataFrame(trend_df_data)
                                        column_order = ["Medição", "Status da Tendência", "Índice de Saúde Preditivo",
                                                        "Desvio Padrão", "Inclinação da Tendência",
                                                        "Qualidade da Tendência (R²)", "Crescimento (% ao dia)"]
                                        trend_df = trend_df[column_order].set_index("Medição")

                                        st.dataframe(trend_df.style.format({
                                            "Índice de Saúde Preditivo": "{:.1f}",
                                            "Desvio Padrão": "{:.4f}",
                                            "Inclinação da Tendência": "{:.6f}",
                                            "Qualidade da Tendência (R²)": "{:.2%}",
                                            "Crescimento (% ao dia)": "{:.2f}%"
                                        }), use_container_width=True)

                                    st.subheader("Análise Estatística Geral")
                                    display_df = device_df.drop(columns=['Dispositivo', 'Período/Job']).set_index(
                                        'Medição')
                                    st.dataframe(display_df.style.format(precision=2), use_container_width=True)
                                else:
                                    health_score = kpis.get('health_index', 0)
                                    color = "green"
                                    if health_score < 60:
                                        color = "red"
                                    elif health_score < 80:
                                        color = "orange"

                                    fig_gauge = go.Figure(go.Indicator(
                                        mode="gauge+number", value=health_score,
                                        title={'text': "Índice de Saúde do Compressor"},
                                        gauge={'axis': {'range': [None, 100]}, 'bar': {'color': color},
                                               'steps': [
                                                   {'range': [0, 60], 'color': "#C70039"},
                                                   {'range': [60, 80], 'color': "#FFC300"},
                                                   {'range': [80, 100], 'color': "#28a745"}],
                                               }))
                                    fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=80, b=10))
                                    # CORREÇÃO: Adicionada key única
                                    st.plotly_chart(fig_gauge, use_container_width=True,
                                                    key=f"gauge_compressor_{current_device}")

                                    st.subheader(f"KPIs de Confiabilidade e Operação")
                                    kpi_cols1 = st.columns(3)
                                    kpi_cols1[0].metric("Disponibilidade", f"{kpis.get('availability', 100):.2f}%")
                                    kpi_cols1[1].metric("Tempo Parado por Falha",
                                                        format_uptime(kpis.get('downtime_due_to_fault', 0)))
                                    kpi_cols1[2].metric("Nº de Paragens por Falha", kpis.get('number_of_faults', 0))
                                    st.divider()
                                    kpi_cols2 = st.columns(3)
                                    kpi_cols2[0].metric("Número de Ciclos", kpis.get('num_cycles', 0))
                                    kpi_cols2[1].metric("Fator de Carga (Duty Cycle)",
                                                        f"{kpis.get('duty_cycle', 0):.2f}%")
                                    kpi_cols2[2].metric("Tempo de Operação Total",
                                                        format_uptime(kpis.get('total_uptime', 0)))
                                    kpi_cols3 = st.columns(3)
                                    kpi_cols3[0].metric("Duração Média do Ciclo",
                                                        format_uptime(kpis.get('mean_cycle_duration', 0)))
                                    kpi_cols3[1].metric("Tempo Médio Entre Ciclos",
                                                        format_uptime(kpis.get('mean_time_between_cycles', 0)))
                                    kpi_cols3[2].metric("Relação de Compressão Média",
                                                        f"{kpis.get('mean_compression_ratio', 0):.2f}")

                                    st.subheader("Análise Estatística por Medição")
                                    display_df = device_df.drop(columns=['Dispositivo', 'Período/Job']).set_index(
                                        'Medição')


                                    def style_thresholds(row):
                                        styles = [''] * len(row)
                                        device_config = st.session_state.params.get(current_device, {})
                                        thresholds = device_config.get('thresholds', {})
                                        if row.name in thresholds:
                                            min_thresh, max_thresh = thresholds[row.name].get('min'), thresholds[
                                                row.name].get('max')
                                            if min_thresh is not None and pd.notna(row['Mínimo']) and row[
                                                'Mínimo'] < min_thresh:
                                                styles[
                                                    display_df.columns.get_loc('Mínimo')] = 'background-color: #8B0000'
                                            if max_thresh is not None and pd.notna(row['Máximo']) and row[
                                                'Máximo'] > max_thresh:
                                                styles[
                                                    display_df.columns.get_loc('Máximo')] = 'background-color: #8B0000'
                                        return styles


                                    st.dataframe(display_df.style.format(precision=2).apply(style_thresholds, axis=1),
                                                 use_container_width=True)

                                if not device_df.empty:
                                    csv = device_df.to_csv(index=False).encode('utf-8')
                                    st.download_button("Exportar para CSV", data=csv,
                                                       file_name=f"analise_{current_device}.csv",
                                                       mime="text/csv", key=f"csv_{current_device}")

                                st.markdown("### Visualizações de Dados")
                                display_df_for_graphs = device_df.drop(
                                    columns=['Dispositivo', 'Período/Job']).set_index(
                                    'Medição')
                                valid_measurements = display_df_for_graphs[
                                    display_df_for_graphs['Ocorrências'] > 0].index.tolist()

                                # Define a lista de abas de gráficos
                                graph_tab_list = ["Série Temporal", "Histograma", "Correlação"]
                                if not kpis.get('is_mkpred'):
                                    # Adiciona as abas específicas de compressor
                                    graph_tab_list.extend(["Assinatura de Ciclo", "Análise de Partida"])
                                if st.session_state.alarm_analysis.get(main_job_label, {}).get(current_device):
                                    graph_tab_list.append("Análise de Alarmes")

                                graph_tabs = st.tabs(graph_tab_list)
                                tab_idx = 0

                                with graph_tabs[tab_idx]:  # Guia Série Temporal
                                    tab_idx += 1
                                    if valid_measurements:
                                        selected_ts = st.multiselect(
                                            "Selecione as medições para o gráfico de série temporal",
                                            options=valid_measurements, default=valid_measurements[:2],
                                            key=f"ts_select_{current_device}")

                                        col1, col2 = st.columns(2)
                                        with col1:
                                            use_ma = st.checkbox("Exibir Média Móvel", key=f"ma_check_{current_device}")
                                        with col2:
                                            show_trend = st.checkbox("Exibir Linha de Tendência",
                                                                     key=f"trend_check_{current_device}",
                                                                     value=True) if kpis.get('is_mkpred') else False

                                        ma_window = 10
                                        if use_ma:
                                            ma_window = st.slider("Janela da Média Móvel (pontos)", min_value=2,
                                                                  max_value=100, value=10,
                                                                  key=f"ma_slider_{current_device}")

                                        if selected_ts:
                                            fig_ts = go.Figure()
                                            for measurement_name in selected_ts:
                                                raw_points = st.session_state.raw_data.get(main_job_label, {}).get(
                                                    current_device, {}).get(measurement_name, [])
                                                if raw_points:
                                                    times, values = zip(*raw_points)
                                                    fig_ts.add_trace(
                                                        go.Scatter(x=list(times), y=list(values), mode='lines',
                                                                   name=measurement_name))

                                                    if use_ma:
                                                        ma_series = pd.Series(values).rolling(window=ma_window).mean()
                                                        fig_ts.add_trace(
                                                            go.Scatter(x=list(times), y=ma_series, mode='lines',
                                                                       name=f'{measurement_name} (Média Móvel)'))

                                                    if show_trend and kpis.get('is_mkpred'):
                                                        trend_indicators = st.session_state.trend_analysis.get(
                                                            main_job_label, {}).get(current_device, {}).get(
                                                            measurement_name, {})
                                                        slope = trend_indicators.get('slope')
                                                        intercept = trend_indicators.get('intercept')

                                                        if slope is not None and intercept is not None:
                                                            numeric_time = [(t - times[0]).total_seconds() for t in
                                                                            times]
                                                            trend_line_values = [slope * t + intercept for t in
                                                                                 numeric_time]
                                                            fig_ts.add_trace(
                                                                go.Scatter(x=list(times), y=trend_line_values,
                                                                           mode='lines',
                                                                           name=f'{measurement_name} (Tendência)',
                                                                           line=dict(dash='dash')))

                                            fig_ts.update_layout(title=f'Série Temporal para {current_device}',
                                                                 xaxis_title='Timestamp', yaxis_title='Valor',
                                                                 legend_title='Medições')
                                            # CORREÇÃO: Adicionada key única
                                            st.plotly_chart(fig_ts, use_container_width=True,
                                                            key=f"ts_chart_{current_device}")
                                        else:
                                            st.info("Selecione pelo menos uma medição para exibir o gráfico.")
                                    else:
                                        st.warning("Nenhuma medição com dados válidos para plotar.")

                                with graph_tabs[tab_idx]:  # Guia Histograma
                                    tab_idx += 1
                                    if valid_measurements:
                                        selected_hist = st.selectbox("Selecione a medição para o histograma",
                                                                     options=valid_measurements,
                                                                     key=f"hist_select_{current_device}")
                                        if selected_hist:
                                            raw_points = st.session_state.raw_data.get(main_job_label, {}).get(
                                                current_device, {}).get(selected_hist, [])
                                            if raw_points:
                                                _, values = zip(*raw_points)
                                                fig_hist = go.Figure(data=[go.Histogram(x=list(values))])
                                                fig_hist.update_layout(
                                                    title=f'Histograma de {selected_hist} para {current_device}',
                                                    xaxis_title='Valor', yaxis_title='Frequência')
                                                # CORREÇÃO: Adicionada key única
                                                st.plotly_chart(fig_hist, use_container_width=True,
                                                                key=f"hist_chart_{current_device}")
                                    else:
                                        st.warning("Nenhuma medição com dados válidos para plotar.")

                                with graph_tabs[tab_idx]:  # Guia Correlação
                                    tab_idx += 1
                                    if len(valid_measurements) >= 2:
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            x_axis_corr = st.selectbox("Eixo X", options=valid_measurements, index=0,
                                                                       key=f"corr_x_{current_device}")
                                        with col2:
                                            y_axis_corr = st.selectbox("Eixo Y", options=valid_measurements, index=1,
                                                                       key=f"corr_y_{current_device}")

                                        if x_axis_corr and y_axis_corr:
                                            x_points = st.session_state.raw_data.get(main_job_label, {}).get(
                                                current_device, {}).get(x_axis_corr, [])
                                            y_points = st.session_state.raw_data.get(main_job_label, {}).get(
                                                current_device, {}).get(y_axis_corr, [])

                                            if x_points and y_points:
                                                df_x = pd.DataFrame(x_points, columns=['time', x_axis_corr]).set_index(
                                                    'time')
                                                df_y = pd.DataFrame(y_points, columns=['time', y_axis_corr]).set_index(
                                                    'time')

                                                df_corr = pd.concat([df_x, df_y], axis=1).interpolate(
                                                    method='time').dropna()

                                                if not df_corr.empty:
                                                    correlation_coef = df_corr[x_axis_corr].corr(df_corr[y_axis_corr])

                                                    fig_corr = go.Figure(
                                                        data=go.Scatter(x=df_corr[x_axis_corr], y=df_corr[y_axis_corr],
                                                                        mode='markers'))
                                                    fig_corr.update_layout(
                                                        title=f'Correlação entre {x_axis_corr} e {y_axis_corr}<br>Coeficiente de Pearson: {correlation_coef:.2f}',
                                                        xaxis_title=x_axis_corr, yaxis_title=y_axis_corr)
                                                    # CORREÇÃO: Adicionada key única
                                                    st.plotly_chart(fig_corr, use_container_width=True,
                                                                    key=f"corr_chart_{current_device}")
                                                else:
                                                    st.warning("Não foi possível alinhar os dados para a correlação.")
                                    else:
                                        st.warning(
                                            "São necessárias pelo menos duas medições com dados para a análise de correlação.")

                                # NOVA ABA: Assinatura de Ciclo
                                if "Assinatura de Ciclo" in graph_tab_list:
                                    with graph_tabs[tab_idx]:
                                        tab_idx += 1
                                        st.subheader("Análise da Assinatura do Ciclo de Operação")

                                        cycle_analysis_data = st.session_state.cycle_signature_analysis.get(
                                            main_job_label, {}).get(current_device, {})

                                        if not cycle_analysis_data:
                                            st.warning(
                                                "Não há dados de assinatura de ciclo para este dispositivo. Verifique se uma medição de carga (ex: MA_01) foi selecionada.")
                                        else:
                                            motor_measurement = next(iter(cycle_analysis_data))
                                            analysis = cycle_analysis_data[motor_measurement]

                                            fig_sig = go.Figure()

                                            x_axis = list(analysis['mean'].keys())
                                            mean_curve = list(analysis['mean'].values())
                                            upper_bound = list(analysis['upper_bound'].values())
                                            lower_bound = list(analysis['lower_bound'].values())

                                            fig_sig.add_trace(go.Scatter(
                                                x=x_axis + x_axis[::-1],
                                                y=upper_bound + lower_bound[::-1],
                                                fill='toself',
                                                fillcolor='rgba(0,100,80,0.2)',
                                                line=dict(color='rgba(255,255,255,0)'),
                                                hoverinfo="skip",
                                                name='Faixa de Normalidade (±1σ)'
                                            ))

                                            fig_sig.add_trace(go.Scatter(
                                                x=x_axis, y=mean_curve,
                                                line=dict(color='rgb(0,100,80)'),
                                                mode='lines',
                                                name='Assinatura Média'
                                            ))

                                            all_cycle_ids = list(analysis['curves'].keys())
                                            selected_cycles = st.multiselect(
                                                "Selecione ciclos individuais para comparar:",
                                                options=all_cycle_ids,
                                                default=[],
                                                key=f"cycle_select_{current_device}"
                                            )

                                            for cycle_id in selected_cycles:
                                                curve_data = analysis['curves'][cycle_id]
                                                fig_sig.add_trace(go.Scatter(
                                                    x=list(curve_data.keys()),
                                                    y=list(curve_data.values()),
                                                    mode='lines',
                                                    line=dict(width=1, dash='dot'),
                                                    name=f'Ciclo {cycle_id.split("_")[-1]}'
                                                ))

                                            fig_sig.update_layout(
                                                title=f'Assinatura de Ciclo para {motor_measurement}',
                                                xaxis_title='Progresso do Ciclo (%)',
                                                yaxis_title='Valor da Medição',
                                                legend_title='Legenda'
                                            )
                                            # CORREÇÃO: Adicionada key única
                                            st.plotly_chart(fig_sig, use_container_width=True,
                                                            key=f"sig_chart_{current_device}")

    st.divider()

    # --- Formulário de Configuração ---
    with st.form("analysis_form"):
        st.header("1. Configurações de Conexão")
        tenant = st.text_input("Tenant (URL)",
                               value=st.session_state.get('tenant', "https://mayekawa.us.cumulocity.com"), key='tenant')
        username = st.text_input("Username", value=st.session_state.get('username', ""), key='username')
        password = st.text_input("Password", type="password")

        if st.form_submit_button("Conectar e Listar Dispositivos"):
            if username and password:
                with st.spinner("Buscando dispositivos..."):
                    st.session_state.structured_device_list = fetch_devices(tenant, username, password)
            else:
                st.warning("Por favor, preencha Username e Password.")

        if 'structured_device_list' in st.session_state and st.session_state.structured_device_list:
            st.header("2. Modo de Análise")
            analysis_mode = st.radio(
                "Escolha o que deseja fazer:",
                ["Análise Detalhada", "Comparar Dispositivos", "Comparar Períodos"],
                key='analysis_mode',
                horizontal=True
            )

            st.subheader("Filtro de Dispositivos")
            col_filter1, col_filter2 = st.columns(2)
            filter_name_serial = col_filter1.text_input("Filtrar por Nome ou S/N", key="filter_name")
            filter_id = col_filter2.text_input("Filtrar por System ID", key="filter_id")

            filtered_list = st.session_state.structured_device_list
            if filter_name_serial:
                filtered_list = [d for d in filtered_list if filter_name_serial.lower() in d['display'].lower()]
            if filter_id:
                filtered_list = [d for d in filtered_list if filter_id.lower() in d['id'].lower()]

            display_options = [d['display'] for d in filtered_list]

            if analysis_mode == "Comparar Períodos":
                st.subheader("Seleção de Dispositivos")
                selected_devices_display = st.multiselect("Selecione os Dispositivos para Comparar", display_options,
                                                          key='selected_devices_display_compare')

                if selected_devices_display:
                    st.subheader("Definição dos Períodos")
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        st.markdown("**Período A**")
                        date_from_a = st.date_input("Data de Início A", datetime.now() - timedelta(days=14),
                                                    key="date_from_a")
                        date_to_a = st.date_input("Data de Fim A", datetime.now() - timedelta(days=7), key="date_to_a")
                    with col_p2:
                        st.markdown("**Período B**")
                        date_from_b = st.date_input("Data de Início B", datetime.now() - timedelta(days=7),
                                                    key="date_from_b")
                        date_to_b = st.date_input("Data de Fim B", datetime.now(), key="date_to_b")

                    st.subheader("Parâmetros de Análise (por dispositivo)")
                    device_tabs = st.tabs(selected_devices_display)
                    all_device_configs = {}

                    for i, device_tab in enumerate(device_tabs):
                        with device_tab:
                            current_device_display = selected_devices_display[i]
                            current_device_obj = next(
                                (d for d in filtered_list if d['display'] == current_device_display), None)
                            if current_device_obj:
                                device_id = current_device_obj['id']
                                series_list = fetch_supported_series(tenant, username, password, device_id)
                                cleaned_series_names = sorted(list(set([s.split('.')[0] for s in series_list])))
                                is_mkpred = is_likely_mkpred(series_list)
                                default_targets = [n for n in
                                                   ['SP_01', 'DP_01', 'OT_01', 'DT_01', 'MA_01', 'v_rms', 'a_rms',
                                                    'a_peak'] if n in cleaned_series_names]
                                target_measurements = st.multiselect("Medições Alvo", options=cleaned_series_names,
                                                                     default=default_targets,
                                                                     key=f"targets_compare_{device_id}")

                                if is_mkpred:
                                    st.info(
                                        "Dispositivo de vibração (MKPRED) detectado. A análise será de tendência contínua.")
                                    load_measurements, op_current, stab_delay, shut_delay, startup_duration = [], 0, 0, 0, 0
                                else:
                                    load_measurements = st.multiselect("Medições de Carga (Gatilho)",
                                                                       cleaned_series_names, default=[
                                            "MA_01"] if "MA_01" in cleaned_series_names else [],
                                                                       key=f"loads_compare_{device_id}")
                                    op_current = st.number_input("Corrente Mín. de Operação (A)", value=1.0, step=0.1,
                                                                 key=f"op_current_compare_{device_id}")
                                    stab_delay = st.number_input("Atraso de Estabilização (s)", value=300,
                                                                 key=f"stab_compare_{device_id}")
                                    shut_delay = st.number_input("Atraso de Desligamento (s)", value=60,
                                                                 key=f"shut_compare_{device_id}")
                                    startup_duration = st.number_input("Duração da Análise de Partida (s)", value=60,
                                                                       key=f"startup_duration_compare_{device_id}")

                                all_device_configs[device_id] = {
                                    'device_id': device_id, 'device_display_name': current_device_display,
                                    'target_measurements_list': target_measurements,
                                    'load_measurement_names': load_measurements,
                                    'operating_current': op_current, 'stabilization_delay': stab_delay,
                                    'shutdown_delay': shut_delay, 'startup_duration': startup_duration,
                                    'is_mkpred': is_mkpred
                                }

            elif analysis_mode in ["Análise Detalhada", "Comparar Dispositivos"]:
                st.subheader("Seleção de Dispositivos")
                selected_devices_display = st.multiselect("Selecione os Dispositivos", display_options,
                                                          key='selected_devices_display')

                if selected_devices_display:
                    st.subheader("Parâmetros de Análise")
                    device_tabs = st.tabs(selected_devices_display)
                    all_device_configs = {}

                    for i, device_tab in enumerate(device_tabs):
                        with device_tab:
                            current_device_display = selected_devices_display[i]
                            current_device_obj = next(
                                (d for d in filtered_list if d['display'] == current_device_display), None)
                            if current_device_obj:
                                device_id = current_device_obj['id']
                                series_list = fetch_supported_series(tenant, username, password, device_id)
                                cleaned_series_names = sorted(list(set([s.split('.')[0] for s in series_list])))
                                is_mkpred = is_likely_mkpred(series_list)
                                default_targets = [n for n in
                                                   ['SP_01', 'DP_01', 'OT_01', 'DT_01', 'MA_01', 'v_rms', 'a_rms',
                                                    'a_peak'] if n in cleaned_series_names]
                                target_measurements = st.multiselect("Medições Alvo", options=cleaned_series_names,
                                                                     default=default_targets,
                                                                     key=f"targets_{device_id}")

                                if is_mkpred:
                                    st.info(
                                        "Dispositivo de vibração (MKPRED) detectado. A análise será de tendência contínua.")
                                    load_measurements, op_current, stab_delay, shut_delay, startup_duration = [], 0, 0, 0, 0
                                else:
                                    load_measurements = st.multiselect("Medições de Carga (Gatilho)",
                                                                       cleaned_series_names, default=[
                                            "MA_01"] if "MA_01" in cleaned_series_names else [],
                                                                       key=f"loads_{device_id}")
                                    op_current = st.number_input("Corrente Mín. de Operação (A)", value=1.0, step=0.1,
                                                                 key=f"op_current_{device_id}")
                                    stab_delay = st.number_input("Atraso de Estabilização (s)", value=300,
                                                                 key=f"stab_{device_id}")
                                    shut_delay = st.number_input("Atraso de Desligamento (s)", value=60,
                                                                 key=f"shut_{device_id}")
                                    startup_duration = st.number_input("Duração da Análise de Partida (s)", value=60,
                                                                       key=f"startup_duration_{device_id}")

                                with st.expander("Definir Limites de Operação (Opcional)"):
                                    threshold_df_data = {'Medição': target_measurements,
                                                         'Mínimo Aceitável': [None] * len(target_measurements),
                                                         'Máximo Aceitável': [None] * len(target_measurements)}
                                    df_thresholds = pd.DataFrame(threshold_df_data)
                                    edited_df = st.data_editor(df_thresholds, key=f"thresholds_editor_{device_id}",
                                                               hide_index=True,
                                                               column_config={
                                                                   "Mínimo Aceitável": st.column_config.NumberColumn(
                                                                       format="%.2f"),
                                                                   "Máximo Aceitável": st.column_config.NumberColumn(
                                                                       format="%.2f")})
                                    all_device_configs[f"editor_{device_id}"] = edited_df

                                all_device_configs[device_id] = {
                                    'device_id': device_id, 'device_display_name': current_device_display,
                                    'target_measurements_list': target_measurements,
                                    'load_measurement_names': load_measurements,
                                    'operating_current': op_current, 'stabilization_delay': stab_delay,
                                    'shutdown_delay': shut_delay, 'startup_duration': startup_duration,
                                    'is_mkpred': is_mkpred
                                }

                    st.subheader("Período Global")
                    col1, col2 = st.columns(2)
                    with col1:
                        date_from = st.date_input("Data de Início", datetime.now() - timedelta(days=7))
                    with col2:
                        date_to = st.date_input("Data de Fim", datetime.now())

            st.divider()
            fetch_alarms = st.checkbox("Buscar alarmes no período", value=True)
            fetch_events = st.checkbox("Buscar eventos no período", value=False)

            if st.form_submit_button("▶️ Iniciar Análise", type="primary"):
                jobs_to_run = []
                base_params = {"tenant_url": tenant, "username": username, "password": password,
                               "fetch_alarms": fetch_alarms, "fetch_events": fetch_events}
                st.session_state.params = {'analysis_mode': analysis_mode}

                if analysis_mode == "Comparar Períodos":
                    if 'all_device_configs' in locals() and all_device_configs:
                        for device_id, config in all_device_configs.items():
                            jobs_to_run.append({'job_label': 'Período A', **base_params, **config,
                                                'date_from': date_from_a.strftime('%Y-%m-%d'),
                                                'date_to': date_to_a.strftime('%Y-%m-%d')})
                            jobs_to_run.append({'job_label': 'Período B', **base_params, **config,
                                                'date_from': date_from_b.strftime('%Y-%m-%d'),
                                                'date_to': date_to_b.strftime('%Y-%m-%d')})
                            st.session_state.params[config['device_display_name']] = config
                else:
                    if 'all_device_configs' in locals() and all_device_configs:
                        job_label = "main"
                        for device_id, config in all_device_configs.items():
                            if not device_id.startswith("editor_"):
                                editor_df = all_device_configs.get(f"editor_{device_id}")
                                thresholds = {}
                                if editor_df is not None:
                                    for _, row in editor_df.iterrows():
                                        min_val = row['Mínimo Aceitável'] if pd.notna(row['Mínimo Aceitável']) else None
                                        max_val = row['Máximo Aceitável'] if pd.notna(row['Máximo Aceitável']) else None
                                        if min_val is not None or max_val is not None:
                                            thresholds[row['Medição']] = {'min': min_val, 'max': max_val}
                                config['thresholds'] = thresholds

                                jobs_to_run.append({'job_label': job_label, **base_params, **config,
                                                    'date_from': date_from.strftime('%Y-%m-%d'),
                                                    'date_to': date_to.strftime('%Y-%m-%d')})
                                st.session_state.params[config['device_display_name']] = config

                if jobs_to_run:
                    st.session_state.jobs = jobs_to_run
                    st.session_state.is_running = True
                    st.session_state.log_messages = []
                    st.session_state.results_df = None
                    st.session_state.raw_data = None
                    st.rerun()
                else:
                    st.warning("Nenhuma análise para iniciar. Verifique as configurações de dispositivo.")
