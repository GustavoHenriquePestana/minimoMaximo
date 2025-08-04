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
# tkcalendar

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
</style>
""", unsafe_allow_html=True)


# --- Funções Auxiliares ---
def format_timestamp_to_brasilia(dt_obj):
    if not dt_obj: return ""
    try:
        if isinstance(dt_obj, str):
            dt_obj = datetime.fromisoformat(dt_obj.replace("Z", "+00:00"))
        brasilia_tz = pytz.timezone("America/Sao_Paulo")
        if dt_obj.tzinfo is None:
            dt_obj = pytz.utc.localize(dt_obj)
        return dt_obj.astimezone(brasilia_tz).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(dt_obj)


def extract_measurement_value(measurement, measurement_type):
    """
    Função restaurada para a lógica original e funcional da versão desktop.
    """
    if measurement_type not in measurement:
        # Fallback para casos onde o tipo é um prefixo (ex: MA_01 para MA_01.T)
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


# --- Funções de API (Thread-safe) ---
@st.cache_data(ttl=300)
def fetch_devices(tenant, user, password):
    """Busca dispositivos e retorna uma lista estruturada para filtragem."""
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
        if hasattr(e, 'response') and e.response is not None:
            st.error(
                f"Erro ao buscar dispositivos: Falha na requisição com status {e.response.status_code}. Resposta: {e.response.text}")
        else:
            st.error(f"Erro inesperado ao buscar dispositivos: {e}")
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
        if hasattr(e, 'response') and e.response is not None:
            st.error(
                f"Erro ao buscar medições: Falha na requisição com status {e.response.status_code}. Resposta: {e.response.text}")
        else:
            st.error(f"Erro inesperado ao buscar medições suportadas: {e}")
        return []


# --- Lógica Principal da Análise (Executa em Thread) ---
def analyze_single_device(device_params, log_queue):
    """Função que analisa um único dispositivo e retorna seus resultados."""
    device_id = device_params['device_id']
    device_display_name = device_params['device_display_name']
    api_call_counter = 0

    try:
        c8y = CumulocityApi(base_url=device_params["tenant_url"],
                            tenant_id=device_params["tenant_url"].split('.')[0].split('//')[1],
                            username=device_params["username"], password=device_params["password"])

        log_queue.put({'type': 'log', 'data': f"[{device_display_name}] Iniciando análise..."})

        results_data = {target: {"min": None, "max": None, "count_valid": 0, "min_time": None, "max_time": None} for
                        target in device_params["target_measurements_list"]}
        raw_data = {}

        # Lógica Padrão de Ciclos com múltiplos gatilhos
        log_queue.put({'type': 'log',
                       'data': f"[{device_display_name}] Buscando dados dos gatilhos: {device_params['load_measurement_names']}"})

        all_points = []
        for trigger_name in device_params['load_measurement_names']:
            measurements = list(
                c8y.measurements.select(source=device_id, type=trigger_name, date_from=device_params['date_from'],
                                        date_to=device_params['date_to']))
            api_call_counter += 1
            for m in measurements:
                ts = datetime.fromisoformat(m.time.replace("Z", "+00:00"))
                val = extract_measurement_value(m, trigger_name)
                if val is not None:
                    all_points.append({'time': ts, 'value': val, 'type': trigger_name})

        all_points.sort(key=lambda p: p['time'])

        summed_trigger_measurements = []
        last_known_values = {name: 0.0 for name in device_params['load_measurement_names']}
        for point in all_points:
            last_known_values[point['type']] = point['value']
            current_sum = sum(last_known_values.values())
            summed_trigger_measurements.append((point['time'], current_sum))

        operational_cycles = []
        if summed_trigger_measurements:
            cycle_start_time = None
            for ts, summed_value in summed_trigger_measurements:
                is_on = summed_value > device_params["operating_current"]
                if is_on and cycle_start_time is None:
                    cycle_start_time = ts
                elif not is_on and cycle_start_time is not None:
                    operational_cycles.append({"start": cycle_start_time, "end": ts})
                    cycle_start_time = None
            if cycle_start_time is not None:
                last_ts = summed_trigger_measurements[-1][0]
                operational_cycles.append({"start": cycle_start_time, "end": last_ts})

        log_queue.put({'type': 'log',
                       'data': f"[{device_display_name}] Mapeamento concluído. {len(operational_cycles)} ciclos de operação encontrados."})
        if not operational_cycles:
            return device_display_name, results_data, raw_data, api_call_counter

        log_queue.put({'type': 'log', 'data': f"[{device_display_name}] Buscando dados das medições alvo..."})
        for target_name in device_params['target_measurements_list']:
            target_measurements = list(
                c8y.measurements.select(source=device_id, type=target_name, date_from=device_params['date_from'],
                                        date_to=device_params['date_to']))
            api_call_counter += 1
            points = [
                (datetime.fromisoformat(m.time.replace("Z", "+00:00")), extract_measurement_value(m, target_name))
                for m in target_measurements if extract_measurement_value(m, target_name) is not None]
            raw_data[target_name] = sorted(points, key=lambda x: x[0])

        for i, cycle in enumerate(operational_cycles):
            analysis_start = cycle['start'] + timedelta(seconds=device_params['stabilization_delay'])
            analysis_end = cycle['end'] - timedelta(seconds=device_params['shutdown_delay'])
            if analysis_start >= analysis_end: continue

            for target_name in device_params['target_measurements_list']:
                for time_obj, value in raw_data.get(target_name, []):
                    if analysis_start <= time_obj <= analysis_end:
                        res = results_data[target_name]
                        if res["min"] is None or value < res["min"]: res["min"], res["min_time"] = value, time_obj
                        if res["max"] is None or value > res["max"]: res["max"], res["max_time"] = value, time_obj
                        res["count_valid"] += 1

        return device_display_name, results_data, raw_data, api_call_counter
    except Exception as e:
        log_queue.put({'type': 'log', 'data': f"ERRO ao analisar {device_display_name}: {e}", 'color': 'error'})
        return device_display_name, {}, {}, api_call_counter


def perform_analysis_master_thread(stop_event, log_queue, params):
    total_api_calls = 0
    final_results = {}
    final_raw_data = {}

    devices_to_analyze = params['selected_devices_configs']

    with ThreadPoolExecutor(max_workers=10) as executor:
        # CORREÇÃO: Monta o dicionário completo de parâmetros para cada tarefa
        tasks = []
        for device_config in devices_to_analyze:
            task_params = params.copy()
            task_params.update(device_config)
            tasks.append(task_params)

        future_to_device = {executor.submit(analyze_single_device, task, log_queue): task for task in tasks}

        for i, future in enumerate(as_completed(future_to_device)):
            if stop_event.is_set():
                log_queue.put({'type': 'log', 'data': "Cancelamento solicitado pelo usuário.", 'color': 'warning'})
                break

            device_display_name, results, raw, api_calls = future.result()
            final_results[device_display_name] = results
            final_raw_data[device_display_name] = raw
            total_api_calls += api_calls
            log_queue.put(
                {'type': 'status', 'data': f"Análise concluída para {i + 1}/{len(devices_to_analyze)} dispositivos.",
                 'progress': (i + 1) / len(devices_to_analyze)})

    log_queue.put({'type': 'log',
                   'data': f"Análise concluída para todos os dispositivos. Total de Chamadas à API: {total_api_calls}."})
    log_queue.put(
        {'type': 'finished', 'data': {'results': final_results, 'raw': final_raw_data, 'api_calls': total_api_calls}})


# --- Funções para Salvar/Carregar Configurações ---
def save_settings():
    settings = {
        'tenant': st.session_state.get('tenant'),
        'username': st.session_state.get('username'),
        'saved_devices_display': st.session_state.get('selected_devices_display')
    }
    with open("analyzer_settings.json", "w") as f:
        json.dump(settings, f, indent=4)
    st.toast("Configurações salvas!", icon="💾")


def load_settings():
    if os.path.exists("analyzer_settings.json"):
        with open("analyzer_settings.json", "r") as f:
            settings = json.load(f)
        for key, value in settings.items():
            st.session_state[key] = value
        st.toast("Configurações carregadas!", icon="✅")


# --- Inicialização do Estado da Sessão ---
if 'is_running' not in st.session_state: st.session_state.is_running = False
if 'log_messages' not in st.session_state: st.session_state.log_messages = []
if 'status_text' not in st.session_state: st.session_state.status_text = "Aguardando início..."
if 'progress_value' not in st.session_state: st.session_state.progress_value = 0.0
if 'results_df' not in st.session_state: st.session_state.results_df = None
if 'raw_data' not in st.session_state: st.session_state.raw_data = None
if 'log_queue' not in st.session_state: st.session_state.log_queue = Queue()
if 'api_call_count' not in st.session_state: st.session_state.api_call_count = 0

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
            stop_event, st.session_state.log_queue, st.session_state.params))
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
            results_by_device = msg['data']['results']
            st.session_state.api_call_count = msg['data']['api_calls']
            df_data = []
            for device_name, results in results_by_device.items():
                for name, data in results.items():
                    df_data.append({
                        "Dispositivo": device_name, "Medição": name, "Mínimo": data['min'],
                        "Timestamp Mínimo": format_timestamp_to_brasilia(data['min_time']),
                        "Máximo": data['max'], "Timestamp Máximo": format_timestamp_to_brasilia(data['max_time']),
                        "Ocorrências": data['count_valid']
                    })

            if not df_data:
                columns = ["Dispositivo", "Medição", "Mínimo", "Timestamp Mínimo", "Máximo", "Timestamp Máximo",
                           "Ocorrências"]
                st.session_state.results_df = pd.DataFrame(columns=columns)
            else:
                st.session_state.results_df = pd.DataFrame(df_data)

            st.session_state.raw_data = msg['data']['raw']
            st.rerun()
        elif msg['type'] == 'critical_error':
            st.session_state.is_running = False
            st.error(msg['data'])
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
        st.success("Análise Concluída!")
        st.metric("Chamadas à API", st.session_state.api_call_count)

        analyzed_devices = st.session_state.results_df['Dispositivo'].unique().tolist()
        if analyzed_devices:
            device_tabs = st.tabs(analyzed_devices)
            for i, tab in enumerate(device_tabs):
                with tab:
                    current_device = analyzed_devices[i]
                    device_df = st.session_state.results_df[
                        st.session_state.results_df['Dispositivo'] == current_device].drop(columns=['Dispositivo'])
                    st.dataframe(device_df)

                    if not device_df.empty:
                        csv = device_df.to_csv(index=False).encode('utf-8')
                        st.download_button("Exportar para CSV", data=csv, file_name=f"analise_{current_device}.csv",
                                           mime="text/csv", key=f"csv_{current_device}")

                    st.markdown("#### Visualizar Gráfico")
                    valid_measurements = device_df[device_df['Ocorrências'] > 0]['Medição'].tolist()

                    if valid_measurements:
                        selected_measurement = st.selectbox("Selecione uma medição", options=valid_measurements,
                                                            key=f"select_{current_device}")

                        plot_data = st.session_state.raw_data.get(current_device, {}).get(selected_measurement)
                        if plot_data:
                            timestamps, values = zip(*plot_data)

                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=list(timestamps), y=list(values), mode='lines+markers',
                                                     name=selected_measurement, marker=dict(size=5),
                                                     line=dict(width=2)))

                            result_row = device_df[device_df['Medição'] == selected_measurement].iloc[0]
                            min_val, max_val = result_row['Mínimo'], result_row['Máximo']
                            min_time, max_time = result_row['Timestamp Mínimo'], result_row['Timestamp Máximo']

                            if pd.notna(min_val):
                                fig.add_trace(go.Scatter(x=[pd.to_datetime(min_time)], y=[min_val], mode='markers',
                                                         name=f'Mínimo: {min_val:.2f}',
                                                         marker=dict(color='#28a745', size=12, symbol='circle')))
                            if pd.notna(max_val):
                                fig.add_trace(go.Scatter(x=[pd.to_datetime(max_time)], y=[max_val], mode='markers',
                                                         name=f'Máximo: {max_val:.2f}',
                                                         marker=dict(color='#ff4b4b', size=12, symbol='x')))

                            fig.update_layout(title=f'Análise da Medição: {selected_measurement}',
                                              xaxis_title='Timestamp', yaxis_title='Valor', template='plotly_dark',
                                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right",
                                                          x=1), margin=dict(l=20, r=20, t=50, b=20))
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(st.session_state.results_df)  # Mostra a tabela vazia se não houver resultados

        st.divider()

    # --- Formulário de Configuração ---
    with st.form("analysis_form"):
        st.header("1. Configurações de Conexão")
        tenant = st.text_input("Tenant (URL)",
                               value=st.session_state.get('tenant', "https://mayekawa.us.cumulocity.com"), key='tenant')
        username = st.text_input("Username", value=st.session_state.get('username', ""), key='username')
        password = st.text_input("Password", type="password")

        connect_button = st.form_submit_button("Conectar e Listar Dispositivos")
        if connect_button and username and password:
            with st.spinner("Buscando dispositivos..."):
                st.session_state.structured_device_list = fetch_devices(tenant, username, password)
        elif connect_button:
            st.warning("Por favor, preencha Username e Password.")

        if 'structured_device_list' in st.session_state and st.session_state.structured_device_list:
            st.subheader("Filtro de Dispositivos")
            col_filter1, col_filter2 = st.columns(2)
            with col_filter1:
                filter_name_serial = st.text_input("Filtrar por Nome ou S/N", key="filter_name")
            with col_filter2:
                filter_id = st.text_input("Filtrar por System ID", key="filter_id")

            filtered_list = st.session_state.structured_device_list
            if filter_name_serial:
                filtered_list = [d for d in filtered_list if filter_name_serial.lower() in d['display'].lower()]
            if filter_id:
                filtered_list = [d for d in filtered_list if filter_id.lower() in d['id'].lower()]

            display_options = [d['display'] for d in filtered_list]

            selected_devices_display = st.multiselect("Selecione os Dispositivos", display_options,
                                                      key='selected_devices_display')

            st.header("2. Parâmetros da Análise")

            if selected_devices_display:
                device_tabs = st.tabs(selected_devices_display)
                all_device_configs = []

                for i, device_tab in enumerate(device_tabs):
                    with device_tab:
                        current_device_display = selected_devices_display[i]
                        current_device_obj = next((d for d in st.session_state.structured_device_list if
                                                   d['display'] == current_device_display), None)

                        if current_device_obj:
                            device_id = current_device_obj['id']
                            series_list = fetch_supported_series(tenant, username, password, device_id)
                            is_mkpred = any('S01_AC_' in s.upper() for s in series_list)

                            if is_mkpred:
                                st.info(
                                    "Dispositivo de vibração (MKPRED) detectado. A análise será feita em todo o período.")
                                cleaned_series_names = sorted(list(set(series_list)))
                            else:
                                cleaned_series_names = sorted(list(set([s.split('.')[0] for s in series_list])))

                            target_measurements = st.multiselect("Medições Alvo", options=cleaned_series_names,
                                                                 default=[n for n in
                                                                          ['SP_01', 'DP_01', 'OT_01', 'DT_01', 'MA_01']
                                                                          if n in cleaned_series_names],
                                                                 key=f"targets_{device_id}")

                            if not is_mkpred:
                                st.subheader("Parâmetros de Ciclo")
                                load_measurements = st.multiselect("Medições de Carga (Gatilho)", cleaned_series_names,
                                                                   default=[
                                                                       "MA_01"] if "MA_01" in cleaned_series_names else [],
                                                                   help="Selecione uma ou mais medições. Os valores serão somados.",
                                                                   key=f"loads_{device_id}")
                                op_current = st.number_input("Corrente Mín. de Operação (A)", value=1.0, step=0.1,
                                                             key=f"op_current_{device_id}")
                                stab_delay = st.number_input("Atraso de Estabilização (s)", value=300,
                                                             key=f"stab_{device_id}")
                                shut_delay = st.number_input("Atraso de Desligamento (s)", value=60,
                                                             key=f"shut_{device_id}")

                            device_config = {
                                'device_id': device_id,
                                'device_display_name': current_device_display,
                                'is_mkpred': is_mkpred,
                                'target_measurements_list': target_measurements  # Passa os nomes limpos
                            }

                            if not is_mkpred:
                                device_config.update({
                                    "load_measurement_names": load_measurements,  # Passa os nomes limpos
                                    "operating_current": op_current,
                                    "stabilization_delay": stab_delay,
                                    "shutdown_delay": shut_delay
                                })
                            all_device_configs.append(device_config)

                st.divider()
                st.subheader("Parâmetros Globais")
                col1_form, col2_form = st.columns(2)
                with col1_form:
                    date_from = st.date_input("Data de Início para TODOS os dispositivos",
                                              datetime.now() - timedelta(days=7))
                with col2_form:
                    date_to = st.date_input("Data de Fim para TODOS os dispositivos", datetime.now())

                debug_mode = st.checkbox("Ativar Log de Diagnóstico (Debug)")

                submitted = st.form_submit_button("▶️ Iniciar Análise", type="primary")
                if submitted:
                    # Validar se todas as abas estão configuradas corretamente
                    valid_submission = True
                    for config in all_device_configs:
                        if not config['target_measurements_list']:
                            st.error(f"Nenhuma medição alvo foi selecionada para {config['device_display_name']}.")
                            valid_submission = False
                        if not config.get('is_mkpred', False) and not config.get('load_measurement_names'):
                            st.error(
                                f"Nenhuma Medição de Carga (Gatilho) foi selecionada para {config['device_display_name']}.")
                            valid_submission = False

                    if valid_submission:
                        params = {
                            "tenant_url": tenant, "username": username, "password": password,
                            "selected_devices_configs": all_device_configs,
                            "date_from": date_from.strftime('%Y-%m-%d'),
                            "date_to": date_to.strftime('%Y-%m-%d'),
                            "debug_mode": debug_mode
                        }
                        st.session_state.params = params
                        st.session_state.is_running = True
                        st.session_state.log_messages = []
                        st.session_state.results_df = None
                        st.session_state.raw_data = None
                        st.rerun()
            elif selected_devices_display:
                st.warning("Pelo menos um dispositivo deve ser selecionado para configurar a análise.")

