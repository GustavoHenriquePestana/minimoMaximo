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
    try:
        base_type = measurement_type.split('.')[0]
        if base_type in measurement:
            fragment = measurement[base_type]
            first_series = next(iter(fragment.values()))
            return float(first_series['value'])
        if measurement_type in measurement:
            fragment = measurement[measurement_type]
            first_series = next(iter(fragment.values()))
            return float(first_series['value'])
    except (StopIteration, KeyError, ValueError, TypeError):
        return None
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


# --- Lógica Principal da Análise (Executa em Thread) ---
def perform_analysis_thread(stop_event, log_queue, params):
    try:
        api_call_counter = 0
        log_queue.put({'type': 'log', 'data': "Iniciando análise..."})
        c8y = CumulocityApi(base_url=params["tenant_url"], tenant_id=params["tenant_url"].split('.')[0].split('//')[1],
                            username=params["username"], password=params["password"])

        log_queue.put({'type': 'log',
                       'data': f"Analisando dispositivo: {params['device_display_name']} (ID: {params['device_id']})"})

        results_data = {target: {"min": None, "max": None, "count_valid": 0, "min_time": None, "max_time": None} for
                        target in params["target_measurements_list"]}
        raw_data = {}

        if params.get('is_mkpred', False):
            # Lógica para MKPRED
            # ...
            pass
        else:
            # Lógica Padrão de Ciclos
            log_queue.put(
                {'type': 'status', 'data': f"Buscando dados do gatilho '{params['load_measurement_name']}'..."})
            trigger_measurements = list(
                c8y.measurements.select(source=params['device_id'], type=params['load_measurement_name'],
                                        date_from=params['date_from'], date_to=params['date_to']))
            api_call_counter += 1

            if params['debug_mode']:
                log_queue.put(
                    {'type': 'log', 'data': f"Encontrados {len(trigger_measurements)} pontos de dados para o gatilho.",
                     'color': 'debug'})
                if trigger_measurements:
                    log_queue.put(
                        {'type': 'log', 'data': "Amostra dos 5 primeiros valores do gatilho:", 'color': 'debug'})
                    for m in trigger_measurements[:5]:
                        value = extract_measurement_value(m, params["load_measurement_name"])
                        log_queue.put(
                            {'type': 'log', 'data': f"  - {format_timestamp_to_brasilia(m.time)}, Valor: {value}",
                             'color': 'debug'})

            operational_cycles = []
            if trigger_measurements:
                trigger_measurements.sort(key=lambda m: m.time)
                cycle_start_time = None
                for m in trigger_measurements:
                    current_time_obj = datetime.fromisoformat(m.time.replace("Z", "+00:00"))
                    motor_current_value = extract_measurement_value(m, params["load_measurement_name"])
                    if motor_current_value is not None:
                        is_on = motor_current_value > params["operating_current"]
                        if is_on and cycle_start_time is None:
                            cycle_start_time = current_time_obj
                        elif not is_on and cycle_start_time is not None:
                            operational_cycles.append({"start": cycle_start_time, "end": current_time_obj})
                            cycle_start_time = None
                if cycle_start_time is not None:
                    last_time_obj = datetime.fromisoformat(trigger_measurements[-1].time.replace("Z", "+00:00"))
                    operational_cycles.append({"start": cycle_start_time, "end": last_time_obj})

            log_queue.put({'type': 'log',
                           'data': f"Mapeamento concluído. {len(operational_cycles)} ciclos de operação encontrados."})
            if not operational_cycles:
                log_queue.put({'type': 'finished',
                               'data': {'results': results_data, 'raw': raw_data, 'api_calls': api_call_counter}})
                return

            log_queue.put({'type': 'status', 'data': "Buscando dados das medições alvo..."})
            for target_name in params['target_measurements_list']:
                if stop_event.is_set(): break
                target_measurements = list(
                    c8y.measurements.select(source=params['device_id'], type=target_name, date_from=params['date_from'],
                                            date_to=params['date_to']))
                api_call_counter += 1
                points = [
                    (datetime.fromisoformat(m.time.replace("Z", "+00:00")), extract_measurement_value(m, target_name))
                    for m in target_measurements if extract_measurement_value(m, target_name) is not None]
                raw_data[target_name] = sorted(points, key=lambda x: x[0])

            for i, cycle in enumerate(operational_cycles):
                if stop_event.is_set(): break
                log_queue.put({'type': 'status', 'data': f"Analisando ciclo {i + 1}/{len(operational_cycles)}...",
                               'progress': (i + 1) / len(operational_cycles)})

                analysis_start = cycle['start'] + timedelta(seconds=params['stabilization_delay'])
                analysis_end = cycle['end'] - timedelta(seconds=params['shutdown_delay'])

                if analysis_start >= analysis_end: continue

                for target_name in params['target_measurements_list']:
                    for time_obj, value in raw_data.get(target_name, []):
                        if analysis_start <= time_obj <= analysis_end:
                            res = results_data[target_name]
                            if res["min"] is None or value < res["min"]: res["min"], res["min_time"] = value, time_obj
                            if res["max"] is None or value > res["max"]: res["max"], res["max_time"] = value, time_obj
                            res["count_valid"] += 1

        log_queue.put({'type': 'log', 'data': f"Análise concluída. Total de Chamadas à API: {api_call_counter}."})
        log_queue.put(
            {'type': 'finished', 'data': {'results': results_data, 'raw': raw_data, 'api_calls': api_call_counter}})

    except Exception as e:
        import traceback
        log_queue.put({'type': 'critical_error', 'data': f"Erro crítico na análise: {e}\n{traceback.format_exc()}"})


# --- Funções para Salvar/Carregar Configurações ---
def save_settings():
    settings = {
        'tenant': st.session_state.get('tenant'),
        'username': st.session_state.get('username'),
        'saved_device_display': st.session_state.get('selected_device_display')
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
        st.session_state.analysis_thread = Thread(target=perform_analysis_thread, args=(
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
            results = msg['data']['results']
            st.session_state.api_call_count = msg['data']['api_calls']
            df_data = []
            for name, data in results.items():
                df_data.append({
                    "Medição": name, "Mínimo": data['min'],
                    "Timestamp Mínimo": format_timestamp_to_brasilia(data['min_time']),
                    "Máximo": data['max'], "Timestamp Máximo": format_timestamp_to_brasilia(data['max_time']),
                    "Ocorrências": data['count_valid']
                })

            if not df_data:
                columns = ["Medição", "Mínimo", "Timestamp Mínimo", "Máximo", "Timestamp Máximo", "Ocorrências"]
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
        st.dataframe(st.session_state.results_df)

        if not st.session_state.results_df.empty and 'Ocorrências' in st.session_state.results_df.columns:
            csv = st.session_state.results_df.to_csv(index=False).encode('utf-8')
            st.download_button("Exportar para CSV", data=csv, file_name="analise_min_max.csv", mime="text/csv")

            st.markdown("### Visualizar Gráfico")
            valid_measurements = st.session_state.results_df[st.session_state.results_df['Ocorrências'] > 0][
                'Medição'].tolist()
            if valid_measurements:
                selected_measurement = st.selectbox("Selecione uma medição para visualizar", options=valid_measurements)

                plot_data = st.session_state.raw_data.get(selected_measurement)
                if plot_data:
                    timestamps, values = zip(*plot_data)

                    fig = go.Figure()

                    # Adiciona a série temporal principal
                    fig.add_trace(
                        go.Scatter(x=list(timestamps), y=list(values), mode='lines+markers', name=selected_measurement,
                                   marker=dict(size=5), line=dict(width=2)))

                    # Adiciona marcadores de Mínimo e Máximo
                    result_row = \
                    st.session_state.results_df[st.session_state.results_df['Medição'] == selected_measurement].iloc[0]
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

                    # Configura o layout do gráfico
                    fig.update_layout(
                        title=f'Análise da Medição: {selected_measurement}',
                        xaxis_title='Timestamp',
                        yaxis_title='Valor',
                        template='plotly_dark',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=20, r=20, t=50, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
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

            saved_device = st.session_state.get('saved_device_display')
            index = display_options.index(saved_device) if saved_device and saved_device in display_options else 0

            selected_device_display = st.selectbox("Selecione o Dispositivo", display_options, index=index,
                                                   key='selected_device_display')

            st.header("2. Parâmetros da Análise")

            selected_device_obj = next(
                (d for d in st.session_state.structured_device_list if d['display'] == selected_device_display), None)
            device_id = selected_device_obj['id'] if selected_device_obj else None

            if device_id:
                series_list = fetch_supported_series(tenant, username, password, device_id)
                is_mkpred = any('S01_AC_' in s.upper() for s in series_list)

                if is_mkpred:
                    st.info("Dispositivo de vibração (MKPRED) detectado. A análise será feita em todo o período.")
                    cleaned_series_names = sorted(list(set(series_list)))
                else:
                    cleaned_series_names = sorted(list(set([s.split('.')[0] for s in series_list])))

                target_measurements = st.multiselect("Medições Alvo", options=cleaned_series_names,
                                                     default=[n for n in ['SP_01', 'DP_01', 'OT_01', 'DT_01', 'MA_01']
                                                              if n in cleaned_series_names])

                col1, col2 = st.columns(2)
                with col1:
                    date_from = st.date_input("Data de Início", datetime.now() - timedelta(days=7))
                with col2:
                    date_to = st.date_input("Data de Fim", datetime.now())

                if not is_mkpred:
                    st.subheader("Parâmetros de Ciclo")
                    load_measurement = st.selectbox("Medição de Carga (Gatilho)", cleaned_series_names,
                                                    index=cleaned_series_names.index(
                                                        "MA_01") if "MA_01" in cleaned_series_names else 0)
                    op_current = st.number_input("Corrente Mín. de Operação (A)", value=1.0, step=0.1)
                    stab_delay = st.number_input("Atraso de Estabilização (s)", value=300)
                    shut_delay = st.number_input("Atraso de Desligamento (s)", value=60)

                st.header("3. Opções Adicionais")
                debug_mode = st.checkbox("Ativar Log de Diagnóstico (Debug)")

                submitted = st.form_submit_button("▶️ Iniciar Análise", type="primary")
                if submitted:
                    if not target_measurements:
                        st.error("Nenhuma medição alvo foi selecionada. Por favor, marque ao menos uma.")
                    else:
                        params = {
                            "tenant_url": tenant, "username": username, "password": password, "device_id": device_id,
                            "device_display_name": selected_device_display, "date_from": date_from.strftime('%Y-%m-%d'),
                            "date_to": date_to.strftime('%Y-%m-%d'), "target_measurements_list": target_measurements,
                            "debug_mode": debug_mode, "is_mkpred": is_mkpred
                        }
                        if not is_mkpred:
                            params.update({
                                "load_measurement_name": load_measurement, "operating_current": op_current,
                                "stabilization_delay": stab_delay, "shutdown_delay": shut_delay
                            })

                        st.session_state.params = params
                        st.session_state.is_running = True
                        st.session_state.log_messages = []
                        st.session_state.results_df = None
                        st.session_state.raw_data = None
                        st.rerun()
            elif selected_device_display:
                st.warning("Dispositivo selecionado não encontrado. Por favor, reconecte ou limpe o filtro.")
