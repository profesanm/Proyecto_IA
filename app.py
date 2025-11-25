import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================== CONFIGURACIÓN DE PÁGINA  =====================
st.set_page_config(
    page_title="Evaluador de Retroalimentación Docente",
    page_icon="teacher",
    layout="centered"
)

# Cargar modelo
@st.cache_resource
def load_model():
    with open('modelo_retroalimentacion.pkl', 'rb') as f:
        data = pickle.load(f)
    return data


data = load_model()
model = data['model']
scaler = data['scaler']
le = data['le']
all_features = data['all_feature_names']

# ===================== ESTILO =====================
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stApp {max-width: 900px; margin: 0 auto;}
    .result-box {padding: 20px; border-radius: 12px; margin: 20px 0;}
    .high {background-color: #d4edda; border-left: 6px solid #28a745;}
    .medium {background-color: #fff3cd; border-left: 6px solid #ffc107;}
    .low {background-color: #f8d7da; border-left: 6px solid #dc3545;}
</style>
""", unsafe_allow_html=True)

# ===================== ENCABEZADO =====================
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://img.icons8.com/fluency/100/teacher.png", width=100)
with col2:
    st.title("LUMINA: Evaluador de Apropiación de Retroalimentación Efectiva")
    st.markdown("**Herramienta predictiva basada en Inteligencia Artificial**")

st.markdown("---")
st.markdown("### Por favor, responde las siguientes afirmaciones según tu percepción actual:")

# ===================== PREGUNTAS =====================
preguntas = [
    ("16_nivel_formacion_retroalimentacion_normalized",
     "1. Me siento bien preparado/a para dar retroalimentación constructiva y oportuna."),
    ("17_concepto_retroalimentacion_count_normalized",
     "2. Considero que dar información para mejorar resume de mejor manera el concepto de retroalimentación."),
    ("18_relevancia_retroalimentacion_normalized",
     "3. La retroalimentación que doy es fundamental para el aprendizaje de mis estudiantes."),
    ("19_momento_entrega_retroalimentacion_count_normalized",
     "4. Entrego retroalimentación clase a clase y después de cada evaluación."),
    ("20_tecnicas_retroalimentacion_count_normalized",
     "5. Utilizo diversas técnicas: escrita, oral, por pares, automatizada, etc."),
    ("21_tiempo_retroalimentacion_clase_normalized",
     "6. Dedico tiempo en clase para dar retroalimentación."),
    ("25_frecuencia_tecnologias_retroalimentacion_normalized",
     "7. Uso frecuentemente herramientas digitales para retroalimentar."),
    ("26_herramientas_tecnologicas_count_normalized",
     "8. Conozco y uso varias herramientas digitales (Google Classroom, Kahoot, etc.)."),
]

# ===================== CONTROL DE ESTADO Y LIMPIEZA =====================
if 'form_key' not in st.session_state:
    st.session_state.form_key = 0
    st.session_state.respuestas = {}
    st.session_state.evaluado = False

# Botón para nueva evaluación (si ya se evaluó)
if st.session_state.get('evaluado', False):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Evaluar a otro docente", type="primary", use_container_width=True):
            st.session_state.form_key += 1
            st.session_state.respuestas = {}
            st.session_state.evaluado = False
            st.rerun()

# ===================== FORMULARIO CON LIMPIEZA REAL =====================
form_key = st.session_state.form_key

for i, (codigo, pregunta) in enumerate(preguntas):
    if codigo in all_features:
        st.markdown(f"**{pregunta}**")

        radio_key = f"pregunta_{i}_{form_key}"

        respuesta = st.radio(
            "Selecciona tu nivel de acuerdo:",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: [
                "Totalmente en desacuerdo",
                "En desacuerdo",
                "Neutral",
                "De acuerdo",
                "Totalmente de acuerdo"
            ][x - 1],
            horizontal=True,
            key=radio_key,
            label_visibility="collapsed"
        )

        st.session_state.respuestas[codigo] = (respuesta - 1) / 4.0
        st.markdown("---")

# ===================== BOTÓN DE PREDICCIÓN =====================
if st.button("Evaluar mi nivel de retroalimentación", type="primary", use_container_width=True):
    with st.spinner("Analizando con Inteligencia Artificial..."):
        # Construir vector completo
        full_vector = np.zeros(len(all_features))
        for codigo, valor_normalizado in st.session_state.respuestas.items():
            if codigo in all_features:
                idx = all_features.index(codigo)
                full_vector[idx] = valor_normalizado

        # Escalar
        X_scaled = scaler.transform(full_vector.reshape(1, -1))
        X_final = X_scaled

        # Predecir
        pred = model.predict(X_final)[0]
        proba = model.predict_proba(X_final)[0]
        nivel = le.inverse_transform([pred])[0]

        idx_alto = np.where(le.classes_ == 'Alto')[0][0]
        prob_alto = proba[idx_alto] * 100

    st.session_state.evaluado = True
    st.session_state.resultado = {
        'nivel': nivel,
        'prob_alto': prob_alto,
        'proba': proba
    }
    st.rerun()

# ===================== MOSTRAR RESULTADO =====================
if st.session_state.get('evaluado', False):
    resultado = st.session_state.resultado
    nivel = resultado['nivel']
    prob_alto = resultado['prob_alto']
    proba = resultado['proba']

    st.markdown("## Resultado de la Evaluación")

    if nivel == "Alto":
        st.success(f"Nivel **{nivel.upper()}**")
        st.markdown(f"""
        <div class="result-box high">
            <h2>Excelente! Nivel ALTO de apropiación</h2>
            <p>Este docente tiene una <strong>probabilidad del {prob_alto:.1f}%</strong> de ser un <strong>gran retroalimentador efectivo</strong>.</p>
            <p>Recomendación: <strong>Líder natural</strong> → Invitarlo a formar a otros docentes.</p>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()

    elif nivel == "Medio":
        st.warning(f"Nivel **{nivel.upper()}**")
        st.markdown(f"""
        <div class="result-box medium">
            <h2>Nivel MEDIO – Buen potencial</h2>
            <p>Probabilidad de alcanzar nivel Alto: <strong>{prob_alto:.1f}%</strong></p>
            <p>Recomendación: Ofrecer <strong>capacitación focalizada</strong> en retroalimentación y uso de herramientas digitales.</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.error(f"Nivel **{nivel.upper()}**")
        st.markdown(f"""
        <div class="result-box low">
            <h2>Nivel BAJO – Prioridad de intervención</h2>
            <p>Probabilidad actual de alto desempeño: <strong>{prob_alto:.1f}%</strong></p>
            <p>Recomendación: Incluir urgentemente en <strong>programa intensivo de formación en retroalimentación efectiva</strong>.</p>
        </div>
        """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(5, 1.5))
    niveles = le.classes_
    barras = ax.bar(niveles, proba * 100, color=['#dc3545', '#ffc107', '#28a745'])

    idx_pred = list(niveles).index(nivel)
    barras[idx_pred].set_color('#198754' if nivel == 'Alto' else '#e63946' if nivel == 'Bajo' else '#f4a261')
    barras[idx_pred].set_edgecolor('black')
    barras[idx_pred].set_linewidth(3)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Probabilidad (%)")
    ax.set_title("Resultado de la predicción", fontsize=12)
    for i, v in enumerate(proba * 100):
        ax.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=10)

    st.pyplot(fig)

    st.success("Evaluación completada. ¡Gracias por participar!")

# ===================== PIE DE PÁGINA =====================
st.markdown("---")
st.caption("Modelo basado en Machine Learning • 90+ docentes evaluados • 2025")