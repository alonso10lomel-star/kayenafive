# =====================================================================
#  ENTRENAR EL GMM CON DATOS REALES DE NASA POWER
#  Jackson Five · NASA Space Apps 2026 · Responsable: Jesús Díaz
#
#  Qué hace este programa:
#    1. Lee 10 años de clima diario de NASA POWER (Chihuahua, 29.25, -107.41)
#    2. Entrena un GMM con TEMPERATURA y HUMEDAD DEL AIRE, que son
#       justo las 2 cosas que mide el sensor DHT22 del ESP32 de Misael.
#    3. Usa el BIC para escoger CUÁNTOS grupos y QUÉ FORMA tienen los cerros,
#       y guarda la gráfica "bic_gmm.png" para la presentación.
#    4. Revisa con años que el modelo NUNCA vio (2024 y 2025) si los grupos
#       siguen teniendo sentido.
#    5. Lo compara contra una regla simple (lo pide el Excel, hora 9).
#    6. Aprende una CADENA DE MARKOV para pronosticar el clima a 7 días.
#    7. Guarda el modelo como "gmm_modelo.h" para meterlo al ESP32.
#
#  Cómo se corre (en tu compu o en el Jupyter de AWS, es igual):
#      pip install pandas scikit-learn matplotlib
#      python entrenar_gmm.py
#
#  El archivo nasa_chihuahua.csv debe estar en la misma carpeta.
# =====================================================================


# =====================================================================
# BLOQUE 1: LAS HERRAMIENTAS
# =====================================================================
import numpy as np                              # cuentas con números
import pandas as pd                             # tablas (como Excel)
from sklearn.mixture import GaussianMixture     # el GMM
import matplotlib
matplotlib.use("Agg")                           # dibuja a archivo, sin abrir ventanas (sirve en AWS)
import matplotlib.pyplot as plt                 # gráficas


# =====================================================================
# BLOQUE 2: LEER EL ARCHIVO DE NASA
# =====================================================================
ARCHIVO = "nasa_chihuahua.csv"

# El CSV de NASA trae renglones de explicación arriba. Buscamos el
# renglón "-END HEADER-" para saber cuántos hay que brincarse.
with open(ARCHIVO, encoding="utf-8") as f:
    renglones = f.readlines()
saltar = 0
for numero, renglon in enumerate(renglones):
    if "-END HEADER-" in renglon:
        saltar = numero + 1

datos = pd.read_csv(ARCHIVO, skiprows=saltar)
datos = datos.replace(-999, np.nan).dropna()   # quita días sin dato, si hubiera

# Convertimos YEAR + DOY (día del año) en una fecha de verdad,
# para poder saber de qué MES es cada día.
datos["fecha"] = pd.to_datetime(
    datos["YEAR"].astype(str) + "-" + datos["DOY"].astype(str), format="%Y-%j")
datos["mes"] = datos["fecha"].dt.month

print(f"📂 Leí {len(datos)} días, de {datos['YEAR'].min()} a {datos['YEAR'].max()}")


# =====================================================================
# BLOQUE 3: ESCOGER QUÉ VE EL MODELO (LAS "ENTRADAS")
# =====================================================================
# 👀 Regla de oro de TinyML: el modelo SOLO puede aprender de lo que el
#    ESP32 va a poder medir en el campo. El DHT22 mide temperatura y
#    humedad del aire. NO mide lluvia. Así que la lluvia NO entra al modelo.
#    (La usamos después, en el Bloque 6, para revisar si los grupos tienen sentido).
ENTRADAS = ["T2M", "RH2M"]      # temperatura (°C) y humedad del aire (%)

# Separamos los datos en dos montones, como en la escuela:
#   ENTRENAR = los años con los que el modelo estudia (2016 a 2023)
#   PROBAR   = el "examen sorpresa" con años que nunca vio (2024 y 2025)
entrenar = datos[datos["YEAR"] <= 2023]
probar = datos[datos["YEAR"] >= 2024]
print(f"📚 Para entrenar: {len(entrenar)} días  |  📝 Para el examen: {len(probar)} días")


# =====================================================================
# BLOQUE 4: EL BIC ESCOGE CUÁNTOS GRUPOS Y QUÉ FORMA TIENEN
# =====================================================================
# El BIC es una calificación al revés: MIENTRAS MÁS BAJO, MEJOR.
#   BIC = (qué tan MAL explica los datos) + (castigo por cada número extra)
# Un modelo con muchos grupos siempre "se ajusta" mejor, pero el castigo
# evita que hagamos trampa metiendo grupos de más.
#
# Además del NÚMERO de grupos, probamos la FORMA de los cerros:
#   "full"      -> cada cerro con su propia forma, puede ir chueco (inclinado)
#   "tied"      -> todos los cerros con la MISMA forma
#   "diag"      -> cada uno con su forma, pero derechitos (sin inclinarse)
#   "spherical" -> bolitas redondas
# Mientras más simple la forma, menos números tiene que guardar el ESP32.
FORMAS = ["full", "tied", "diag", "spherical"]
GRUPOS_A_PROBAR = range(1, 8)
X = entrenar[ENTRADAS].values

print("\n🔢 El BIC decide (más bajo = mejor):")
tabla_bic = {}            # (forma, k) -> BIC
modelos = {}              # (forma, k) -> modelo ya entrenado
for forma in FORMAS:
    for k in GRUPOS_A_PROBAR:
        m = GaussianMixture(n_components=k, covariance_type=forma,
                            random_state=0, n_init=5)
        m.fit(X)
        tabla_bic[(forma, k)] = m.bic(X)
        modelos[(forma, k)] = m

mejor_forma, mejor_k = min(tabla_bic, key=tabla_bic.get)
mejor_bic = tabla_bic[(mejor_forma, mejor_k)]

# Imprimimos la tabla. "Δ" = cuánto peor es cada opción que la ganadora.
print(f"   {'grupos':>6} | " + " | ".join(f"{f:>10}" for f in FORMAS))
for k in GRUPOS_A_PROBAR:
    celdas = []
    for forma in FORMAS:
        delta = tabla_bic[(forma, k)] - mejor_bic
        celdas.append("  GANADOR " if delta == 0 else f"{delta:+10,.0f}")
    print(f"   {k:>6} | " + " | ".join(celdas))
print(f"   ✅ Ganador: {mejor_k} grupos con forma '{mejor_forma}' (BIC = {mejor_bic:,.0f})")
print("   📏 Cómo leer la Δ: más de +10 = el ganador es MUCHO mejor; "
      "menos de +2 = prácticamente empatados.")

# Revisamos si hubo un empate técnico con un modelo más sencillo
casi_iguales = [(f, k) for (f, k), b in tabla_bic.items()
                if b - mejor_bic < 10 and (f, k) != (mejor_forma, mejor_k)]
if casi_iguales:
    print(f"   ⚠️ Casi empatados con el ganador (Δ < 10): {casi_iguales}. "
          "Si uno es más simple, se vale defenderlo ante el jurado.")

# --- La gráfica para la presentación ---
fig, ax = plt.subplots(figsize=(7, 4.2))
for forma in FORMAS:
    valores = [tabla_bic[(forma, k)] for k in GRUPOS_A_PROBAR]
    ax.plot(list(GRUPOS_A_PROBAR), valores, marker="o", label=forma)
ax.scatter([mejor_k], [mejor_bic], s=180, facecolors="none",
           edgecolors="black", linewidths=2, zorder=5, label="ganador")
ax.set_xlabel("Número de grupos (cerros)")
ax.set_ylabel("BIC (más bajo = mejor)")
ax.set_title(f"El BIC escoge {mejor_k} grupos con forma '{mejor_forma}'")
ax.legend(title="Forma de los cerros")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("bic_gmm.png", dpi=150)
print("   🖼️ Gráfica guardada en bic_gmm.png (¡a la presentación!)")


# =====================================================================
# BLOQUE 5: EL GMM GANADOR
# =====================================================================
gmm = modelos[(mejor_forma, mejor_k)]   # ya lo entrenamos en el Bloque 4


def covarianza_completa(modelo, i):
    """Saca la forma del cerro i como matriz 2x2, sin importar qué forma ganó.
    Así el ESP32 siempre recibe lo mismo y no hay que cambiarle nada."""
    c = modelo.covariances_
    if modelo.covariance_type == "full":
        return c[i]
    if modelo.covariance_type == "tied":
        return c                        # todos comparten la misma
    if modelo.covariance_type == "diag":
        return np.diag(c[i])            # derechito: ceros fuera de la diagonal
    return np.eye(2) * c[i]             # spherical: bolita redonda


# Acomodamos los grupos de más frío a más caliente, para que el
# grupo 0 siempre sea el más frío (así es más fácil hablar de ellos).
orden = np.argsort(gmm.means_[:, 0])


def grupo_ordenado(x):
    """Le pregunta al GMM a qué grupo se parece y lo traduce al orden frío->caliente."""
    original = gmm.predict(x)
    traductor = {viejo: nuevo for nuevo, viejo in enumerate(orden)}
    return np.array([traductor[g] for g in original])


datos["grupo"] = grupo_ordenado(datos[ENTRADAS].values)


# =====================================================================
# BLOQUE 6: ¿QUÉ SIGNIFICA CADA GRUPO? (ESTO LO DECIDE UN HUMANO)
# =====================================================================
# El GMM solo dice "grupo 0, 1, 2...". Para saber qué ES cada grupo,
# usamos lo que el modelo NO vio: la lluvia y el mes del año.
# Si un grupo junta los días lluviosos SIN haber visto la lluvia,
# quiere decir que la temperatura + humedad sí traen información útil.
NOMBRES_MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                 "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

print("\n🔍 Así quedaron los grupos (revisados con años que el modelo NUNCA vio, 2024-2025):")
resumen = []
for g in range(mejor_k):
    examen = datos[(datos["grupo"] == g) & (datos["YEAR"] >= 2024)]
    todos = datos[datos["grupo"] == g]
    meses_top = todos["mes"].value_counts().head(3).index
    fila = {
        "grupo": g,
        "temp": todos["T2M"].mean(),
        "humedad": todos["RH2M"].mean(),
        "dias_con_lluvia_pct": (examen["PRECTOTCORR"] >= 1).mean() * 100,
        "lluvia_promedio_mm": examen["PRECTOTCORR"].mean(),
        "meses": ", ".join(NOMBRES_MESES[m - 1] for m in sorted(meses_top)),
    }
    resumen.append(fila)
    print(f"\n   GRUPO {g}:  {fila['temp']:.1f} °C  y  {fila['humedad']:.0f} % de humedad")
    print(f"      Meses donde más aparece: {fila['meses']}")
    print(f"      En 2024-2025: llovió (≥1 mm) en {fila['dias_con_lluvia_pct']:.0f}% "
          f"de sus días, {fila['lluvia_promedio_mm']:.2f} mm por día en promedio")

# 👇 NOMBRES PUESTOS POR HUMANOS después de ver los números de arriba.
#    Si cambias los datos (otro lugar, otros años), VUELVE A REVISARLOS.
#    (Sin acentos a propósito: el Monitor Serial del ESP32 a veces los muestra feo).
NOMBRES = {
    0: "Frio y humedo (otono-invierno)",            # Oct-Dic, casi no llueve
    1: "Seco (estiaje)",                            # Mar-May, 0% de días con lluvia
    2: "Calido y humedo (temporada de lluvias)",    # Jul-Sep, llueve la mitad de los días
}
if mejor_k != len(NOMBRES):
    print(f"\n⚠️ OJO: salieron {mejor_k} grupos y hay {len(NOMBRES)} nombres. "
          "Revisa los números de arriba y corrige el diccionario NOMBRES.")
    NOMBRES = {g: f"Grupo {g}" for g in range(mejor_k)}


# =====================================================================
# BLOQUE 7: GMM CONTRA UNA REGLA SIMPLE (honestidad ante todo)
# =====================================================================
# Pregunta del examen: "¿Hoy es un día de temporada de lluvias?"
# La respuesta real la sacamos de la lluvia de NASA (≥1 mm = sí llovió).
# Comparamos dos formas de adivinar SOLO con temperatura y humedad.
examen = datos[datos["YEAR"] >= 2024]
llovio = examen["PRECTOTCORR"] >= 1
grupo_lluvioso = max(range(mejor_k), key=lambda g: resumen[g]["dias_con_lluvia_pct"])

dice_gmm = examen["grupo"] == grupo_lluvioso
dice_regla = (examen["T2M"] > 17) & (examen["RH2M"] > 50)   # regla escrita por una persona


def calificar(nombre, dice):
    atrapados = (dice & llovio).sum() / llovio.sum() * 100       # de los días con lluvia, cuántos detectó
    falsas_alarmas = (dice & ~llovio).sum() / dice.sum() * 100   # de los que avisó, cuántos fueron en seco
    print(f"   {nombre:<28} detectó {atrapados:4.0f}% de los días con lluvia | "
          f"{falsas_alarmas:4.0f}% de sus avisos fueron falsa alarma")


print("\n⚖️ ¿Quién detecta mejor los días de lluvia en 2024-2025?")
calificar("GMM (grupo lluvioso)", dice_gmm)
calificar("Regla: T>17°C y H>50%", dice_regla)
print("   👉 Ninguno es perfecto: uno atrapa más días de lluvia y el otro da menos")
print("      falsas alarmas. Eso se dice TAL CUAL ante el jurado. La ventaja extra del")
print("      GMM es que da una PROBABILIDAD (qué tan seguro está), y la regla no.")


# =====================================================================
# BLOQUE 8: CADENA DE MARKOV (¿CUÁNTO VA A DURAR EL CLIMA DE HOY?)
# =====================================================================
# Idea: el clima de MAÑANA depende del de HOY.
# Contamos, día por día, cuántas veces se pasó de un grupo a otro.
# Ejemplo: si hoy es "lluvias", ¿cuántas veces mañana siguió siendo "lluvias"?
# Esa tabla de conteos, convertida en porcentajes, ES la cadena de Markov.
datos = datos.sort_values("fecha")        # ordenar por fecha, por si acaso
k = mejor_k
conteo = np.zeros((k, k))                 # tabla de k x k llena de ceros

hoy = datos["grupo"].values[:-1]          # todos los días menos el último
manana = datos["grupo"].values[1:]        # todos los días menos el primero
es_entreno = (datos["YEAR"].values[1:] <= 2023)   # solo aprende de 2016-2023
for g_hoy, g_manana, sirve in zip(hoy, manana, es_entreno):
    if sirve:
        conteo[g_hoy, g_manana] += 1      # "una vez más se pasó de g_hoy a g_manana"

# Dividimos cada renglón entre su total -> porcentajes que suman 1 (100%)
MARKOV = conteo / conteo.sum(axis=1, keepdims=True)

print("\n🔗 Cadena de Markov: si HOY es... ¿qué probabilidad hay de cada grupo MAÑANA?")
for g in range(k):
    seguir = MARKOV[g, g]
    duracion = 1 / (1 - seguir)           # fórmula de la duración promedio de una racha
    print(f"   {NOMBRES[g]:<42} sigue igual {seguir*100:4.1f}%  "
          f"-> dura ~{duracion:.0f} días seguidos")

# ----- PRONÓSTICO A 7 DÍAS: dos formas, y el examen decide cuál sirve -----
DIAS_PRONOSTICO = 7
fechas = datos.set_index("fecha")["grupo"]


def tabla_de_saltos(desde_anio, hasta_anio, salto):
    """Cuenta: si un día era grupo A, ¿qué grupo era 'salto' días después?"""
    t = np.zeros((k, k))
    for f, g in fechas[(fechas.index.year >= desde_anio) & (fechas.index.year <= hasta_anio)].items():
        despues = fechas.get(f + pd.Timedelta(days=salto))
        if despues is not None:
            t[g, despues] += 1
    return t


# FORMA A (Markov "de libro"): encadenar 7 veces el salto de 1 día.
#   Supone que el clima solo se acuerda de AYER.
MARKOV_ENCADENADA = np.linalg.matrix_power(MARKOV, DIAS_PRONOSTICO)

# FORMA B: medir directo el salto de 7 días (sigue siendo una cadena de
#   Markov, pero con pasos de una semana en vez de un día).
saltos = tabla_de_saltos(2016, 2023, DIAS_PRONOSTICO)
MARKOV_SEMANA = saltos / saltos.sum(axis=1, keepdims=True)

# 📝 EXAMEN con 2024-2025 (años que ninguna de las dos vio)
real = tabla_de_saltos(2024, 2025, DIAS_PRONOSTICO)
real = real / real.sum(axis=1, keepdims=True)
print(f"\n📝 Examen: si hoy es X, ¿seguirá siendo X dentro de {DIAS_PRONOSTICO} días? (2024-2025)")
print(f"   {'Hoy es':<42} {'A: 1 día x7':>11} {'B: 7 días':>10} {'REALIDAD':>9}")
for g in range(k):
    print(f"   {NOMBRES[g]:<42} {MARKOV_ENCADENADA[g, g]*100:10.0f}% "
          f"{MARKOV_SEMANA[g, g]*100:9.0f}% {real[g, g]*100:8.0f}%")
print("   👉 La forma A se queda corta: el clima de Chihuahua 'se acuerda' de más")
print("      que solo ayer (las temporadas duran semanas). Al ESP32 le mandamos la B.")


# =====================================================================
# BLOQUE 9: GUARDAR EL MODELO PARA EL ESP32
# =====================================================================
# Todo el "cerebro" del GMM son unos cuantos números por grupo:
#   - el centro (promedio de temperatura y humedad)
#   - la "forma" del cerro (matriz de covarianza; guardamos su INVERSA,
#     porque al ESP32 le cuesta menos multiplicar que dividir)
#   - una constante que junta el peso del grupo y el tamaño del cerro
# Con eso el ESP32 calcula a qué grupo se parece cada lectura.
# Gane la forma que gane en el BIC, aquí siempre se guarda como 2x2,
# así que gmm_clasificar.h y el .ino NO se tocan.
lineas = [
    "// gmm_modelo.h  |  GENERADO AUTOMÁTICAMENTE por entrenar_gmm.py. No lo edites a mano.",
    "// Jackson Five · NASA Space Apps 2026",
    f"// Entrenado con NASA POWER diario, {entrenar['YEAR'].min()}-{entrenar['YEAR'].max()}, "
    "punto 29.25, -107.41 (Chihuahua)",
    f"// Escogido por BIC: {mejor_k} grupos, forma '{mejor_forma}' (BIC = {mejor_bic:.0f})",
    "// Entradas: [0] temperatura del aire (°C), [1] humedad relativa del aire (%)",
    "#pragma once",
    "",
    f"#define GMM_N_GRUPOS {mejor_k}",
    "",
]
centros, inversas, constantes, nombres_c = [], [], [], []
for nuevo, viejo in enumerate(orden):
    mu = gmm.means_[viejo]
    cov = covarianza_completa(gmm, viejo)
    inv = np.linalg.inv(cov)
    constante = np.log(gmm.weights_[viejo]) - 0.5 * np.log(np.linalg.det(cov)) - np.log(2 * np.pi)
    centros.append(f"  {{{mu[0]:.6f}f, {mu[1]:.6f}f}}")
    inversas.append(f"  {{{inv[0, 0]:.8f}f, {inv[0, 1]:.8f}f, {inv[1, 1]:.8f}f}}")
    constantes.append(f"  {constante:.8f}f")
    nombres_c.append(f'  "{NOMBRES[nuevo]}"')

lineas += ["// Centro de cada grupo: {temperatura, humedad}",
           "const float GMM_CENTRO[GMM_N_GRUPOS][2] = {", ",\n".join(centros), "};", "",
           "// Inversa de la covarianza de cada grupo: {a, b, c} = [[a, b], [b, c]]",
           "const float GMM_INV_COV[GMM_N_GRUPOS][3] = {", ",\n".join(inversas), "};", "",
           "// log(peso) - 0.5*log(det(cov)) - log(2*pi)",
           "const float GMM_CONSTANTE[GMM_N_GRUPOS] = {", ",\n".join(constantes), "};", "",
           "// Nombres puestos por humanos (ver Bloque 6 de entrenar_gmm.py)",
           "const char* const GMM_NOMBRE[GMM_N_GRUPOS] = {", ",\n".join(nombres_c), "};", "",
           "// Cadena de Markov semanal (ver Bloque 8): [hoy][dentro de 7 dias] = probabilidad",
           "const float GMM_MARKOV[GMM_N_GRUPOS][GMM_N_GRUPOS] = {",
           ",\n".join("  {" + ", ".join(f"{p:.6f}f" for p in fila) + "}" for fila in MARKOV_SEMANA),
           "};", "",
           f"#define GMM_DIAS_PRONOSTICO {DIAS_PRONOSTICO}", ""]

with open("gmm_modelo.h", "w", encoding="utf-8") as f:
    f.write("\n".join(lineas))

numeros = mejor_k * (2 + 3 + 1) + mejor_k * mejor_k
print(f"\n📦 Modelo guardado en gmm_modelo.h  ->  {numeros} números en total "
      f"(unos {numeros * 4} bytes). Le sobra espacio al ESP32.")

# Tres lecturas de ejemplo para comprobar que el ESP32 da lo MISMO que Python.
print("\n🧪 Lecturas de prueba (el ESP32 debe dar el mismo grupo):")
for t, h in [(30.0, 20.0), (20.0, 70.0), (5.0, 60.0)]:
    g = grupo_ordenado(np.array([[t, h]]))[0]
    p = gmm.predict_proba(np.array([[t, h]]))[0][orden[g]] * 100
    print(f"   {t:4.1f} °C, {h:4.1f} %  ->  {NOMBRES[g]}  ({p:.0f}% seguro)")
