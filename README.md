# kayenafive
Adapting Farms
# Jackson Five · Rotación de cultivos con datos NASA POWER

Jackson Five construye un prototipo que, a partir de una ubicación, usa datos climáticos de NASA POWER y fuentes agronómicas (FAO e INIFAP) para comparar tres alternativas de rotación de cultivos y explicar por qué conviene o no cada una. El caso de prueba está en el estado de Chihuahua (latitud 29.25, longitud -107.41), con datos diarios de 2016 a 2025. Es un proyecto para NASA Space Apps (14 y 15 de noviembre de 2026). El criterio de éxito es un prototipo honesto: compara con reglas que se pueden rastrear hasta su fuente y dice con claridad lo que no sabe.

## Problema y usuario

Un productor que decide qué sembrar después de su cultivo actual casi nunca tiene a la mano el clima histórico de su zona ni una forma clara de comparar opciones. Repetir el mismo cultivo cansa el suelo, y elegir uno que no aguanta la temperatura o el agua de la región sale caro.

- **Usuario:** productor o técnico agrícola de una región semiárida del norte de México.
- **Lo que da al sistema:** su ubicación, el cultivo anterior y su prioridad (por ejemplo, ahorrar agua).
- **Lo que recibe:** tres rotaciones posibles, cada una con su razón, la condición que la favorece y su limitación.

## Objetivos

**Objetivo general:** comparar tres alternativas de rotación de cultivos para una ubicación, usando datos climáticos de NASA POWER y criterios agronómicos con fuente, y explicar cada resultado con sus limitaciones.

**Objetivos específicos:**

1. Obtener y limpiar un CSV diario de NASA POWER con temperatura, precipitación y humedad, documentando fuente, periodo y faltantes.
2. Calcular indicadores climáticos básicos, como temperatura promedio y lluvia acumulada.
3. Definir tres cultivos candidatos y tres criterios trazables a FAO o INIFAP: temperatura, agua y temporada.
4. Mostrar en una app el recorrido entrada → datos → tres rotaciones, con al menos una razón por opción.
5. Guardar los datos en un bucket privado de S3 y leerlos desde Jupyter sin exponer claves.

**Extensiones:**

1. Conectar una lectura local de un sensor con ESP32, etiquetada como práctica.
2. Comparar un modelo pequeño de TinyML (GMM) contra las reglas, sin presentarlo como predicción validada.
3. Publicar la app en línea y tener un video de respaldo.

## Nuestros datos

La fuente principal es un CSV diario de [NASA POWER](https://power.larc.nasa.gov/) para la latitud 29.25 y la longitud -107.41 (Chihuahua), del 1 de enero de 2016 al 31 de diciembre de 2025. Son unos 3,650 días, uno por fila.

| Columna | Qué mide | Unidad | Para qué la usamos |
| --- | --- | --- | --- |
| `YEAR`, `DOY` | Año y día del año (1 a 366) | — | Armar la fecha y separar temporadas |
| `T2M` | Temperatura media a 2 m | °C | Si el cultivo tolera el clima de la zona |
| `T2M_MAX`, `T2M_MIN` | Temperatura máxima y mínima a 2 m | °C | Riesgo de calor extremo y de heladas |
| `T2MDEW` | Punto de rocío o escarcha a 2 m | °C | Apoyo para estimar humedad y heladas |
| `RH2M` | Humedad relativa a 2 m | % | Contexto de humedad y riesgo de plagas |
| `PRECTOTCORR` | Precipitación corregida | mm/día | Lluvia acumulada por temporada, el criterio de agua |

- **Formato:** el archivo trae un encabezado de NASA que termina en `-END HEADER-`. Al leerlo con pandas hay que saltar esas líneas.
- **Faltantes:** NASA marca con `-999` los días sin dato. Se reemplazan por vacío antes de calcular promedios.
- **Alcance espacial:** es un dato regional de malla, no la medición de una parcela. Describe la zona, no el suelo de un rancho en particular.
- **Fuentes agronómicas:** las necesidades de temperatura, agua y temporada de cada cultivo salen de [FAO](https://www.fao.org/) e [INIFAP](https://www.gob.mx/inifap). Lo que no tenga fuente se marca como ilustrativo.
- **Sensor local (extensión):** lecturas de un ESP32 con nombre de variable y unidad. Se presentan como demostración, no como humedad de suelo calibrada.
- **Dónde viven:** el CSV original y el limpio se guardan en un bucket privado de S3 y en las carpetas `datos/crudos` y `datos/limpios` del repositorio.

## Cómo funciona

El usuario da tres datos y recibe tres rotaciones explicadas. Todo lo que muestra la app se puede rastrear hasta el CSV de NASA o hasta una fuente agronómica.

1. **Entrada:** el usuario elige la ubicación, el cultivo anterior y su prioridad.
2. **Datos:** la app carga el CSV limpio de NASA POWER de esa zona y calcula los indicadores, como temperatura promedio, riesgo de helada y lluvia acumulada por temporada.
3. **Criterios:** compara cada cultivo candidato contra tres reglas con fuente (temperatura, agua y temporada).
4. **Salida:** muestra tres rotaciones, cada una con su razón, la condición que la favorece y su limitación.
5. **Extensiones:** si hay tiempo, se suma una lectura local del ESP32 y una comparación experimental con TinyML.

**Por dónde viajan los datos:** Sebaspan descarga el CSV de NASA POWER y lo sube a un bucket privado de S3. Sebastin lo lee en Jupyter para revisarlo y limpiarlo, sin exponer claves de AWS. Jesús lo carga en la app (Streamlit o plantilla web), y si AWS o internet fallan, la app corre en local con el CSV del repositorio.

## Equipo y roles

| Integrante | Rol | Aporta al proyecto |
| --- | --- | --- |
| Kevin Alejandro Chávez Juárez | Coordinación e integración técnica | Alcance, decisiones de plan B y simulacro final |
| Daniel Sebastian Torres Ontiveros | Análisis de datos NASA | CSV de NASA POWER limpio, indicadores y diccionario de variables |
| César Yair Chavez Frias | Investigación agronómica y validación | Cultivos candidatos, tres rotaciones y criterios con fuente FAO o INIFAP |
| Sebastian Alonso Lopez Melendez | AWS/Jupyter, datos compartidos y comunicación | Bucket S3, lectura en Jupyter, orden de evidencias, guía de uso y pitch |
| Oscar Ramses Carrasco Cordova | App y prueba de TinyML/GMM | Aplicación, comparación por reglas y prueba experimental del GMM |
| Misael Arturo Zepeda Juarez | ESP32, sensores y pruebas físicas | Lectura local del sensor y demostración física |
