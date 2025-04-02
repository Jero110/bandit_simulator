<head>
  <meta charset="utf-8">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.body);"></script>
</head>
  
  
clave unica __206002____________________________  

# Problema de Multi-Bandas (Multi-Armed Bandit): Teoría e Implementación

La tarea se entrega por discord antes del miercoles de la siguiente clase. Incluye llenar cuidadosamente en latex todos los snippets mencionados aqui, mas el codigo ya sea con link a colab o al repositorio. No olviden poner su clave unica. La idea es que investiguen, entiendan y proponga una solucion al problema. Utilicen chatgpt y los tutoriales de la tarea (cursor especialmente) para hacer codigo y entender el problema.  

**Nota**  
No pueden utilizar machine learning salvo regresion lineal si asi lo desean (no arboles, deep learning, etc..). 

La proxima clase vamos a continuar con un ejercicio parecido, pero usando cadenas de markov. Vamos a modificar el bandit para que sea mas interesante ante cadenas de markov.  

**Examen**  
El lunes hay examen sobre estos ejercicios a papel y lapiz, la calificacion sera el $min\{examen, ejercicios\}$, si $|examen - ejercicios|<1$ entonces sera el $maximo$. 


## 1. Introducción a los Problemas de Multi-Bandas

### 1.1 Definición y Enunciado del Problema

El problema de Multi-Bandas (MAB, por sus siglas en inglés) es un problema clásico en teoría de la decisión y aprendizaje por refuerzo. Su nombre surge del escenario de un jugador que enfrenta múltiples máquinas tragamonedas (a veces llamadas "bandidos de un solo brazo"), cada una con diferentes probabilidades de recompensa desconocidas. El jugador debe decidir qué máquinas jugar, en qué orden y cuántas veces, para maximizar su recompensa total.

En este modelo:
- Existen $K$ brazos (o acciones) diferentes.
- Cada brazo, cuando se jala, otorga una recompensa extraída de una distribución de probabilidad específica de ese brazo.
- Las distribuciones de recompensa son inicialmente desconocidas para el tomador de decisiones.
- El objetivo es maximizar la recompensa acumulada a lo largo de una serie de jugadas.

El problema captura la disyuntiva fundamental entre **exploración** (probar diferentes brazos para reunir información sobre sus distribuciones de recompensa) y **explotación** (elegir el brazo que actualmente parece ser el mejor).

### 1.2 Dilema de Exploración vs. Explotación

Este dilema está en el corazón del problema de multi-bandas:

- **Exploración**: Seleccionar brazos para aprender más sobre sus distribuciones de recompensa, potencialmente sacrificando recompensas inmediatas.
- **Explotación**: Seleccionar el brazo que actualmente parece ofrecer la mayor recompensa esperada en función de la información reunida hasta el momento.

Equilibrar estos dos aspectos es crucial. Demasiada exploración desperdicia recursos en brazos subóptimos. Demasiada explotación puede impedir descubrir un brazo mejor.

### 1.3 Formulación Matemática General

Formalicemos el problema estándar de bandas estocásticas:

- Sea $K$ el número de brazos.
- Para cada brazo $i \in \{1, 2, \ldots, K\}$, existe una distribución de probabilidad desconocida $\mathcal{D}_i$ con media $\mu_i$.
- En cada paso de tiempo $t \in \{1, 2, \ldots, T\}$:
  - El agente selecciona un brazo $a_t \in \{1, 2, \ldots, K\}$.
  - El agente recibe una recompensa $r_t \sim \mathcal{D}_{a_t}$.
- El objetivo es maximizar la recompensa acumulada $\sum_{t=1}^{T} r_t$.

Alternativamente, el problema puede enmarcarse en términos de minimizar **el arrepentimiento**. El arrepentimiento se define como la diferencia entre la recompensa obtenida al seleccionar siempre el brazo óptimo y la recompensa realmente obtenida por el agente:

$\text{Regret}(T) = T \cdot \max_{i} \mu_i - \mathbb{E}\left[\sum_{t=1}^{T} r_t\right]$

## 2. Escenarios de Información en Nuestro Entorno de Bandas

En nuestro entorno de multi-bandas, exploramos tres escenarios de información distintos, cada uno proporcionando al agente diferentes niveles de conocimiento:

### 2.1 Escenario de Información Completa

En este escenario, el agente observa:
- El número de turno actual.
- El número total de turnos T.
- La probabilidad de recompensa para el brazo 1 (p1).
- El historial completo de acciones y recompensas pasadas.

Este es el escenario más informativo, ya que el agente conoce la probabilidad de uno de los brazos directamente y puede inferir la del otro con base en las recompensas observadas.

### 2.2 Escenario de Información Parcial

En este escenario, el agente observa:
- El número de turno actual.
- El número total de turnos T.
- La probabilidad de recompensa para el brazo 1 (p1).
- El historial de acciones y recompensas pasadas.

El agente conoce la probabilidad de un brazo pero debe aprender la del otro a través de la experimentación.

### 2.3 Escenario de Solo Recompensa

En este escenario, el agente observa:
- El número de turno actual.
- El historial de acciones y recompensas pasadas.

Este es el escenario más desafiante porque:
1. El agente no conoce la probabilidad de ninguno de los dos brazos.
2. El agente no conoce el número total de turnos T.

El agente debe aprender las probabilidades de ambos brazos mediante la experimentación y no puede optimizar su estrategia en función de la duración conocida del juego.

## 3. Entornos de Bandas en Nuestro Playground

Nuestro entorno implementa cuatro tipos diferentes de entornos de multi-bandas, cada uno con características distintas que afectan cómo cambian las probabilidades de los brazos a lo largo del tiempo.

### 3.1 Entorno de Banda Fija

#### Descripción
En el entorno de Banda Fija, cada brazo tiene una probabilidad constante de recompensa durante todo el juego. Estas probabilidades se asignan aleatoriamente al inicio de cada juego (uniforme entre 0.01 y 0.99) y permanecen sin cambios.

#### Formulación Matemática
- Dos brazos: $a \in \{0, 1\}$
- Probabilidades fijas: $p_1, p_2 \in [0.01, 0.99]$
- En el turno $t$, al seleccionar el brazo $a$:
  - Se recibe recompensa $r_t = 1$ con probabilidad $p_{a+1}$
  - Se recibe recompensa $r_t = 0$ con probabilidad $1 - p_{a+1}$

#### Decisión (T Fijo)

### **EJERCICIO**  
Definir el problema de decisión para la Banda Fija con horizonte de tiempo conocido T = 100. ¿Cuál es la función objetivo? ¿Cuáles son las restricciones? ¿Cuál es la política óptima?
**RESPUESTA**  
---
# Análisis del Problema de Bandas Bernoulli: Del Caso General a Casos Particulares

## 1. Caso General: Bandas Bernoulli con Probabilidades Uniformes

Consideramos un problema de decisión con dos bandas (brazos) que proporcionan recompensas distribuidas según Bernoulli:

- **Banda 1**: Recompensa 1 con probabilidad $p_1 \sim \text{Uniforme}(0.01, 0.99)$, 0 con probabilidad $1 - p_1$
- **Banda 2**: Recompensa 1 con probabilidad $p_2 \sim \text{Uniforme}(0.01, 0.99)$, 0 con probabilidad $1 - p_2$

### 1.1 Función Objetivo
$$\max \mathbb{E}\left[ \sum_{t=1}^{T} X_t \right]$$

Donde:
- $X_t$ es la recompensa obtenida en el turno $t$
- $T$ es el horizonte temporal (en nuestro ejemplo, $T = 100$)

### 1.2 Restricciones
1. En cada turno $t$, se debe elegir exactamente una de las dos bandas disponibles
2. Las probabilidades $p_1$ y $p_2$ son desconocidas a priori
3. Ambas probabilidades siguen distribuciones uniformes en el intervalo $[0.01, 0.99]$

### 1.3 Política Óptima para el Caso General

En este escenario más general, donde ambas probabilidades son desconocidas y siguen distribuciones uniformes, la política óptima debe balancear exploración y explotación de manera adaptativa:

#### Enfoque Bayesiano
1. **Modelado Inicial**: 
   - Para cada banda $i$, inicializamos una distribución Beta($\alpha_i$, $\beta_i$) como prior
   - Para distribuciones uniformes en $[0.01, 0.99]$, podemos aproximar con Beta(1, 1)

2. **Actualización Bayesiana**:
   - Después de cada observación de la banda $i$, actualizamos:
     - Si observamos recompensa 1: $\alpha_i \leftarrow \alpha_i + 1$
     - Si observamos recompensa 0: $\beta_i \leftarrow \beta_i + 1$

3. **Política de Selección**:
   - **Thompson Sampling**: En cada turno, muestreamos $\tilde{p}_i$ de cada distribución Beta($\alpha_i$, $\beta_i$) y seleccionamos la banda con mayor valor muestreado
   - **Índice de Gittins**: Calcula y selecciona la banda con el mayor índice de Gittins, que tiene en cuenta el valor de exploración futuro

#### Balance Óptimo Exploración-Explotación
Para $T = 100$, el número óptimo de turnos de exploración para cada banda depende de la incertidumbre inicial y la diferencia esperada entre $p_1$ y $p_2$.

## 2. Casos Particulares

### 2.1 Banda Fija con $p_1$ Conocido (Horizonte $T = 100$)

#### Restricciones Específicas
1. La probabilidad $p_1$ de la primera banda es conocida explícitamente
2. La probabilidad $p_2$ sigue siendo desconocida y requiere estimación
3. El horizonte temporal está fijo en $T = 100$ turnos

#### Política Óptima
1. **Fase de Exploración**:
   - Dedicar $n_2$ turnos para explorar la banda 2 y estimar $p_2$
   - Usando la desigualdad de Hoeffding: 
     $$n_2 \geq \frac{\ln(2/\delta)}{2\varepsilon^2}$$
   - Donde $\varepsilon = |p_1 - p_2|/4$ para minimizar el riesgo de error
   - $\delta$ es el nivel de confianza deseado (típicamente 0.05)

2. **Fase de Explotación**:
   - Para los $100 - n_2$ turnos restantes:
     - Si $\hat{p}_2 > p_1$: elegir siempre la banda 2
     - Si $\hat{p}_2 < p_1$: elegir siempre la banda 1
     - Si $\hat{p}_2 = p_1$: indiferente

### 2.3 Horizonte Desconocido (Solo Recompensa)

#### Restricciones Específicas
1. No se conoce el horizonte temporal $T$
2. No se conocen las probabilidades $p_1$ ni $p_2$
3. Solo se observan las recompensas después de cada elección

#### Política Óptima
Al no conocer ni $T$ ni las probabilidades iniciales, debemos aplicar algoritmos que equilibren exploración y explotación constantemente:

1. **UCB (Upper Confidence Bound)**:
   - Selecciona la banda $i$ que maximiza:
     $$\hat{p}_i + \sqrt{\frac{2 \ln t}{n_i}}$$
   - donde $\hat{p}_i$ es la estimación actual de $p_i$, $t$ es el número total de jugadas y $n_i$ es el número de veces que se ha jugado la banda $i$

2. **Thompson Sampling**:
   - Inicializar distribuciones Beta(1, 1) para ambas bandas
   - En cada turno:
     - Muestrear $\tilde{p}_1$ y $\tilde{p}_2$ de sus respectivas distribuciones
     - Elegir la banda con mayor valor muestreado
     - Actualizar la distribución correspondiente según la recompensa observada

3. **$\varepsilon$-greedy**:
   - Con probabilidad $\varepsilon$, explorar aleatoriamente
   - Con probabilidad $1 - \varepsilon$, explotar la banda con mayor $\hat{p}_i$ estimada
   - Opcionalmente, disminuir $\varepsilon$ gradualmente con el tiempo


---
#### Decisión (T Aleatorio)

### **EJERCICIO**  
**RESPUESTA**  
Definir el problema de decisión para la Banda Fija con horizonte de tiempo desconocido T ~ Uniform(1, 300). ¿Cómo afecta el horizonte de tiempo aleatorio la estrategia óptima?
# Cambios en la Estrategia Óptima con Horizonte de Tiempo Aleatorio T ~ Uniform(1, 300)

## 1. Modificaciones en la Función Objetivo

La función objetivo ahora debe considerar la expectativa sobre el horizonte temporal aleatorio:

$$\max \mathbb{E}_T\left[\mathbb{E}\left[ \sum_{t=1}^{T} X_t \right]\right]$$

Donde $T \sim \text{Uniforme}(1, 300)$ y la expectativa exterior es sobre la distribución del horizonte temporal.

## 2. Cambios en las Restricciones

- El horizonte temporal ya no es fijo en 100, sino aleatorio siguiendo una distribución uniforme entre 1 y 300
- No conocemos exactamente cuándo terminará el proceso de decisión
- El valor esperado de $T$ es 150.5

## 3. Cambios en la Estrategia Óptima

### 3.1 Exploración y Explotación

- **Mayor incertidumbre sobre la duración**: No podemos determinar con certeza cuántos turnos dedicar a la exploración
- **Exploración más conservadora**: Debemos limitar la fase de exploración inicial debido al riesgo de un horizonte corto
- **Exploración continua**: A diferencia del caso con $T$ fijo, puede ser óptimo mantener un componente exploratorio a lo largo del tiempo

### 3.2 Adaptación de la Política para Banda Fija con $p_1$ Conocido

- **Enfoque más robusto**: En lugar de dividir claramente entre fases de exploración y explotación, necesitamos una política que se adapte en tiempo real

- **Actualización del cálculo de $n_2$**:
  - El valor óptimo de $n_2$ ahora debe considerar la posibilidad de un horizonte muy corto
  - Una aproximación razonable es usar el valor esperado $\mathbb{E}[T] = 150.5$ con un factor de corrección:
    $$n_2 \approx \min\left(\frac{\ln(2/\delta)}{2\varepsilon^2}, \beta \cdot \mathbb{E}[T]\right)$$
  - Donde $\beta < 0.5$ es un factor de ajuste que refleja la incertidumbre sobre $T$

### 3.3 Incorporación de Políticas en Tiempo Real

- **Thompson Sampling Modificado**: Más adecuado para horizontes aleatorios que una política fija de dos fases
  - Inicializar con prior informativo basado en $p_1$ conocido
  - Actualizar el posterior de $p_2$ después de cada observación
  - Seleccionar la banda con mayor valor esperado

- **Índice de Gittins con descuento**: Incorporar un factor de descuento que refleje la probabilidad de finalización en cada turno:
  - Factor de descuento $\gamma = 1 - \frac{1}{300-t+1}$ para el turno $t$ (si $t \leq 300$)
  - El índice de Gittins considera el valor potencial de la información futura con descuento

### 3.4 Modelado de la Incertidumbre Sobre $T$

- **Distribución de probabilidad de finalización**: En cada turno $t$:
  - Si $t < 1$: Prob(finalizar) = 0
  - Si $1 \leq t \leq 300$: Prob(finalizar en $t$ | no finalizado hasta $t-1$) = $\frac{1}{300-t+1}$
  - Si $t > 300$: Prob(finalizar) = 1

- **Valor esperado con horizonte aleatorio**: Para cualquier política $\pi$:
  $$V^\pi = \sum_{t=1}^{300} \mathbb{E}^\pi[X_t] \cdot \mathbb{P}(T \geq t)$$


### 3.2 Entorno de Banda Periódica

#### Descripción
En el entorno de Banda Periódica, la probabilidad de recompensa de cada brazo cambia cada k turnos (por defecto, k=10). En cada punto de cambio, se asignan nuevas probabilidades aleatorias (uniforme entre 0.01 y 0.99) a ambos brazos.

#### Formulación Matemática
- Dos brazos: $a \in \{0, 1\}$
- En el turno $t$, las probabilidades son:
  - $p_1(t) = p_1^{\lfloor t/k \rfloor}$, donde $p_1^j \sim \text{Uniform}(0.01, 0.99)$
  - $p_2(t) = p_2^{\lfloor t/k \rfloor}$, donde $p_2^j \sim \text{Uniform}(0.01, 0.99)$
- El superíndice $j = \lfloor t/k \rfloor$ indica el número de "período".
- En cada punto de cambio (cuando $t$ es divisible por $k$), se asignan nuevos valores aleatorios.

#### Decisión (T Fijo)

### **EJERCICIO**  
**RESPUESTA**  
Definir el problema de decisión para la Banda Periódica con horizonte de tiempo conocido T = 100 y período k = 10. ¿Cómo abordarías la búsqueda de una estrategia óptima? ¿Qué información adicional sería valiosa rastrear?
# Problema de Decisión para la Banda Periódica con Horizonte T = 100 y Período k = 10

## 1. Definición del Problema

En el problema de Banda Periódica, las probabilidades de recompensa cambian periódicamente:

- **Banda 1**: Recompensa 1 con probabilidad $p_1(t)$, 0 con probabilidad $1 - p_1(t)$
- **Banda 2**: Recompensa 1 con probabilidad $p_2(t)$, 0 con probabilidad $1 - p_2(t)$

Donde $p_i(t) = p_i(t \mod k)$ para cada banda $i$, con período $k = 10$.

## 2. Función Objetivo

La función objetivo sigue siendo maximizar la recompensa total esperada:

$$\max \mathbb{E}\left[ \sum_{t=1}^{100} X_t \right]$$

Donde $X_t$ es la recompensa obtenida en el turno $t$.

## 3. Restricciones

1. En cada turno $t$, se debe elegir exactamente una de las dos bandas
2. El horizonte temporal es $T = 100$ turnos
3. Las probabilidades de recompensa $p_i(t)$ son periódicas con período $k = 10$
4. Para cada valor de $t \mod 10$, las probabilidades $p_i(t)$ pueden ser diferentes

## 4. Cambios en la Estrategia Óptima

### 4.1 Consideración de la Periodicidad

- **Tratamiento como sub-problemas**: El problema se puede dividir en 10 sub-problemas independientes (uno para cada valor de $t \mod 10$)
- **Patrón cíclico**: La estrategia óptima seguirá un patrón cíclico con período 10
- **Expectativa a futuro**: Debemos considerar que ciertas posiciones dentro del ciclo pueden aparecer más veces que otras debido al horizonte finito

### 4.2 Estructura de la Política Óptima

Para cada posición $j \in \{0, 1, 2, ..., 9\}$ dentro del ciclo:

1. **Estimación de probabilidades**: Estimar $p_1(j)$ y $p_2(j)$ utilizando solo observaciones de turnos $t$ donde $t \mod 10 = j$
2. **Comparación en cada posición**: Para cada posición $j$ del ciclo, determinar qué banda tiene mayor probabilidad
3. **Decisión específica por posición**: La decisión óptima en el turno $t$ dependerá de $t \mod 10$

### 4.3 Exploración y Explotación

- **Exploración inicial por posición**: Dedicar los primeros ciclos a explorar ambas bandas en cada posición del ciclo
- **Explotación posterior**: Explotar la banda con mayor probabilidad estimada para cada posición
- **Distribución del presupuesto de exploración**: Al tener 10 sub-problemas, debemos distribuir el esfuerzo de exploración entre ellos

## 5. Información Adicional Valiosa a Rastrear

### 5.1 Estadísticas por Posición en el Ciclo

- **Contadores separados**: Mantener contadores separados de éxitos/intentos para cada banda y cada posición del ciclo
- **Estadísticas para cada $(i,j)$**: Para cada banda $i \in \{1,2\}$ y posición $j \in \{0,1,...,9\}$:
  - $n_{i,j}$ = número de veces que se ha jugado la banda $i$ en posición $j$
  - $w_{i,j}$ = número de éxitos obtenidos con la banda $i$ en posición $j$
  - $\hat{p}_{i,j} = \frac{w_{i,j}}{n_{i,j}}$ (estimación de probabilidad)

### 5.2 Métricas de Confianza

- **Intervalos de confianza**: Calcular intervalos de confianza para cada $\hat{p}_{i,j}$
- **UCB por posición**: Adaptar el algoritmo UCB para cada posición del ciclo:
  $$\text{UCB}_{i,j} = \hat{p}_{i,j} + \sqrt{\frac{2\ln(t/k)}{n_{i,j}}}$$
- **Probabilidad de error**: Estimar la probabilidad de tomar una decisión incorrecta en cada posición

### 5.3 Correlaciones y Patrones

- **Correlaciones entre posiciones**: Detectar posibles correlaciones entre diferentes posiciones del ciclo
- **Tendencias dentro del ciclo**: Identificar patrones como tendencias crecientes/decrecientes dentro del ciclo
- **Similitud entre bandas**: Detectar si ambas bandas siguen patrones similares o opuestos

### 5.4 Proyecciones de Recompensa

- **Recompensa esperada por ciclo**: Calcular la recompensa esperada para un ciclo completo bajo diferentes políticas
- **Recompensa restante**: Estimar la recompensa esperada para los turnos restantes
- **Contribución por posición**: Identificar qué posiciones del ciclo contribuyen más a la recompensa total

## 6. Algoritmo Propuesto

1. **Fase inicial** (primeros 2-3 ciclos completos):
   - Explorar ambas bandas en cada posición del ciclo para obtener estimaciones iniciales

2. **Fase intermedia** (aproximadamente hasta turno 70):
   - Para cada posición $j$ del ciclo:
     - Si $\hat{p}_{1,j}$ y $\hat{p}_{2,j}$ están bien separadas (diferencia significativa), explotar la mejor banda
     - Si están cercanas, continuar explorando ambas

3. **Fase final** (últimos 30 turnos):
   - Explotar exclusivamente la banda con mayor probabilidad estimada en cada posición
   - No realizar más exploración en esta fase

## 7. Complejidad Adicional

- **Número de ciclos completos**: Con $T = 100$ y $k = 10$, habrá exactamente 10 ciclos completos
- **Consistencia de datos**: Algunas posiciones del ciclo tendrán más observaciones que otras si $T$ no fuera múltiplo de $k$
- **Transferencia de información**: Evaluar si es posible transferir información entre posiciones para mejorar las estimaciones
#### Decisión (T Aleatorio)

### **EJERCICIO**  
**RESPUESTA**  
Definir el problema de decisión para la Banda Periódica con horizonte de tiempo desconocido T ~ Uniform(1, 300) y período k = 10. ¿Cómo interactúa la aleatoriedad en T con la naturaleza periódica del entorno?
# Banda Periódica con Horizonte Aleatorio T ~ Uniform(1, 300) y Período k = 10

## 1. Definición del Problema Modificado

En este escenario, enfrentamos dos fuentes de variabilidad:
- **Periodicidad**: Las probabilidades de recompensa siguen un patrón cíclico con período k = 10
- **Horizonte aleatorio**: El número total de turnos T sigue una distribución uniforme entre 1 y 300

Para cada banda i ∈ {1, 2}:
- La probabilidad de recompensa es $p_i(t) = p_i(t \mod 10)$
- El horizonte T ~ Uniform(1, 300) es desconocido hasta la finalización
- El número esperado de turnos es E[T] = 150.5

## 2. Función Objetivo Modificada

La función objetivo ahora incorpora la incertidumbre sobre el horizonte temporal:

$$\max \mathbb{E}_T\left[\mathbb{E}\left[ \sum_{t=1}^{T} X_t \right]\right]$$

Donde la expectativa exterior es sobre la distribución de T, y la interior sobre las recompensas.

## 3. Interacción entre Aleatoriedad del Horizonte y Periodicidad

### 3.1 Efectos de la Interacción

1. **Incertidumbre sobre ciclos completos**: 
   - Con T ~ Uniform(1, 300), tendremos entre 0 y 30 ciclos completos
   - El valor esperado es de 15.05 ciclos completos
   - Existe incertidumbre sobre cuántas veces aparecerá cada posición del ciclo

2. **Distribución no uniforme de posiciones finales**:
   - La probabilidad de que el proceso termine en la posición j del ciclo no es uniforme
   - Para horizontes no múltiplos de k, algunas posiciones aparecerán más veces que otras

3. **Valor de la información por posición**:
   - El valor de explorar en la posición j depende de:
     - Cuántas veces más se espera que aparezca esa posición
     - La incertidumbre actual sobre cuál es la mejor banda en esa posición

4. **Priorización adaptativa**:
   - Las primeras posiciones del ciclo tienen mayor probabilidad de aparecer más veces
   - Las últimas posiciones tienen menor valor esperado de exploración

### 3.2 Probabilidades Condicionales Relevantes

1. **Probabilidad de continuación**:
   - En el turno t, la probabilidad de que el proceso continúe es:
     $$P(\text{continuar después de } t) = \frac{\max(0, 300-t)}{300}$$

2. **Número esperado de apariciones restantes**:
   - Para la posición j del ciclo en el turno t:
     $$E[\text{apariciones restantes de j}] \approx \frac{\max(0, 300-t)}{10} \cdot \frac{1}{10}$$

## 4. Estrategia Óptima Modificada

### 4.1 Consideraciones Estratégicas

1. **Exploración frontal cargada**:
   - Realizar más exploración al inicio cuando hay mayor certeza de ciclos futuros
   - Reducir gradualmente la exploración a medida que avanza el tiempo

2. **Ponderación por valor esperado**:
   - Asignar más recursos de exploración a las posiciones iniciales del ciclo
   - La ponderación debe ser proporcional al número esperado de apariciones futuras

3. **Umbral de exploración dinámico**:
   - El umbral para decidir entre exploración y explotación debe disminuir con el tiempo
   - En las últimas etapas, favorecer fuertemente la explotación

### 4.2 Política Adaptativa Propuesta

Para cada turno t con posición j = t mod 10:

1. **Calcular valor de información**:
   - $V_{info}(i,j,t) = P(\text{mejor decisión cambia}) \times \text{ganancia esperada} \times \text{apariciones esperadas restantes}$

2. **Calcular valor de explotación**:
   - $V_{expl}(i,j,t) = \hat{p}_{i,j} - \hat{p}_{3-i,j}$ (diferencia entre bandas)

3. **Decisión**:
   - Si $V_{info}(i^*,j,t) > V_{expl}(i^*,j,t)$: explorar la banda con mayor valor de información
   - De lo contrario: explotar la banda con mayor probabilidad estimada

### 4.3 Matriz de Decisión Dinámica

- Mantener una matriz de decisión D[j] para cada posición j del ciclo
- Actualizar dinámicamente basándose en:
  1. La confianza actual en las estimaciones $\hat{p}_{i,j}$
  2. El número esperado de veces que la posición j aparecerá nuevamente
  3. La probabilidad de que el proceso termine pronto

## 5. Información Adicional a Rastrear

### 5.1 Métricas por Posición del Ciclo

- Todas las métricas del caso periódico con T fijo:
  - Contadores $n_{i,j}$ y $w_{i,j}$ para cada banda i y posición j
  - Estimaciones $\hat{p}_{i,j}$ e intervalos de confianza

### 5.2 Métricas Específicas para Horizonte Aleatorio

1. **Estimación de apariciones restantes**:
   - Para cada posición j, estimar cuántas veces más aparecerá
   - Actualizar esta estimación a medida que avanza el tiempo

2. **Valor de exploración ajustado por horizonte**:
   - Calcular el valor esperado de la exploración considerando la aleatoriedad en T
   - $V_{exp}(i,j,t) = \Delta\hat{p}_{i,j} \times E[\text{apariciones restantes}| t]$

3. **Indicadores de certidumbre relativa**:
   - Comparar la confianza en estimaciones entre diferentes posiciones
   - Priorizar exploración donde haya más incertidumbre y mayor valor esperado

### 5.3 Meta-parámetros Adaptativos

1. **Factor de descuento temporal**:
   - $\gamma(t) = \frac{\max(0, 300-t)}{300}$ para reflejar la probabilidad de continuación

2. **Índice UCB ajustado por horizonte**:
   - $\text{UCB}_{i,j}(t) = \hat{p}_{i,j} + \gamma(t) \times \sqrt{\frac{2\ln(t/k)}{n_{i,j}}}$

3. **Ratio exploración/explotación dinámico**:
   - Iniciar con mayor énfasis en exploración
   - Reducir gradualmente según avanza t y aumenta la probabilidad de finalización

## 6. Algoritmo de Decisión Propuesto

1. **Inicialización**:
   - Explorar ambas bandas al menos una vez en cada posición del ciclo
   - Inicializar estimaciones $\hat{p}_{i,j}$ para todo i,j

2. **Fase principal** (para cada turno t):
   - Identificar la posición actual j = t mod 10
   - Calcular $\gamma(t) = \frac{\max(0, 300-t)}{300}$
   - Para cada banda i, calcular:
     - $\text{UCB}_{i,j}(t) = \hat{p}_{i,j} + \gamma(t) \times \sqrt{\frac{2\ln(t/k)}{n_{i,j}}}$
   - Elegir la banda con mayor UCB ajustado
   - Actualizar contadores y estimaciones

3. **Adaptación temporal**:
   - A medida que t aumenta, reducir el componente de exploración
   - Para t > 200, favorecer fuertemente la explotación

### 3.3 Entorno de Banda Dinámica

#### Descripción
En el entorno de Banda Dinámica, las probabilidades de recompensa para ambos brazos cambian en cada turno. Cada turno se asignan probabilidades aleatorias completamente nuevas (uniforme entre 0.01 y 0.99) a ambos brazos.

#### Formulación Matemática
- Dos brazos: $a \in \{0, 1\}$
- En el turno $t$, las probabilidades son:
  - $p_1(t) \sim \text{Uniform}(0.01, 0.99)$
  - $p_2(t) \sim \text{Uniform}(0.01, 0.99)$
- Se generan nuevos valores aleatorios en cada turno.

La estrategia óptima es simple: selección aleatoria uniforme (50%-50%) entre ambas bandas en cada turno. No existe algoritmo que pueda superar consistentemente esta estrategia debido a la renovación completa e independiente de las probabilidades en cada turno.
#### Decisión (T Fijo)

### **EJERCICIO**  
**RESPUESTA**  
Definir el problema de decisión para la Banda Dinámica con horizonte de tiempo conocido T = 100. ¿Hay una forma significativa de aprender de observaciones pasadas en este entorno? ¿Cuál sería la estrategia óptima?
# Problema de Decisión para la Banda Dinámica con Horizonte T = 100

## 1. Definición del Problema

En la Banda Dinámica, las probabilidades de recompensa cambian completamente en cada turno:
- **Banda 1**: Recompensa con probabilidad $p_1(t) \sim \text{Uniform}(0.01, 0.99)$
- **Banda 2**: Recompensa con probabilidad $p_2(t) \sim \text{Uniform}(0.01, 0.99)$
- Las probabilidades se regeneran independientemente en cada turno

La estrategia óptima es simple: selección aleatoria uniforme (50%-50%) entre ambas bandas en cada turno. No existe algoritmo que pueda superar consistentemente esta estrategia debido a la renovación completa e independiente de las probabilidades en cada turno.


### **EJERCICIO**  
**RESPUESTA**  
# Problema de Decisión para la Banda Dinámica con Horizonte T ~ Uniform(1, 300)

## 1. Definición del Problema

- **Banda Dinámica**: Las probabilidades $p_1(t)$ y $p_2(t)$ se generan aleatoriamente (Uniform(0.01, 0.99)) cada turno
- **Horizonte aleatorio**: T ~ Uniform(1, 300), desconocido hasta la finalización
- **Objetivo**: Maximizar la recompensa total esperada

## 2. Análisis del Enfoque Óptimo

Idéntica al caso con T fijo: selección aleatoria uniforme (50%-50%). La incertidumbre adicional sobre el horizonte temporal no modifica la estrategia óptima, ya que la independencia entre turnos hace que cualquier intento de aprendizaje sea inútil.


### 3.4 Entorno de Banda Totalmente Aleatorio

#### Descripción
En el entorno de Banda Totalmente Aleatorio, las probabilidades de los brazos se inicializan de forma aleatoria y luego cambian aleatoriamente con una pequeña probabilidad (5%) en cada turno. Esto crea un entorno donde los cambios son impredecibles pero ocurren con menos frecuencia que en el entorno Dinámico.

#### Formulación Matemática
- Dos brazos: $a \in \{0, 1\}$
- Probabilidades iniciales: $p_1(0), p_2(0) \sim \text{Uniform}(0.01, 0.99)$
- En el turno $t > 0$, con probabilidad 0.05:
  - $p_1(t) \sim \text{Uniform}(0.01, 0.99)$
  - $p_2(t) \sim \text{Uniform}(0.01, 0.99)$
- De lo contrario (con probabilidad 0.95):
  - $p_1(t) = p_1(t-1)$
  - $p_2(t) = p_2(t-1)$

#### Decisión (T Fijo)

### **EJERCICIO**  
**RESPUESTA**  
Definir el problema de decisión para la Banda Totalmente Aleatoria con horizonte de tiempo conocido T = 100. ¿Cómo equilibrarías la exploración y explotación sabiendo que las probabilidades de los brazos podrían cambiar repentinamente?

## Estrategia Óptima: Banda Totalmente Aleatoria con T = 100

**Política con ventanas deslizantes adaptativas:**

1. **Algoritmo SW-UCB adaptativo**:
   - Mantener ventana de 10-20 observaciones recientes
   - Para cada banda i en turno t: $UCB_{i,t} = \hat{p}_i + \sqrt{\frac{3\ln(t)}{2n_i}}$ usando solo datos de la ventana

2. **Detección de cambios**:
   - Implementar test CUSUM o Page-Hinkley para detectar cambios
   - Reiniciar contadores y estimaciones cuando se detecta un cambio

3. **Reducción progresiva de exploración**:
   - Fase inicial (turnos 1-20): Exploración equilibrada
   - Fase media (turnos 21-80): Reducción gradual de exploración
   - Fase final (turnos 81-100): Explotación priorizada con ventana pequeña (5-10 observaciones)

Esta estrategia balancea la necesidad de adaptarse rápidamente a cambios mientras maximiza la explotación de conocimiento reciente.
#### Decisión (T Aleatorio)

### **EJERCICIO**  
**RESPUESTA**  
Definir el problema de decisión para la Banda Totalmente Aleatoria con horizonte de tiempo desconocido T ~ Uniform(1, 300). ¿Cómo interactúan las dos formas de aleatoriedad (en las probabilidades de los brazos y en el horizonte de tiempo)?
## Estrategia Óptima: Banda Totalmente Aleatoria con T ~ Uniform(1, 300)

### Definición del Problema

- **Banda Totalmente Aleatoria**: 
  - Probabilidades iniciales: $p_1(0), p_2(0) \sim \text{Uniform}(0.01, 0.99)$
  - En cada turno t, con probabilidad 0.05: $p_i(t) \sim \text{Uniform}(0.01, 0.99)$
  - Con probabilidad 0.95: $p_i(t) = p_i(t-1)$
- **Horizonte aleatorio**: T ~ Uniform(1, 300), desconocido hasta finalización

### Estrategia Óptima

**Política adaptativa con factor de urgencia temporal:**

1. **Factor de urgencia temporal**:
   - $\alpha(t) = 1 - \frac{\max(0, 300-t)}{300}$ (aumenta linealmente con t)

2. **Ventana adaptativa**:
   - $w(t) = \max(5, 20 \cdot (1-\alpha(t)))$ (reduce tamaño con t)

3. **Índice de decisión**:
   - $D_i(t) = \hat{p}_i^{w(t)} + (1-\alpha(t)) \cdot \sqrt{\frac{\ln(t)}{n_i^{w(t)}}}$
   - Seleccionar banda con mayor valor $D_i(t)$

4. **Detección de cambios**:
   - Implementar tests estadísticos (CUSUM)
   - Al detectar cambio, reiniciar contadores con Factor de Olvido: $\gamma(t) = 0.95 \cdot (1 - \alpha(t))$

5. **Fases de ejecución**:
   - **Inicial** (t < 50): Alta exploración, ventana amplia
   - **Intermedia** (50 < t < 200): Reducción gradual de exploración
   - **Final** (t > 200): Exploración mínima, ventana pequeña (5-10 observaciones)

Esta estrategia balancea la adaptación a cambios impredecibles con la incertidumbre sobre el horizonte temporal, reduciendo progresivamente la exploración a medida que aumenta la probabilidad de finalización.
## 4. Implementación de Agentes

En nuestro entorno, implementarás tres tipos de agentes correspondientes a los tres escenarios de información descritos anteriormente. Esto es lo que cada agente debe manejar:

### 4.1 Agente de Información Completa

**Entrada:**
```python
env_info = {
    'current_turn': int,        # Número de turno actual
    'total_turns': int,         # Número total de turnos en el juego
    'p1': float,                # Probabilidad de recompensa del brazo 1
    'history': {
        'actions': [int, ...],   # Acciones pasadas (0 para brazo 1, 1 para brazo 2)
        'rewards': [float, ...], # Recompensas pasadas
        'p1': [float, ...],      # Historial de probabilidades del brazo 1
        'p2': [float, ...]       # Historial de probabilidades del brazo 2 (solo para evaluación)
    }
}
```

**Salida:**
```python
action = 0 or 1  # 0 para el brazo 1, 1 para el brazo 2
```

### 4.2 Agente de Información Parcial

**Entrada:**
```python
env_info = {
    'current_turn': int,        # Número de turno actual
    'total_turns': int,         # Número total de turnos en el juego
    'p1': float,                # Probabilidad de recompensa del brazo 1
    'history': {
        'actions': [int, ...],   # Acciones pasadas (0 para brazo 1, 1 para brazo 2)
        'rewards': [float, ...]  # Recompensas pasadas
    }
}
```

**Salida:**
```python
action = 0 or 1  # 0 para el brazo 1, 1 para el brazo 2
```

### 4.3 Agente de Solo Recompensa

**Entrada:**
```python
env_info = {
    'current_turn': int,        # Número de turno actual
    'history': {
        'actions': [int, ...],   # Acciones pasadas (0 para brazo 1, 1 para brazo 2)
        'rewards': [float, ...]  # Recompensas pasadas
    }
}
```

**Salida:**
```python
action = 0 or 1  # 0 para el brazo 1, 1 para el brazo 2
```

## 5. Métricas de Rendimiento

El entorno evalúa el rendimiento de los agentes usando varias métricas clave:

### 5.1 Recompensa Promedio

Esta es la recompensa media obtenida por turno, calculada como:

$\text{Recompensa Promedio} = \frac{1}{T} \sum_{t=1}^{T} r_t$

Esta métrica mide directamente qué tan bien el agente está maximizando su función objetivo. Valores más altos indican un mejor rendimiento.

### 5.2 Porcentaje de Acciones Óptimas

Esta métrica mide el porcentaje de veces que el agente seleccionó el brazo con la mayor probabilidad de recompensa:

$\text{Acciones Óptimas (\%)} = \frac{100}{T} \sum_{t=1}^{T} \mathbf{1}\{a_t = \arg\max_i p_i(t)\}$

Donde $\mathbf{1}$ es la función indicadora que vale 1 cuando la condición es verdadera y 0 en caso contrario.

Esta métrica muestra con qué frecuencia el agente elige el mejor brazo, independientemente de la recompensa real recibida. Valores más altos indican una mejor selección de brazos.

### 5.3 Arrepentimiento (Regret)

El arrepentimiento mide la diferencia entre la recompensa esperada de elegir siempre el brazo óptimo y la recompensa esperada de las elecciones del agente:

$\text{Regret} = \sum_{t=1}^{T} \max_i p_i(t) - \sum_{t=1}^{T} p_{a_t+1}(t)$

Valores más bajos de arrepentimiento indican un mejor rendimiento.

### 5.4 Distribución de Recompensas

El entorno visualiza la distribución de recompensas en diferentes entornos usando diagramas de caja (boxplots) y diagramas de violín (violin plots). Estas visualizaciones ayudan a entender:
- La mediana del rendimiento
- La variabilidad en el rendimiento
- La presencia de valores atípicos
- La forma general de la distribución de recompensas

## 6. Pautas de Estrategia

### 6.1 Enfoques Generales

Aquí hay algunos enfoques generales a considerar para la implementación de tus agentes:

1. **Selección Aleatoria**: Elegir brazos aleatoriamente (enfoque de referencia).
2. **Greedy (Codicioso)**: Elegir siempre el brazo con la recompensa estimada más alta.
3. **ε-Greedy**: Casi siempre elegir el mejor brazo, pero explorar ocasionalmente.
4. **UCB (Upper Confidence Bound)**: Elegir brazos basados en estimaciones optimistas de su valor.
5. **Thompson Sampling**: Elegir brazos basados en emparejar probabilidades con distribuciones a posteriori.
6. **Enfoques Bayesianos**: Mantener distribuciones de probabilidad sobre los valores de los brazos.

### 6.2 Consideraciones Específicas del Entorno

#### Banda Fija
- Enfocarse en identificar rápidamente el mejor brazo.
- La exploración se vuelve menos valiosa conforme avanza el juego.
- Con T conocido, se puede planificar un programa decreciente de exploración.

#### Banda Periódica
- Detectar la estructura periódica (k=10).
- Restablecer estimaciones al comienzo de cada período.
- Asignar más exploración al inicio de cada período.

#### Banda Dinámica
- Las observaciones recientes valen más que las antiguas.
- Considerar el uso de una ventana deslizante de observaciones.
- Podría necesitar alta capacidad de respuesta a los cambios.

#### Banda Totalmente Aleatoria
- Estar alerta a cambios repentinos en los patrones de recompensa.
- Equilibrar la persistencia (usar historial) con la adaptabilidad.
- Considerar métodos de detección de cambios.

### 6.3 Consideraciones Específicas de la Información

#### Agente de Información Completa
- Aprovechar el valor conocido p1.
- Enfocarse en estimar p2 con eficiencia.
- Ajustar la estrategia dinámicamente con base en los valores relativos.

#### Agente de Información Parcial
- Similar a información completa, pero más limitado.
- Podría requerir más exploración en ciertos entornos.

#### Agente de Solo Recompensa
- Debe estimar las probabilidades de ambos brazos.
- Necesita lidiar con el horizonte de tiempo desconocido.
- Considerar estrategias adaptativas en el tiempo.

## 7. Conclusión

El problema de Multi-Bandas ofrece un marco fundamental para estudiar la toma de decisiones secuenciales bajo incertidumbre. Los entornos y escenarios de información en este playground brindan un conjunto rico de desafíos que resaltan diferentes aspectos del dilema exploración-explotación.

Al implementar agentes para estos escenarios, obtendrás experiencia práctica con conceptos clave en aprendizaje por refuerzo y teoría de la decisión, y desarrollarás intuición para equilibrar la recolección de información con la maximización de recompensas en diversos contextos.

Mientras trabajas en tus implementaciones, considera cómo se extenderían tus estrategias a:
- Bandas con más de dos brazos.
- Espacios de acción continuos.
- Distribuciones de recompensa no estacionarias con diferentes patrones.
- Bandas contextuales donde se dispone de información adicional.

