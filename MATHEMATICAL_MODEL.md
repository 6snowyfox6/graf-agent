# Математическая спецификация проекта `graf-agent`

## 1. Обозначения

- \(u\): текстовый запрос пользователя (prompt).
- \(r\): набор референсов (текст, JSON, image-analysis).
- \(D\): черновая диаграмма (`draft`), JSON-граф.
- \(C\): критика (`critique`) в фиксированном JSON-контракте.
- \(\Pi\): план исправлений (`patch_plan`), из `must_fix`/`optional`.
- \(F\): улучшенная диаграмма (`final`).
- \(V\): результат верификации применения критики (`verify`).
- \(G=(N,E)\): граф диаграммы, где \(N\) — узлы, \(E\) — ориентированные ребра.

## 2. Формализация пайплайна

Кодовая оркестрация (main):

1. Генерация:
   \[
   D = \mathcal{G}(u, r)
   \]
2. Критика:
   \[
   C = \mathcal{C}(u, D, r)
   \]
3. План исправлений:
   \[
   \Pi = \mathcal{P}(C)
   \]
4. Улучшение:
   \[
   F = \mathcal{I}(u, D, C, \Pi, r)
   \]
5. Верификация:
   \[
   V = \mathcal{V}(u, D, \Pi, F)
   \]
6. При частично/неисправленных пунктах:
   \[
   \Pi' = \mathcal{P}_{followup}(\Pi, V), \quad F'=\mathcal{I}(u,F,C,\Pi',r), \quad V'=\mathcal{V}(u,D,\Pi,F')
   \]
   Прием \(F'\) выполняется, если улучшилась верификация:
   \[
   \text{accept}(F') \iff \big(\text{fixed}_{V'}>\text{fixed}_{V}\big)\ \lor\ \big(\text{ignored}_{V'}<\text{ignored}_{V}\big)
   \]

## 3. Выбор режима и роутинг рендера

### 3.1 Детекция режима
Режим выбирается по:

- первому совпадению keyword в `diagram_types/*.json`;
- иначе эвристически по маркерам архитектур (`resnet`, `unet`, `qwen`, `anfis`, `3d`, `plotneuralnet`, ...), что форсирует `model_architecture`;
- иначе `general`.

### 3.2 Роутер рендера
Функция \(R(\cdot)\):

\[
R(F)=
\begin{cases}
\text{PlotNeuralNet}, & \text{if } renderer=\texttt{plotneuralnet} \lor layout\_hint=\texttt{model\_architecture}\\
\text{Infographic}, & \text{if } renderer=\texttt{infographic} \lor layout\_hint=\texttt{infographic}\\
\text{Pipeline (Graphviz)}, & \text{if } renderer=\texttt{pipeline} \lor layout\_hint=\texttt{pipeline}\\
\text{General (Graphviz)}, & \text{otherwise}
\end{cases}
\]

## 4. Целевая функция изменения диаграммы

В `critic_influence.py` используется интегральная метрика изменений:

\[
\Delta_N = |\,|N_F|-|N_D|\,|,\quad
\Delta_E = |\,|E_F|-|E_D|\,|
\]
\[
S_{sem} = \#\text{changed\_labels} + \#\text{changed\_kinds}
\]

\[
\text{change\_score} =
1.5\Delta_N + 1.2\Delta_E
 + 1.4(\#\text{added\_nodes}+\#\text{removed\_nodes})
 + 1.1(\#\text{added\_edges}+\#\text{removed\_edges})
 + 1.8S_{sem}
\]

Это целевая переменная суррогатной модели SHAP.

## 5. Оценка кандидатов генерации

### 5.1 `general`-кандидат
Базовый скоринг:
\[
\text{score}_{general}=100-\text{penalties}
\]
штрафы за:

- слишком мало/много узлов;
- запрещенные видимые токены;
- слишком длинные label;
- семантически подозрительный `kind`;
- «обратные»/слишком дальние по уровню связи.

### 5.2 Гибридный запрос (`ResNet+Qwen+ANFIS`)
Для кандидата вычисляется покрытие обязательных компонент:
\[
\text{coverage} = \sum_{k \in \{\text{resnet,qwen,anfis,fusion}\}} \mathbf{1}[k\ \text{найден}]
 + \min(2,0.08|N|)+\min(1,0.05|E|)
\]
Если обязательные части отсутствуют, применяется штраф:
\[
\text{score}\leftarrow \text{score} - (5 + \#\text{missing})
\]

## 6. Математика критики и применения правок

## 6.1 План правок \(\Pi\)
`must_fix` собирается из:

- `missing_requirements`, `wrong_interpretations`, `fixes`, `problems`.

`optional`:

- `extra_elements`, `visual_problems`.

После этого dedupe и hard-cap по длине списков.

## 6.2 Верификация правок
Для `must_fix` верификатор выдает статусы \(\{\text{fixed,partial,ignored}\}\).

Сводка:
\[
\text{fixed}=\sum \mathbf{1}[s_i=\text{fixed}],\quad
\text{partial}=\sum \mathbf{1}[s_i=\text{partial}],\quad
\text{ignored}=\sum \mathbf{1}[s_i=\text{ignored}]
\]

Флаги деградации `invalid_final=True` если:

- \( |N_F|=0 \) или \( |E_F|=0 \);
- \( |N_F| < \max(1,\lfloor |N_D|/3 \rfloor)\) при непустом `draft`.

## 7. Метрики «генератор слушает критика»

Есть два режима.

### 7.1 Verify-based режим (если есть валидный verify)
\[
\text{fixes\_coverage}=\frac{\text{fixed}+0.5\cdot\text{partial}}{\max(1,\text{total})}
\]
\[
\text{ignored\_rate}=\frac{\text{ignored}}{\max(1,\text{total})}
\]
\[
\text{alignment}=
\mathrm{clip}_{[0,1]}\left(
0.75\frac{\text{fixed}}{\text{total}}
+0.15\frac{\text{partial}}{\text{total}}
+0.10(1-\text{contradiction\_rate})
\right)
\]

В этом режиме:
\[
\text{precision\_proxy}=\text{recall\_proxy}=\text{listening\_f1}=\text{fixes\_coverage}
\]

### 7.2 Token-based fallback
Токенизация по regex на алфанумерике длиной \(\ge 3\).  
Пусть \(T_{chg}\) — токены изменений между `draft` и `final`,  
\(T_{crit}\) — токены actionable-критики.

- Coverage fixes/problems via совпадение токенов.
- Proxy precision:
\[
\text{precision\_proxy}=\frac{|T_{chg}\cap T_{crit}|}{\max(1,|T_{chg}|)}
\]
- Proxy recall:
\[
\text{recall\_proxy}=\frac{\#\text{matched\_actionable}}{\max(1,\#\text{actionable})}
\]
- F1:
\[
F_1=
\begin{cases}
\frac{2PR}{P+R}, & P+R>0\\
0, & \text{иначе}
\end{cases}
\]
- Confidence:
\[
\text{evidence\_scale}=\min\left(1,\frac{\#\text{actionable}+|T_{chg}|}{20}\right)
\]
\[
\text{listening\_confidence}=\mathrm{clip}_{[0,1]}\left(\text{evidence\_scale}\cdot (1-0.5\cdot \text{contradiction\_rate})\right)
\]
- Alignment:
\[
\text{alignment}=\mathrm{clip}_{[0,1]}\left(
0.55\cdot\text{fixes\_coverage}
+0.35\cdot\text{problems\_addressed}
+0.10\cdot(1-\text{contradiction\_rate})
\right)
\]

## 8. Traceability (fix \(\rightarrow\) change)

Для каждого пункта критики ищется лучшее изменение по overlap:
\[
\text{overlap\_score}(i,j)=\frac{|T_i\cap T_j|}{\max(1,|T_i|)}
\]
Смягчающий фактор применяется для длинных фраз при слабом overlap.

\[
\text{match\_rate}=
\frac{\#\text{matched\_items}}{\max(1,\#\text{actionable\_items})}
\]

## 9. SHAP и суррогатная модель влияния критика

### 9.1 Датасет
История запусков хранится в `outputs/_critic_influence_history.jsonl`:

- признаки \(x\): `features` (из `critique`);
- цель \(y\): `change_score` (из §4) + диагностические цели.

### 9.2 Модель
Используется `RandomForestRegressor`:
\[
\hat{f}: x \mapsto \widehat{\text{change\_score}}
\]

### 9.3 SHAP
Через `shap.TreeExplainer`:

- локальные вклады:
\[
\hat{f}(x) = \phi_0 + \sum_{i=1}^{d} \phi_i(x)
\]
- глобальная важность:
\[
I_i = \mathbb{E}_{x}|\phi_i(x)|
\]

### 9.4 Fallback-режимы
- Если данных < `min_samples` (8): `insufficient_data`.
- Если нет sklearn: `degraded_no_sklearn`.
- Если нет shap: `degraded_no_shap`, локальные вклады аппроксимируются.

## 10. A/B replay как квази-каузальный тест

Фактический прогон: с реальной критикой \(C\).  
Контрфакт: с нейтральной критикой \(C_0\) (все списки пустые, score=1).

\[
F = \mathcal{I}(u,D,C,\Pi(C),r),\quad
F_0 = \mathcal{I}(u,D,C_0,\Pi(C_0),r)
\]

Дельты:
\[
\Delta_{align} = \text{alignment}(F)-\text{alignment}(F_0)
\]
\[
\Delta_{ignored} = \text{ignored}(F_0)-\text{ignored}(F)
\]
\[
\Delta_{chg} = \text{change\_score}(F)-\text{change\_score}(F_0)
\]

Положительные \(\Delta_{align},\Delta_{ignored}\) трактуются как полезный вклад критики.

## 11. Алгоритмы устойчивости JSON

`pipeline/json_ops.py`:

1. Удаление markdown-fences.
2. Скан первого валидного JSON-объекта `raw_decode`.
3. Fallback: поиск первой балансной `{...}` подстроки.
4. Попытка `json.loads`, затем `ast.literal_eval`.
5. Иначе ошибка извлечения.

Это минимизирует вероятность срыва пайплайна на «грязных» ответах LLM.

## 12. Математика взаимодействия агентов (Generator ↔ Critic ↔ Improver)

В проекте реализована итеративная схема кооперативной оптимизации:

- Generator предлагает \(D\) (кандидат структуры).
- Critic строит вектор ошибок \(C\).
- Improver выполняет constrained-update:
\[
D \xrightarrow[]{C,\Pi} F
\]
с ограничениями контракта (`kind`, `renderer`, `layout_hint`, JSON schema).

Дальше `verify` играет роль «дискриминатора применения критики», а SHAP — роль пост-хок интерпретатора чувствительности изменений к свойствам критики.

## 13. Что является «целевой функцией» в проекте

В коде нет одной глобальной scalar objective, но фактически оптимизируется несколько величин:

1. Структурная состоятельность диаграммы (non-empty, no collapse).
2. Исправление `must_fix` по verify.
3. Метрики слушания критики (`alignment`, `ignored_rate`, `F1`).
4. Рендеропригодность (успешная компиляция/визуализация).

То есть это многокритериальная оптимизация с жесткими ограничениями контракта.

## 14. Практический вывод

Математически система уже поддерживает:

- структурную метрику изменения (\(\text{change\_score}\)),
- измерение «влияния критика» в двух режимах (verify-based и token-based),
- квази-каузальный A/B replay,
- SHAP-интерпретацию как локальную (\(\phi_i\)), так и глобальную (\(\mathbb{E}|\phi_i|\)).

Это достаточный базис для научного/диссертационного описания архитектуры как интерпретируемого multi-agent pipeline.
