"""Render a reproducible narrative and two PNG figures from v0.5 CSV files."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reports'
REPORT=ROOT/'research'/'13_robustness_results_v05.md'
PROTOCOL={"legacy_anchor":"Исходная выборка v0.4", "temporal_forward":"Новые временные периоды", "cohort_holdout":"Новые синтетические когорты"}
MECH={"complete":"Без пропусков", "mar_attendance":"MAR (посещаемость)","mnar_value":"MNAR (значение)","contiguous_terms":"Блочные пропуски"}
MODEL={"logistic_regression":"Логистическая регрессия", "random_forest":"Случайный лес"}


def render() -> None:
    meta=json.loads((OUT/'robustness_summary_synthetic.json').read_text(encoding='utf-8'))
    runs=pd.read_csv(OUT/'robustness_runs_synthetic.csv')
    agg=pd.read_csv(OUT/'robustness_aggregates_synthetic.csv')
    deltas=pd.read_csv(OUT/'robustness_delta_summary_synthetic.csv')
    missing=pd.read_csv(OUT/'robustness_missingness_synthetic.csv')
    seeds=pd.read_csv(OUT/'robustness_selections_synthetic.csv')
    lines=[
        '# EduRisk v0.5 — исследование устойчивости и переноса прогнозирования',
        '',
        '**Тип исследования:** контролируемый вычислительный эксперимент на синтетических траекториях студентов. Реальные данные не использовались.',
        '',
        '## Цель и проверяемая гипотеза',
        '',
        'Оценить, насколько адаптивное правило выбора длины истории v0.4 сохраняет преимущество или уступает моделям с фиксированными окнами при временном переносе, смене группы обучающихся и структурных пропусках.',
        '',
        'Рабочая гипотеза: качество адаптивного подхода зависит от механизма пропусков и выбранного семейства предикторов, поэтому усреднение оценок риска может служить конкурентной альтернативой. На данном этапе это **гипотеза**, а не установленное новое научное положение.',
        '',
        '## Дизайн эксперимента',
        '',
        f'- **{meta["n_folds"]} разбиений:** одно контрольное на прежней синтетической выборке, три временных сдвига на новом сгенерированном массиве и два разбиения без пересечения студентов между train/validation/test.',
        f'- **{meta["n_trained_models"]} обучений:** пять фиксированных окон истории × два семейства моделей × четыре сценария пропусков × шесть разбиений.',
        '- Целевой признак: возникновение средней оценки менее 3,0 за следующий синтетический семестр.',
        '- Выбор фиксированного окна и параметров адаптивного правила производится только по **валидационному** AP (при равенстве — Brier).',
        '- Сравнение проводится на **отложенном** тестовом периоде; отдельные обученные модели обрабатывают окна от 1 до 5 семестров.',
        '- При обучении медианная импутация и масштабирование оцениваются только по обучающей части.',
        '- Рассчитываются Average Precision (основной показатель), ROC-AUC, Brier score. Первые две метрики выше — лучше, Brier ниже — лучше.',
        '- Конкуренты: k=1, k=5, окно по валидации, адаптивное правило v0.4, равномерное усреднение вероятностей, взвешенное усреднение по AP валидации.',
        '',
        '### Тестовые сценарии',
        '',
        '| Механизм | Модель пропусков |', '|---|---|',
        '| Без пропусков | Контрольный сценарий |',
        '| MAR | Вероятность отсутствия оценок/заданий связана с **наблюдаемой** посещаемостью; посещаемость всегда остаётся известной |',
        '| MNAR | Вероятность исчезновения конкретного значения зависит от самого значения, известного только симулятору |',
        '| Блочные пропуски | У каждого синтетического студента случайным образом отсутствует один непрерывный блок семестров |',
        '',
        'Запрошенный уровень пропусков для ненулевых сценариев — 25%; это ориентир, фактические доли приведены далее.',
        '',
        '### Независимость разбиений',
        '',
        '| Протокол / проверка | Обучение | Валидация | Тест | Доля риска train | Доля риска val | Доля риска test |', '|---|---|---|---|---:|---:|---:|',
    ]
    for fold in meta['protocols']:
        d=fold['description']
        lines.append(f'| {fold["fold_id"]} | {fold["train_rows"]} строк | {fold["validation_rows"]} | {fold["test_rows"]} | {fold["train_prevalence"]:.1%} | {fold["validation_prevalence"]:.1%} | {fold["test_prevalence"]:.1%} |')
    lines += ['','Для протокола «новые когорты» идентификаторы обучающихся не пересекаются между обучением, валидацией и тестом. Но все когорты создаются **одним синтетическим генератором** с разными случайными зернами, поэтому переноса на реальную внешнюю популяцию не проверено. Доля положительного класса меняется между разбиениями (см. таблицу), а AP чувствительна к распространённости положительного класса — это необходимо учитывать при интерпретации изменений.','',
              '## Основные результаты: разница AP относительно фиксированного окна','',
              'Δ AP = AP выбранного подхода − AP фиксированного окна, выбранного по валидации. Положительная разница означает улучшение на тесте в данном эксперименте.','',
              '| Протокол | Пропуски | Модель | Адаптивный Δ AP | Равномерное Δ AP | Взвешенное Δ AP |',
              '|---|---|---|---:|---:|---:|']
    for (protocol, mechanism, model),g in deltas.groupby(['protocol','mechanism','model'], sort=True):
        d={z.method:z.mean_delta_ap for _,z in g.iterrows()}
        if len(d)!=3: raise RuntimeError('Incomplete methods')
        lines.append(f'| {PROTOCOL[protocol]} | {MECH[mechanism]} | {MODEL[model]} | {d["adaptive_gate"]:+.4f} | {d["uniform_ensemble"]:+.4f} | {d["weighted_ensemble"]:+.4f} |')

    # Explain the zero-gap gate empirically without overclaiming.
    original=runs[(runs.protocol=='legacy_anchor') & (runs.mechanism=='complete')]
    lines += ['','### Контроль воспроизводимости прежнего эксперимента','',
              'При отсутствии пропусков использован тот же исходный CSV `data/synthetic_long_10_terms.csv` и то же временное разбиение, что в v0.4; это позволяет проверить воспроизводимость базовых моделей.',
              '', '| Модель | AP: одно окно | AP: окно 5 | AP: адаптивное |','|---|---:|---:|---:|']
    for fam in ('logistic_regression','random_forest'):
        z=original[original.model==fam].set_index('method')
        lines.append(f'| {MODEL[fam]} | {z.loc["snapshot_1","test_average_precision"]:.4f} | {z.loc["longest","test_average_precision"]:.4f} | {z.loc["adaptive_gate","test_average_precision"]:.4f} |')
    lines += ['','Нулевая разница адаптивного и лучшего фиксированного окна здесь означает, что оба метода дали одинаковые тестовые вероятности на данном контрольном разбиении. Это не доказательство эквивалентности методов в целом.','',
              '### Фактически реализованные пропуски','', '| Сценарий | Минимум | Среднее | Максимум |','|---|---:|---:|---:|']
    for key,g in missing.groupby('mechanism',sort=True):
        lines.append(f'| {MECH[key]} | {g.missing_rate_actual.min():.1%} | {g.missing_rate_actual.mean():.1%} | {g.missing_rate_actual.max():.1%} |')
    lines += ['',
              '## Интерпретация и ограничения','',
              '1. Знак изменения AP различается между условиями: при искусственной смене когорт и MAR-пропусках адаптивный вариант улучшал логистическую регрессию, но ухудшал случайный лес. Это указывает на зависимость результата от модели и наблюдаемости признаков.',
              '2. Взвешенное усреднение в отдельных сценариях улучшает AP, в других — нет. Оно служит конкурентом предлагаемому правилу; выбор стратегии по **тестовой** метрике недопустим.',
              '3. Протоколы проверяют различающиеся типы обобщения: будущие семестры тех же синтетических студентов и ранее невиденные синтетические студенты. Результаты нельзя напрямую усреднять между ними и принимать за оценку эффекта в другом вузе.',
              '4. Временные разбиения частично пересекаются по студентам; две когортные проверки используют общие синтетические наборы. Стандартные отклонения между разбиениями **не** являются доверительными интервалами.',
              '5. Осуществлён один случайный рисунок пропусков на каждый сценарий/разбиение. Взвешенный ансамбль использует AP той же валидационной части, на которой сравниваются кандидаты, что повышает риск переобучения на валидации. Для подтверждающих выводов нужны вложенная валидация, повторы, интервальные оценки, более разнообразные генераторы и проверка на правомерно полученных обезличенных данных.',
              '6. Сам факт разработки адаптивного правила не доказывает научной новизны. Необходимо изучить научную литературу, включая существующие методы динамического выбора моделей, ансамбли и методы работы с пропусками.',
              '',
              '## Следующий этап (v0.6)','',
              'Разработать **мета-стратегию**, которая по метаданным доступной истории и результатам внутренней валидации выбирает между фиксированным окном, адаптивным правилом и ансамблем **без обращения к итоговой тестовой выборке**. Провести повторные испытания на нескольких независимо сгенерированных когортах/масках, оценить интервалы неопределённости и чувствительность к изменению механизма пропусков. Затем согласовать с научным руководителем возможность апробации на законно предоставленных обезличенных данных.',
              '',
              '### Воспроизведение','',
              '```powershell',
              '$env:PYTHONPATH = "src"',
              '.\\.venv\\Scripts\\python.exe -m edurisk.robustness --legacy data/synthetic_long_10_terms.csv --output reports',
              '.\\.venv\\Scripts\\python.exe make_report_v05.py',
              '.\\.venv\\Scripts\\python.exe -m pytest -q',
              '```',
              '',
              'Файлы `reports/robustness_*_synthetic.csv` сохраняют результаты отдельных моделей, выбранные по валидации настройки и агрегированные сравнения. `robustness_summary_synthetic.json` сохраняет протокол и ограничения.',
              '']
    REPORT.write_text('\n'.join(lines),encoding='utf-8')

    # Plot 1: adaptive vs fixed across protocols/missingness mechanisms.
    subset=deltas[deltas.method=='adaptive_gate'].sort_values(['protocol','mechanism','model'])
    labels=[f"{PROTOCOL[r.protocol]}\n{MECH[r.mechanism]} · {'Логит' if r.model=='logistic_regression' else 'Лес'}" for r in subset.itertuples()]
    fig,ax=plt.subplots(figsize=(12,10))
    ypos=np.arange(len(subset))
    ax.barh(ypos,subset.mean_delta_ap)
    ax.set_yticks(ypos,labels,fontsize=8)
    ax.axvline(0,linewidth=1)
    ax.invert_yaxis()
    ax.set_xlabel('Изменение Average Precision относительно фиксированного окна (Δ AP)')
    ax.set_title('EduRisk v0.5: адаптивное правило против фиксированного окна\nСинтетические данные; средние по доступным разбиениям')
    fig.tight_layout()
    fig.savefig(OUT/'robustness_adaptive_delta_synthetic.png',dpi=160,bbox_inches='tight')
    plt.close(fig)

    # Plot 2: gap of uniform/weighted ensembles for holdout, per missingness.
    subset=deltas[(deltas.protocol=='cohort_holdout') & deltas.method.isin(['uniform_ensemble','weighted_ensemble'])].copy()
    cats=[]
    for mechanism in ['complete','mar_attendance','mnar_value','contiguous_terms']:
        for family in ['logistic_regression','random_forest']:
            cats.append((mechanism,family))
    u=[]; w=[]
    for mech,fam in cats:
        group=subset[(subset.mechanism==mech) & (subset.model==fam)].set_index('method')
        u.append(group.loc['uniform_ensemble','mean_delta_ap'])
        w.append(group.loc['weighted_ensemble','mean_delta_ap'])
    x=np.arange(len(cats))
    fig,ax=plt.subplots(figsize=(12,5))
    ax.bar(x-0.2,u,width=.4,label='Равномерный ансамбль')
    ax.bar(x+0.2,w,width=.4,label='Взвешенный ансамбль (валидация)')
    ax.set_xticks(x,[f'{MECH[m]}\n{"Логит" if f=="logistic_regression" else "Лес"}' for m,f in cats],fontsize=8)
    ax.axhline(0,linewidth=1)
    ax.set_ylabel('Δ AP относительно фиксированного окна')
    ax.set_title('Новые синтетические когорты: сравнение ансамблей')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT/'robustness_ensembles_cohorts_synthetic.png',dpi=160,bbox_inches='tight')
    plt.close(fig)
    print('Wrote',REPORT,'and 2 PNG figures')


if __name__=='__main__':
    render()
