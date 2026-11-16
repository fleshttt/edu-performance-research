"""Render an auditable Russian report and two figures from saved v0.6 CSV artifacts."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'reports'
RESEARCH=ROOT/'research'
R=json.loads((RESULTS/'nested_summary_synthetic.json').read_text(encoding='utf-8'))
S=pd.read_csv(RESULTS/'nested_summary_synthetic.csv')
SELECTIONS=pd.read_csv(RESULTS/'nested_selections_synthetic.csv')
frequency=pd.read_csv(RESULTS/'nested_method_frequencies_synthetic.csv')
mechanisms={'complete':'Без пропусков','mar_attendance':'MAR','mnar_value':'MNAR','contiguous_terms':'Пропущенные семестры'}
families={'logistic_regression':'Логистическая регрессия','random_forest':'Случайный лес'}
methods={'fixed_val':'Фиксированное окно','adaptive_gate':'Адаптивное окно',
         'uniform_ensemble':'Равномерный ансамбль','weighted_ensemble':'Взвешенный ансамбль',
         'selected_nested':'Выбранный метод'}
fmt=lambda x: f'{x:+.4f}' if pd.notna(x) else '—'

# Figure 1: paired deltas, CI, with positive/negative areas separated by x=0.
fig,ax=plt.subplots(figsize=(10,5.5))
rows=S[S.method.eq('selected_nested')].sort_values(['mechanism','family'])
y=np.arange(len(rows))
delta=rows['delta_ap_vs_fixed'].to_numpy(float)
lo=rows['delta_ap_ci_low'].to_numpy(float)
hi=rows['delta_ap_ci_high'].to_numpy(float)
ax.errorbar(delta,y,xerr=np.vstack((delta-lo,hi-delta)),fmt='o',capsize=4)
ax.axvline(0,lw=1,ls='--')
ax.set_yticks(y,[mechanisms[x.mechanism]+' · '+ ('LR' if x.family=='logistic_regression' else 'RF') for x in rows.itertuples()])
ax.set_xlabel('Δ AP относительно фиксированного окна (больше — лучше)')
ax.set_title('EduRisk v0.6: выбранный метод против фиксированного окна\nСинтетические генерации, парный бутстрэп по 6 seed')
ax.grid(axis='x',alpha=.15)
fig.tight_layout()
fig.savefig(RESULTS/'nested_delta_ci_synthetic.png',dpi=170,bbox_inches='tight')
plt.close(fig)

# Figure 2: method selection frequencies, two algorithms and four scenarios.
fig,ax=plt.subplots(figsize=(11,5))
sub=frequency.copy()
keys=[]
for mechanism in mechanisms:
    for family in families:
        keys.append((mechanism,family))
meths=['fixed_val','adaptive_gate','uniform_ensemble','weighted_ensemble']
bottom=np.zeros(len(keys))
for method in meths:
    vals=[]
    for m,f in keys:
        match=sub[(sub.mechanism==m)&(sub.family==f)&(sub.chosen_method==method)]
        vals.append(int(match.n_selected.sum()) if len(match) else 0)
    ax.bar(np.arange(len(keys)),vals,bottom=bottom,label=methods[method])
    bottom+=np.array(vals)
ax.set_xticks(np.arange(len(keys)),[('Блочные' if m=='contiguous_terms' else mechanisms[m])+'\n'+('LR' if f=='logistic_regression' else 'RF') for m,f in keys],rotation=30,ha='right')
ax.set_ylabel('Количество генераций (из 6)')
ax.set_title('Какую стратегию выбирала внутренняя валидация')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2)
fig.tight_layout()
fig.savefig(RESULTS/'nested_strategy_counts_synthetic.png',dpi=170,bbox_inches='tight')
plt.close(fig)

text_method=r'''# EduRisk v0.6 — математическая постановка и протокол исследования

## 1. Цель и границы применения

Проверить, способен ли алгоритм выбора стратегии прогнозирования по результатам *внутренней временной валидации* улучшить качество предсказания учебных затруднений по сравнению с фиксированным окном исторических данных. **Данные полностью синтетические.** Исследование не устанавливает научную новизну и не допускает автоматическое применение прогнозов к реальным студентам.

## 2. Математическая постановка

Для студента `i` на учебном периоде `t` наблюдаются пять показателей `x_{i,t}`: оценка, посещаемость, доля выполненных заданий, доля освоенных кредитов и доля просроченных работ. Целевой признак `y_{i,t}` — индикатор среднего балла ниже 3 на периоде `t+1`. Прогноз производится по `x_{i,s}` при `s <= t`; ни одна будущая оценка не используется в конструировании признаков.

Для длины истории `k ∈ {1,2,3,4,5}` строится вектор признаков `φ_k(x_{i,t},...,x_{i,t-k+1})` и прогноз `p_k(x_{i,t}) ∈ [0,1]` от логистической регрессии или случайного леса. Признаки включают среднее, тренд, вариативность, лаги и долю пропущенных наблюдений; пропуск периода не заменяется предыдущим доступным периодом.

Четыре стратегии прогнозирования:

1. **Фиксированная:** `p_fixed(i,t)=p_{k*}(i,t)`, где `k* = argmax_k AP_inner(k)`.
2. **Адаптивная:** `p_adaptive(i,t)=p_{k_i(θ*)}(i,t)`, `k_i(θ)=argmax_k Q_i(k;θ)` и `Q_i(k;θ)=ln(1+n_{ik})+α c_{ik}-β v_{ik}-γ(k-1)/4`; параметры `θ*` выбираются по внутренней валидации.
3. **Равномерный ансамбль:** `p_uniform = (1/K) Σ_k p_k`.
4. **Взвешенный ансамбль:** `p_weighted = Σ_k w_k p_k`, `w_k = exp(AP_inner(k)/T) / Σ_j exp(AP_inner(j)/T)`, `T=0.05` задан заранее.

Затем **выбирается стратегия** `m* = argmax_m AP_inner(m)` среди четырёх стратегий, при равенстве AP меньший Brier, затем лексикографическое имя. Оценка AP на внутренней проверке используется исключительно для подбора, а не в качестве беспристрастной оценки результата.

## 3. Вложенная временная валидация

- **Внутренние expanding-window разбиения:** train 4–5 → validation 6; train 4–6 → validation 7; train 4–7 → validation 8. Все три валидационных набора объединяются в единый out-of-fold вектор прогнозов. Внутренняя процедура выбирает `k*`, `θ*`, `w_k` и `m*` без доступа к меткам семестра 9.
- **Внешний эксперимент:** после выбора параметров модели для каждого `k` заново обучаются на семестрах 4–8 и формируют прогноз на **отложенный семестр 9**. Только в конце вычисляется AP, ROC-AUC и Brier, и только для анализа.
- В лабораторном протоколе есть одна последняя тестовая временная точка на каждую независимую синтетическую выборку. Это **повторённый nested holdout**, а не полный цикл вложенной k-fold CV в традиционном случайном разбиении.
- Каждый синтетический источник воспроизводится фиксированным `seed`. Данных реальных университетов нет.

## 4. Моделирование пропусков

Контроль без пропусков; MAR — пропуски отдельных измерений зависят от наблюдаемой посещаемости; MNAR — от латентного скрываемого значения; блочный сценарий — подряд идущие учебные периоды. Показатели пропусков рассчитываются относительно пяти исходных измерений. Измеренная доля может отличаться от номинальной 25%. Скрытие данных не меняет `student_id`, `term_index` и целевые метки.

## 5. Повторы и неопределённость

Шесть независимых генераций по `seed` на каждую комбинацию механизма и семейства. На каждой генерации все методы используют одинаковые данные и маски. Парная разница `Δ_s = AP_selected,s - AP_fixed,s`. Оценка эффекта — средняя `Δ_s`. Исследовательский 95%-интервал: 2000 бутстрэп-выборок из шести **целых генераций** с возвращением; 2,5-й и 97,5-й процентили средних разниц.

Это описательные интервалы **только для данного генератора данных**. Малое число генераций (n=6) и собственный механизм симуляции не позволяют переносить их на реальные популяции и не доказывают превосходства метода. Тестовую выборку нельзя использовать для выбора лучшего подхода в будущих изменениях проекта: потребуется новая заранее отложенная выборка.

## 6. Воспроизведение

```
python -m edurisk.nested_selection --output reports --n-students 160 --seeds 103,211,307,419,523,631
python make_report_v06.py
pytest -q
```

Артефакты: `nested_runs_synthetic.csv` (все тестовые метрики), `nested_selections_synthetic.csv` (решения, принятые на inner-OOF), `nested_paired_deltas_synthetic.csv` (парные разницы), `nested_summary_synthetic.csv/json` (сводка и бутстрэп-интервалы), `nested_missingness_synthetic.csv` (доли масок). Предыдущие эксперименты v0.2–v0.5 сохранены отдельно.
'''
(RESEARCH/'14_nested_strategy_protocol_v06.md').write_text(text_method,encoding='utf-8')

lines=['# EduRisk v0.6 — результаты повторённой вложенной временной проверки','','## Объём эксперимента',
      '',f'- Независимых синтетических генераций: **{R["n_seed_replicates"]}**.',
      f'- Сравнений механизм × модель × seed: **{R["n_experiments"]}**.',
      f'- Всего обучений базовых моделей на внутренних и внешних разбиениях: **{R["n_model_fits"]}**.',
      '- Сценарии: отсутствие пропусков, MAR, MNAR, подряд идущие пустые семестры.',
      '- Отложенный тест: 9-й семестр (только после выбора параметров на семестрах 4–8).',
      '', '## Сравнение с фиксированным окном', '',
      'Средняя AP на тесте и парная разница AP относительно фиксированного окна; интервалы 95% — бутстрэп по шести генерациям, а не по отдельным наблюдениям.',
      '', '| Механизм | Модель | AP фикс. | AP выбранной | Δ AP [2,5%; 97,5%] | Сколько из 6 улучшений |',
      '|---|---|---:|---:|---:|---:|']
for mechanism in mechanisms:
    for family in families:
        base=S[(S.mechanism==mechanism)&(S.family==family)&(S.method=='fixed_val')].iloc[0]
        chosen=S[(S.mechanism==mechanism)&(S.family==family)&(S.method=='selected_nested')].iloc[0]
        ci=f'{chosen.delta_ap_vs_fixed:+.4f} [{chosen.delta_ap_ci_low:+.4f}; {chosen.delta_ap_ci_high:+.4f}]'
        lines.append(f'| {mechanisms[mechanism]} | {families[family]} | {base.test_ap_mean:.4f} | {chosen.test_ap_mean:.4f} | {ci} | {int(chosen.paired_wins)}/6 |')

lines.extend(['','## Интерпретация','',
'Нельзя утверждать, что новый селектор доказанно превосходит фиксированное окно: эффекты меняют знак в зависимости от механизма пропусков и семейства модели. Отрицательные результаты важны и сохраняются без отбора «успешных» случаев.',
'', 'На полных данных селектор в среднем уступил фиксированному окну в обоих семействах. При пропущенных семестрах небольшой положительный средний эффект наблюдался у обеих моделей, но интервалы включают ноль. Для MNAR со случайным лесом отмечено среднее ухудшение; вывод относится исключительно к симуляции.',
'', '## Частота выбора стратегии по шести генерациям','',
'| Механизм | Модель | Фиксированное | Адаптивное | Равномерное | Взвешенное |',
'|---|---|---:|---:|---:|---:|'])
for mechanism in mechanisms:
 for family in families:
    counts={m:0 for m in methods if m!='selected_nested'}
    relevant=SELECTIONS[(SELECTIONS.mechanism==mechanism)&(SELECTIONS.family==family)]
    for meth,count in relevant.chosen_method.value_counts().items(): counts[meth]=int(count)
    lines.append(f'| {mechanisms[mechanism]} | {families[family]} | '+ ' | '.join(str(counts.get(m,0)) for m in ['fixed_val','adaptive_gate','uniform_ensemble','weighted_ensemble'])+' |')
lines.extend(['','## Что необходимо для кандидатской диссертации','',
'1. Повторить эксперименты на *разрешённых к исследовательскому использованию, юридически корректно обезличенных* реальных данных и проверить временные, институциональные и когортные сдвиги.',
'2. Увеличить число независимых генераций и проверить интервалы, устойчивость выбора, калибровку вероятностей и различия распространённости классов.',
'3. Сопоставить селектор с простыми стратегиями и мета-алгоритмами на заранее отложенной новой внешней выборке; исключить повторную подгонку под уже рассмотренные тесты.',
'4. Отдельно оценить справедливость и риски ошибочных прогнозов, особенно при неоднородной полноте образовательных данных.',
'', '## Ограничения','',
'Значения AP и интервалов получены на 160 синтетических студентах на seed, шести seed и одной тестовой временной точке. Эти результаты не являются доказательством причинности, научной новизны или пригодности к внедрению. Результаты v0.5 на других разбиениях нельзя напрямую вычитать из v0.6: корректные парные сравнения находятся внутри v0.6.',
'', 'Графики: `reports/nested_delta_ci_synthetic.png`, `reports/nested_strategy_counts_synthetic.png`. Результаты эксперимента: `reports/nested_*.csv/json`.'])
(RESEARCH/'15_nested_strategy_results_v06.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Report and figures generated')
