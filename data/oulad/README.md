# OULAD: данные для внешней проверки

`fixture/` — **полностью искусственно сгенерированные CSV с похожей схемой**. Имена файлов похожи на официальный OULAD, содержимое не имеет отношения к реальным студентам.

`raw/` — сюда пользователь помещает **оригинальные** семь файлов:

- `studentInfo.csv`
- `studentRegistration.csv`
- `assessments.csv`
- `studentAssessment.csv`
- `studentVle.csv`
- `courses.csv`
- `vle.csv`

Источник: https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset

**В ZIP релиза оригинальных CSV нет.** Запуск на `fixture/` без `--fixture` приведёт к неверной маркировке происхождения данных; не делайте так. Для демонстрации используйте `--fixture`, для оригинала — `--raw data/oulad/raw` без этого флага.
