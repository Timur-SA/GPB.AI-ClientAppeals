
import pandas as pd
from pathlib import Path
import seaborn as sns

# Находим главную папку проекта
project_dir = Path(__file__).resolve().parent.parent

# Указываем путь к нашему датасету
file_path = project_dir / "data" / "Bank_support_dataset.csv"

# Загружаем таблицу
df = pd.read_csv(file_path)

# Смотрим размер таблицы
print("Размер датасета:")
print(df.shape)

# Смотрим названия всех столбцов
print("\nНазвания столбцов:")
print(df.columns.tolist())

# Выводим первые 5 строк таблицы
print("\nПервые 5 обращений:")
print(df.head().to_string())



# Проверяем пропущенные значения
print("\nПропущенные значения в каждом столбце:")
print(df.isnull().sum())

# Считаем размеченные обращения
labeled = df[df["label_source"] == "human"]

print("\nКоличество размеченных обращений:")
print(len(labeled))

# Считаем обращения без разметки
unlabeled = df[df["label_source"].isnull()]

print("\nКоличество обращений без разметки:")
print(len(unlabeled))

# Анализ распределения четырёх таргетов


targets = [
    "category",
    "subcategory",
    "complexity"
]


for target in targets:
    print("\n" + "=" * 40)
    print("Распределение:", target)
    print("=" * 40)

    counts = labeled[target].value_counts()

    print(counts)

    print("\nКоличество уникальных классов:")
    print(labeled[target].nunique())

    print("\nРедкие классы (меньше 15 примеров):")
    rare = counts[counts < 15]

    if rare.empty:
        print("Редких классов нет")
    else:
        print(rare)

import matplotlib.pyplot as plt

# Считаем количество обращений по категориям
category_counts = labeled["category"].value_counts()

# Создаём график
plt.figure(figsize=(12, 6))

category_counts.plot(kind="bar")

# Добавляем заголовок и подписи
plt.title("Распределение обращений по категориям")
plt.xlabel("Категория")
plt.ylabel("Количество обращений")

# Поворачиваем подписи
plt.xticks(rotation=45, ha="right")

# Делаем расположение элементов удобным
plt.tight_layout()

# Сохраняем график в папку results
plt.savefig(project_dir / "results" / "category_distribution.png")

# Показываем график
plt.show()

# Считаем обращения по сложности
complexity_counts = labeled["complexity"].value_counts()

# Создаём график
plt.figure(figsize=(8, 5))

# Рисуем столбцы
complexity_counts.plot(kind="bar")

# Подписываем график
plt.title("Распределение обращений по сложности")
plt.xlabel("Сложность обращения")
plt.ylabel("Количество обращений")

# Делаем подписи горизонтальными
plt.xticks(rotation=0)

# Настраиваем расположение
plt.tight_layout()

# Сохраняем картинку
plt.savefig(project_dir / "results" / "complexity_distribution.png")

# Показываем график
plt.show()

# Считаем обращения по приоритету
priority_counts = labeled["priority"].value_counts()

# Создаём график
plt.figure(figsize=(8, 5))

# Рисуем столбцы
priority_counts.plot(kind="bar")

# Добавляем заголовок и подписи
plt.title("Распределение обращений по приоритету")
plt.xlabel("Приоритет")
plt.ylabel("Количество обращений")

# Делаем подписи горизонтальными
plt.xticks(rotation=0)

# Настраиваем расположение
plt.tight_layout()

# Сохраняем график
plt.savefig(project_dir / "results" / "priority_distribution.png")

# Показываем график
plt.show()

# Считаем обращения по подкатегориям
subcategory_counts = labeled["subcategory"].value_counts()

# Создаём график
plt.figure(figsize=(12, 9))

# Рисуем горизонтальные столбцы
subcategory_counts.sort_values().plot(kind="barh")

# Добавляем заголовок и подписи
plt.title("Распределение обращений по подкатегориям")
plt.xlabel("Количество обращений")
plt.ylabel("Подкатегория")

# Настраиваем расположение
plt.tight_layout()

# Сохраняем график
plt.savefig(project_dir / "results" / "subcategory_distribution.png")

# Показываем график
plt.show()


# Проверка дубликатов

print("\nПРОВЕРКА ДУБЛИКАТОВ")

# Проверяем повторяющиеся номера обращений
id_duplicates = df["request_id"].duplicated().sum()

print("Повторяющиеся ID:", id_duplicates)

# Проверяем полностью одинаковые строки
full_duplicates = df.duplicated().sum()

print("Полностью одинаковые строки:", full_duplicates)

# Проверяем одинаковые сообщения клиентов
text_duplicates = df["text"].duplicated().sum()

print("Повторяющиеся тексты:", text_duplicates)


# Проверка противоречий в разметке

print("\nПРОВЕРКА ПРОТИВОРЕЧИЙ")

targets = [
    "category",
    "subcategory",
    "complexity",
    "priority"
]

for target in targets:

    # Считаем разные метки для одинаковых текстов
    unique_labels = labeled.groupby("text")[target].nunique()

    # Находим тексты с несколькими разными метками
    conflicts = unique_labels[unique_labels > 1]

    print("\nСтолбец:", target)
    print("Текстов с противоречивыми метками:", len(conflicts))


print("\nАНАЛИЗ УНИКАЛЬНЫХ ТЕКСТОВ")

print("Уникальных текстов всего:")
print(df["text"].nunique())

print("\nУникальных размеченных текстов:")
print(labeled["text"].nunique())

print("\nПовторов среди размеченных обращений:")
print(labeled["text"].duplicated().sum())

print("\nРаспределение подкатегорий без повторов:")
print(labeled.drop_duplicates(subset="text")["subcategory"].value_counts())

print("\nПРОВЕРКА СЕГМЕНТОВ КЛИЕНТОВ")

print("\nСегменты до удаления повторов:")
print(labeled["client_segment"].value_counts())

print("\nСегменты после удаления повторов:")
print(
    labeled.drop_duplicates(subset="text")
    ["client_segment"].value_counts()
)

print("\nКоличество уникальных текстов, "
      "встречающихся в разных сегментах:")

print(
    (labeled.groupby("text")["client_segment"]
     .nunique() > 1).sum()
)
print("\nОЧИСТКА ДАТАСЕТА")

cleaned = labeled.drop_duplicates(
    subset="text"
).copy()

print("Размер очищенного датасета:")
print(cleaned.shape)

print("\nКоличество оставшихся повторов:")
print(cleaned["text"].duplicated().sum())

cleaned.to_csv(
    "../data/Bank_support_cleaned.csv",
    index=False
)

print("\nОчищенный датасет сохранён")


# Круговая диаграмма сегментов клиентов
plt.figure(figsize=(8, 8))

cleaned["client_segment"].value_counts().plot(
    kind="pie", #круговая диаграмма
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Распределение по сегментам клиентов")
plt.ylabel("")
plt.tight_layout()

plt.savefig(
    "../results/client_segment_pie.png",
    dpi=300
)

plt.close()

print("Пайчарт сегментов клиентов сохранён")
categorical_columns = [
    "category",
    "subcategory",
    "complexity",
    "priority",
    "channel",
    "resolution_type",
    "has_pii_masked"
]

for column in categorical_columns:
    plt.figure(figsize=(10, 10))

    cleaned[column].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title(f"Распределение: {column}")
    plt.ylabel("")
    plt.tight_layout()

    plt.savefig(
        f"../results/{column}_pie.png",
        dpi=300
    )

    plt.close()

    print(f"Пайчарт {column} сохранён")
plt.figure(figsize=(12, 6))

sns.barplot(
    data=cleaned,
    x="category",
    y="request_char_len",
    errorbar=None
)

plt.title("Средняя длина обращений по категориям")
plt.xlabel("Категория")
plt.ylabel("Количество символов")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "../results/text_length_barplot.png",
    dpi=300
)

plt.close()

print("Барплот длины текста сохранён")
plt.figure(figsize=(10, 6))

sns.histplot(
    data=cleaned,
    x="request_char_len",
    bins=30,
    kde=True
)

plt.title("Распределение длины обращений")
plt.xlabel("Количество символов")
plt.ylabel("Количество обращений")
plt.tight_layout()

plt.savefig(
    "../results/text_length_histogram.png",
    dpi=300
)

plt.close()

print("Гистограмма длины текста сохранена")
print("\nАНАЛИЗ РЕДКИХ КЛАССОВ")

for column in ["category", "subcategory",
               "complexity", "priority"]:

    counts = cleaned[column].value_counts()
    rare = counts[counts < 15]

    print(f"\nРедкие классы: {column}")
    print(rare)

a = labeled["subcategory"].value_counts()
b = cleaned["subcategory"].value_counts()

comparison = pd.DataFrame({
    "До очистки": a,
    "После очистки": b
}).fillna(0)

comparison.plot(
    kind="bar",
    figsize=(15, 7)
)

plt.title("Распределение подкатегорий до и после очистки")
plt.xlabel("Подкатегория")
plt.ylabel("Количество обращений")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    "../results/subcategory_comparison.png",
    dpi=300
)

plt.close()

print("Сравнительный график сохранён")
print("\nПроверка контекста одинаковых текстов")

for c in ["client_segment", "active_products",
          "channel", "resolution_type"]:
    a = labeled.groupby("text")[c].nunique()
    print(c, ":", (a > 1).sum())
print("\nОчистка с учётом контекста")

new = labeled.drop_duplicates(
    subset=["text", "client_segment",
            "active_products", "channel",
            "resolution_type"]
).copy()

print("До очистки:", len(labeled))
print("После очистки:", len(new))
print("Удалено повторов:", len(labeled) - len(new))
print("Уникальных текстов:", new["text"].nunique())
print("\nРедкие классы после новой очистки")

for c in ["category", "subcategory",
          "complexity", "priority",
          "resolution_type"]:
    a = new[c].value_counts()
    print("\n", c)
    print(a[a < 15])
print("\nСтроки и уникальные тексты по подкатегориям")

a = new.groupby("subcategory")["text"].agg(
    ["count", "nunique"]
)

a.columns = ["Всего строк", "Уникальных текстов"]

print(a.sort_values("Уникальных текстов"))
new.to_csv(
    "../data/Bank_support_context_cleaned.csv",
    index=False
)

print("Новый датасет сохранён")
print("Размер:", new.shape)