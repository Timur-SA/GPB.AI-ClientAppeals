import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt
p = Path(__file__).resolve().parent.parent

df = pd.read_csv(p / "data" / "Bank_support_dataset.csv")
df = df[df["label_source"] == "human"].copy()

# Проверяем различия CRM-контекста
cols = ["client_segment", "active_products",
        "channel", "resolution_type"]

for c in cols:
    n = df.groupby("text")[c].nunique()
    print(c, ":", (n > 1).sum())

# Разделяем уникальные тексты
u = df["text"].unique()

tr, temp = train_test_split(
    u, test_size=0.3, random_state=42
)

v, te = train_test_split(
    temp, test_size=0.5, random_state=42
)

# Возвращаем все контекстные варианты
train = df[df["text"].isin(tr)]
val = df[df["text"].isin(v)]
test = df[df["text"].isin(te)]

print("\nРазмеры выборок:")

for name, x in [("Train", train),
                ("Val", val),
                ("Test", test)]:
    print(name, len(x),
          "Уникальных:", x["text"].nunique())

print("\nПересечения:")
print("Train / Val:", len(set(tr) & set(v)))
print("Train / Test:", len(set(tr) & set(te)))
print("Val / Test:", len(set(v) & set(te)))
print("\nРаспределение подкатегорий:")

r = pd.DataFrame({
    "Train": train.groupby("subcategory")["text"].nunique(),
    "Val": val.groupby("subcategory")["text"].nunique(),
    "Test": test.groupby("subcategory")["text"].nunique()
})

r = r.fillna(0).astype(int)
print(r)

print("\nОтсутствующие классы:")

for c in r.columns:
    print(c, r.index[r[c] == 0].tolist())
# Сохранение выборок
for name, x in [("train", train),
                ("val", val),
                ("test", test)]:

    # Исходный поток с повторами
    x.to_csv(
        p / "data" / f"grouped_{name}_full.csv",
        index=False
    )

    # Уникальные тексты
    x.drop_duplicates("text").to_csv(
        p / "data" / f"grouped_{name}_unique.csv",
        index=False
    )

    # С сохранением различий CRM-контекста
    x.drop_duplicates(
        subset=["text"] + cols
    ).to_csv(
        p / "data" / f"grouped_{name}_context.csv",
        index=False
    )

r.to_csv(p / "results" / "grouped_distribution.csv")

print("\nВсе выборки сохранены")
# Проверка контекстных вариантов
print("\nКонтекстные выборки:")

for name, x in [("Train", train),
                ("Val", val),
                ("Test", test)]:

    y = x.drop_duplicates(subset=["text"] + cols)

    print(name, len(y))
    print("Удалено:", len(x) - len(y))

# Проверка дополнительных CRM-признаков
g = df.groupby(["text"] + cols)

print("\nДополнительный CRM-контекст:")

for c in ["client_id", "tenure_months"]:
    print(c, (g[c].nunique() > 1).sum())
cols = ["text", "client_segment", "active_products",
        "channel", "resolution_type", "client_id",
        "tenure_months"]

print("\nИсправленные контекстные выборки:")

for name, x in [("train", train),
                ("val", val),
                ("test", test)]:

    y = x.drop_duplicates(subset=cols)

    y.to_csv(
        p / "data" / f"grouped_{name}_context.csv",
        index=False
    )

    print(name, len(y))
    print("Удалено:", len(x) - len(y))
def template(t):
    t = t.lower()
    t = re.sub(r"\*{2,}\s*\d{4}\b", "CARD", t)
    t = re.sub(r"(?:\+7|8)[ -]?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{2}[ -]?\d{2}\b", "PHONE", t)
    t = re.sub(r"[\w.+-]+@[\w.-]+\.\w+", "EMAIL", t)
    t = re.sub(r"\b\d{2}\.\d{2}\.\d{4}\b", "DATE", t)
    t = re.sub(r"\d+(?:[ \u00a0]\d{3})*(?:[.,]\d+)?\s*₽", "SUM", t)
    t = re.sub(r"\b\d{5}CARD\b", "ACCOUNT", t)
    return t

a = set(train["text"].apply(template))
b = set(val["text"].apply(template))
c = set(test["text"].apply(template))

print("\nКоличество шаблонов:")
print("Train:", len(a))
print("Val:", len(b))
print("Test:", len(c))

print("\nПересечения шаблонов:")
print("Train / Val:", len(a & b))
print("Train / Test:", len(a & c))
print("Val / Test:", len(b & c))
df["template_text"] = df["text"].apply(template)

u = df["template_text"].unique()

tr, temp = train_test_split(
    u, test_size=0.3, random_state=42
)

v, te = train_test_split(
    temp, test_size=0.5, random_state=42
)

print("\nРазделение по шаблонам:")

for name, t in [("train", tr),
                ("val", v),
                ("test", te)]:

    x = df[df["template_text"].isin(t)]

    print(name, len(x),
          "Шаблонов:", len(t),
          "Классов:", x["subcategory"].nunique())

    x.drop(columns="template_text").to_csv(
        p / "data" / f"template_{name}_full.csv",
        index=False
    )

print("\nПересечения:")
print("Train / Val:", len(set(tr) & set(v)))
print("Train / Test:", len(set(tr) & set(te)))
print("Val / Test:", len(set(v) & set(te)))
r = pd.DataFrame({
    "Exact text split": [21, 24, 21],
    "Template split": [0, 0, 0]
}, index=["Train / Val", "Train / Test", "Val / Test"])

ax = r.plot(kind="bar", color=["tomato", "seagreen"])

for c in ax.containers:
    ax.bar_label(c)

plt.title("Template leakage comparison")
plt.ylabel("Overlapping templates")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    p / "results" / "split_leakage_comparison.png",
    dpi=300
)

plt.show()

r.to_csv(p / "results" / "split_leakage_comparison.csv")
print("\nПересечения клиентов:")

a = set(df[df["template_text"].isin(tr)]["client_id"])
b = set(df[df["template_text"].isin(v)]["client_id"])
c = set(df[df["template_text"].isin(te)]["client_id"])

print("Train / Val:", len(a & b))
print("Train / Test:", len(a & c))
print("Val / Test:", len(b & c))
tr, v, te = set(tr), set(v), set(te)

while True:
    a = set(df[df["template_text"].isin(tr)]["client_id"])
    b = set(df[df["template_text"].isin(v)]["client_id"])
    c = set(df[df["template_text"].isin(te)]["client_id"])

    common = (a & b) | (a & c) | (b & c)

    if not common:
        break

    t = set(df[df["client_id"].isin(common)]["template_text"])

    tr.update(t)
    v.difference_update(t)
    te.difference_update(t)

print("\nИтоговые выборки:")

for name, t in [("train", tr),
                ("val", v),
                ("test", te)]:

    x = df[df["template_text"].isin(t)]

    print(name, len(x),
          "Шаблонов:", len(t),
          "Классов:", x["subcategory"].nunique())

    x.drop(columns="template_text").to_csv(
        p / "data" / f"template_{name}_full.csv",
        index=False
    )

print("\nПересечения клиентов:")
print("Train / Val:", len(a & b))
print("Train / Test:", len(a & c))
print("Val / Test:", len(b & c))

print("\nПересечения шаблонов:")
print("Train / Val:", len(tr & v))
print("Train / Test:", len(tr & te))
print("Val / Test:", len(v & te))
for name, t in [("train", tr),
                ("val", v),
                ("test", te)]:

    x = df[df["template_text"].isin(t)]
    x = x.drop_duplicates("text")
    x = x.drop(columns="template_text")

    x.to_csv(
        p / "data" / f"template_{name}_unique.csv",
        index=False
    )

    print(name, "Уникальных текстов:", len(x))

print("Готово!")