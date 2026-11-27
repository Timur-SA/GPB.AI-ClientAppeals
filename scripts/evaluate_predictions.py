import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

p = Path(__file__).resolve().parent.parent

train = pd.read_csv(
    p / "data" / "template_train_unique.csv"
)

val = pd.read_csv(
    p / "data" / "template_val_unique.csv"
)

test = pd.read_csv(
    p / "data" / "template_test_unique.csv"
)

print("Train:", len(train))
print("Val:", len(val))
print("Test:", len(test))

# Проверка пересечений текстов
assert not set(train["text"]) & set(val["text"])
assert not set(train["text"]) & set(test["text"])
assert not set(val["text"]) & set(test["text"])

print("Пересечения отсутствуют")

# Определение редких категорий по Train
r = train["category"].value_counts()

print("\nРаспределение категорий в Train:")
print(r)

print("\nРедкие категории:")
print(r[r < 15])
def evaluate(data, predictions):
    # Сопоставляем ответы по request_id
    x = data[["request_id", "category"]].merge(
        predictions[["request_id", "category"]],
        on="request_id",
        suffixes=("_true", "_pred"),
        validate="one_to_one"
    )

    assert len(x) == len(data) == len(predictions)
    assert not x.isna().any().any()

    a = x["category_true"]
    b = x["category_pred"]

    labels = sorted(a.unique())

    print("\nAccuracy:", round(accuracy_score(a, b), 4))

    print("Macro-F1:", round(f1_score(
        a, b, labels=labels,
        average="macro", zero_division=0
    ), 4))

    report = classification_report(
        a, b, labels=labels,
        output_dict=True, zero_division=0
    )

    report = pd.DataFrame(report).T

    print("\nМетрики по категориям:")
    print(report)

    print("\nКачество редких категорий:")
    rare = r[r < 15].index
    print(report.reindex(rare))

    return report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

p = Path(__file__).resolve().parent.parent

# 1. Загрузка выборок

train = pd.read_csv(p / "data" / "template_train_unique.csv")
val = pd.read_csv(p / "data" / "template_val_unique.csv")
test = pd.read_csv(p / "data" / "template_test_unique.csv")

print("Train:", len(train))
print("Val:", len(val))
print("Test:", len(test))

# 2. Проверка пересечений

assert not set(train["text"]) & set(val["text"])
assert not set(train["text"]) & set(test["text"])
assert not set(val["text"]) & set(test["text"])

print("Пересечения текстов отсутствуют")

# 3. Определение редких категорий

r = train["category"].value_counts()
rare = r[r < 15].index.tolist()

print("\nРаспределение категорий:")
print(r)

print("\nРедкие категории:", rare)


# 4. Функция оценки модели

def evaluate(data, predictions, name):

    x = data[["request_id", "category"]].merge(
        predictions[["request_id", "category"]],
        on="request_id",
        suffixes=("_true", "_pred"),
        validate="one_to_one"
    )

    assert len(x) == len(data) == len(predictions)
    assert not x.isna().any().any()

    a = x["category_true"]
    b = x["category_pred"]

    assert set(b).issubset(set(r.index))

    labels = sorted(set(a) | set(b))
    macro_labels = sorted(a.unique())

    # Основные метрики

    m = {
        "Accuracy": accuracy_score(a, b),

        "Macro-Precision": precision_score(
            a, b, labels=labels,
            average="macro", zero_division=0
        ),

        "Macro-Recall": recall_score(
            a, b, labels=labels,
            average="macro", zero_division=0
        ),

        "Macro-F1": f1_score(
            a, b, labels=labels,
            average="macro", zero_division=0
        )
    }

    metrics = pd.DataFrame(
        list(m.items()),
        columns=["metric", "value"]
    )

    print("\nОсновные метрики:")
    print(metrics.round(4))

    metrics.to_csv(
        p / "results" / f"{name}_metrics.csv",
        index=False
    )

    # 5. Метрики по категориям

    report = classification_report(
        a, b,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    report = pd.DataFrame(report).T

    print("\nМетрики по категориям:")
    print(report.round(4))

    report.to_csv(
        p / "results" / f"{name}_report.csv"
    )

    # 6. Оценка редких категорий

    rr = report.reindex(rare).copy()

    rr["train_support"] = r.reindex(rare)
    rr["support"] = a.value_counts().reindex(
        rare, fill_value=0
    )

    rr.loc[
        rr["support"] == 0,
        ["precision", "recall", "f1-score"]
    ] = float("nan")

    print("\nМетрики редких категорий:")
    print(rr.round(4))

    rr.to_csv(
        p / "results" / f"{name}_rare_classes.csv"
    )

    # 7. Матрица ошибок

    cm = confusion_matrix(a, b, labels=labels)

    cm = pd.DataFrame(
        cm, index=labels, columns=labels
    )

    cm.to_csv(
        p / "results" / f"{name}_confusion_matrix.csv"
    )

    plt.figure(figsize=(11, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted category")
    plt.ylabel("True category")
    plt.tight_layout()

    plt.savefig(
        p / "results" / f"{name}_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # 8. График основных метрик

    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        metrics["metric"],
        metrics["value"],
        color="steelblue"
    )

    plt.bar_label(bars, fmt="%.3f")
    plt.ylim(0, 1.1)
    plt.title(f"Classification metrics: {name}")
    plt.ylabel("Score")
    plt.tight_layout()

    plt.savefig(
        p / "results" / f"{name}_metrics.png",
        dpi=300
    )

    plt.close()

    # 9. График F1 по категориям

    q = report.loc[labels, "f1-score"]

    plt.figure(figsize=(10, 6))

    bars = plt.barh(
        q.index,
        q.values,
        color="seagreen"
    )

    plt.bar_label(bars, fmt="%.3f")
    plt.xlim(0, 1.15)
    plt.title(f"F1-score by category: {name}")
    plt.tight_layout()

    plt.savefig(
        p / "results" / f"{name}_category_f1.png",
        dpi=300
    )

    plt.close()

    return metrics, report


# 10. Демонстрационная проверка стенда

if __name__ == "__main__":

    pred = val[["request_id", "category"]].copy()

    pred.loc[
        pred.index[::5], "category"
    ] = r.idxmax()

    print("\nДЕМОНСТРАЦИОННАЯ ОЦЕНКА")
    print("Не является результатом реальной модели")

    evaluate(val, pred, "demo_val")

    print("\nВсе метрики и графики сохранены")
    print("Стенд оценки успешно отработал")
import re


def template(t):
    t = t.lower()
    t = re.sub(r"\*{2,}\s*\d{4}\b", "CARD", t)
    t = re.sub(r"(?:\+7|8)[ -]?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{2}[ -]?\d{2}\b", "PHONE", t)
    t = re.sub(r"[\w.+-]+@[\w.-]+\.\w+", "EMAIL", t)
    t = re.sub(r"\b\d{2}\.\d{2}\.\d{4}\b", "DATE", t)
    t = re.sub(r"\d+(?:[ \u00a0]\d{3})*(?:[.,]\d+)?\s*₽", "SUM", t)
    t = re.sub(r"\b\d{5}CARD\b", "ACCOUNT", t)
    return t


def evaluate_templates(data, predictions, name):

    x = data.copy()
    x["template_text"] = x["text"].apply(template)

    x = x.sort_values("request_id")
    x = x.drop_duplicates("template_text")

    y = predictions[
        predictions["request_id"].isin(x["request_id"])
    ]

    print("\nОценка по уникальным шаблонам")
    print("Количество шаблонов:", len(x))

    return evaluate(x, y, name)


if __name__ == "__main__":

    evaluate_templates(
        val, pred, "demo_template_val"
    )

    print("\nОценка по шаблонам завершена")