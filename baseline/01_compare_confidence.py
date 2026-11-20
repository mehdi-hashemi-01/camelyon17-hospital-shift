import csv
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parents[1]
with (root / "baseline/data/toy_predictions.csv").open() as file:
    rows = list(csv.DictReader(file))

with (root / "results/toy_summary.csv").open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["hospital", "model", "correct", "total", "accuracy", "confidence", "gap"])
    for hospital in ["source", "unseen"]:
        samples = [row for row in rows if row["hospital"] == hospital]
        for model in ["a", "b"]:
            correct = 0
            confidence = []
            for row in samples:
                p1 = float(row[f"p1_model_{model}"])
                prediction = int(p1 >= 0.5)
                correct += int(prediction == int(row["label"]))
                confidence.append(max(p1, 1 - p1))
            accuracy = correct / len(samples)
            average = mean(confidence)
            gap = round(average - accuracy, 4)
            writer.writerow([hospital, model, correct, len(samples), accuracy, average, gap])
            print(hospital, model, f"accuracy={accuracy:.0%} confidence={average:.0%} gap={gap:+.2f}")
