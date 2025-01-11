import json
import re
import sys


MODELS = {
    "dummy": "Dummy",
    "knn": "kNN",
    "svm": "SVM",
    "log_reg": "Logistic Regression",
    "dec_tree": "Decision Tree",
    "rnd_forest": "Random Forest",
    "adaboost": "AdaBoost Decision Tree",
    "stacked": "Stacked kNN, SVM, LogReg, DecTree",
}

PROBLEMS = {
    "all vs. all": "All features, 3 classes",
    "all vs. dropout": "All features, dropout",
    "all vs. did_not_graduate": "All features, did not graduate in time",
    "early vs. all": "Early features, 3 classes",
    "early vs. dropout": "Early features, dropout",
    "early vs. did_not_graduate": "Early features, did not graduate in time",
    "highlights vs. all": "EDA highlights, 3 classes",
    "highlights vs. dropout": "EDA highlights, dropout",
    "highlights vs. did_not_graduate": "EDA highlights, did not graduate in time",
    "highlights_early vs. all": "Early EDA highlights, 3 classes",
    "highlights_early vs. dropout": "Early EDA highlights, dropout",
    "highlights_early vs. did_not_graduate": "Early EDA highlights, did not graduate in time",
}

CVS = {
    "kfcv": "5-fold CV",
    "skfcv": "stratified 5-fold CV",
}

CVS_SHORT = {
    "kfcv": "",
    "skfcv": "strf",
}

TUNINGS = {
    "untuned": "",
    "tuned": "tuned",
}


def main(argv):
    best_n = None

    try:
        best_n = int(argv[1])
    except:
        best_n = 5

    results, best_early, best_all = collect_results()

    print(f"#### The best results by mean F1 score")
    print("""
Each problem formulation and model pair is listed with the best score from its
4 variants.
""")

    print_best_models(results, best_all[:best_n], "All")
    print_best_models(results, best_early[:best_n], "Early")

    print("""
The cells corresponding to the top 3 problem formulation and model pairs are
typeset in bold in the following tables, separately for all features and for
early features.
""")

    best = {
        (problem, model, cv, tuning)
        for f1, problem, model, cv, tuning in (best_early[:3] + best_all[:3])
    }

    print_score_table(results, 0, "F1", best)
    print_score_table(results, 1, "Precision", best)
    print_score_table(results, 2, "Recall", best)

    return 0


def collect_results():
    model_re = re.compile(r"^(.*) evaluation:$")
    problem_re = re.compile(r"^ +(.* vs\. .*) with (.*)$")
    result_re = re.compile(
        r"^ +(tuned )?([a-zA-Z]+) +F1: .* / ([0-9]+\.[0-9]+) +P: .* / ([0-9]+\.[0-9]+) +R: .* / ([0-9]+\.[0-9]+)$"
    )

    model = None
    problem = None
    cv = None

    results = {}
    early_ftr_scores = {}
    all_ftr_scores = {}

    for line in collect_outputs():
        line = line.rstrip()

        if m := model_re.match(line):
            model = m[1]
        elif m := problem_re.match(line):
            problem = m[1]
            cv = m[2]
        elif m := result_re.match(line):
            is_tuned = bool(m[1].strip()) if m[1] else False
            tuning = "tuned" if is_tuned else "untuned"
            result_type = m[2]
            f1 = float(m[3])
            prec = float(m[4])
            rec = float(m[5])

            if model is not None and problem is not None:
                (
                    results
                        .setdefault(cv, {})
                        .setdefault(problem, {})
                        .setdefault(tuning, {})
                        .setdefault(model, {})
                )[result_type] = (f1, prec, rec)

                if result_type == "Mean":
                    scores = early_ftr_scores if "early vs." in problem else all_ftr_scores
                    scores_key = (problem, model)
                    scores_entry = (f1, cv, tuning)

                    if scores_key not in scores or scores[scores_key][0] < f1:
                        scores[scores_key] = scores_entry

    best_early = sort_scores(early_ftr_scores)
    best_all = sort_scores(all_ftr_scores)

    return results, best_early, best_all


def collect_outputs():
    with open("math-modeling-practice.ipynb", "r") as f:
        notebook = json.load(f)

        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue

            for output in cell["outputs"]:
                if "name" in output and output["name"] == "stdout":
                    yield from output["text"]


def sort_scores(scores):
    return sorted(
        [
            (f1, problem, model, cv, tuning)
            for (problem, model), (f1, cv, tuning) in scores.items()
        ],
        key=lambda e: e[0],
        reverse=True
    )


def print_best_models(results, best_models, title):
    print(f"##### {title} features")
    print("")

    for idx, (f1, problem, model, cv, tuning) in enumerate(best_models):
        model_str = MODELS[model]
        problem_str = PROBLEMS[problem]
        cv_str = CVS[cv]
        tuning_str = TUNINGS[tuning]

        comments = [cv_str]

        if tuning_str:
            comments.append(tuning_str)

        comments = "" if not comments else (" (" + ", ".join(comments) + ")")

        print(f"{idx + 1}. **{model_str}** on the \"*{problem_str}*\" problem{comments}:")

        for score_idx, score_name in ((0, "F1"), (1, "Precision"), (2, "Recall")):
            score_min = results[cv][problem][tuning][model]["Min"][score_idx]
            score_mean = results[cv][problem][tuning][model]["Mean"][score_idx]
            score_max = results[cv][problem][tuning][model]["Max"][score_idx]
            print(
                f"    * {score_name} (min, mean, max): {score_min}, {score_mean}, {score_max}."
            )

    print("")


def print_score_table(results, score_idx, title, best):
    print(f"#### {title}")
    print("")

    model_order = [
        "dummy",
        "knn",
        "svm",
        "log_reg",
        "dec_tree",
        "rnd_forest",
        "adaboost",
        "stacked",
    ]
    problem_order = [
        "all vs. all",
        "all vs. dropout",
        "all vs. did_not_graduate",
        "early vs. all",
        "early vs. dropout",
        "early vs. did_not_graduate",
        "highlights vs. all",
        "highlights vs. dropout",
        "highlights vs. did_not_graduate",
        "highlights_early vs. all",
        "highlights_early vs. dropout",
        "highlights_early vs. did_not_graduate",
    ]
    width = 24

    print_header(["Problem&nbsp;↓&nbsp;/&nbsp;Model&nbsp;→"] + [MODELS[col] for col in model_order])
    print_header([":-----"] + [(":-----:") for col in model_order])

    for problem in problem_order:
        for tuning in ("untuned", "tuned"):
            for cv in ("kfcv", "skfcv"):
                cv_str = CVS_SHORT[cv]
                tuning_str = TUNINGS[tuning]
                comments = []

                if cv_str:
                    comments.append(cv_str)

                if tuning_str:
                    comments.append(tuning_str)

                comments = "" if not comments else (" (" + ", ".join(comments) + ")")
                problem_str = PROBLEMS[problem] + comments

                row = [problem_str]

                for model in model_order:
                    if model not in results[cv][problem][tuning]:
                        row.append("")

                        continue

                    score_min = results[cv][problem][tuning][model]["Min"][score_idx]
                    score_mean = results[cv][problem][tuning][model]["Mean"][score_idx]
                    score_max = results[cv][problem][tuning][model]["Max"][score_idx]
                    stats = f"{score_min:.2f},&nbsp;{score_mean:.2f},&nbsp;{score_max:.2f}"

                    if (problem, model, cv, tuning) in best:
                        stats = f"**{stats}**"

                    row.append(stats)

                print_row(row)

        print_row([""] * len(row))
        print_row([""] * len(row))

    print("")


def print_header(row):
    print(f"| {row[0]} | " + " | ".join(f"{cell}" for cell in row[1:]) + " |")


def print_row(row):
    print(f"| {row[0]} | " + " | ".join(f"{cell}" for cell in row[1:]) + " |")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
