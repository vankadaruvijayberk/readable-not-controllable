"""Summarize the intervention outputs: per-combination effects, paired tests, and the pooled estimates of Table 5.

Reads <interventions>/<model>__<dataset>.parquet and writes analysis.json and summary.csv to <out>.
A combination is one model on one dataset. "top" is the selected neuron set, "random" the control set.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from actground.config import add_config_arg, load_config
from actground.stats import bootstrap_diff_ci, bootstrap_mean_ci, holm, mcnemar_p


class Analyzer:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng

    def hall_rate(self, df):
        """(rate, ci low, ci high, n) over the examples with a parseable answer."""
        labels = df["hallucinated"].dropna().astype(float).values
        return (*bootstrap_mean_ci(labels, self.cfg["n_bootstrap"], self.rng), len(labels))

    def correct_logprob(self, df):
        values = df["ca_logprob"].replace([np.inf, -np.inf], np.nan).dropna().values
        return bootstrap_mean_ci(values, self.cfg["n_bootstrap"], self.rng)

    def dose_rows(self, df, baseline, base_rate, base_logprob):
        rows = []
        for alpha in sorted(df["alpha"].dropna().unique()):
            cond = df[df["alpha"] == alpha]
            rate, low, high, n = self.hall_rate(cond)
            logprob, lp_low, lp_high = self.correct_logprob(cond)
            direction = "suppress" if alpha < 1 else "amplify" if alpha > 1 else "sham"
            intended, opposite, n_paired = (
                (np.nan, np.nan, 0) if direction == "sham" else reliability(cond, baseline, direction)
            )
            rows.append(
                {
                    "alpha": float(alpha),
                    "hall_rate": rate,
                    "hr_ci": [low, high],
                    "n": n,
                    "d_hall_rate": rate - base_rate,
                    "ca_logprob": logprob,
                    "ca_logprob_ci": [lp_low, lp_high],
                    "d_ca_logprob": logprob - base_logprob,
                    "parse_rate": parse_rate(cond),
                    "degeneration": degeneration_rate(cond),
                    "reliability": intended,
                    "anti_steerable": opposite,
                    "reliability_n": n_paired,
                    "direction": direction,
                }
            )
        return rows

    def dose_response(self, df, baseline, base_rate, base_logprob):
        dose = df[df["experiment"] == "dose_scale"]
        top, random = dose[dose["set_type"] == "top"], dose[dose["set_type"] == "random"]
        top_rows = self.dose_rows(top, baseline, base_rate, base_logprob)
        random_rows = self.dose_rows(random, baseline, base_rate, base_logprob)

        differences = []
        for alpha in sorted(set(top["alpha"].dropna()) & set(random["alpha"].dropna())):
            t, r = top[top["alpha"] == alpha], random[random["alpha"] == alpha]
            diff, low, high = bootstrap_diff_ci(
                t["hallucinated"].dropna().astype(float).values,
                r["hallucinated"].dropna().astype(float).values,
                self.cfg["n_bootstrap"],
                self.rng,
            )
            b, c, n_paired = discordant_pairs(t, r)
            differences.append(
                {
                    "alpha": float(alpha),
                    "top_minus_random": diff,
                    "ci": [low, high],
                    "mcnemar_b": b,
                    "mcnemar_c": c,
                    "mcnemar_p": mcnemar_p(b, c),
                    "n_paired": n_paired,
                }
            )

        hall_rho, hall_p = trend(top_rows, "alpha", "hall_rate")
        logprob_rho, logprob_p = trend(top_rows, "alpha", "ca_logprob")
        control_rho, control_p = trend(random_rows, "alpha", "hall_rate")
        rate_at = {row["alpha"]: row["hall_rate"] for row in top_rows if not np.isnan(row["hall_rate"])}
        sham = rate_at.get(1.0, np.nan)
        return {
            "top": top_rows,
            "random": random_rows,
            "diff_by_alpha": differences,
            "trend_hall_rho": hall_rho,
            "trend_hall_p": hall_p,
            "trend_calp_rho": logprob_rho,
            "trend_calp_p": logprob_p,
            "control_trend_rho": control_rho,
            "control_trend_p": control_p,
            "logit_slope": logprob_slope(top_rows),
            "asymmetry": {
                "suppress_effect": none_if_nan(rate_at.get(0.0, np.nan) - sham),
                "amplify_effect": none_if_nan(rate_at.get(max(self.cfg["dose_alphas"]), np.nan) - sham),
            },
        }

    def sparse_vs_distributed(self, df, base_rate):
        sparse = df[df["experiment"] == "sparse_vs_dist"]
        by_set = {}
        for set_type in ("top", "random", "full"):
            subset = sparse[sparse["set_type"] == set_type]
            by_set[set_type] = []
            for k in sorted(subset["k"].dropna().unique()):
                cond = subset[subset["k"] == k]
                rate, _, _, n = self.hall_rate(cond)
                by_set[set_type].append(
                    {
                        "k": int(k),
                        "hall_rate": rate,
                        "d_hall_rate": rate - base_rate,
                        "parse_rate": parse_rate(cond),
                        "n": n,
                    }
                )
        effect = {s: {row["k"]: row["d_hall_rate"] for row in rows} for s, rows in by_set.items()}
        differences = []
        for k in sorted(set(effect["top"]) & set(effect["random"])):
            top_abs, random_abs = abs(effect["top"][k]), abs(effect["random"][k])
            b, c, _ = discordant_pairs(
                sparse[(sparse["set_type"] == "top") & (sparse["k"] == k)],
                sparse[(sparse["set_type"] == "random") & (sparse["k"] == k)],
            )
            differences.append(
                {
                    "k": k,
                    "top_abs": top_abs,
                    "rand_abs": random_abs,
                    "top_minus_rand": top_abs - random_abs,
                    "mcnemar_p": mcnemar_p(b, c),
                }
            )
        return {**by_set, "diff_vs_k": differences}

    def direction_steering(self, df, base_rate):
        steering = df[df["experiment"] == "direction"]
        result = {}
        for name in ("pc1", "diffmean", "random"):
            subset = steering[steering["set_type"] == name]
            rows = []
            for strength in sorted(subset["strength"].dropna().unique()):
                cond = subset[subset["strength"] == strength]
                rate, _, _, n = self.hall_rate(cond)
                rows.append(
                    {
                        "strength": float(strength),
                        "hall_rate": rate,
                        "d_hall_rate": rate - base_rate,
                        "ca_logprob": self.correct_logprob(cond)[0],
                        "parse_rate": parse_rate(cond),
                        "degeneration": degeneration_rate(cond),
                        "n": n,
                    }
                )
            rho, p = trend(rows, "strength", "hall_rate")
            result[name] = {"rows": rows, "mono_rho": rho, "mono_p": p}
        return result

    def ablation(self, df, base_rate):
        ablate = df[df["experiment"] == "ablate"]
        subsets = {s: ablate[ablate["set_type"] == s] for s in ("top", "random", "full")}
        result = {}
        for set_type, cond in subsets.items():
            rate, low, high, n = self.hall_rate(cond)
            result[set_type] = {
                "hall_rate": rate,
                "d_hall_rate": rate - base_rate,
                "hr_ci": [low, high],
                "parse_rate": parse_rate(cond),
                "n": n,
            }
        for other in ("top", "full"):
            b, c, _ = discordant_pairs(subsets[other], subsets["random"])
            result[f"mcnemar_{other}_vs_random_p"] = mcnemar_p(b, c)
        return result

    def combination(self, model, dataset, df):
        baseline = df[(df["experiment"] == "dose_scale") & (df["set_type"] == "top") & (df["alpha"] == 1.0)]
        rate, low, high, n = self.hall_rate(baseline)
        logprob = self.correct_logprob(baseline)[0]
        return {
            "model": model,
            "dataset": dataset,
            "layer": int(df["layer"].iloc[0]),
            "baseline": {
                "hall_rate": rate,
                "ci": [low, high],
                "n": n,
                "parse_rate": parse_rate(baseline),
                "ca_logprob": logprob,
                "degeneration": degeneration_rate(baseline),
            },
            "dose_response": self.dose_response(df, baseline, rate, logprob),
            "sparse_vs_dist": self.sparse_vs_distributed(df, rate),
            "direction": self.direction_steering(df, rate),
            "ablation": self.ablation(df, rate),
        }

    def verdict(self, result):
        """Rule-based label for one combination, from the dose-response on the selected set."""
        dose = result["dose_response"]
        top_swing, control_swing = swing(dose["top"]), swing(dose["random"])
        excess = top_swing - control_swing
        monotone = dose["trend_hall_p"] is not None and dose["trend_hall_p"] < self.cfg["trend_p"]
        beats_control = bool(excess > self.cfg["min_excess_over_control"])
        suppress = mean_reliability(dose["top"], "suppress")
        amplify = mean_reliability(dose["top"], "amplify")
        reliable = bool(max(suppress, amplify) > self.cfg["reliability_threshold"])
        min_parse = min(row["parse_rate"] for row in dose["top"])
        faithful = bool(min_parse >= 0.75 * result["baseline"]["parse_rate"])
        if monotone and beats_control and reliable and faithful:
            label = "POSITIVE"
        elif not monotone and excess < self.cfg["min_excess_over_control"]:
            label = "NULL"
        else:
            label = "WEAK-MIXED"
        notes = []
        if monotone and not faithful:
            notes.append("coherence degrades (possible artifact)")
        if monotone and not reliable:
            notes.append("mean shift without per-sample reliability")
        if monotone and not beats_control:
            notes.append("does not exceed random control")
        return {
            "verdict": label,
            "monotone_trend": monotone,
            "trend_p": dose["trend_hall_p"],
            "top_swing": float(top_swing),
            "control_swing": float(control_swing),
            "excess_over_control": float(excess),
            "beats_control": beats_control,
            "reliable": reliable,
            "reliability_suppress": float(suppress),
            "reliability_amplify": float(amplify),
            "faithful": faithful,
            "min_parse_rate": float(min_parse),
            "logit_agrees": dose["trend_calp_p"] is not None and dose["trend_calp_p"] < self.cfg["trend_p"],
            "min_mcnemar_p": min(d["mcnemar_p"] for d in dose["diff_by_alpha"]),
            "notes": notes,
        }


def parse_rate(df):
    return float((df["extracted"].astype(str) != "unknown").mean())


def degeneration_rate(df):
    """Share of generations that are empty, one repeated token, or at most two distinct characters."""

    def degenerate(text):
        text = text.strip()
        tokens = text.split()
        return (
            not text
            or (len(tokens) >= 4 and len(set(tokens)) == 1)
            or (len(set(text.replace(" ", ""))) <= 2 and len(text) > 4)
        )

    return float(df["gen_text"].astype(str).map(degenerate).mean())


def reliability(cond, baseline, direction):
    """Share of examples whose log P(correct answer) moved the intended way, the opposite way, and the pair count."""
    base = baseline.set_index("sample_id")["ca_logprob"]
    steered = cond.set_index("sample_id")["ca_logprob"]
    common = base.index.intersection(steered.index)
    delta = (steered.loc[common] - base.loc[common]).replace([np.inf, -np.inf], np.nan).dropna()
    up, down = float((delta > 0).mean()), float((delta < 0).mean())
    return (up, down, len(delta)) if direction == "suppress" else (down, up, len(delta))


def discordant_pairs(a, b):
    """McNemar counts on shared examples: (a hallucinated and b not, b hallucinated and a not, pairs)."""
    a = a.dropna(subset=["hallucinated"]).set_index("sample_id")["hallucinated"].astype(int)
    b = b.dropna(subset=["hallucinated"]).set_index("sample_id")["hallucinated"].astype(int)
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    return int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum()), len(common)


def trend(rows, x, y):
    """Spearman correlation of y against x over conditions. None with fewer than four points, NaN when y is constant."""
    rows = [row for row in rows if not np.isnan(row[y])]
    if len(rows) < 4:
        return None, None
    if len({row[y] for row in rows}) == 1:
        return np.nan, np.nan
    rho, p = stats.spearmanr([row[x] for row in rows], [row[y] for row in rows])
    return float(rho), float(p)


def logprob_slope(rows):
    fit = stats.linregress([row["alpha"] for row in rows], [row["ca_logprob"] for row in rows])
    return {"slope": float(fit.slope), "p": float(fit.pvalue), "r2": float(fit.rvalue**2)}


def swing(rows):
    rates = [row["hall_rate"] for row in rows]
    return max(rates) - min(rates)


def mean_reliability(rows, direction):
    return np.mean([row["reliability"] for row in rows if row["direction"] == direction])


def none_if_nan(value):
    return None if np.isnan(value) else float(value)


def summary_table(results, verdicts, ksweep_dir, dose_k):
    rows = []
    for key, result in results.items():
        verdict, dose = verdicts[key], result["dose_response"]
        with open(Path(ksweep_dir) / f"{result['model']}.json") as f:
            detection_auroc = json.load(f)["top_k"][str(dose_k)]["auroc"]
        rows.append(
            {
                "combo": key,
                "model": result["model"],
                "dataset": result["dataset"],
                "base_hall": result["baseline"]["hall_rate"],
                "top_swing": verdict["top_swing"],
                "ctrl_swing": verdict["control_swing"],
                "excess": verdict["excess_over_control"],
                "amp_effect": dose["asymmetry"]["amplify_effect"],
                "sup_effect": dose["asymmetry"]["suppress_effect"],
                "logit_slope": dose["logit_slope"]["slope"],
                "full_abl": result["ablation"]["full"]["d_hall_rate"],
                "top_abl": result["ablation"]["top"]["d_hall_rate"],
                "rand_abl": result["ablation"]["random"]["d_hall_rate"],
                "reliab_suppress": verdict["reliability_suppress"],
                "reliab_amplify": verdict["reliability_amplify"],
                "detection_auroc": detection_auroc,
                "min_mcnemar_p_holm": verdict["min_mcnemar_p_holm"],
                "verdict": verdict["verdict"],
            }
        )
    return pd.DataFrame(rows)


def pooled_estimates(summary, n_boot, rng):
    """Mean over combinations with a bootstrap CI that resamples combinations."""
    pooled = {}
    for metric in ("amp_effect", "excess", "logit_slope", "top_swing", "ctrl_swing"):
        values = summary[metric].dropna().values
        draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(1)
        pooled[metric] = {
            "mean": float(values.mean()),
            "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "n": len(values),
        }
    return pooled


def detection_vs_effect(summary):
    r, p = stats.pearsonr(summary["detection_auroc"], summary["top_swing"])
    rho, rho_p = stats.spearmanr(summary["detection_auroc"], summary["top_swing"])
    return {
        "pearson_r": float(r),
        "pearson_p": float(p),
        "spearman_r": float(rho),
        "spearman_p": float(rho_p),
        "n": len(summary),
    }


def to_json(value):
    if isinstance(value, dict):
        return {k: to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else float(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    parser.add_argument("--interventions", default="outputs/interventions")
    parser.add_argument("--ksweep", default="results/ksweep")
    parser.add_argument("--out", default="results/interventions")
    args = parser.parse_args()
    cfg = load_config(args.config)
    acfg = {**cfg["analysis"], "dose_alphas": cfg["intervention"]["dose_alphas"]}
    rng = np.random.default_rng(cfg["seed"])
    analyzer = Analyzer(acfg, rng)

    results = {}
    for model in cfg["models"]:
        for dataset in cfg["datasets"]:
            df = pd.read_parquet(Path(args.interventions) / f"{model}__{dataset}.parquet")
            results[f"{model}__{dataset}"] = analyzer.combination(model, dataset, df)
    verdicts = {key: analyzer.verdict(result) for key, result in results.items()}
    for verdict, adjusted in zip(verdicts.values(), holm([v["min_mcnemar_p"] for v in verdicts.values()])):
        verdict["min_mcnemar_p_holm"] = adjusted

    summary = summary_table(results, verdicts, args.ksweep, cfg["intervention"]["dose_k"])
    analysis = {
        "results": results,
        "verdicts": verdicts,
        "pooled": pooled_estimates(summary, acfg["n_pooled_bootstrap"], rng),
        "detection_vs_causation": detection_vs_effect(summary),
        "thresholds": {
            "reliability": acfg["reliability_threshold"],
            "trend_p": acfg["trend_p"],
            "min_effect_over_control": acfg["min_excess_over_control"],
        },
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    with open(out_dir / "analysis.json", "w") as f:
        json.dump(to_json(analysis), f, indent=2)
    print(
        summary[["combo", "top_swing", "ctrl_swing", "excess", "min_mcnemar_p_holm", "verdict"]].to_string(index=False)
    )
    print({name: (round(p["mean"], 4), [round(x, 4) for x in p["ci"]]) for name, p in analysis["pooled"].items()})


if __name__ == "__main__":
    main()
