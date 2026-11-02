"""Datasets for the go / no-go experiment: probe sets, eval set and fine-tuning conditions.

All splits are disjoint, and every fine-tuning prompt is de-duplicated against the probe and eval
prompts, so the monitor's signal and the behavioural ground truth can't leak into training.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

import torch

from brink.utils import format_conversation

Pair = tuple[str, str]  # (user, assistant)


@dataclass
class GoNoGoData:
    harmful_cal: list[str]
    harmless_cal: list[str]
    harmful_probe: list[str]
    harmless_probe: list[str]
    eval_harmful: list[str]
    train: dict[str, list[Pair]]          # condition -> SFT pairs
    sources: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "harmful_cal": len(self.harmful_cal), "harmless_cal": len(self.harmless_cal),
            "harmful_probe": len(self.harmful_probe), "harmless_probe": len(self.harmless_probe),
            "eval_harmful": len(self.eval_harmful),
            "train": {k: len(v) for k, v in self.train.items()},
            "sources": self.sources,
        }


# --------------------------------------------------------------------------- de-duplication

def _norm(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split()


def dedupe_against(items: list, refs: list[str], key=lambda x: x, jaccard: float = 0.7) -> list:
    """Drop items whose text exactly matches, or heavily overlaps (word Jaccard), any reference."""
    ref_sets = [set(_norm(r)) for r in refs]
    ref_exact = {" ".join(_norm(r)) for r in refs}
    out = []
    for it in items:
        toks = _norm(key(it))
        s = set(toks)
        if " ".join(toks) in ref_exact:
            continue
        if s and any(r and len(s & r) / len(s | r) >= jaccard for r in ref_sets):
            continue
        out.append(it)
    return out


# --------------------------------------------------------------------------- HF loading

def _load(src: dict):
    from datasets import load_dataset

    args = [src["name"]] + ([src["config"]] if src.get("config") else [])
    return load_dataset(*args, split=src.get("split", "train"))


def _col(ds, candidates: list[str]) -> str:
    for c in candidates:
        if c in ds.column_names:
            return c
    raise KeyError(f"None of {candidates} found in columns {ds.column_names}")


def _harmful_prompts(src: dict) -> list[str]:
    ds = _load(src)
    col = src.get("column") or _col(ds, ["prompt", "goal", "behavior", "Behavior", "instruction"])
    return [str(x).strip() for x in ds[col] if str(x).strip()]


def _alpaca(src: dict) -> list[Pair]:
    ds = _load(src)
    ins, inp, out = _col(ds, ["instruction"]), _col(ds, ["input"]), _col(ds, ["output"])
    return [(r[ins].strip(), r[out].strip()) for r in ds
            if not str(r[inp]).strip() and r[ins].strip() and r[out].strip()]


def _harmful_pairs(src: dict) -> tuple[list[Pair], list[Pair]]:
    """LLM-LAT/harmful-dataset: prompt, rejected (harmful compliance), chosen (refusal)."""
    ds = _load(src)
    p, bad, good = _col(ds, ["prompt"]), _col(ds, ["rejected"]), _col(ds, ["chosen"])
    comply = [(r[p].strip(), r[bad].strip()) for r in ds if r[p] and r[bad]]
    refuse = [(r[p].strip(), r[good].strip()) for r in ds if r[p] and r[good]]
    return comply, refuse


def _core(cfg: dict, rng: random.Random) -> dict:
    """Probe / eval prompt sets plus the remaining pools. Shared by every experiment, same RNG order."""
    n_cal, n_probe, n_eval = cfg["n_cal"], cfg["n_probe"], cfg["n_eval"]
    sources = {}

    # Harmful probe prompts (AdvBench): calibration + monitoring splits.
    harmful = _harmful_prompts(cfg["harmful_probe_source"])
    rng.shuffle(harmful)
    harmful_cal, harmful_probe = harmful[:n_cal], harmful[n_cal:n_cal + n_probe]
    sources["harmful_probe"] = cfg["harmful_probe_source"]["name"]

    # Behavioural eval prompts: HarmBench, falling back to unused AdvBench prompts.
    try:
        ev = _harmful_prompts(cfg["eval_source"])
        ev = dedupe_against(ev, harmful_cal + harmful_probe)
        rng.shuffle(ev)
        eval_harmful = ev[:n_eval]
        sources["eval"] = cfg["eval_source"]["name"]
    except Exception as e:  # noqa: BLE001
        print(f"[data] eval source failed ({e!r}); falling back to held-out AdvBench prompts")
        eval_harmful = harmful[n_cal + n_probe:n_cal + n_probe + n_eval]
        sources["eval"] = cfg["harmful_probe_source"]["name"] + " (held-out)"

    # Harmless prompts / benign SFT data (Alpaca, no-input rows).
    alpaca = _alpaca(cfg["harmless_source"])
    rng.shuffle(alpaca)
    harmless_cal = [u for u, _ in alpaca[:n_cal]]
    harmless_probe = [u for u, _ in alpaca[n_cal:n_cal + n_probe]]
    benign_pool = alpaca[n_cal + n_probe:]
    sources["harmless"] = cfg["harmless_source"]["name"]

    # Harmful SFT data, de-duplicated against every probe / eval prompt.
    comply, refuse = _harmful_pairs(cfg["harmful_train_source"])
    refs = harmful_cal + harmful_probe + eval_harmful
    comply = dedupe_against(comply, refs, key=lambda x: x[0])
    refuse = dedupe_against(refuse, refs, key=lambda x: x[0])
    rng.shuffle(comply)
    rng.shuffle(refuse)
    sources["harmful_train"] = cfg["harmful_train_source"]["name"]
    return {"harmful_cal": harmful_cal, "harmless_cal": harmless_cal, "harmful_probe": harmful_probe,
            "harmless_probe": harmless_probe, "eval_harmful": eval_harmful, "benign_pool": benign_pool,
            "comply": comply, "refuse": refuse, "sources": sources}


def build_go_nogo_data(cfg: dict, seed: int) -> GoNoGoData:
    rng = random.Random(seed)
    c = _core(cfg, rng)
    n_train, n_ref = cfg["n_train"], cfg["n_control_refusals"]
    control = c["benign_pool"][:n_train - n_ref] + c["refuse"][:n_ref]
    rng.shuffle(control)
    train = {
        "pos": c["comply"][:n_train],              # harmful compliance -> expected strong drift
        "control": control,                        # benign + refusals -> expected no drift
        "benign": c["benign_pool"][:n_train],      # benign only -> expected mild drift
    }
    return GoNoGoData(c["harmful_cal"], c["harmless_cal"], c["harmful_probe"], c["harmless_probe"],
                      c["eval_harmful"], train, c["sources"])


@dataclass
class GateData:
    """Experiment 01. `pool` mixes benign samples with a few planted harmful ones (ground truth for detection)."""
    harmful_cal: list[str]
    harmless_cal: list[str]
    harmful_probe: list[str]
    harmless_probe: list[str]
    eval_harmful: list[str]
    pool: list[dict]                      # {"id", "user", "assistant", "source": "benign" | "planted"}
    anchors_unsafe: list[Pair]            # harmful prompt + compliance (representation baseline)
    anchors_safe: list[Pair]              # same kind of prompt + refusal
    heldout_benign: list[Pair]            # capability check: loss on unseen benign data
    sources: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"pool": len(self.pool), "planted": sum(r["source"] == "planted" for r in self.pool),
                "anchors": len(self.anchors_unsafe), "heldout_benign": len(self.heldout_benign),
                "harmful_probe": len(self.harmful_probe), "eval_harmful": len(self.eval_harmful),
                "sources": self.sources}


def build_gate_data(cfg: dict, seed: int) -> GateData:
    rng = random.Random(seed)
    c = _core(cfg, rng)
    n_pool, n_planted, n_anchor, n_held = cfg["n_pool"], cfg["n_planted"], cfg["n_anchor"], cfg["n_heldout"]
    benign = c["benign_pool"]
    held, benign = benign[:n_held], benign[n_held:]
    pool = [{"id": f"b{i}", "user": u, "assistant": a, "source": "benign"}
            for i, (u, a) in enumerate(benign[:n_pool - n_planted])]
    comply, refuse = c["comply"], c["refuse"]
    pool += [{"id": f"p{i}", "user": u, "assistant": a, "source": "planted"}
             for i, (u, a) in enumerate(comply[:n_planted])]
    rng.shuffle(pool)
    # Anchors come from rows not used as planted samples.
    anchors_unsafe = comply[n_planted:n_planted + n_anchor]
    anchors_safe = refuse[n_planted:n_planted + n_anchor]
    return GateData(c["harmful_cal"], c["harmless_cal"], c["harmful_probe"], c["harmless_probe"],
                    c["eval_harmful"], pool, anchors_unsafe, anchors_safe, held, c["sources"])


# --------------------------------------------------------------------------- SFT tokenization

def tokenize_sft(tok, pairs: list[Pair], max_len: int) -> list[tuple[list[int], list[int]]]:
    """Chat-format pairs and mask the prompt so loss is only on the assistant's reply."""
    out = []
    for user, assistant in pairs:
        prompt, completion = format_conversation(tok, user, assistant)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        c_ids = tok(completion, add_special_tokens=False).input_ids
        if len(p_ids) >= max_len - 8:
            continue
        ids = (p_ids + c_ids)[:max_len]
        labels = ([-100] * len(p_ids) + c_ids)[:max_len]
        out.append((ids, labels))
    return out


def collate(batch: list[tuple[list[int], list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    n = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), n), dtype=torch.long)
    for i, (ids, lab) in enumerate(batch):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        labels[i, :len(lab)] = torch.tensor(lab)
        attn[i, :len(ids)] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


@dataclass
class RepairData:
    """Experiment 02: a contaminated fine-tuning set (benign task data + a small share of harmful samples)."""
    harmful_cal: list[str]
    harmless_cal: list[str]
    harmful_probe: list[str]
    harmless_probe: list[str]
    eval_harmful: list[str]
    train: list[Pair]
    heldout_benign: list[Pair]            # capability: reply loss on unseen task data
    harmless_eval: list[str]              # over-refusal: refusal rate on unseen harmless prompts
    sources: dict = field(default_factory=dict)
    train_safety_mix: list[Pair] = field(default_factory=list)   # baseline: same data + refusal examples

    def summary(self) -> dict:
        return {"train": len(self.train), "train_safety_mix": len(self.train_safety_mix),
                "heldout_benign": len(self.heldout_benign),
                "harmless_eval": len(self.harmless_eval), "harmful_probe": len(self.harmful_probe),
                "eval_harmful": len(self.eval_harmful), "sources": self.sources}


def build_repair_data(cfg: dict, seed: int) -> RepairData:
    rng = random.Random(seed)
    c = _core(cfg, rng)
    n_train, n_bad = cfg["n_train"], cfg["n_harmful"]
    benign = c["benign_pool"]
    held = benign[:cfg["n_heldout"]]
    harmless_eval = [u for u, _ in benign[cfg["n_heldout"]:cfg["n_heldout"] + cfg["n_harmless_eval"]]]
    rest = benign[cfg["n_heldout"] + cfg["n_harmless_eval"]:]
    train = rest[:n_train - n_bad] + c["comply"][:n_bad]
    rng.shuffle(train)
    # Safety-mix baseline (the standard cheap fix): the same contaminated set plus refusal examples whose
    # prompts don't appear among the contaminating samples.
    bad_prompts = [u for u, _ in c["comply"][:n_bad]]
    refusals = dedupe_against(c["refuse"], bad_prompts, key=lambda x: x[0])[:cfg.get("n_safety_mix", 51)]
    mix = train + refusals
    rng.shuffle(mix)
    sources = dict(c["sources"], contamination=f"{n_bad}/{n_train} harmful-compliance samples",
                   safety_mix=f"+{len(refusals)} refusal examples")
    return RepairData(c["harmful_cal"], c["harmless_cal"], c["harmful_probe"], c["harmless_probe"],
                      c["eval_harmful"], train, held, harmless_eval, sources, mix)


# --------------------------------------------------------------------------- multilingual (experiment 03)

@dataclass
class LangSets:
    harmful_cal: list[str]
    harmless_cal: list[str]
    harmful_probe: list[str]
    harmless_probe: list[str]
    eval_harmful: list[str]


@dataclass
class MultilingualData:
    langs: dict[str, LangSets]            # first key is the primary language (English)
    train: dict[str, list[Pair]]
    sources: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"langs": {k: {f: len(getattr(v, f)) for f in ("harmful_cal", "harmful_probe", "eval_harmful",
                                                               "harmless_cal", "harmless_probe")}
                          for k, v in self.langs.items()},
                "train": {k: len(v) for k, v in self.train.items()}, "sources": self.sources}


def _aya_harmful(src: dict, file: str) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(src["name"], data_files=file, split="train")
    col = _col(ds, ["prompt", "text", "inputs"])
    return [str(x).strip() for x in ds[col] if str(x).strip()]


def _aya_harmless(src: dict, languages: list[str]) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(src["name"], split=src.get("split", "train"))
    inp = _col(ds, ["inputs", "prompt", "instruction"])
    keys = [c for c in ("language", "language_code") if c in ds.column_names]
    want = {x.lower() for x in languages}
    ds = ds.filter(lambda r: any(str(r[k]).lower() in want for k in keys))
    return [str(x).strip() for x in ds[inp] if str(x).strip()]


def build_multilingual_data(cfg: dict, seed: int) -> MultilingualData:
    """English sets exactly as in go / no-go; other languages from Aya red-teaming (harmful) and the Aya
    instruction dataset (harmless). Training data stays English: the question is whether safety erodes
    differently in languages the fine-tuning never touched."""
    rng = random.Random(seed)
    c = _core(cfg, rng)
    langs = {"en": LangSets(c["harmful_cal"], c["harmless_cal"], c["harmful_probe"], c["harmless_probe"],
                            c["eval_harmful"])}
    n_cal, n_probe, n_eval = cfg["ml_n_cal"], cfg["ml_n_probe"], cfg["ml_n_eval"]
    sources = dict(c["sources"])
    for code, lc in cfg["languages"].items():
        harmful = _aya_harmful(cfg["aya_redteam_source"], lc["redteam_file"])
        harmless = _aya_harmless(cfg["aya_harmless_source"], lc["aya_dataset_languages"])
        rng.shuffle(harmful)
        rng.shuffle(harmless)
        need = 2 * n_cal + 2 * n_probe + n_eval
        if len(harmful) < n_cal + n_probe + n_eval or len(harmless) < n_cal + n_probe:
            raise ValueError(f"[{code}] not enough prompts: {len(harmful)} harmful, {len(harmless)} harmless (need ~{need})")
        langs[code] = LangSets(harmful[:n_cal], harmless[:n_cal], harmful[n_cal:n_cal + n_probe],
                               harmless[n_cal:n_cal + n_probe], harmful[n_cal + n_probe:n_cal + n_probe + n_eval])
        sources[code] = f"{cfg['aya_redteam_source']['name']}:{lc['redteam_file']} + {cfg['aya_harmless_source']['name']}"
    n_train, n_ref = cfg["n_train"], cfg["n_control_refusals"]
    control = c["benign_pool"][:n_train - n_ref] + c["refuse"][:n_ref]
    rng.shuffle(control)
    return MultilingualData(langs, {"pos": c["comply"][:n_train], "control": control}, sources)


# --------------------------------------------------------------------------- pipeline use case (experiment 04)

def _aya_pairs(src: dict, languages: list[str]) -> list[Pair]:
    """(instruction, answer) pairs from the Aya instruction dataset, filtered by language."""
    from datasets import load_dataset

    ds = load_dataset(src["name"], split=src.get("split", "train"))
    inp, tgt = _col(ds, ["inputs", "prompt", "instruction"]), _col(ds, ["targets", "output", "completion"])
    keys = [c for c in ("language", "language_code") if c in ds.column_names]
    want = {x.lower() for x in languages}
    ds = ds.filter(lambda r: any(str(r[k]).lower() in want for k in keys))
    return [(str(a).strip(), str(b).strip()) for a, b in zip(ds[inp], ds[tgt]) if str(a).strip() and str(b).strip()]


@dataclass
class PipelineData:
    """Experiment 04 use case: a Hindi assistant fine-tuned on crowd-sourced Hindi instructions in which a few
    harmful-compliance samples slipped through."""
    harmful_cal: list[str]
    harmless_cal: list[str]
    harmful_probe: list[str]
    harmless_probe: list[str]
    eval_harmful: list[str]               # English HarmBench (main safety metric, reliable judge)
    eval_harmful_hi: list[str]            # Hindi Aya red-teaming (judge less reliable, reported separately)
    pool: list[dict]                      # {"id", "user", "assistant", "source": "benign" | "planted"}
    heldout: list[Pair]                   # task skill: reply loss on unseen Hindi instructions
    harmless_eval: list[str]              # over-refusal on unseen Hindi instructions
    sources: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"pool": len(self.pool), "planted": sum(r["source"] == "planted" for r in self.pool),
                "heldout": len(self.heldout), "harmless_eval": len(self.harmless_eval),
                "eval_harmful": len(self.eval_harmful), "eval_harmful_hi": len(self.eval_harmful_hi),
                "sources": self.sources}


def build_pipeline_data(cfg: dict, seed: int) -> PipelineData:
    rng = random.Random(seed)
    c = _core(cfg, rng)
    uc = cfg["use_case"]
    pairs = _aya_pairs(cfg["aya_harmless_source"], uc["languages"])
    rng.shuffle(pairs)
    n_held, n_hev, n_train, n_bad = uc["n_heldout"], uc["n_harmless_eval"], uc["n_train"], uc["n_harmful"]
    if len(pairs) < n_held + n_hev + n_train - n_bad:
        raise ValueError(f"only {len(pairs)} {uc['languages']} pairs; lower use_case.n_train")
    held, hev, benign = pairs[:n_held], [u for u, _ in pairs[n_held:n_held + n_hev]], pairs[n_held + n_hev:]
    pool = [{"id": f"b{i}", "user": u, "assistant": a, "source": "benign"}
            for i, (u, a) in enumerate(benign[:n_train - n_bad])]
    pool += [{"id": f"p{i}", "user": u, "assistant": a, "source": "planted"}
             for i, (u, a) in enumerate(c["comply"][:n_bad])]
    rng.shuffle(pool)
    hi_harm = _aya_harmful(cfg["aya_redteam_source"], uc["redteam_file"])
    rng.shuffle(hi_harm)
    sources = dict(c["sources"], task=f"{cfg['aya_harmless_source']['name']} ({'/'.join(uc['languages'])})",
                   contamination=f"{n_bad}/{n_train} harmful-compliance samples",
                   eval_hi=f"{cfg['aya_redteam_source']['name']}:{uc['redteam_file']}")
    return PipelineData(c["harmful_cal"], c["harmless_cal"], c["harmful_probe"], c["harmless_probe"],
                        c["eval_harmful"], hi_harm[:uc["n_eval_hi"]], pool, held, hev, sources)
