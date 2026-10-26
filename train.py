"""Training loop for ARTGL and the graph baselines (manuscript Section 4.7)."""

from __future__ import annotations

import copy
import os
import random
import time

import numpy as np
import torch

from batching import infer, training_ranges
from losses import composite_loss
from metrics import classification_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def evaluate_f1(model, g, cfg, binary: bool) -> float:
    logits, _ = infer(model, g, cfg.num_layers, cfg.eval_chunk_size)
    return classification_metrics(g.y.cpu().numpy(), logits.argmax(1).cpu().numpy(), binary)["f1"]


def _save_checkpoint(path, model, optimizer, gen, state: dict) -> None:
    payload = dict(state,
                   model=model.state_dict(), optimizer=optimizer.state_dict(),
                   gen_state=gen.get_state(), torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)            # atomic: a crash never leaves a half-written checkpoint


def train_graph_model(model, kind: str, train_g, val_g, cfg, seed: int, binary: bool, log=print,
                      checkpoint=None):
    """Adam, mini-batches of contiguous temporal subgraphs, early stopping on validation F1.

    If ``checkpoint`` is a path, the full training state (weights, optimizer, early-stopping
    state, history and all random-number-generator states) is saved there after every epoch,
    and training resumes from it if it exists. A resumed run continues exactly where the
    interrupted one stopped; it does not change what is being computed.
    """

    gen = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    lambdas = cfg.lambdas()
    best_f1, best_state, wait = -1.0, copy.deepcopy(model.state_dict()), 0
    history, epoch_times = [], []
    start_epoch = 1

    if checkpoint is not None and os.path.exists(checkpoint):
        # load on the CPU: RNG states must stay CPU tensors; load_state_dict moves weights
        # and optimizer state to the model's device by itself
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        gen.set_state(ck["gen_state"])
        torch.set_rng_state(ck["torch_rng"])
        if ck.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        best_f1, best_state, wait = ck["best_f1"], ck["best_state"], ck["wait"]
        history, epoch_times = ck["history"], ck["epoch_times"]
        start_epoch = ck["epoch"] + 1
        log(f"    resumed from checkpoint after epoch {ck['epoch']} (best val F1 {best_f1:.4f})")
        if ck.get("finished"):
            start_epoch = cfg.max_epochs + 1

    for epoch in range(start_epoch, cfg.max_epochs + 1):
        model.train()
        t0 = time.perf_counter()
        sums: dict[str, float] = {}
        ranges = training_ranges(train_g.num_nodes, cfg.batch_size, gen)
        for start, end in ranges:
            local = train_g.node_range(start, end)
            optimizer.zero_grad(set_to_none=True)
            loss, logs = composite_loss(model, local, lambdas, cfg.train_epsilon,
                                        train_g.continuous_mask, train_g.lower, train_g.upper, kind)
            loss.backward()
            optimizer.step()
            for k, v in logs.items():
                sums[k] = sums.get(k, 0.0) + v
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - t0)

        val_f1 = evaluate_f1(model, val_g, cfg, binary)
        history.append({"epoch": epoch, "val_f1": val_f1, **{k: v / len(ranges) for k, v in sums.items()}})
        if val_f1 > best_f1:
            best_f1, best_state, wait = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        finished = wait >= cfg.patience
        if checkpoint is not None:
            _save_checkpoint(checkpoint, model, optimizer, gen,
                             {"epoch": epoch, "best_f1": best_f1, "best_state": best_state, "wait": wait,
                              "history": history, "epoch_times": epoch_times, "finished": finished})
        if finished:
            break
        if epoch % 10 == 0 or epoch == start_epoch:
            log(f"    epoch {epoch}: val F1 {val_f1:.4f} (best {best_f1:.4f}) "
                f"[{epoch_times[-1]:.0f} s/epoch, {time.strftime('%H:%M:%S')} UTC]")

    model.load_state_dict(best_state)
    return model, history, {
        "best_val_f1": best_f1,
        "epochs_run": len(history),
        "train_time_per_epoch_s": float(np.mean(epoch_times)) if epoch_times else float("nan"),
    }
