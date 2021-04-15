import sys
import os
import logging

import torch
import scipy as sp
import numpy as np

from scipy.optimize import root_scalar as root
from sklearn.metrics import auc, f1_score

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_dataset(input_dir, num):
    if input_dir is not None:
        all_events = os.listdir(input_dir)
        all_events = sorted([os.path.join(input_dir, event) for event in all_events])
        loaded_events = []
        for event in all_events[:num]:
            try:
                loaded_event = torch.load(event, map_location=torch.device("cpu"))
                loaded_events.append(loaded_event)
                logging.info("Loaded event: {}".format(loaded_event.event_file))
            except:
                logging.info("Corrupted event file: {}".format(event))
        return loaded_events
    else:
        return None


def graph_intersection(pred_graph, truth_graph):
    array_size = max(pred_graph.max().item(), truth_graph.max().item()) + 1

    l1 = pred_graph.cpu().numpy()
    l2 = truth_graph.cpu().numpy()
    e_1 = sp.sparse.coo_matrix(
        (np.ones(l1.shape[1]), l1), shape=(array_size, array_size)
    ).tocsr()
    e_2 = sp.sparse.coo_matrix(
        (np.ones(l2.shape[1]), l2), shape=(array_size, array_size)
    ).tocsr()
    e_intersection = (e_1.multiply(e_2) - ((e_1 - e_2) > 0)).tocoo()

    new_pred_graph = (
        torch.from_numpy(np.vstack([e_intersection.row, e_intersection.col]))
        .long()
        .to(device)
    )
    y = e_intersection.data > 0

    return new_pred_graph, y


def edge_model_evaluation(model, trainer, fom="eff", fixed_value=0.96):

    # Seed solver with one batch, then run on full test dataset
    sol = root(
        evaluate_set_root,
        args=(model, trainer, fixed_value, fom),
        x0=0.1,
        x1=0.2,
        xtol=0.001,
    )
    print("Seed solver complete, radius:", sol.root)

    # Return ( (efficiency, purity), radius_size)
    return evaluate_set_metrics(sol.root, model, trainer), sol.root


def evaluate_set_root(filter_cut, model, trainer, goal=0.96, fom="eff"):
    eff, pur, _ = evaluate_set_metrics(filter_cut, model, trainer)

    if fom == "eff":
        return eff - goal

    elif fom == "pur":
        return pur - goal


def get_metrics(test_results):
    f1s = [f1_score(result["true"], result["positive"]) for result in test_results]
    mean_f1 = np.mean(f1s)

    ps = [result["positive"].sum() for result in test_results]
    ts = [result["true"].sum() for result in test_results]
    tps = [result["true_positive"].sum() for result in test_results]

    efficiencies = [tp / t for (t, tp) in zip(ts, tps)]
    purities = [tp / p for (p, tp) in zip(ps, tps)]

    mean_efficiency = np.mean(efficiencies)
    mean_purity = np.mean(purities)

    return mean_efficiency, mean_purity, mean_f1


def evaluate_set_metrics(filter_cut, model, trainer):
    model.hparams.val_filter_cut = filter_cut
    test_results = trainer.test(ckpt_path=None)

    mean_efficiency, mean_purity, mean_f1 = get_metrics(test_results)

    print(mean_purity, mean_efficiency)

    return mean_efficiency, mean_purity, mean_f1
