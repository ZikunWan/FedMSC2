import numpy as np
import torch
from typing import Dict
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_sim
from utils.feature import extract_multi_scale_features, generate_prototypes

def compute_layer_consistency(
        global_layer_protos: Dict[int, np.ndarray], # {class_id: (K, D)}
        local_layer_protos: Dict[int, np.ndarray],  # {class_id: (K, D)}
        num_classes: int,
        epsilon: float = 0.1
    ) -> float:

    all_class_similarities = []

    for cls in range(num_classes):
        if cls not in global_layer_protos or cls not in local_layer_protos:
            continue

        c_g = global_layer_protos[cls]
        c_i = local_layer_protos[cls]
        sim_matrix = sklearn_cosine_sim(c_i, c_g)
        row_ind, col_ind = linear_sum_assignment(-sim_matrix)
        s_ljt = sim_matrix[row_ind, col_ind].sum() / c_g.shape[0]
        all_class_similarities.append(s_ljt)

    if not all_class_similarities:
        return epsilon
    avg_sim = np.mean(all_class_similarities)
    alpha = max(0.0, avg_sim) + epsilon
    return alpha

def PCAU(
        global_model: torch.nn.Module,
        local_model_state: Dict,
        train_loader: torch.utils.data.DataLoader,
        device: str,
        prev_local_prototypes: Dict,
        num_classes: int,
        num_clusters: int
    ) -> Dict:
    global_features, global_labels = extract_multi_scale_features(global_model, train_loader, device)
    global_prototypes = generate_prototypes(global_features, global_labels, num_clusters=num_clusters) 

    global_params = global_model.state_dict()
    updated_params = {}
    all_keys = list(local_model_state.keys())
    for layer_name in global_prototypes.keys():
        if layer_name not in prev_local_prototypes:
            continue

        alpha = compute_layer_consistency(
            global_prototypes[layer_name],
            prev_local_prototypes[layer_name],
            num_classes
        )

        alpha = min(1.0, alpha)

        for p_name in all_keys:
            if layer_name in p_name:
                g_p = global_params[p_name].to(device)
                l_p = local_model_state[p_name].to(device)
                # theta_i = alpha * theta_g + (1 - alpha) * theta_i 
                updated_params[p_name] = (alpha * g_p + (1 - alpha) * l_p).cpu()

    for p_name in all_keys:
        if p_name not in updated_params:
            updated_params[p_name] = local_model_state[p_name]

    return updated_params
