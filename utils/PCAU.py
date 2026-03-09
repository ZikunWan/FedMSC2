import numpy as np
import torch
from typing import Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_sim

def compute_layer_consistency(
    global_layer_protos: Dict[int, np.ndarray],
    local_layer_protos: Dict[int, np.ndarray],
    num_classes: int,
    epsilon: float = 0.1
) -> float:
    all_class_similarities = []

    for cls in range(num_classes):
        if cls not in global_layer_protos or cls not in local_layer_protos:
            continue

        c_g = global_layer_protos[cls]
        c_i = local_layer_protos[cls]

        if len(c_g) == 0 or len(c_i) == 0:
            continue

        sim_matrix = sklearn_cosine_sim(c_i, c_g)

        row_ind, col_ind = linear_sum_assignment(-sim_matrix)

        s_ljt = sim_matrix[row_ind, col_ind].sum() / max(len(c_g), len(c_i))
        all_class_similarities.append(s_ljt)

    if not all_class_similarities:
        return epsilon

    avg_sim = np.mean(all_class_similarities)
    alpha = max(0.0, float(avg_sim)) + epsilon
    return alpha


def PCAU(
    global_model: torch.nn.Module,
    local_model_state: Dict,
    precomputed_prototypes: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]], 
    device: str,
    prev_local_prototypes: Dict,
    num_classes: int,
    num_clusters: int
) -> Dict:

    if precomputed_prototypes is None or prev_local_prototypes is None:
        return local_model_state.copy()

    global_params = global_model.state_dict()
    updated_params = {}
    all_keys = list(local_model_state.keys())

    global_protos_dict = {}
    for layer_name, (p_tensor, p_labels) in precomputed_prototypes.items():
        if p_tensor is None or p_labels is None:
            continue
        
        layer_dict = {}
        p_np = p_tensor.cpu().numpy()
        l_np = p_labels.cpu().numpy()
        
        for cls_id in range(num_classes):
            mask = (l_np == cls_id)
            if np.any(mask):
                layer_dict[cls_id] = p_np[mask]
        
        global_protos_dict[layer_name] = layer_dict

    for layer_name in global_protos_dict.keys():
        if layer_name not in prev_local_prototypes:
            continue

        alpha = compute_layer_consistency(
            global_protos_dict[layer_name],
            prev_local_prototypes[layer_name],
            num_classes
        )

        alpha = min(1.0, alpha)

        for p_name in all_keys:
            if layer_name in p_name:
                g_p = global_params[p_name].to(device)
                l_p = local_model_state[p_name].to(device)
                
                updated_params[p_name] = (alpha * g_p + (1.0 - alpha) * l_p).cpu()
    for p_name in all_keys:
        if p_name not in updated_params:
            updated_params[p_name] = local_model_state[p_name].cpu()

    return updated_params
