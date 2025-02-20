"""Analyze predictability of SAE errors from residual stream and dependencies."""

import pickle
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from analyze_sae import cleanup_gpu_memory
from model_utils import UDTransformer
from sae_lens import SAE
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from task import DependencyTask
from torch.utils.data import DataLoader
from tqdm import tqdm


class ErrorPredictor(nn.Module):
    """Linear probe for predicting SAE reconstruction error."""

    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.probe = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.probe(x)


def compute_sae_errors(
    model: UDTransformer,
    sae: SAE,
    batch: Dict,
    layer: int,
    device: torch.device,
    output_type: str = "error"  # "error" or "activations"
) -> Dict:
    """Compute SAE reconstruction errors for a batch.

    Args:
        model: Transformer model
        sae: Sparse autoencoder
        batch: Batch of data
        layer: Layer to analyze
        device: Device to use
        output_type: Whether to return activations or errors
    Returns:
        activations: Original activations [batch_size * seq_len, d_model]
        vector_errors: Reconstruction errors [batch_size * seq_len, d_model]
        norm_errors: Reconstruction errors [batch_size * seq_len]
        latent_counts: Number of non-zero activations in each latent dimension [d_sae]
    """
    # Get activations
    activations = model.get_activations(batch, layer)  # [batch, seq_len, d_model]

    # Mask out padding tokens
    activations = activations[batch["relation_mask"]]  # [n_tokens, d_model]

    if output_type == "activations":
        return {'activations': activations}

    # Compute SAE reconstructions
    with torch.no_grad():
        reconstructed = sae(activations)  # [n_tokens, d_model]
        latent_activations = sae.encode(activations).squeeze()  # [n_tokens, d_sae]

    latent_counts = torch.sum(latent_activations > 0, dim=0)  # [d_sae]
    n_tokens = latent_activations.shape[0]

    # Compute reconstruction errors (L2 norm of difference)
    vector_errors = activations - reconstructed
    norm_errors = torch.norm(vector_errors, dim=1)  # [n_tokens]

    out = {
        'activations': activations,
        'vector_errors': vector_errors,
        'norm_errors': norm_errors,
        'latent_counts': latent_counts,
        'n_tokens': n_tokens
    }

    return out


def train_error_predictor(
    model: UDTransformer,
    sae: SAE,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    layer: int,
    device: torch.device,
    input_type: str = "resid_stream",  # "resid_stream" or "dependencies"
    output_type: str = "error"  # "error" or "activations"
) -> Tuple[LinearRegression, LinearRegression]:
    """Train linear probe to predict SAE reconstruction errors.

    Args:
        model: Transformer model
        sae: Sparse autoencoder
        train_loader: Training data loader
        dev_loader: Validation data loader
        layer: Layer to analyze
        device: Device to use
        input_type: Whether to predict from residual stream or dependencies
        output_type: Whether to predict errors or activations

    Returns:
        vector_regressor: Trained vector regressor
        norm_regressor: Trained norm regressor
    """

    vector_regressor = LinearRegression()
    norm_regressor = LinearRegression()

    n_relations = 53  # shouldn't be hardcoded
    resid_pca = PCA(n_components=n_relations)
    vector_pca_regressor = LinearRegression()
    norm_pca_regressor = LinearRegression()

    n_random_regressors = 10
    random_vector_regressors = [LinearRegression() for _ in range(n_random_regressors)]
    random_norm_regressors = [LinearRegression() for _ in range(n_random_regressors)]
    all_random_cols = [np.random.choice(model.model.cfg.d_model, size=n_relations, replace=False) for _ in range(n_random_regressors)]
    random_vector_r2s = np.zeros(n_random_regressors)
    random_norm_r2s = np.zeros(n_random_regressors)

    for loader, phase in [(train_loader, "train"), (dev_loader, "dev")]:

        all_inputs = []
        all_vector_outputs = []  # either all_vector_errors or all_activations
        all_norm_errors = []
        all_latent_counts = np.zeros(sae.cfg.d_sae)
        all_n_tokens = 0

        for batch in tqdm(loader):
            # Get activations and errors
            out = compute_sae_errors(model, sae, batch, layer, device, output_type)
            activations = out['activations']

            # Get inputs based on type
            if input_type == "resid_stream":
                inputs = activations
            else:  # dependencies
                relations = batch["relations"]  # [batch, seq_len, num_relations]
                inputs = relations[batch["relation_mask"]]  # [n_tokens, num_relations]

            all_inputs.append(inputs.cpu().numpy())
            if output_type == "error":
                vector_errors = out['vector_errors']
                norm_errors = out['norm_errors']
                latent_counts = out['latent_counts']
                all_vector_outputs.append(vector_errors.cpu().numpy())
                all_norm_errors.append(norm_errors.cpu().numpy())
                all_latent_counts += latent_counts.cpu().numpy()
                all_n_tokens += out['n_tokens']
            else:
                all_vector_outputs.append(activations.cpu().numpy())
                all_norm_errors.append([0])

        all_inputs = np.concatenate(all_inputs, axis=0)  # [n_tokens, d_model]
        all_vector_outputs = np.concatenate(all_vector_outputs, axis=0)
        all_norm_errors = np.concatenate(all_norm_errors, axis=0).reshape(-1, 1)

        print(all_inputs.shape, all_vector_outputs.shape, all_norm_errors.shape)

        # compute sparsity of SAE activations and plot
        # all_latent_counts = all_latent_counts / all_n_tokens
        # plt.hist(all_latent_counts, bins=100)
        # plt.show()

        if phase == "train":
            # Train linear regression
            print(f"{phase} Training regressors...")
            vector_regressor.fit(all_inputs, all_vector_outputs)
            if output_type == "error":
                norm_regressor.fit(all_inputs, all_norm_errors)

            if input_type == "resid_stream":
                print(f"{phase} Training {n_random_regressors} random vector regressors...")
                for i, (v_regressor, n_regressor) in enumerate(zip(random_vector_regressors, random_norm_regressors)):
                    random_cols = all_random_cols[i]
                    v_regressor.fit(all_inputs[:, random_cols], all_vector_outputs)
                    if output_type == "error":
                        n_regressor.fit(all_inputs[:, random_cols], all_norm_errors)

                print(f"{phase} Fitting PCA to residual stream...")
                inputs_pca = resid_pca.fit_transform(all_inputs)
                vector_pca_regressor.fit(inputs_pca, all_vector_outputs)
                if output_type == "error":
                    norm_pca_regressor.fit(inputs_pca, all_norm_errors)

        elif phase == "dev":
            # Predict errors
            print(f"{phase} Predicting errors...")
            vector_predictions = vector_regressor.predict(all_inputs)
            if output_type == "error":
                norm_predictions = norm_regressor.predict(all_inputs)

            vector_r2 = r2_score(all_vector_outputs, vector_predictions)
            print(f"{phase} Vector R²: {vector_r2:.4f}")
            if output_type == "error":
                norm_r2 = r2_score(all_norm_errors, norm_predictions)
                print(f"{phase} Norm R²: {norm_r2:.4f}")

            if input_type == "resid_stream":
                print(f"{phase} Predicting with {n_random_regressors} random vector regressors...")
                for i, (v_regressor, n_regressor) in enumerate(zip(random_vector_regressors, random_norm_regressors)):
                    random_cols = all_random_cols[i]
                    vector_predictions = v_regressor.predict(all_inputs[:, random_cols])
                    random_vector_r2s[i] = r2_score(all_vector_outputs, vector_predictions)
                    if output_type == "error":
                        norm_predictions = n_regressor.predict(all_inputs[:, random_cols])
                        random_norm_r2s[i] = r2_score(all_norm_errors, norm_predictions)
                print(f"{phase} Random Vector Avg. R²: {random_vector_r2s.mean():.4f}")
                if output_type == "error":
                    print(f"{phase} Random Norm Avg. R²: {random_norm_r2s.mean():.4f}")

                print(f"{phase} Predicting with PCA...")
                inputs_pca = resid_pca.transform(all_inputs)
                vector_pca_predictions = vector_pca_regressor.predict(inputs_pca)
                vector_pca_r2 = r2_score(all_vector_outputs, vector_pca_predictions)
                print(f"{phase} PCA Vector R²: {vector_pca_r2:.4f}")
                if output_type == "error":
                    norm_pca_predictions = norm_pca_regressor.predict(inputs_pca)
                    norm_pca_r2 = r2_score(all_norm_errors, norm_pca_predictions)
                    print(f"{phase} PCA Norm R²: {norm_pca_r2:.4f}")

    return vector_regressor, norm_regressor


def main(
    model: UDTransformer,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    device: torch.device,
    layer: int = 20,
    sae_width: int = 16
):
    """Main analysis function."""
    print(f"\nAnalyzing SAE errors for layer {layer}...")

    # Load pretrained SAE
    sae_release = "gemma-scope-2b-pt-res-canonical"
    sae_id = f"layer_{layer}/width_{sae_width}k/canonical"
    sae = SAE.from_pretrained(sae_release, sae_id, device=str(device))[0]

    # Train predictors for both approaches
    for input_type, output_type in [("dependencies", "activations"), ("resid_stream", "error"), ("dependencies", "error")]:
        print(f"\nTraining {input_type}-based predictor for {output_type}...")
        vector_regressor, norm_regressor = train_error_predictor(
            model=model,
            sae=sae,
            train_loader=train_loader,
            dev_loader=dev_loader,
            layer=layer,
            device=device,
            input_type=input_type,
            output_type=output_type
        )

        # Save predictor
        with open(f"models/error_predictor_layer_{layer}_{input_type}_width_{sae_width}k_vector.pkl", "wb") as f:
            pickle.dump(vector_regressor, f)
        with open(f"models/error_predictor_layer_{layer}_{input_type}_width_{sae_width}k_norm.pkl", "wb") as f:
            pickle.dump(norm_regressor, f)

    # Print comparison
    # print("\nResults:")
    # print(f"Residual Stream Vector R²: {results['residual']['vector_r2']:.4f}")
    # print(f"Dependencies Vector R²: {results['dependencies']['vector_r2']:.4f}")
    # print(f"Residual Stream Norm R²: {results['residual']['norm_r2']:.4f}")
    # print(f"Dependencies Norm R²: {results['dependencies']['norm_r2']:.4f}")
    # print(f"Residual Stream Random Vector Avg. R²: {results['residual']['random_vector_r2s'].mean():.4f}")
    # print(f"Residual Stream Random Norm Avg. R²: {results['residual']['random_norm_r2s'].mean():.4f}")

    cleanup_gpu_memory()
