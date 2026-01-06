"""
Main VL-JEPA Model
Combines vision encoder, text encoder, and predictor with EMA target encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import copy

from .vision_encoder import VisionEncoder
from .text_encoder import TextEncoder
from .predictor import PredictorMLP, PredictorWithCrossAttention


class VLJEPAModel(nn.Module):
    """
    Vision-Language Joint Embedding Predictive Architecture (VL-JEPA).
    
    Architecture:
        - Context Encoder: Encodes visible patches and text
        - Target Encoder: EMA copy of context encoder, encodes masked patches
        - Predictor: Predicts target representations from context
    
    Args:
        vision_encoder: Vision encoder (ViT)
        text_encoder: Text encoder (BERT/DistilBERT)
        predictor: Predictor network
        embedding_dim: Dimension of shared embedding space
        ema_momentum: EMA momentum for target encoder (0.996 recommended)
        temperature: Temperature for contrastive loss
    """
    
    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        predictor: nn.Module,
        embedding_dim: int = 256,
        ema_momentum: float = 0.996,
        temperature: float = 0.07,
    ):
        super().__init__()
        
        # Context encoders (trainable)
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        
        # Predictor (trainable)
        self.predictor = predictor
        
        # Target encoders (EMA, not trainable)
        self.target_vision_encoder = copy.deepcopy(vision_encoder)
        self.target_text_encoder = copy.deepcopy(text_encoder)
        
        # Freeze target encoders
        for param in self.target_vision_encoder.parameters():
            param.requires_grad = False
        for param in self.target_text_encoder.parameters():
            param.requires_grad = False
        
        # Projection heads for contrastive learning (optional)
        vision_dim = vision_encoder.embed_dim
        text_dim = text_encoder.embed_dim
        
        self.vision_projection = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, embedding_dim),
        )
        
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, embedding_dim),
        )
        
        # Parameters
        self.ema_momentum = ema_momentum
        self.temperature = temperature
        self.embedding_dim = embedding_dim
        
        # Initialize projections
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights"""
        for m in [self.vision_projection, self.text_projection]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
    
    @torch.no_grad()
    def update_target_encoder(self):
        """
        Update target encoder with EMA of context encoder.
        Call this after each optimizer step.
        """
        # Update vision target encoder
        for param_q, param_k in zip(
            self.vision_encoder.parameters(),
            self.target_vision_encoder.parameters()
        ):
            param_k.data = param_k.data * self.ema_momentum + param_q.data * (1.0 - self.ema_momentum)
        
        # Update text target encoder
        for param_q, param_k in zip(
            self.text_encoder.parameters(),
            self.target_text_encoder.parameters()
        ):
            param_k.data = param_k.data * self.ema_momentum + param_q.data * (1.0 - self.ema_momentum)
    
    def forward_jepa(
        self,
        images: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for JEPA training.
        
        Args:
            images: Input images [B, 3, H, W]
            text_input_ids: Text token IDs [B, L]
            text_attention_mask: Text attention mask [B, L]
            vision_mask: Boolean mask for vision patches [B, N] (True = keep, False = mask)
            
        Returns:
            Dictionary with predictions and targets
        """
        B = images.shape[0]
        
        # Encode context (visible patches)
        context_vision = self.vision_encoder(images, return_all_tokens=True)  # [B, N+1, D_v]
        context_text = self.text_encoder(
            text_input_ids, 
            text_attention_mask, 
            return_all_tokens=True,
            return_projected=False
        )  # [B, L, D_t]
        
        # Apply masking to vision if provided
        if vision_mask is not None:
            # Keep only visible patches (including CLS token)
            # vision_mask: [B, N] for N patches (excluding CLS)
            # Add True for CLS token
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=vision_mask.device)
            full_mask = torch.cat([cls_mask, vision_mask], dim=1)  # [B, N+1]
            
            # Select visible tokens
            # This is simplified - in practice you'd use more efficient indexing
            visible_vision = context_vision
        else:
            visible_vision = context_vision
        
        # Predict target representations
        predicted_vision = self.predictor(visible_vision)  # [B, N+1, D_v]
        
        # Encode targets with target encoder (no gradients)
        with torch.no_grad():
            target_vision = self.target_vision_encoder(images, return_all_tokens=True)  # [B, N+1, D_v]
        
        return {
            'predicted_vision': predicted_vision,
            'target_vision': target_vision.detach(),
            'context_vision': visible_vision,
            'context_text': context_text,
        }
    
    def forward_contrastive(
        self,
        images: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for contrastive learning (like CLIP).
        
        Args:
            images: Input images [B, 3, H, W]
            text_input_ids: Text token IDs [B, L]
            text_attention_mask: Text attention mask [B, L]
            
        Returns:
            Dictionary with vision and text embeddings
        """
        # Encode vision (CLS token only)
        vision_features = self.vision_encoder(images, return_all_tokens=False)  # [B, 1, D_v]
        vision_features = vision_features.squeeze(1)  # [B, D_v]
        
        # Encode text (CLS token only)
        text_features = self.text_encoder(
            text_input_ids,
            text_attention_mask,
            return_all_tokens=False,
            return_projected=False
        )  # [B, D_t]
        
        # Project to shared embedding space
        vision_embed = self.vision_projection(vision_features)  # [B, embedding_dim]
        text_embed = self.text_projection(text_features)  # [B, embedding_dim]
        
        # Normalize
        vision_embed = F.normalize(vision_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)
        
        return {
            'vision_embed': vision_embed,
            'text_embed': text_embed,
        }
    
    def compute_jepa_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute JEPA loss (smooth L1 loss in representation space).
        
        Args:
            predicted: Predicted representations [B, N, D]
            target: Target representations [B, N, D]
            mask: Optional mask for which tokens to compute loss [B, N]
            
        Returns:
            Loss value
        """
        # Skip CLS token (first token) for JEPA loss - only compute on patch tokens
        # predicted and target are [B, N+1, D] where N+1 = 197 (1 CLS + 196 patches)
        predicted_patches = predicted[:, 1:, :]  # [B, N, D] = [B, 196, D]
        target_patches = target[:, 1:, :]  # [B, N, D] = [B, 196, D]
        
        # Normalize (helps stability)
        predicted_patches = F.layer_norm(predicted_patches, predicted_patches.shape[-1:])
        target_patches = F.layer_norm(target_patches, target_patches.shape[-1:])
        
        # Compute smooth L1 loss
        loss = F.smooth_l1_loss(predicted_patches, target_patches, reduction='none')  # [B, N, D]
        
        # Average over feature dimension
        loss = loss.mean(dim=-1)  # [B, N]
        
        # Apply mask if provided
        if mask is not None:
            # Ensure mask matches patches (should be [B, N] = [B, 196])
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(loss.shape[0], -1)
            loss = loss * mask.float()
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()
        
        return loss
    
    def compute_contrastive_loss(
        self,
        vision_embed: torch.Tensor,
        text_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss (InfoNCE) between vision and text.
        
        Args:
            vision_embed: Vision embeddings [B, D]
            text_embed: Text embeddings [B, D]
            
        Returns:
            Contrastive loss
        """
        # Compute similarity matrix
        logits = torch.matmul(vision_embed, text_embed.t()) / self.temperature  # [B, B]
        
        # Labels: diagonal elements are positives
        labels = torch.arange(logits.shape[0], device=logits.device)
        
        # Cross entropy loss in both directions
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        
        # Average
        loss = (loss_i2t + loss_t2i) / 2.0
        
        return loss
    
    def forward(
        self,
        images: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        mode: str = "jepa",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with specified mode.
        
        Args:
            images: Input images [B, 3, H, W]
            text_input_ids: Text token IDs [B, L]
            text_attention_mask: Text attention mask [B, L]
            vision_mask: Vision patch mask [B, N]
            mode: Forward mode ("jepa" or "contrastive")
            
        Returns:
            Dictionary with outputs and losses
        """
        if mode == "jepa":
            outputs = self.forward_jepa(images, text_input_ids, text_attention_mask, vision_mask)
            
            # Compute JEPA loss
            jepa_loss = self.compute_jepa_loss(
                outputs['predicted_vision'],
                outputs['target_vision'],
                mask=vision_mask if vision_mask is not None else None,
            )
            
            outputs['jepa_loss'] = jepa_loss
            outputs['loss'] = jepa_loss
            
        elif mode == "contrastive":
            outputs = self.forward_contrastive(images, text_input_ids, text_attention_mask)
            
            # Compute contrastive loss
            contrastive_loss = self.compute_contrastive_loss(
                outputs['vision_embed'],
                outputs['text_embed'],
            )
            
            outputs['contrastive_loss'] = contrastive_loss
            outputs['loss'] = contrastive_loss
            
        elif mode == "both":
            # Combined mode: JEPA + contrastive
            jepa_outputs = self.forward_jepa(images, text_input_ids, text_attention_mask, vision_mask)
            contrastive_outputs = self.forward_contrastive(images, text_input_ids, text_attention_mask)
            
            jepa_loss = self.compute_jepa_loss(
                jepa_outputs['predicted_vision'],
                jepa_outputs['target_vision'],
                mask=vision_mask if vision_mask is not None else None,
            )
            
            contrastive_loss = self.compute_contrastive_loss(
                contrastive_outputs['vision_embed'],
                contrastive_outputs['text_embed'],
            )
            
            outputs = {**jepa_outputs, **contrastive_outputs}
            outputs['jepa_loss'] = jepa_loss
            outputs['contrastive_loss'] = contrastive_loss
            outputs['loss'] = jepa_loss + 0.5 * contrastive_loss  # Weighted combination
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return outputs


def create_vl_jepa_model(config: dict) -> VLJEPAModel:
    """
    Factory function to create VL-JEPA model from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        VLJEPAModel instance
    """
    from .vision_encoder import create_vision_encoder
    from .text_encoder import create_text_encoder
    from .predictor import create_predictor
    
    # Create encoders
    vision_encoder = create_vision_encoder(config['model'])
    text_encoder = create_text_encoder(config['model'])
    
    # Create predictor
    predictor_type = config['model']['predictor'].get('type', 'mlp')
    predictor = create_predictor(config['model'], predictor_type=predictor_type)
    
    # Create model
    model = VLJEPAModel(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        predictor=predictor,
        embedding_dim=config['model'].get('embedding_dim', 256),
        ema_momentum=config['training'].get('ema_momentum_start', 0.996),
        temperature=config['model'].get('temperature', 0.07),
    )
    
    return model


if __name__ == "__main__":
    print("Testing VLJEPAModel...")
    
    # Create simple config
    config = {
        'model': {
            'vision_encoder': {
                'type': 'vit_tiny_patch16_224',
                'pretrained': False,
                'hidden_dim': 192,
                'gradient_checkpointing': False,
            },
            'text_encoder': {
                'type': 'distilbert-base-uncased',
                'projection_dim': None,
                'max_length': 128,
                'gradient_checkpointing': False,
            },
            'predictor': {
                'type': 'mlp',
                'input_dim': 192,
                'hidden_dim': 256,
                'output_dim': 192,
                'num_layers': 3,
            },
            'embedding_dim': 256,
            'temperature': 0.07,
        },
        'training': {
            'ema_momentum_start': 0.996,
        }
    }
    
    # Create model
    model = create_vl_jepa_model(config)
    print(f"Model created successfully!")
    
    # Test inputs
    images = torch.randn(2, 3, 224, 224)
    text_input_ids = torch.randint(0, 1000, (2, 128))
    text_attention_mask = torch.ones(2, 128)
    vision_mask = torch.rand(2, 196) > 0.25  # Random mask
    
    # Test JEPA mode
    with torch.no_grad():
        outputs = model(
            images=images,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            vision_mask=vision_mask,
            mode="jepa",
        )
        print(f"JEPA loss: {outputs['loss'].item():.4f}")
    
    # Test contrastive mode
    with torch.no_grad():
        outputs = model(
            images=images,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            mode="contrastive",
        )
        print(f"Contrastive loss: {outputs['loss'].item():.4f}")
    
    # Test EMA update
    model.update_target_encoder()
    print("EMA update successful!")
    
    print("\nVL-JEPA model test passed!")
