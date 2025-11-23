"""
Configuration management using Pydantic for type validation and dot notation access.
Provides structured configuration with automatic validation.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Literal, Dict
import yaml
from pathlib import Path


class ModelConfig(BaseModel):
    """Model architecture configuration"""
    name: str = Field(..., description="Model name")
    embed_dim: int = Field(512, description="Embedding dimension")
    hidden_dim: int = Field(512, description="Hidden dimension")
    num_layers: int = Field(2, description="Number of layers (fallback if encoder_layers/decoder_layers not specified)")
    dropout: float = Field(0.1, ge=0.0, le=1.0, description="Dropout rate")
    
    # Separate encoder/decoder layers (optional, falls back to num_layers)
    encoder_layers: Optional[int] = Field(None, description="Number of encoder layers")
    decoder_layers: Optional[int] = Field(None, description="Number of decoder layers")
    
    # For LSTM models
    attention_type: Optional[str] = Field(None, description="Attention type: bahdanau, general, dot, concat")
    
    # For Transformer models
    num_heads: Optional[int] = Field(8, description="Number of attention heads")
    ff_dim: Optional[int] = Field(2048, description="Feed-forward dimension")
    
    @validator('attention_type')
    def validate_attention_type(cls, v):
        """Validate attention type"""
        if v is not None and v not in ['bahdanau', 'general', 'dot', 'concat']:
            raise ValueError(f"Invalid attention_type: {v}. Must be one of: bahdanau, general, dot, concat")
        return v


class TrainingConfig(BaseModel):
    """Training hyperparameters"""
    batch_size: int = Field(32, gt=0, description="Batch size")
    num_epochs: int = Field(50, gt=0, description="Number of epochs")
    learning_rate: float = Field(0.001, gt=0.0, description="Learning rate")
    optimizer: Literal["adam", "sgd", "adamw"] = Field("adam", description="Optimizer type")
    scheduler: Optional[str] = Field("cosine", description="Learning rate scheduler")
    clip_grad_norm: float = Field(5.0, gt=0.0, description="Gradient clipping threshold")
    warmup_steps: Optional[int] = Field(None, description="Warmup steps for transformer")
    
    # Evaluation
    eval_every: int = Field(1000, gt=0, description="Evaluate every N steps")
    save_every: int = Field(5000, gt=0, description="Save checkpoint every N steps")


class DataConfig(BaseModel):
    """Data paths and vocabulary configuration"""
    sos_id: int = Field(1, description="Start of sentence token ID")
    eos_id: int = Field(2, description="End of sentence token ID")
    pad_id: int = Field(0, description="Padding token ID")
    unk_id: int = Field(3, description="Unknown token ID")
    
    # Level: 'word' or 'phoneme' (applies to both source and target)
    source_level: Literal["word", "phoneme"] = Field("word", description="Sequence level: word or phoneme (applies to both source and target)")
    target_level: Literal["word", "phoneme"] = Field("word", description="Sequence level: word or phoneme (should match source_level)")
    
    @validator('target_level')
    def validate_levels_match(cls, v, values):
        """Ensure source_level and target_level match (only word or phoneme, not mixed)"""
        if 'source_level' in values:
            source_level = values.get('source_level')
            if v != source_level:
                raise ValueError(
                    f"source_level and target_level must match. "
                    f"Got source_level={source_level}, target_level={v}. "
                    f"Only 'word' or 'phoneme' level is supported (not mixed)."
                )
        return v
    
    # Direct paths to training data files
    train_src: str = Field(..., description="Direct path to training source data file (.en)")
    train_tgt: str = Field(..., description="Direct path to training target data file (.vi)")
    
    # Direct paths to dev data files
    dev_src: str = Field(..., description="Direct path to dev source data file (.en)")
    dev_tgt: str = Field(..., description="Direct path to dev target data file (.vi)")
    
    # Direct paths to test data files
    test_src: str = Field(..., description="Direct path to test source data file (.en)")
    test_tgt: str = Field(..., description="Direct path to test target data file (.vi)")
    
    # Direct paths to English IPA vocabulary JSON files (for English phoneme-level source)
    vocab_json_train: Optional[str] = Field(None, description="Direct path to English IPA vocabulary JSON file (for EN phoneme-level)")
    vocab_json_dev: Optional[str] = Field(None, description="Direct path to English IPA vocabulary JSON file (for EN phoneme-level)")
    vocab_json_test: Optional[str] = Field(None, description="Direct path to English IPA vocabulary JSON file (for EN phoneme-level)")
    
    # Vocabulary and sequence settings
    min_count: int = Field(3, ge=1, description="Minimum word count for vocabulary")
    max_seq_len: int = Field(100, gt=0, description="Maximum sequence length")


class Config(BaseModel):
    """
    Main configuration class for NMT models.
    
    Usage:
        config = Config.from_yaml('configs/lstm_bahdanau.yaml')
        print(config.model.embed_dim)
        print(config.training.batch_size)
    """
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    device: str = Field("cuda", description="Device: cuda or cpu")
    seed: Optional[int] = Field(42, description="Random seed")
    
    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> 'Config':
        """
        Load configuration from YAML file with Pydantic validation.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            Config object with validated fields
            
        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValidationError: If configuration is invalid
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return cls(**config_data)
    
    def save_yaml(self, yaml_path: str | Path):
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML file
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                self.model_dump(),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False
            )
    
    def get(self, key: str, default=None):
        """
        Get nested configuration value using dot notation.
        
        Args:
            key: Dot-separated key (e.g., 'model.embed_dim')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self
        
        try:
            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                else:
                    return default
            return value
        except (AttributeError, KeyError):
            return default
    
    class Config:
        """Pydantic config."""
        extra = "allow"  # Allow extra fields
        validate_assignment = True  # Validate on assignment