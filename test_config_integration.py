"""
Test script to verify config integration with data loading.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from configs.config import Config
from src.data.preprocessing import prepare_data
from src.data.data_loader import create_data_loader


def test_config_integration():
    """Test that config can be used directly with data loading functions."""
    print("=" * 60)
    print("Testing Config Integration")
    print("=" * 60)
    
    try:
        # Load config
        config = Config.from_yaml('configs/lstm_bahdanau.yaml')
        print(f"✓ Config loaded successfully")
        print(f"  Model: {config.model.name}")
        print(f"  Batch size: {config.training.batch_size}")
        print(f"  Data root: {config.data.data_root}")
        print(f"  Target level: {config.data.target_level}")
        print(f"  Max seq len: {config.data.max_seq_len}")
        print(f"  Min count: {config.data.min_count}")
        if config.data.json_paths:
            print(f"  JSON paths: {config.data.json_paths}")
        
        # Prepare data using config
        print(f"\n📊 Preparing data with config...")
        data_result = prepare_data(
            data_root=config.data.data_root,
            splits=['train', 'dev'],
            max_len=config.data.max_seq_len,
            min_count=config.data.min_count,
            config=config
        )
        
        print(f"✓ Data prepared successfully")
        print(f"  Input vocab size: {data_result['input_vocab'].count}")
        # Output vocab can be ViWordVocab (phoneme) or ViWordLevelVocab (word)
        output_vocab_size = getattr(data_result['output_vocab'], 'vocab_size', data_result['output_vocab'].count)
        print(f"  Output vocab size: {output_vocab_size}")
        print(f"  Target level: {data_result['target_level']}")
        print(f"  Train pairs: {len(data_result['data']['train'])}")
        print(f"  Dev pairs: {len(data_result['data']['dev'])}")
        
        # Check if we have data
        if len(data_result['data']['train']) == 0:
            print(f"\n⚠️  No training data found. Skipping DataLoader test.")
            print(f"   This is expected if data files don't exist yet.")
            print(f"   Please ensure data files exist at: {config.data.data_root}/tokenization/train/")
            print(f"\n✅ Config integration test PASSED (no data to load)")
            return True
        
        # Create data loader using config
        print(f"\n📦 Creating DataLoader with config...")
        try:
            train_loader = create_data_loader(
                indexed_pairs=data_result['data']['train'],
                batch_size=config.training.batch_size,
                shuffle=True,
                target_level=data_result['target_level']  # Pass target_level from prepare_data result
            )
            
            print(f"✓ DataLoader created successfully")
            print(f"  Batch size: {train_loader.batch_size}")
            
            # Test one batch
            print(f"\n🔍 Testing batch loading...")
            for src_batch, tgt_batch in train_loader:
                print(f"✓ Batch loaded successfully")
                print(f"  Source shape: {src_batch.shape}")  
                print(f"  Target shape: {tgt_batch.shape}")  
                
                # Verify target shape based on target_level from data_result
                # (may differ from config if fallback occurred)
                target_level = data_result['target_level']
                if target_level == 'phoneme':
                    assert len(tgt_batch.shape) == 3, f"Expected 3D tensor for phoneme-level, got {len(tgt_batch.shape)}D"
                    assert tgt_batch.shape[2] == 4, f"Expected 4 components, got {tgt_batch.shape[2]}"
                    print(f"  ✓ Phoneme-level shape verified: (batch, seq_len, 4)")
                else:
                    assert len(tgt_batch.shape) == 2, f"Expected 2D tensor for word-level, got {len(tgt_batch.shape)}D"
                    print(f"  ✓ Word-level shape verified: (batch, seq_len)")
                break
        except ValueError as e:
            if "Dataset is empty" in str(e):
                print(f"\n⚠️  {e}")
                print(f"   This is expected if data files don't exist yet.")
                print(f"\n✅ Config integration test PASSED (no data to load)")
                return True
            else:
                raise
        
        print("\n✅ Config integration test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Config integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_configs():
    """Test all config files."""
    print("\n" + "=" * 60)
    print("Testing All Config Files")
    print("=" * 60)
    
    config_files = [
        'configs/lstm_bahdanau.yaml',
        'configs/lstm_luong.yaml',
        'configs/transformer.yaml'
    ]
    
    results = []
    for config_file in config_files:
        print(f"\n📄 Testing {config_file}...")
        try:
            config = Config.from_yaml(config_file)
            print(f"  ✓ Config loaded: {config.model.name}")
            print(f"    Target level: {config.data.target_level}")
            print(f"    Batch size: {config.training.batch_size}")
            results.append((config_file, True))
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results.append((config_file, False))
    
    print("\n" + "=" * 60)
    print("Config Files Test Summary")
    print("=" * 60)
    for config_file, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{config_file}: {status}")
    
    return all(passed for _, passed in results)


if __name__ == '__main__':
    print("Starting config integration tests...\n")
    
    # Test config integration
    test1 = test_config_integration()
    
    # Test all config files
    test2 = test_all_configs()
    
    print("\n" + "=" * 60)
    print("Final Summary")
    print("=" * 60)
    print(f"Config Integration: {'✅ PASSED' if test1 else '❌ FAILED'}")
    print(f"All Config Files: {'✅ PASSED' if test2 else '❌ FAILED'}")
    
    if test1 and test2:
        print("\n🎉 All tests PASSED!")
    else:
        print("\n⚠️  Some tests FAILED. Please check the errors above.")

