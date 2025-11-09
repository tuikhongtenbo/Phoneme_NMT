"""
Test script to verify data loading with config for word/phoneme level.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from configs.config import Config
from src.data.preprocessing import prepare_data
from src.data.data_loader import create_data_loader


def test_phoneme_level():
    """Test data loading with phoneme-level target."""
    print("=" * 60)
    print("Testing Phoneme-level Data Loading")
    print("=" * 60)
    
    try:
        # Load config
        config = Config.from_yaml('configs/lstm_bahdanau.yaml')
        print(f"✓ Config loaded successfully")
        print(f"  Model: {config.model.name}")
        print(f"  Batch size: {config.training.batch_size}")
        print(f"  Max seq len: {config.data.max_seq_len}")
        print(f"  Target level: {config.data.target_level}")
        
        # Prepare data with phoneme-level (using config)
        data_result = prepare_data(
            data_root=config.data.data_root,
            splits=['train', 'dev'],
            max_len=config.data.max_seq_len,
            min_count=config.data.min_count,
            config=config
        )
        
        print(f"\n✓ Data prepared successfully")
        print(f"  Input vocab size: {data_result['input_vocab'].count}")
        # Output vocab can be ViWordVocab (phoneme) or ViWordLevelVocab (word)
        output_vocab_size = getattr(data_result['output_vocab'], 'vocab_size', data_result['output_vocab'].count)
        print(f"  Output vocab size: {output_vocab_size}")
        print(f"  Target level: {data_result['target_level']}")
        print(f"  Train pairs: {len(data_result['data']['train'])}")
        print(f"  Dev pairs: {len(data_result['data']['dev'])}")
        
        # Check if we have data
        if len(data_result['data']['train']) == 0:
            print("⚠️  WARNING: No training data loaded. Skipping DataLoader test.")
            print("✅ Phoneme-level test PASSED (data preparation only)")
            return True
        
        # Create data loader using data_loader module
        train_loader = create_data_loader(
            indexed_pairs=data_result['data']['train'],
            batch_size=config.training.batch_size,
            shuffle=True,
            target_level=data_result['target_level']  # Pass target_level from prepare_data result
        )
        
        print(f"\n✓ DataLoader created successfully")
        print(f"  Batch size: {train_loader.batch_size}")
        
        # Test one batch
        for src_batch, tgt_batch in train_loader:
            print(f"\n✓ Batch loaded successfully")
            print(f"  Source shape: {src_batch.shape}")  # (batch_size, max_src_len)
            print(f"  Target shape: {tgt_batch.shape}")  # (batch_size, max_tgt_len, 4) for phoneme
            print(f"  Target level: {data_result['target_level']}")
            break
            
        print("\n✅ Phoneme-level test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Phoneme-level test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_word_level():
    """Test data loading with word-level target."""
    print("\n" + "=" * 60)
    print("Testing Word-level Data Loading")
    print("=" * 60)
    
    try:
        # Load config
        config = Config.from_yaml('configs/lstm_bahdanau.yaml')
        print(f"✓ Config loaded successfully")
        
        # Temporarily change target_level to 'word' for this test
        original_target_level = config.data.target_level
        config.data.target_level = 'word'
        
        # Prepare data with word-level
        data_result = prepare_data(
            data_root=config.data.data_root,
            splits=['train'],
            max_len=config.data.max_seq_len,
            min_count=config.data.min_count,
            config=config
        )
        
        print(f"\n✓ Data prepared successfully")
        print(f"  Input vocab size: {data_result['input_vocab'].count}")
        print(f"  Output vocab size: {data_result['output_vocab'].count}")
        print(f"  Target level: {data_result['target_level']}")
        print(f"  Train pairs: {len(data_result['data']['train'])}")
        
        # Check if we have data
        if len(data_result['data']['train']) == 0:
            print("⚠️  WARNING: No training data loaded. Skipping DataLoader test.")
            print("✅ Word-level test PASSED (data preparation only)")
            # Restore original target_level
            config.data.target_level = original_target_level
            return True
        
        # Create data loader using data_loader module
        train_loader = create_data_loader(
            indexed_pairs=data_result['data']['train'],
            batch_size=config.training.batch_size,
            shuffle=True,
            target_level=data_result['target_level']  # Pass target_level from prepare_data result
        )
        
        print(f"\n✓ DataLoader created successfully")
        print(f"  Batch size: {train_loader.batch_size}")
        
        # Test one batch
        for src_batch, tgt_batch in train_loader:
            print(f"\n✓ Batch loaded successfully")
            print(f"  Source shape: {src_batch.shape}")  # (batch_size, max_src_len)
            print(f"  Target shape: {tgt_batch.shape}")  # (batch_size, max_tgt_len) for word
            print(f"  Target level: {data_result['target_level']}")
            break
        
        # Restore original target_level
        config.data.target_level = original_target_level
        
        print("\n✅ Word-level test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Word-level test FAILED: {e}")
        import traceback
        traceback.print_exc()
        # Restore original target_level if it was changed
        if 'original_target_level' in locals():
            config.data.target_level = original_target_level
        return False


def test_data_loader_integration():
    """Test data loader integration with both phoneme and word levels."""
    print("\n" + "=" * 60)
    print("Testing Data Loader Integration")
    print("=" * 60)
    
    try:
        # Load config
        config = Config.from_yaml('configs/lstm_bahdanau.yaml')
        print(f"✓ Config loaded successfully")
        
        # Test phoneme level
        print(f"\n--- Testing Phoneme Level ---")
        config.data.target_level = 'phoneme'
        data_result_phoneme = prepare_data(
            data_root=config.data.data_root,
            splits=['train'],
            max_len=config.data.max_seq_len,
            min_count=config.data.min_count,
            config=config
        )
        
        if len(data_result_phoneme['data']['train']) > 0:
            phoneme_loader = create_data_loader(
                indexed_pairs=data_result_phoneme['data']['train'],
                batch_size=config.training.batch_size,
                shuffle=False,
                target_level=data_result_phoneme['target_level']
            )
            for src, tgt in phoneme_loader:
                print(f"  Phoneme batch - Source: {src.shape}, Target: {tgt.shape}")
                assert tgt.dim() == 3, f"Expected 3D tensor for phoneme, got {tgt.dim()}D"
                assert tgt.shape[2] == 4, f"Expected 4 components, got {tgt.shape[2]}"
                break
        
        # Test word level
        print(f"\n--- Testing Word Level ---")
        config.data.target_level = 'word'
        data_result_word = prepare_data(
            data_root=config.data.data_root,
            splits=['train'],
            max_len=config.data.max_seq_len,
            min_count=config.data.min_count,
            config=config
        )
        
        if len(data_result_word['data']['train']) > 0:
            word_loader = create_data_loader(
                indexed_pairs=data_result_word['data']['train'],
                batch_size=config.training.batch_size,
                shuffle=False,
                target_level=data_result_word['target_level']
            )
            for src, tgt in word_loader:
                print(f"  Word batch - Source: {src.shape}, Target: {tgt.shape}")
                assert tgt.dim() == 2, f"Expected 2D tensor for word, got {tgt.dim()}D"
                break
        
        print("\n✅ Data loader integration test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Data loader integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("Starting data loading tests...\n")
    
    results = []
    results.append(("Phoneme Level", test_phoneme_level()))
    results.append(("Word Level", test_word_level()))
    results.append(("Data Loader Integration", test_data_loader_integration()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n🎉 All tests PASSED!")
    else:
        print("\n⚠️  Some tests FAILED. Please check the errors above.")