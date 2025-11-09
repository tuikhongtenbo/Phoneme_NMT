"""
Test script to load test data from dataset/vocabs/clean/
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from configs.config import Config
from src.data.preprocessing import prepare_data
from src.data.data_loader import create_data_loader


def test_load_test_data():
    """Test loading test data."""
    print("=" * 60)
    print("Testing Test Data Loading")
    print("=" * 60)
    
    try:
        # Load config
        config = Config.from_yaml('configs/lstm_bahdanau.yaml')
        print(f"✓ Config loaded successfully")
        print(f"  Data root: {config.data.data_root}")
        print(f"  Target level: {config.data.target_level}")
        
        # Prepare data with train and test splits
        # Need train split to build vocabularies
        print(f"\n📊 Preparing data with train and test splits...")
        print(f"   (Train split needed to build vocabularies)")
        data_result = prepare_data(
            data_root=config.data.data_root,
            splits=['train', 'test'],  # Load train to build vocab, test to test
            max_len=config.data.max_seq_len,
            min_count=config.data.min_count,
            config=config
        )
        
        print(f"\n✓ Data prepared successfully")
        print(f"  Input vocab size: {data_result['input_vocab'].count}")
        output_vocab_size = getattr(data_result['output_vocab'], 'vocab_size', data_result['output_vocab'].count)
        print(f"  Output vocab size: {output_vocab_size}")
        print(f"  Target level: {data_result['target_level']}")
        print(f"  Test pairs: {len(data_result['data']['test'])}")
        
        # Check if we have data
        if len(data_result['data']['test']) == 0:
            print(f"\n⚠️  No test data loaded.")
            return False
        
        # Show first few examples
        print(f"\n📝 First 3 examples:")
        for i, (en_indices, vi_indices) in enumerate(data_result['data']['test'][:3]):
            print(f"\n  Example {i+1}:")
            print(f"    EN indices (first 10): {en_indices[:10]}...")
            if data_result['target_level'] == 'phoneme':
                print(f"    VI indices (first 3 syllables): {vi_indices[:3]}...")
                print(f"    VI shape: {len(vi_indices)} syllables, each with 4 components")
            else:
                print(f"    VI indices (first 10): {vi_indices[:10]}...")
        
        # Create data loader
        print(f"\n📦 Creating DataLoader...")
        test_loader = create_data_loader(
            indexed_pairs=data_result['data']['test'],
            batch_size=min(8, len(data_result['data']['test'])),  # Small batch for test
            shuffle=False,
            target_level=data_result['target_level']
        )
        
        print(f"✓ DataLoader created successfully")
        print(f"  Batch size: {test_loader.batch_size}")
        print(f"  Number of batches: {len(test_loader)}")
        
        # Test one batch
        print(f"\n🔍 Testing batch loading...")
        for batch_idx, (src_batch, tgt_batch) in enumerate(test_loader):
            print(f"✓ Batch {batch_idx + 1} loaded successfully")
            print(f"  Source shape: {src_batch.shape}")
            print(f"  Target shape: {tgt_batch.shape}")
            
            if data_result['target_level'] == 'phoneme':
                print(f"  ✓ Phoneme-level: (batch, seq_len, 4)")
            else:
                print(f"  ✓ Word-level: (batch, seq_len)")
            
            # Only show first batch
            break
        
        print("\n✅ Test data loading PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Test data loading FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_load_test_data()
    if success:
        print("\n🎉 Test data loaded successfully!")
    else:
        print("\n⚠️  Test data loading failed. Please check the errors above.")

