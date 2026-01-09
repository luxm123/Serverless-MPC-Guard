
import os
import sys
import time
import threading
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Mock boto3 before importing middleware
with patch('boto3.client') as mock_boto:
    from src.mpc import middleware
    from src.mpc.middleware import MPCMiddleware

def test_async_sqs_update():
    # Setup
    os.environ['MPC_MAIN_QUEUE_URL'] = 'https://sqs.us-east-1.amazonaws.com/123/my-queue'
    middleware.MAIN_QUEUE_URL = 'https://sqs.us-east-1.amazonaws.com/123/my-queue' # Force update global var
    
    # Mock SQS response with delay to prove async
    mock_sqs = MagicMock()
    def delayed_sqs(*args, **kwargs):
        time.sleep(1.0) # Simulate network lag
        return {'Attributes': {'ApproximateNumberOfMessages': '42'}}
    
    mock_sqs.get_queue_attributes.side_effect = delayed_sqs
    middleware.sqs = mock_sqs # Replace module-level sqs client
    
    # Reset Cache
    middleware._L1_CACHE = {
        'params': None,
        'version': 0,
        'last_sync': 0,
        'last_backlog': None, # Start empty
        'last_backlog_sync': 0,
        'updating_backlog': False
    }

    mw = MPCMiddleware()
    
    # Test 1: First call - Cache Miss, Async Trigger
    event = {
        'metrics': {'p90': 100.0}, # No queue_backlog provided
        'task': {'id': 't1'}
    }
    
    print("Step 1: Calling decide() with empty cache...")
    start_t = time.time()
    decision, dbg = mw.decide(event)
    duration = time.time() - start_t
    
    print(f"Decide took {duration*1000:.2f}ms")
    print(f"DEBUG INFO: {dbg}")
    
    # Should return default cold value immediately (because SQS is sleeping)
    assert dbg['queue_backlog_source'] == 'default_cold'
    assert dbg['queue_backlog'] == 0.0
    
    # Thread should have been started
    # Give the thread a moment to finish sleeping and update
    print("Waiting for background thread...")
    time.sleep(1.5)
    
    # Verify SQS was called
    mock_sqs.get_queue_attributes.assert_called_once()
    print("SQS was called asynchronously!")
    
    # Verify Cache was updated
    assert middleware._L1_CACHE['last_backlog'] == 42.0
    print(f"Cache updated to: {middleware._L1_CACHE['last_backlog']}")
    
    # Test 2: Second call - Should use cached value
    print("\nStep 2: Calling decide() again...")
    decision, dbg = mw.decide(event)
    
    assert dbg['queue_backlog_source'] == 'sqs_cache'
    assert dbg['queue_backlog'] == 42.0
    print("Used cached value successfully!")

    print("\nAsync SQS Test Passed!")

if __name__ == "__main__":
    test_async_sqs_update()
