import pytest
import asyncio
import time
from unittest.mock import MagicMock, patch
from llm_inference_engine.utils import ConcurrencyLimiter, SlideWindowRateLimiter

# -----------------------------------------------------------------------------
# Test ConcurrencyLimiter
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrency_limiter_strict_limit():
    """
    Verify that the ConcurrencyLimiter strictly enforces the maximum number 
    of concurrent tasks.
    """
    max_concurrent = 2
    total_tasks = 10
    limiter = ConcurrencyLimiter(max_concurrent_requests=max_concurrent)
    
    # Shared counter to track currently active tasks
    active_tasks = 0
    max_active_seen = 0
    
    async def worker():
        nonlocal active_tasks, max_active_seen
        
        await limiter.acquire()
        try:
            active_tasks += 1
            max_active_seen = max(max_active_seen, active_tasks)
            # Simulate work
            await asyncio.sleep(0.01)
        finally:
            active_tasks -= 1
            limiter.release()

    # Launch all tasks simultaneously
    tasks = [asyncio.create_task(worker()) for _ in range(total_tasks)]
    await asyncio.gather(*tasks)

    # Assertions
    assert max_active_seen <= max_concurrent, \
        f"Concurrency limit exceeded! Seen {max_active_seen}, Limit {max_concurrent}"
    assert active_tasks == 0, "All tasks should have finished."


@pytest.mark.asyncio
async def test_concurrency_limiter_no_block_under_limit():
    """
    Verify that tasks run immediately if under the limit.
    """
    limiter = ConcurrencyLimiter(max_concurrent_requests=5)
    
    start_time = time.time()
    await limiter.acquire()
    limiter.release()
    duration = time.time() - start_time
    
    # Should be nearly instantaneous
    assert duration < 0.1


# -----------------------------------------------------------------------------
# Test SlideWindowRateLimiter
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rate_limiter_allow_requests_under_limit():
    """
    Verify requests pass through when under the rate limit.
    """
    limit = 5
    limiter = SlideWindowRateLimiter(max_requests_per_minute=limit)
    
    # Mock time to stay constant (instantaneous bursts)
    with patch('time.monotonic', return_value=100.0):
        # We should be able to acquire 'limit' times without blocking
        for _ in range(limit):
            # We wrap in wait_for to ensure it doesn't hang (block)
            await asyncio.wait_for(limiter.acquire(), timeout=0.1)
            
    assert len(limiter.request_timestamps) == limit


@pytest.mark.asyncio
async def test_rate_limiter_blocks_over_limit():
    """
    Verify that the limiter blocks when the rate limit is exceeded, 
    and releases after the window passes.
    """
    limit = 2
    limiter = SlideWindowRateLimiter(max_requests_per_minute=limit)
    
    # Helper to manage virtual time
    current_time = 0.0
    
    def mock_time():
        return current_time

    async def mock_sleep(seconds):
        nonlocal current_time
        # Add a tiny epsilon to ensure we cross the threshold
        current_time += seconds + 0.001

    with patch('time.monotonic', side_effect=mock_time), \
         patch('asyncio.sleep', side_effect=mock_sleep) as mocked_sleep:
        
        # 1. First request (Time 0.0) -> Allowed
        await limiter.acquire()
        
        # 2. Advance time manually so the second request has a distinct timestamp
        current_time = 10.0
        
        # 3. Second request (Time 10.0) -> Allowed (Limit is 2)
        await limiter.acquire()
        
        assert mocked_sleep.call_count == 0
        assert len(limiter.request_timestamps) == 2
        
        # 4. Third request (Time 10.0) -> Should Block
        # The oldest request was at 0.0. Window is 60.0.
        # It needs to wait until T=60.0 to clear the first request.
        # Wait needed = (0.0 + 60.0) - 10.0 = 50.0s.
        await limiter.acquire()
        
        # Verify it slept
        assert mocked_sleep.call_count == 1
        
        # It should sleep 50.0s. Mock updates current_time: 10.0 + 50.0 + 0.001 = 60.001
        assert current_time >= 60.0
        
        # The first request (0.0) should be popped: 60.001 - 0.0 > 60.0 (True)
        # The second request (10.0) should REMAIN: 60.001 - 10.0 = 50.001 < 60.0 (False)
        # The third request (60.001) is added.
        
        assert len(limiter.request_timestamps) == 2
        assert limiter.request_timestamps[0] == 10.0 
        assert abs(limiter.request_timestamps[1] - 60.0) < 0.1


@pytest.mark.asyncio
async def test_rate_limiter_refills_after_window():
    """
    Verify that the rate limiter completely resets after the window passes.
    """
    limit = 5
    limiter = SlideWindowRateLimiter(max_requests_per_minute=limit)
    
    current_time = 0.0
    
    # Simple lambda won't work if we need to update current_time, 
    # but here we manually update it between calls.
    with patch('time.monotonic', side_effect=lambda: current_time):
        # Fill the bucket
        for _ in range(limit):
            await limiter.acquire()
            
        assert len(limiter.request_timestamps) == limit
        
        # Advance time past the window manually
        current_time += 61.0
        
        # Next acquire should trigger the cleanup logic first
        await limiter.acquire()
        
        # Should now only have 1 active timestamp (the one we just added)
        # because the previous 5 expired.
        assert len(limiter.request_timestamps) == 1