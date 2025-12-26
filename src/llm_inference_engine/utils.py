import abc
from typing import List, Dict
import asyncio
import time
from collections import deque


class ConcurrencyLimiter:
    def __init__(self, max_concurrent_requests: int):
        """
        This class implements a concurrency limiter using asyncio.Semaphore.

        Parameters:
        -----------
        max_concurrent_requests : int
            Maximum number of concurrent requests allowed.
        """
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def acquire(self):
        """
        Acquires permission to proceed. 
        If the maximum number of concurrent requests is reached, this method blocks until a slot is available.
        """
        await self.semaphore.acquire()

    def release(self):
        """
        Releases a previously acquired permission.
        """
        self.semaphore.release()


class RateLimiter:
    @abc.abstractmethod
    def __init__(self):
        """
        This is an abstract class for rate limiters.
        """
        return NotImplemented

    @abc.abstractmethod
    async def acquire(self):
        """
        Acquires permission to proceed. 
        """
        return NotImplemented


class SlideWindowRateLimiter(RateLimiter):
    def __init__(self, max_requests_per_minute: int):
        """
        This class implements a sliding window rate limiter. It checks the number of requests in the last 60 seconds
        and allows or blocks new requests based on the specified rate limit (requests per minute).

        Parameters:
        -----------
        max_requests_per_minute : int
            Maximum number of requests allowed per minute.
        """

        self.max_requests_per_minute = max_requests_per_minute
        self.window = 60.0
        self.request_timestamps = deque()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """
        Acquires permission to proceed. 
        If the rate limit is exceeded, this method blocks (sleeps) until a slot is available.
        """
        async with self.lock:
            while True:
                now = time.monotonic()
                
                # Remove all timestamps outside the sliding window
                while self.request_timestamps and now - self.request_timestamps[0] > self.window:
                    self.request_timestamps.popleft()

                # Allow: Under limit
                if len(self.request_timestamps) < self.max_requests_per_minute:
                    self.request_timestamps.append(now)
                    return
                # Wait: Over limit
                else:
                    oldest_timestamp = self.request_timestamps[0]
                    # wait until the oldest request is outside the window
                    wait_time = (oldest_timestamp + self.window) - now
                    
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)


class MessagesLogger:
    def __init__(self, store_images: bool = False):
        """
        This class is used to log the messages for InferenceEngine.chat().

        Parameters:
        -----------
        store_images : bool
            Whether to store images in the log or not. If False, image URLs will be replaced with a placeholder "[image]".
        """
        self.messages_log = []
        self.store_images = store_images

    def log_messages(self, messages : List[Dict[str,str]]):
        """
        This method logs the messages to a list.
        """
        self.messages_log.append(messages)

    def get_messages_log(self) -> List[List[Dict[str,str]]]:
        """
        This method returns a copy of the current messages log
        """
        return self.messages_log.copy()
    
    def clear_messages_log(self):
        """
        This method clears the current messages log
        """
        self.messages_log.clear()
