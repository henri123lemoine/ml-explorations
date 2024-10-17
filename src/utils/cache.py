import functools
import os
import pickle
from pathlib import Path
from typing import Any, Callable

from src.settings import CACHE_PATH


class Cache:
    def __init__(self, cache_dir: str = CACHE_PATH, maxsize: int = 128):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.maxsize = maxsize

    def disk_cache(self, func: Callable) -> Callable:
        @functools.lru_cache(maxsize=self.maxsize)
        def wrapper(*args, **kwargs):
            key = self._make_key(func, args, kwargs)
            result = self._load_from_disk(key)
            if result is None:
                result = func(*args, **kwargs)
                self._save_to_disk(key, result)
            return result

        return wrapper

    def _make_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        key = f"{func.__name__}_{hash(args)}_{hash(frozenset(kwargs.items()))}"
        return key

    def _load_from_disk(self, key: str) -> Any:
        file_path = self.cache_dir / f"{key}.pkl"
        if file_path.exists():
            with open(file_path, "rb") as f:
                return pickle.load(f)
        return None

    def _save_to_disk(self, key: str, value: Any) -> None:
        file_path = self.cache_dir / f"{key}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(value, f)

    def clear_cache(self, func: Callable = None):
        if func is None:
            # Clear all disk cache
            for file in self.cache_dir.glob("*.pkl"):
                os.remove(file)
        else:
            # Clear disk cache for specific function
            for file in self.cache_dir.glob(f"{func.__name__}_*.pkl"):
                os.remove(file)

        if func is not None and hasattr(func, "cache_clear"):
            func.cache_clear()


if __name__ == "__main__":
    cache = Cache()

    # Example usage
    @cache.disk_cache
    def expensive_operation(x: int, y: int) -> int:
        print(
            f"Calculating {x} + {y}"
        )  # This will only print when the calculation is actually performed
        return x + y

    result1 = expensive_operation(5, 3)
    print(result1)  # Output: 8
