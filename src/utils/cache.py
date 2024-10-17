import functools
import json
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Union

from src.settings import CACHE_PATH


class Cache:
    def __init__(self, cache_dir: str = CACHE_PATH, maxsize: int = 128):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.maxsize = maxsize

    def disk_cache(
        self, func: Callable = None, *, key_prefix: str = "", serializer: str = "auto"
    ) -> Union[Callable, Callable[[Callable], Callable]]:
        def decorator(f):
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                key = self._make_key(key_prefix, f, args, kwargs)
                result = self._load_from_disk(key, serializer)
                if result is None:
                    result = f(*args, **kwargs)
                    self._save_to_disk(key, result, serializer)
                return result

            return functools.lru_cache(maxsize=self.maxsize)(wrapper)

        return decorator if func is None else decorator(func)

    def get(self, key: str, default: Any = None, serializer: str = "json") -> Any:
        return self._load_from_disk(key, serializer) or default

    def set(self, key: str, value: Any, serializer: str = "json") -> None:
        self._save_to_disk(key, value, serializer)

    def _make_key(self, prefix: str, func: Callable, args: tuple, kwargs: dict) -> str:
        key = f"{prefix}_{func.__name__}_{hash(args)}_{hash(frozenset(kwargs.items()))}"
        return key

    def _load_from_disk(self, key: str, serializer: str) -> Any:
        file_path = self.cache_dir / f"{key}.{serializer}"
        if file_path.exists():
            with open(file_path, "rb" if serializer == "pickle" else "r") as f:
                if serializer == "json":
                    return json.load(f)
                elif serializer == "pickle":
                    return pickle.load(f)
                else:  # "auto"
                    try:
                        return json.load(f)
                    except json.JSONDecodeError:
                        f.seek(0)
                        return pickle.load(f)
        return None

    def _save_to_disk(self, key: str, value: Any, serializer: str) -> None:
        file_path = self.cache_dir / f"{key}.{serializer}"
        mode = "wb" if serializer == "pickle" else "w"
        with open(file_path, mode) as f:
            if serializer == "json":
                json.dump(value, f, ensure_ascii=False, indent=2)
            elif serializer == "pickle":
                pickle.dump(value, f)
            else:  # "auto"
                try:
                    json.dump(value, f, ensure_ascii=False, indent=2)
                except TypeError:
                    f.close()
                    with open(file_path, "wb") as f:
                        pickle.dump(value, f)

    def clear_cache(self, key_prefix: str = ""):
        for file in self.cache_dir.glob(f"{key_prefix}*.*"):
            os.remove(file)


cache = Cache()


if __name__ == "__main__":

    @cache.disk_cache
    def expensive_operation(x: int, y: int) -> int:
        print(
            f"Calculating {x} + {y}"
        )  # This will only print when the calculation is actually performed
        return x + y

    result1 = expensive_operation(5, 3)
    print(result1)  # Output: 8
