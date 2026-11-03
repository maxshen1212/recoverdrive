import threading

import numpy as np

from recoverdrive.cache import KVCache, digest, image_digest


def test_kvcache_roundtrip_threadsafe(tmp_path):
    cache = KVCache(tmp_path / "c.sqlite")

    def worker(t):
        for i in range(50):
            cache.set(f"{t}:{i}", {"v": [t, i]})

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(cache) == 400
    assert cache.get("3:7") == {"v": [3, 7]}
    assert cache.get("missing") is None
    assert KVCache(tmp_path / "c.sqlite").get("7:49") == {"v": [7, 49]}  # persisted


def test_digest_stable():
    assert digest({"a": 1, "b": [1, 2]}) == digest({"b": [1, 2], "a": 1})
    assert digest({"a": 1}) != digest({"a": 2})
    img = np.zeros((4, 4, 3), np.uint8)
    other = img.copy()
    assert image_digest(img) == image_digest(other)
    other[0, 0, 0] = 1
    assert image_digest(img) != image_digest(other)
    assert image_digest(np.zeros((2, 8, 3), np.uint8)) != image_digest(np.zeros((4, 4, 3), np.uint8))
