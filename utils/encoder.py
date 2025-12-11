import os
import pickle
from typing import Dict
import numpy as np
import torch
from functools import lru_cache

class iFunction:
    _feature_caches: Dict[str, Dict[str, np.ndarray]] = {}
    _cache_loaded: Dict[str, bool] = {}
    _feature_paths = {
        'esm2': 'data/suc/esm2_features.pkl',
        'onehot': 'data/suc/onehot_features.pkl',
    }

    @classmethod
    def _ensure_cache_loaded(cls, feature_type: str, pkl_path: str = None) -> None:
        if cls._cache_loaded.get(feature_type, False):
            return
        pkl_path = pkl_path or cls._feature_paths.get(feature_type)
        if not pkl_path or not os.path.exists(pkl_path):
            print(f"警告: {feature_type}特征文件 '{pkl_path}' 未找到")
            cls._feature_caches[feature_type] = {}
            cls._cache_loaded[feature_type] = True
            return
        try:
            with open(pkl_path, 'rb') as f:
                cls._feature_caches[feature_type] = pickle.load(f)
            cls._cache_loaded[feature_type] = True
            print(f"成功加载 {len(cls._feature_caches[feature_type])} 个{feature_type}特征")
        except Exception as e:
            print(f"加载{feature_type}特征失败: {e}")
            cls._feature_caches[feature_type] = {}
            cls._cache_loaded[feature_type] = True

    @classmethod
    def _get_feature(cls, seq: str, feature_type: str, expected_dim: int, pkl_path: str = None) -> torch.Tensor:
        cls._ensure_cache_loaded(feature_type, pkl_path)
        cache = cls._feature_caches.get(feature_type, {})
        if seq in cache:
            return torch.from_numpy(cache[seq]).float()
        print(f"警告: 序列 '{seq[:10]}...' 未在{feature_type}特征中找到")
        return torch.empty((0, expected_dim), dtype=torch.float32)

    @staticmethod
    def to_onehot(seq: str, pkl_path: str = None) -> torch.Tensor:
        return iFunction._get_feature(seq, 'onehot', 23, pkl_path)

    @staticmethod
    def load_esm2_from_file(seq: str, pkl_path: str = None) -> np.ndarray:
        iFunction._ensure_cache_loaded('esm2', pkl_path)

        cache = iFunction._feature_caches.get('esm2', {})
        if seq in cache:
            return cache[seq]
        print(f"警告: 序列 '{seq[:10]}...' 未在ESM-2特征中找到")
        if cache:
            any_feature = next(iter(cache.values()))
            return np.zeros_like(any_feature)
        return np.zeros((33, 320), dtype=np.float32)

    @classmethod
    def clear_cache(cls):
        cls._feature_caches.clear()
        cls._cache_loaded.clear()

    @classmethod
    def get_cache_info(cls):
        info = {}
        for feature_type, cache in cls._feature_caches.items():
            info[feature_type] = len(cache)
        return info

    @staticmethod
    @lru_cache(maxsize=1000)
    def load_onehot_esm2(seq: str, esm2_pkl_path: str = None) -> torch.Tensor:
        esm2_feat_np = iFunction.load_esm2_from_file(seq, esm2_pkl_path)
        esm2_feat = torch.from_numpy(esm2_feat_np).float()
        onehot_feat = iFunction.to_onehot(seq)
        max_len = esm2_feat.shape[0]
        def pad_to_len(feat: torch.Tensor, target_len: int) -> torch.Tensor:
            if feat.shape[0] < target_len:
                padding = torch.zeros((target_len - feat.shape[0], feat.shape[1]), dtype=feat.dtype)
                return torch.cat([feat, padding], dim=0)
            return feat[:target_len, :]
        onehot_padded = pad_to_len(onehot_feat, max_len)
        return torch.cat([esm2_feat, onehot_padded], dim=1)