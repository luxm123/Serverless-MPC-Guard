"""
Azure 2019 轨迹窗口筛选器
从原始 Azure Functions 数据中提取 Stable 和 Bursty 场景
"""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# 动态路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.append(PROJECT_ROOT)

from experiments.serverless_test.trace_experiment.experiment_config import (
    ExperimentConfig,
    MEAN_RPS_MIN,
    MEAN_RPS_MAX,
    STABLE_CV_THRESHOLD,
    BURSTY_CV_THRESHOLD
)


class AzureTraceWindowSelector:
    """
    从 Azure Functions 2019 数据中筛选 30 分钟窗口
    分类为 Stable (CV < 0.3) 和 Bursty (CV ≥ 0.5)
    """

    def __init__(self, azure_data_path: str, output_dir: str):
        """
        Args:
            azure_data_path: Azure CSV 文件目录（包含 invocations_per_function_md.anon.dXX.csv）
            output_dir: 筛选后窗口保存目录
        """
        self.azure_data_path = Path(azure_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 场景统计
        self.stable_windows = []
        self.bursty_windows = []

    def _compute_cv(self, time_series: np.ndarray) -> float:
        """计算变异系数 CV = std / mean"""
        mean_val = np.mean(time_series)
        if mean_val <= 0:
            return 0.0
        std_val = np.std(time_series)
        return std_val / mean_val

    def _compute_rps(self, time_series: np.ndarray) -> float:
        """将每分钟调用次数转换为 RPS"""
        total_invocations = np.sum(time_series)
        total_seconds = len(time_series) * 60
        return total_invocations / total_seconds

    def _extract_30min_windows(self, function_series: np.ndarray,
                               step_minutes: int = 30) -> List[Tuple[int, np.ndarray]]:
        """
        从单函数的时间序列中提取滑动窗口（30分钟）
        Returns: List of (start_minute, window_array)
        """
        total_minutes = len(function_series)
        windows = []

        for start in range(0, total_minutes - step_minutes + 1, step_minutes // 2):  # 50% 重叠
            window = function_series[start:start + step_minutes]
            windows.append((start, window))

        return windows

    def _classify_window(self, window: np.ndarray) -> Optional[str]:
        """
        分类单个窗口
        Returns: 'stable', 'bursty', 或 None（不满足条件）
        """
        cv = self._compute_cv(window)
        mean_rps = self._compute_rps(window)

        # 检查 RPS 范围
        if not (MEAN_RPS_MIN <= mean_rps <= MEAN_RPS_MAX):
            return None

        # 分类
        if cv < STABLE_CV_THRESHOLD:
            return 'stable'
        elif cv >= BURSTY_CV_THRESHOLD:
            return 'bursty'
        else:
            return None  # 中间状态（如平稳型）不采用

    def select_windows_from_azure(self, days: int = 14,
                                   max_windows_per_class: int = 5) -> Dict[str, List[Dict]]:
        """
        主函数：从 Azure 数据中筛选窗口

        Args:
            days: 加载的天数（1-14）
            max_windows_per_class: 每类最多选几个窗口（论文要求5个）

        Returns:
            {
                'stable': [{'id': 'stable_01', 'start_min': 0, 'data': [...], 'cv': 0.2, 'rps': 10.5}, ...],
                'bursty': [{'id': 'bursty_01', ...}, ...]
            }
        """
        print(f"[Info] Loading Azure traces from {self.azure_data_path}...")

        # 加载所有天数据
        all_series = []
        for day in range(1, days + 1):
            filename = f"invocations_per_function_md.anon.d{day:02d}.csv"
            filepath = self.azure_data_path / filename

            if not filepath.exists():
                print(f"[Warning] File not found: {filepath}")
                continue

            print(f"[Info] Loading day {day}...")
            df = pd.read_csv(filepath)

            # 提取每个函数的时间序列（1-1440分钟）
            for _, row in df.iterrows():
                minute_cols = [str(i) for i in range(1, 1441)]
                series = np.array([row[col] for col in minute_cols], dtype=float)
                all_series.append(series)

        print(f"[Info] Loaded {len(all_series)} functions.")

        # 提取并分类所有窗口
        candidate_windows = {'stable': [], 'bursty': []}

        for func_idx, series in enumerate(all_series):
            windows = self._extract_30min_windows(series)

            for start_min, window in windows:
                window_class = self._classify_window(window)
                if window_class is None:
                    continue

                cv = self._compute_cv(window)
                rps = self._compute_rps(window)

                candidate_windows[window_class].append({
                    'func_id': func_idx,
                    'start_minute': start_min,
                    'data': window.tolist(),
                    'cv': round(cv, 4),
                    'mean_rps': round(rps, 2),
                    'total_invocations': int(np.sum(window))
                })

        print(f"[Info] Found {len(candidate_windows['stable'])} stable candidates")
        print(f"[Info] Found {len(candidate_windows['bursty'])} bursty candidates")

        # 选择最具代表性的窗口（按 RPS 分布均匀采样）
        selected = {'stable': [], 'bursty': []}

        for class_type in ['stable', 'bursty']:
            candidates = candidate_windows[class_type]
            if not candidates:
                print(f"[Warning] No {class_type} windows found!")
                continue

            # 按 RPS 排序并均匀采样
            candidates.sort(key=lambda x: x['mean_rps'])

            n_select = min(max_windows_per_class, len(candidates))
            step = max(1, len(candidates) // n_select)

            for i in range(n_select):
                idx = i * step
                if idx >= len(candidates):
                    break
                win = candidates[idx]
                win['id'] = f"{class_type}_{i+1:02d}"

                # 保存窗口数据
                window_df = pd.DataFrame({
                    'minute_offset': range(len(win['data'])),
                    'invocations': win['data']
                })
                out_path = self.output_dir / f"window_{win['id']}.csv"
                window_df.to_csv(out_path, index=False)

                selected[class_type].append(win)
                print(f"  Selected {win['id']}: CV={win['cv']:.3f}, RPS={win['mean_rps']:.2f}")

        # 保存元数据
        metadata = {
            'config': {
                'window_minutes': ExperimentConfig.WINDOW_MINUTES,
                'stable_cv_threshold': STABLE_CV_THRESHOLD,
                'bursty_cv_threshold': BURSTY_CV_THRESHOLD,
                'mean_rps_range': [MEAN_RPS_MIN, MEAN_RPS_MAX],
                'days_used': days
            },
            'windows': selected
        }

        meta_path = self.output_dir / 'windows_metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n[Success] Selected {len(selected['stable'])} stable + {len(selected['bursty'])} bursty windows")
        print(f"Metadata saved to {meta_path}")

        return selected


def convert_selected_windows_to_trace_format(windows_dir: str, output_dir: str):
    """
    将选中的窗口转换为实验用的 trace 格式
    格式：timestamp(ms), duration(ms), memory(MB)
    """
    windows_dir = Path(windows_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取元数据
    with open(windows_dir / 'windows_metadata.json') as f:
        metadata = json.load(f)

    for class_type in ['stable', 'bursty']:
        for win in metadata['windows'][class_type]:
            window_id = win['id']
            invocations = win['data']  # 每分钟调用次数

            # 转换为 trace 格式（每 invocation 一个记录）
            trace_records = []
            timestamp_ms = 0

            for minute, count in enumerate(invocations):
                # 均匀分布这一分钟的请求
                if count == 0:
                    continue

                interval_ms = 60000.0 / count
                for i in range(int(count)):
                    # 任务持续时间（模拟不同任务类型）
                    # 基于 Azure 2019 真实分布：median ~100-200ms
                    duration_ms = np.random.choice([
                        np.random.randint(50, 150),    # 短任务
                        np.random.randint(150, 500),   # 中等
                        np.random.randint(500, 2000),  # 长任务（低频）
                    ], p=[0.6, 0.3, 0.1])

                    # 内存需求（基于函数类型）
                    memory_mb = np.random.choice([128, 256, 512, 1024], p=[0.5, 0.3, 0.15, 0.05])

                    trace_records.append({
                        'timestamp': int(timestamp_ms),
                        'duration': int(duration_ms),
                        'memory': int(memory_mb)
                    })

                    timestamp_ms += interval_ms

            # 保存
            df = pd.DataFrame(trace_records)
            out_path = output_dir / f"trace_{window_id}.csv"
            df.to_csv(out_path, index=False)
            print(f"  Generated trace: {out_path.name} ({len(df)} requests)")

    print(f"\nAll traces saved to {output_dir}")


if __name__ == "__main__":
    # 配置路径
    AZURE_DATA_DIR = Path(PROJECT_ROOT) / 'datasets' / 'azure_raw'
    WINDOWS_DIR = Path(PROJECT_ROOT) / 'datasets' / 'processed' / 'selected_windows'
    TRACES_DIR = Path(PROJECT_ROOT) / 'datasets' / 'processed' / 'traces'

    # 步骤1：筛选窗口
    print("="*60)
    print("Step 1: Selecting representative windows from Azure data...")
    print("="*60)

    selector = AzureTraceWindowSelector(str(AZURE_DATA_DIR), str(WINDOWS_DIR))
    selected = selector.select_windows_from_azure(days=14)

    # 步骤2：转换为 trace 格式
    print("\n" + "="*60)
    print("Step 2: Converting windows to trace format...")
    print("="*60)

    convert_selected_windows_to_trace_format(str(WINDOWS_DIR), str(TRACES_DIR))

    print("\n[Done] Ready for experiments!")
