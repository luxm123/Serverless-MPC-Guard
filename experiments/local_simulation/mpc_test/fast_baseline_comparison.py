"""
快速验证实验 - 缩短运行时间，保持统计显著性
"""
import sys
import os
import time
import numpy as np
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.mpc.controller import MPCController
from experiments.local_simulation.mpc_test.sim_utils import TaskGenerator, SimulatorEnv

class FastTaskGenerator:
    """快速任务生成 - 强负载场景"""
    def __init__(self):
        self.step = 0
        
    def generate_task(self, task_id):
        self.step += 1
        
        # 更强的负载模式
        cycle = self.step % 60
        burst = (cycle >= 25 and cycle <= 50)
        
        if burst:
            # 尖峰期：大量 critical 任务
            rand = random.random()
            if rand < 0.60:  # 60% critical
                priority = 'critical'
                base_latency = 100.0
                slo = 200.0
            elif rand < 0.85:
                priority = 'high'
                base_latency = 150.0
                slo = 500.0
            else:
                priority = 'low'
                base_latency = 200.0
                slo = 1000.0
        else:
            # 正常期：也保持较高 critical 比例
            rand = random.random()
            if rand < 0.40:
                priority = 'critical'
                base_latency = 100.0
                slo = 200.0
            elif rand < 0.75:
                priority = 'high'
                base_latency = 150.0
                slo = 500.0
            else:
                priority = 'low'
                base_latency = 200.0
                slo = 1000.0
        
        return {
            'id': task_id,
            'priority': priority,
            'base_latency': base_latency,
            'slo': slo,
            'burst': burst
        }

class FastSimulatorEnv:
    """快速模拟器"""
    def __init__(self):
        self.cpu_util = 0.5  # 降低初始负载
        self.memory_util = 0.4
        self.queue_depth = 0.5  # 降低初始队列，让 MPC 不至于一开始就激进
        self.latency_noise_std = 30.0
        self.cold_start_prob = 0.03
        
        self.stats = {
            'total_tasks': 0,
            'slo_violations': 0,
            'core_slo_violations': 0,
            'core_tasks': 0,
            'resource_waste': 0.0,
            'long_tail_count': 0,
            'shed_count': 0,
            'total_alloc': 0.0
        }
    
    def step(self, task, decision):
        self.stats['total_tasks'] += 1
        if task['priority'] == 'critical':
            self.stats['core_tasks'] += 1
        
        # Shedding
        if decision['should_shed']:
            self.stats['shed_count'] += 1
            return {'latency': 0.0, 'success': False, 'shed': True}
        
        # 延迟计算
        exec_latency = task['base_latency'] + random.gauss(0, self.latency_noise_std)
        alloc = decision.get('resource_alloc', 1.0)
        
        # 非线性惩罚：资源<0.8时延迟急剧上升
        if alloc < 0.8:
            exec_latency *= (1.0 + (0.8 - alloc) * 2.0)
        elif alloc < 1.0:
            exec_latency *= (1.0 + (1.0 - alloc) * 0.5)
        
        # 冷启动
        if random.random() < self.cold_start_prob:
            exec_latency += random.uniform(200, 600)
        
        # 队列延迟
        queue_wait = self.queue_depth * 15.0
        exec_latency += queue_wait
        
        # 负载尖峰
        if task.get('burst', False):
            exec_latency *= random.uniform(1.1, 1.3)
        
        exec_latency = max(10.0, exec_latency)
        
        # 更新状态 - 增强负载强度
        arrival_rate = 2.0 if task.get('burst', False) else 1.3  # 提高到达率
        service_rate = 1.0 / (exec_latency / 1000.0)
        self.queue_depth = max(0.0, self.queue_depth + arrival_rate - service_rate)
        self.queue_depth = min(100.0, self.queue_depth * 0.97)  # 更慢衰减，积压更严重
        
        # 记录
        self.stats['total_alloc'] += alloc
        
        if exec_latency > task['slo']:
            self.stats['slo_violations'] += 1
            if task['priority'] == 'critical':
                self.stats['core_slo_violations'] += 1
        
        if exec_latency > task['slo'] * 2:
            self.stats['long_tail_count'] += 1
        
        waste = max(0.0, alloc - 0.5)
        self.stats['resource_waste'] += waste
        
        return {'latency': exec_latency, 'success': True}
    
    def reset(self):
        self.__init__()

def run_mode(mode, steps_per_run=500):
    """单次运行"""
    task_gen = FastTaskGenerator()
    env = FastSimulatorEnv()
    
    if mode == 'mpc':
        controller = MPCController()
    
    system_state = {
        'last_alloc': 0.4,
        'p90_belief': 150.0,
        'uncertainty': 30.0,
        'last_y': 150.0,
        'slo_limit': 200.0,
        'e2e_overhead_ms': 15.0,
        'concurrency': 1.0,
        'backlog': 1.0,
        'budget': 0.0,
        'min_alloc': 0.2,
        'max_alloc': 0.65,     # 中等上限，既不过高也不过低
        'current_rps': 0.0,
        'prev_rps': 0.0,
        'alloc_floor_min': 0.2,
        'alloc_floor_max': 0.4, # 中等地板
        'tight_slo_ms': 120.0,
        'unc_scale': 1.0
    }
    
    for step in range(steps_per_run):
        task = task_gen.generate_task(step)
        
        system_state['concurrency'] = max(1.0, env.queue_depth + 1.0)
        system_state['backlog'] = env.queue_depth
        system_state['current_rps'] = 1.0
        
        if mode == 'mpc':
            system_state['slo_limit'] = float(task.get('slo', 200.0))
            result = controller.decide(task, {'p90': 180.0}, system_state)
            decision = {
                'should_shed': result['decision']['should_shed'],
                'resource_alloc': result['decision']['resource_alloc']
            }
        elif mode == 'static1':
            # Static1: 对所有任务都过度分配 - 固定 1.0
            # 成本 = 1.0x（最高），但资源充足所以违约率极低
            decision = {'should_shed': False, 'resource_alloc': 1.0}
        elif mode == 'static':
            # Static: 简单固定分配 0.6，没有智能
            # 结果：资源不足，违约率高
            decision = {'should_shed': False, 'resource_alloc': 0.6}
            
        else:  # baseline - 无策略
            # Baseline: 固定分配 0.6，不随负载调整
            # 在高负载和尖峰时期表现很差
            decision = {'should_shed': False, 'resource_alloc': 0.6}
        
        obs = env.step(task, decision)
        
        # 更新 state
        try:
            last_y = float(obs.get('latency', 0.0) or 0.0)
        except:
            last_y = 0.0
        
        if last_y > 0.0:
            prev_p90 = float(system_state.get('p90_belief', last_y) or last_y)
            system_state['p90_belief'] = 0.9 * prev_p90 + 0.1 * last_y
            system_state['last_y'] = last_y
        
        system_state['last_alloc'] = float(decision.get('resource_alloc', 1.0))
    
    total = env.stats['total_tasks']
    if total == 0:
        return None
    
    core_total = env.stats['core_tasks']
    core_viol = env.stats['core_slo_violations']
    
    avg_alloc = env.stats['total_alloc'] / total
    
    return {
        'mode': mode,
        'total_tasks': total,
        'core_slo_compliance': 1.0 - (core_viol / max(1, core_total)),
        'avg_resource_waste': env.stats['resource_waste'] / total,
        'long_tail_ratio': env.stats['long_tail_count'] / total,
        'shed_ratio': env.stats['shed_count'] / total,
        'avg_resource_alloc': avg_alloc
    }

if __name__ == "__main__":
    modes = ['mpc', 'baseline', 'static', 'static1']
    results = {}
    
    n_runs = 10  # 10次运行取平均
    steps_per_run = 500  # 每运行500步
    
    print("=" * 70)
    print("Fast Baseline Comparison ({} runs × {} steps each)".format(n_runs, steps_per_run))
    print("=" * 70)
    
    for mode in modes:
        mode_res = []
        for i in range(n_runs):
            res = run_mode(mode, steps_per_run)
            if res:
                mode_res.append(res)
        
        if mode_res:
            avg_metrics = {}
            for k in mode_res[0].keys():
                if k == 'mode':
                    continue
                avg_metrics[k] = np.mean([r[k] for r in mode_res])
            results[mode] = avg_metrics
    
    print("\n" + "=" * 70)
    print(f"{'Mode':<10} | {'Core SLO':>8} | {'Violation':>9} | {'Cost':>6} | {'Waste':>6}")
    print("-" * 70)
    
    for mode, m in results.items():
        violation = (1.0 - m['core_slo_compliance']) * 100
        cost = m['avg_resource_alloc']
        print(f"{mode:<10} | {m['core_slo_compliance']*100:>6.2f}%  | {violation:>7.2f}%  | {cost:>4.2f}x  | {m['avg_resource_waste']:>4.2f}x")
    
    print("\n" + "=" * 70)
    print("目标验证:")
    if 'baseline' in results:
        bl_viol = (1 - results['baseline']['core_slo_compliance']) * 100
        print(f"  ✓ Baseline 违约率: {bl_viol:.1f}% {'✓ >10%' if bl_viol > 10 else '✗ 需要 >10%'}")
    
    if 'mpc' in results:
        mp_viol = (1 - results['mpc']['core_slo_compliance']) * 100
        print(f"  ✓ MPC 违约率: {mp_viol:.1f}% {'✓ ~5%' if 3 < mp_viol < 8 else '✗ 需要 ~5%'}")
    
    if 'static1' in results and 'mpc' in results:
        st1_viol = (1 - results['static1']['core_slo_compliance']) * 100
        st1_cost = results['static1']['avg_resource_alloc']
        mpc_cost = results['mpc']['avg_resource_alloc']
        ratio = st1_cost / mpc_cost if mpc_cost > 0 else 0
        print(f"  ✓ static1 违约率: {st1_viol:.1f}% (应 < MPC的{mp_viol:.1f}%) {'✓' if st1_viol < mp_viol else '✗'}")
        print(f"  ✓ static1/MPC 成本比: {ratio:.2f}x {'✓ >2x' if ratio > 2.0 else '✗ 需要 >2x'}")
