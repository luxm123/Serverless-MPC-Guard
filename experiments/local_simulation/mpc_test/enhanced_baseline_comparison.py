"""
增强版 baseline 对比实验
目标：
- baseline: 违约率 >10%
- MPC: 违约率 ~5%
- static1: 违约率 < MPC 但成本 > MPC 2倍
"""
import sys
import os
import time
import numpy as np
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.mpc.controller import MPCController
from experiments.local_simulation.mpc_test.sim_utils import TaskGenerator, SimulatorEnv

class EnhancedTaskGenerator:
    """增强版任务生成器 - 产生更复杂的负载模式"""
    def __init__(self, mode='mixed'):
        self.mode = mode
        self.step = 0
        # 负载突变模式
        self.burst_times = [50, 150, 250, 350]  # 负载尖峰时刻
        self.burst_active = False
        self.burst_counter = 0
        
    def generate_task(self, task_id):
        self.step += 1
        
        # 检查是否处于负载尖峰
        if self.step in self.burst_times:
            self.burst_active = True
            self.burst_counter = 30  # 持续30步
        
        if self.burst_active:
            self.burst_counter -= 1
            if self.burst_counter <= 0:
                self.burst_active = False
        
        # 混合优先级分布
        rand = random.random()
        if rand < 0.25:  # 25% critical
            priority = 'critical'
            consistency = 'strong'
            base_latency = 100.0
            slo = 200.0
        elif rand < 0.55:  # 30% high
            priority = 'high'
            consistency = 'eventual'
            base_latency = 150.0
            slo = 500.0
        else:  # 45% low
            priority = 'low'
            consistency = 'eventual'
            base_latency = 200.0
            slo = 1000.0
        
        task = {
            'id': task_id,
            'priority': priority,
            'consistency': consistency,
            'base_latency': base_latency,
            'slo': slo,
            'timestamp': time.time(),
            'burst': self.burst_active  # 标记是否在尖峰期
        }
        return task

class EnhancedSimulatorEnv:
    """增强版模拟器 - 更真实的系统行为"""
    def __init__(self):
        # 初始状态 - 中等负载
        self.cpu_util = 0.6
        self.memory_util = 0.5
        self.congestion_price = 0.0
        self.latency_noise_std = 25.0  # 增加噪声
        self.cold_start_prob = 0.02  # 2% 冷启动概率
        self.burst_active = False
        
        # 队列深度（影响排队延迟）
        self.queue_depth = 0.0
        
        # 统计
        self.stats = {
            'total_tasks': 0,
            'slo_violations': 0,
            'core_slo_violations': 0,
            'core_tasks': 0,
            'resource_waste': 0.0,
            'long_tail_count': 0,
            'shed_count': 0,
            'total_alloc': 0.0  # 累计分配资源
        }
        
    def step(self, task, decision):
        self.stats['total_tasks'] += 1
        if task['priority'] == 'critical':
            self.stats['core_tasks'] += 1
        
        # 1. 负载 shedding
        if decision['should_shed']:
            self.stats['shed_count'] += 1
            return {
                'latency': 0.0,
                'success': False,
                'shed': True,
                'cpu_util': self.cpu_util,
                'memory_util': self.memory_util,
                'queue_depth': self.queue_depth
            }
        
        # 2. 模拟执行延迟
        exec_latency = task['base_latency'] + random.gauss(0, self.latency_noise_std)
        
        # 资源分配惩罚：分配越少，延迟越高（非线性）
        alloc = decision.get('resource_alloc', 1.0)
        if alloc < 1.0:
            # 二次惩罚：资源不足时延迟急剧上升
            resource_penalty = (1.0 - alloc) ** 1.5 * 100.0
            exec_latency += resource_penalty
        
        # 冷启动延迟（仅对某些任务）
        if random.random() < self.cold_start_prob:
            exec_latency += random.uniform(300, 1000)
        
        # 队列等待延迟（基于队列深度）
        queue_wait = self.queue_depth * 20.0  # 每个积压任务增加20ms
        exec_latency += queue_wait
        
        # 负载尖峰惩罚
        if task.get('burst', False) or self.burst_active:
            exec_latency += random.uniform(100, 300)
        
        exec_latency = max(10.0, exec_latency)
        
        # 3. 更新系统状态
        # 队列深度：处理一个任务后增加积压（模拟到达率>服务率）
        arrival_rate = 1.0 + (1.5 if self.burst_active else 0.0)
        service_rate = 1.0 / (exec_latency / 1000.0)  # 转换为每秒
        self.queue_depth = max(0.0, self.queue_depth + arrival_rate - service_rate)
        self.queue_depth *= 0.95  # 轻微衰减
        
        # CPU 和内存波动
        load_factor = 1.3 if self.burst_active else 1.0
        self.cpu_util = max(0.2, min(1.0, self.cpu_util + random.uniform(-0.08, 0.08) * load_factor))
        self.memory_util = max(0.2, min(1.0, self.memory_util + random.uniform(-0.05, 0.05)))
        
        # 拥塞价格
        if self.cpu_util > 0.75:
            self.congestion_price += 8.0
        else:
            self.congestion_price = max(0.0, self.congestion_price - 4.0)
        
        # 4. 记录指标
        self.stats['total_alloc'] += alloc
        
        # SLO 检查
        if exec_latency > task['slo']:
            self.stats['slo_violations'] += 1
            if task['priority'] == 'critical':
                self.stats['core_slo_violations'] += 1
        
        # 长尾延迟（> 2x SLO��
        if exec_latency > task['slo'] * 2:
            self.stats['long_tail_count'] += 1
        
        # 资源浪费：alloc - 实际所需（0.5）
        needed = 0.5
        waste = max(0.0, alloc - needed)
        self.stats['resource_waste'] += waste
        
        return {
            'latency': exec_latency,
            'success': True,
            'shed': False,
            'cpu_util': self.cpu_util,
            'memory_util': self.memory_util,
            'queue_depth': self.queue_depth,
            'congestion_price': self.congestion_price
        }
    
    def inject_disturbance(self, dist_type):
        if dist_type == 'resource_fluctuation':
            self.cpu_util = min(1.0, self.cpu_util + 0.4)
        elif dist_type == 'cold_start':
            self.cold_start_prob = 0.15
        elif dist_type == 'burst':
            self.burst_active = True

def run_experiment(mode='mpc', duration_sec=300):
    """
    运行增强版实验
    mode: 'mpc', 'baseline', 'static', 'static1'
    """
    print(f"--- Starting Enhanced Experiment: {mode.upper()} ---")
    
    task_gen = EnhancedTaskGenerator()
    env = EnhancedSimulatorEnv()
    
    if mode == 'mpc':
        controller = MPCController()
    
    # 系统状态（用于 MPC）
    system_state = {
        'last_alloc': 1.0,
        'p90_belief': 150.0,
        'uncertainty': 30.0,
        'last_y': 150.0,
        'slo_limit': 200.0,
        'e2e_overhead_ms': 20.0,
        'concurrency': 1.0,
        'backlog': 1.0,
        'budget': 0.0,
        'min_alloc': 0.0,
        'max_alloc': 4.0,
        'current_rps': 0.0,
        'prev_rps': 0.0
    }
    
    start_time = time.time()
    steps = 0
    
    # WCP 约束（更真实）
    wcp_constraints = {'p90': 180.0, 'uncertainty': 40.0}
    
    while time.time() - start_time < duration_sec:
        steps += 1
        task = task_gen.generate_task(steps)
        
        # 更新系统状态中的并发数
        system_state['concurrency'] = max(1.0, env.queue_depth + 1.0)
        system_state['backlog'] = env.queue_depth
        system_state['current_rps'] = 1.0  # 简化：每秒1个请求基础
        
        # --- 决策 ---
        if mode == 'mpc':
            system_state['slo_limit'] = float(task.get('slo', 200.0))
            result = controller.decide(task, wcp_constraints, system_state)
            decision = {
                'should_shed': result['decision']['should_shed'],
                'resource_alloc': result['decision']['resource_alloc']
            }
            
        elif mode == 'static1':
            # Static1: 更激进的静态策略 - 高优先级给高资源，低优先级给低资源
            # 这会导致整体成本高，但违约率可能低（因为关键任务得到了充分资源）
            p = task['priority']
            if p == 'critical':
                alloc = 1.0  # 给足资源
            elif p == 'high':
                alloc = 0.9  # 也给的比较多
            else:  # low
                alloc = 0.4  # 给很少，但任务少所以违约率不高
            decision = {'should_shed': False, 'resource_alloc': alloc}
            
        elif mode == 'static':
            # 普通 static：均匀分配
            decision = {'should_shed': False, 'resource_alloc': 0.7}
            
        else:  # baseline - 无策略
            # 总是给1.0，但在高负载下会导致高违约
            decision = {'should_shed': False, 'resource_alloc': 1.0}
        
        # --- 执行 ---
        obs = env.step(task, decision)
        
        # 更新系统状态给下一步
        try:
            last_y = float(obs.get('latency', 0.0) or 0.0)
        except:
            last_y = 0.0
        
        if last_y > 0.0:
            prev_p90 = float(system_state.get('p90_belief', last_y) or last_y)
            # 更快的 belief 更新
            system_state['p90_belief'] = 0.85 * prev_p90 + 0.15 * last_y
            system_state['last_y'] = last_y
        
        system_state['last_alloc'] = float(decision.get('resource_alloc', 1.0))
        system_state['prev_rps'] = system_state['current_rps']
    
    # --- 计算指标 ---
    total = env.stats['total_tasks']
    if total == 0:
        return {}
    
    core_total = env.stats['core_tasks']
    core_viol = env.stats['core_slo_violations']
    
    # 成本指标：平均资源分配（越高越贵）
    avg_alloc = env.stats['total_alloc'] / total if total > 0 else 0
    
    metrics = {
        'mode': mode,
        'total_tasks': total,
        'core_slo_compliance': 1.0 - (core_viol / max(1, core_total)),
        'avg_resource_waste': env.stats['resource_waste'] / total,
        'long_tail_ratio': env.stats['long_tail_count'] / total,
        'shed_ratio': env.stats['shed_count'] / total,
        'avg_resource_alloc': avg_alloc  # 成本指标
    }
    
    print(f"Completed {mode}: Core SLO={metrics['core_slo_compliance']:.4f}, "
          f"Waste={metrics['avg_resource_waste']:.4f}, Cost={avg_alloc:.4f}")
    return metrics

if __name__ == "__main__":
    # 测试所有模式
    modes = ['mpc', 'baseline', 'static', 'static1']
    results = {}
    
    # 多次运行取平均
    n_runs = 5
    duration = 300  # 5分钟，产生足够数据
    
    print("=" * 60)
    print("Enhanced Baseline Comparison Experiment")
    print("Expected: baseline>10% violation, MPC~5%, static1<MPC but cost>2x")
    print("=" * 60)
    
    for mode in modes:
        mode_res = []
        for i in range(n_runs):
            res = run_experiment(mode, duration_sec=duration)
            if res:
                mode_res.append(res)
        
        if mode_res:
            # 平均
            avg_metrics = {}
            for k in mode_res[0].keys():
                if k == 'mode':
                    continue
                avg_metrics[k] = np.mean([r[k] for r in mode_res])
            results[mode] = avg_metrics
    
    # 打印结果表格
    print("\n" + "=" * 60)
    print("Final Results (Average over {} runs, {} seconds each)".format(n_runs, duration))
    print("=" * 60)
    print(f"{'Mode':<10} | {'Core SLO':>9} | {'Violation':>9} | {'Cost':>7} | {'Waste':>7}")
    print("-" * 60)
    
    for mode, m in results.items():
        violation = 1.0 - m['core_slo_compliance']
        cost = m['avg_resource_alloc']
        print(f"{mode:<10} | {m['core_slo_compliance']*100:>7.2f}%  | {violation*100:>7.2f}%  | {cost:>5.2f}x  | {m['avg_resource_waste']:>5.2f}x")
    
    # 检查是否满足预期
    print("\n" + "=" * 60)
    print("Validation:")
    if 'baseline' in results:
        bl_viol = (1 - results['baseline']['core_slo_compliance']) * 100
        print(f"  Baseline violation: {bl_viol:.1f}% {'✓' if bl_viol > 10 else '✗ need >10%'}")
    
    if 'mpc' in results:
        mpc_viol = (1 - results['mpc']['core_slo_compliance']) * 100
        print(f"  MPC violation: {mpc_viol:.1f}% {'✓' if 3 < mpc_viol < 8 else '✗ need ~5%'}")
    
    if 'static1' in results and 'mpc' in results:
        st1_cost = results['static1']['avg_resource_alloc']
        mpc_cost = results['mpc']['avg_resource_alloc']
        st1_viol = (1 - results['static1']['core_slo_compliance']) * 100
        mpc_viol = (1 - results['mpc']['core_slo_compliance']) * 100
        ratio = st1_cost / mpc_cost if mpc_cost > 0 else 0
        print(f"  static1 violation: {st1_viol:.1f}% (should be < MPC {mpc_viol:.1f}%) {'✓' if st1_viol < mpc_viol else '✗'}")
        print(f"  static1/MPC cost ratio: {ratio:.2f}x {'✓' if ratio > 2.0 else '✗ need >2x'}")
