import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class LassoSolvers:
    def __init__(self, X, y, lambda_val):
        self.X = X
        self.y = y
        self.lambda_val = lambda_val
        self.n, self.p = X.shape

        # 预计算常用值
        self.XT = X.T
        self.XTy = X.T @ y
        # 使用更稳定的方法计算Lipschitz常数
        if self.n < self.p:
            # 当n<p时，使用特征值分解更稳定
            X_squared = X.T @ X
            self.L = np.max(np.linalg.eigvalsh(X_squared)) / self.n
        else:
            # 使用SVD计算最大奇异值
            self.L = np.max(np.linalg.svd(X, compute_uv=False)) ** 2 / self.n

    def objective(self, beta):
        """计算Lasso目标函数值"""
        residual = self.y - self.X @ beta
        mse = 0.5 * np.sum(residual ** 2) / self.n
        l1_penalty = self.lambda_val * np.sum(np.abs(beta))
        return mse + l1_penalty


def proximal_gradient(X, y, n, p, lam, max_iter, f_star):
    """临近点梯度法 (ISTA)"""
    try:
        solver = LassoSolvers(X, y, lam)
        beta = np.zeros(p)
        objectives = []
        XTX = X.T @ X  # 预计算，减少重复计算

        for k in range(max_iter):
            # 计算梯度: -X^T(y - Xβ) = X^TXβ - X^Ty
            gradient = (XTX @ beta - solver.XTy) / n

            # 梯度步
            beta_temp = beta - (1 / solver.L) * gradient

            # 临近算子 (软阈值)
            threshold = lam / solver.L
            beta_new = np.sign(beta_temp) * np.maximum(np.abs(beta_temp) - threshold, 0)

            # 记录次优性
            obj_val = solver.objective(beta_new)
            objectives.append(max(obj_val - f_star, 1e-10))  # 确保正值

            beta = beta_new

        return objectives
    except Exception as e:
        print(f"Error in proximal_gradient: {e}")
        return [1.0] * max_iter  # 返回默认值


def coordinate_descent(X, y, n, p, lam, max_iter, f_star):
    """高效的最速坐标下降法"""
    try:
        solver = LassoSolvers(X, y, lam)
        beta = np.zeros(p)
        objectives = []

        # 预计算X_j的平方和，用于步长
        X_sq = np.sum(X ** 2, axis=0) / n  # (p,)

        # 初始化残差
        r = y.copy()  # 初始残差 = y，因为beta=0

        for k in range(max_iter):
            beta_old = beta.copy()

            # 循环更新每个坐标
            for j in range(p):
                if X_sq[j] < 1e-10:  # 跳过零方差特征
                    continue

                # 计算rho_j = X_j^T * r + X_sq[j] * beta_j
                X_j = X[:, j]
                rho_j = X_j @ r / n + X_sq[j] * beta[j]

                # 软阈值操作
                if rho_j > lam:
                    beta_j_new = (rho_j - lam) / X_sq[j]
                elif rho_j < -lam:
                    beta_j_new = (rho_j + lam) / X_sq[j]
                else:
                    beta_j_new = 0.0

                # 更新残差（高效更新）
                if beta_j_new != beta[j]:
                    r += X_j * (beta[j] - beta_j_new)
                    beta[j] = beta_j_new

            # 记录次优性
            obj_val = solver.objective(beta)
            objectives.append(max(obj_val - f_star, 1e-10))

            # 提前终止检查（可选）
            if np.linalg.norm(beta - beta_old) < 1e-8:
                # 填充剩余迭代
                objectives.extend([objectives[-1]] * (max_iter - k - 1))
                break

        return objectives
    except Exception as e:
        print(f"Error in coordinate_descent: {e}")
        return [1.0] * max_iter


def admm(X, y, n, p, lam, rho, max_iter, f_star):
    """交替方向乘子法 (ADMM)"""
    try:
        solver = LassoSolvers(X, y, lam)
        beta = np.zeros(p)
        z = np.zeros(p)
        u = np.zeros(p)
        objectives = []

        # 预计算矩阵逆
        XTX = X.T @ X / n
        I = np.eye(p)
        # 使用Cholesky分解提高稳定性和效率
        try:
            # 尝试Cholesky分解
            inv_matrix = np.linalg.inv(XTX + rho * I)
        except np.linalg.LinAlgError:
            # 如果不可逆，添加小正则化项
            inv_matrix = np.linalg.inv(XTX + rho * I + 1e-8 * np.eye(p))

        for k in range(max_iter):
            # 更新beta
            beta = inv_matrix @ (solver.XTy / n + rho * (z - u))

            # 更新z (软阈值)
            z_old = z.copy()
            z = np.sign(beta + u) * np.maximum(np.abs(beta + u) - lam / rho, 0)

            # 更新对偶变量
            u = u + beta - z

            # 记录次优性
            obj_val = solver.objective(beta)
            objectives.append(max(obj_val - f_star, 1e-10))

        return objectives
    except Exception as e:
        print(f"Error in admm (rho={rho}): {e}")
        return [1.0] * max_iter


def smoothed_gradient(X, y, n, p, lam, max_iter, f_star, mu=0.01):
    """平滑化的梯度下降法 (使用Huber平滑L1范数)"""
    try:
        def huber_gradient(x, mu):
            """Huber平滑的梯度"""
            return np.where(np.abs(x) <= mu, x / mu, np.sign(x))

        solver = LassoSolvers(X, y, lam)
        beta = np.zeros(p)
        objectives = []

        # 调整步长
        step_size = 1.0 / (solver.L + lam / mu)

        for k in range(max_iter):
            # 计算梯度
            residual = y - X @ beta
            gradient_mse = -X.T @ residual / n
            gradient_l1 = lam * huber_gradient(beta, mu)

            gradient_total = gradient_mse + gradient_l1

            # 梯度下降
            beta = beta - step_size * gradient_total

            # 记录次优性
            obj_val = solver.objective(beta)
            objectives.append(max(obj_val - f_star, 1e-10))

        return objectives
    except Exception as e:
        print(f"Error in smoothed_gradient: {e}")
        return [1.0] * max_iter


def fista(X, y, n, lam, max_iter, f_star):
    """FISTA算法"""
    try:
        solver = LassoSolvers(X, y, lam)
        beta = np.zeros(X.shape[1])
        beta_prev = beta.copy()
        t = 1.0
        objectives = []
        XTX = X.T @ X  # 预计算

        for k in range(max_iter):
            # 计算梯度: -X^T(y - Xβ) = X^TXβ - X^Ty
            gradient = (XTX @ beta - solver.XTy) / n

            # 梯度步
            beta_temp = beta - (1 / solver.L) * gradient

            # 临近算子
            threshold = lam / solver.L
            beta_new = np.sign(beta_temp) * np.maximum(np.abs(beta_temp) - threshold, 0)

            # FISTA加速
            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            beta = beta_new + ((t - 1) / t_new) * (beta_new - beta_prev)
            beta_prev = beta_new.copy()
            t = t_new

            # 记录次优性
            obj_val = solver.objective(beta)
            objectives.append(max(obj_val - f_star, 1e-10))

        return objectives
    except Exception as e:
        print(f"Error in FISTA: {e}")
        return [1.0] * max_iter


def fista_restart(X, y, n, lam, max_iter, f_star):
    """带重启的FISTA算法"""
    try:
        solver = LassoSolvers(X, y, lam)
        beta = np.zeros(X.shape[1])
        beta_prev = beta.copy()
        t = 1.0
        objectives = []
        XTX = X.T @ X  # 预计算

        # 保存函数值用于重启判断
        f_prev = solver.objective(beta)

        for k in range(max_iter):
            # 计算梯度: -X^T(y - Xβ) = X^TXβ - X^Ty
            gradient = (XTX @ beta - solver.XTy) / n

            # 梯度步
            beta_temp = beta - (1 / solver.L) * gradient

            # 临近算子
            threshold = lam / solver.L
            beta_new = np.sign(beta_temp) * np.maximum(np.abs(beta_temp) - threshold, 0)

            # 计算当前函数值
            f_new = solver.objective(beta_new)

            # 检查重启条件：函数值增加或梯度内积条件
            restart_condition = (f_new > f_prev) or (k > 0 and
                                                     np.dot(beta_new - beta_prev, beta_prev - beta) > 0)

            if restart_condition:
                # 重启
                beta = beta_new
                beta_prev = beta.copy()
                t = 1.0
            else:
                # FISTA加速
                t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
                beta = beta_new + ((t - 1) / t_new) * (beta_new - beta_prev)
                beta_prev = beta_new.copy()
                t = t_new

            f_prev = f_new

            # 记录次优性
            objectives.append(max(f_new - f_star, 1e-10))

        return objectives
    except Exception as e:
        print(f"Error in FISTA (Restart): {e}")
        return [1.0] * max_iter


def huber_gradient_accel(X, y, n, p, lam, max_iter, f_star, mu=0.01):
    """加速的Huber梯度法"""
    try:
        def huber_gradient(x, mu):
            return np.where(np.abs(x) <= mu, x / mu, np.sign(x))

        solver = LassoSolvers(X, y, lam)
        beta = np.zeros(p)
        v = beta.copy()
        t = 1.0
        objectives = []

        step_size = 1.0 / (solver.L + lam / mu)
        XTX = X.T @ X  # 预计算

        for k in range(max_iter):
            beta_prev = beta.copy()

            # 加速梯度步
            gradient_mse = (XTX @ v - solver.XTy) / n
            gradient_l1 = lam * huber_gradient(v, mu)
            gradient_total = gradient_mse + gradient_l1

            beta = v - step_size * gradient_total

            # 更新辅助变量
            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            v = beta + ((t - 1) / t_new) * (beta - beta_prev)
            t = t_new

            # 记录次优性
            obj_val = solver.objective(beta)
            objectives.append(max(obj_val - f_star, 1e-10))

        return objectives
    except Exception as e:
        print(f"Error in Huber Gradient (Accel): {e}")
        return [1.0] * max_iter


def huber_gradient_accel_restart(X, y, n, p, lam, max_iter, f_star, mu=0.01):
    """带重启的加速Huber梯度法"""
    try:
        def huber_gradient(x, mu):
            return np.where(np.abs(x) <= mu, x / mu, np.sign(x))

        solver = LassoSolvers(X, y, lam)
        beta = np.zeros(p)
        v = beta.copy()
        t = 1.0
        objectives = []

        step_size = 1.0 / (solver.L + lam / mu)
        XTX = X.T @ X  # 预计算

        for k in range(max_iter):
            beta_prev = beta.copy()

            # 加速梯度步
            gradient_mse = (XTX @ v - solver.XTy) / n
            gradient_l1 = lam * huber_gradient(v, mu)
            gradient_total = gradient_mse + gradient_l1

            beta_new = v - step_size * gradient_total

            # 计算函数值
            f_new = solver.objective(beta_new)
            f_prev = solver.objective(beta_prev) if k > 0 else float('inf')

            # 检查重启条件：函数值增加
            if f_new > f_prev * 0.999:  # 允许轻微增加
                # 重启
                v = beta_new
                t = 1.0
            else:
                # 更新辅助变量
                t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
                v = beta_new + ((t - 1) / t_new) * (beta_new - beta)
                t = t_new

            beta = beta_new

            # 记录次优性
            objectives.append(max(f_new - f_star, 1e-10))

        return objectives
    except Exception as e:
        print(f"Error in Huber (Accel + Restart): {e}")
        return [1.0] * max_iter


def generate_data(n, p, sparsity=0.1, noise_std=0.1, random_state=None):
    """生成测试数据"""
    if random_state is not None:
        np.random.seed(random_state)

    # 生成稀疏系数
    beta_true = np.zeros(p)
    n_nonzero = max(1, int(sparsity * p))  # 确保至少有一个非零元素
    nonzero_indices = np.random.choice(p, n_nonzero, replace=False)
    beta_true[nonzero_indices] = np.random.normal(0, 1, n_nonzero)

    # 生成特征矩阵和目标变量
    X = np.random.normal(0, 1, (n, p))
    y = X @ beta_true + np.random.normal(0, noise_std, n)

    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = (y - np.mean(y)) / np.std(y)

    return X, y, beta_true


def run_comparison_with_trials(n_trials, n_samples, n_features, lambda_val, max_iter):
    """运行多轮随机试验并比较算法性能"""

    # 算法配置 - 添加了临近点梯度法
    algo_configs = {
        'Coordinate Desc': {'color': 'blue', 'style': '-', 'width': 3},
        'Proximal Gradient': {'color': 'darkblue', 'style': '--', 'width': 3},  # 添加临近点梯度法
        'FISTA (Restart)': {'color': 'red', 'style': '-', 'width': 3},
        'Huber Gradient': {'color': 'green', 'style': '-', 'width': 3},
        'ADMM (rho=0.5)': {'color': 'orange', 'style': '-', 'width': 3},
        'Huber Gradient (Accel)': {'color': 'purple', 'style': '-', 'width': 3},
        'ADMM (rho=1)': {'color': 'brown', 'style': '-', 'width': 3},
        'Huber (Accel + Restart)': {'color': 'pink', 'style': '-', 'width': 3},
        'ADMM (rho=2)': {'color': 'gray', 'style': '-', 'width': 3},
        'FISTA': {'color': 'olive', 'style': '-', 'width': 3},
        'ADMM (rho=5)': {'color': 'cyan', 'style': '-', 'width': 3}
    }

    # 初始化结果字典
    results = {name: [] for name in algo_configs.keys()}
    performance_stats = {name: [] for name in algo_configs.keys()}

    print(f"Starting {n_trials} random trials...")
    start_time = time.time()

    for trial in range(n_trials):
        if (trial + 1) % 10 == 0:
            print(f"Trial {trial + 1}/{n_trials}")
            elapsed = time.time() - start_time
            remaining = elapsed / (trial + 1) * (n_trials - trial - 1)
            print(f"  Elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s")

        # 生成随机数据
        X, y, beta_true = generate_data(n_samples, n_features, random_state=trial)

        # 使用sklearn Lasso计算最优值f*
        try:
            lasso_sklearn = Lasso(alpha=lambda_val, max_iter=10000, tol=1e-8, fit_intercept=False)
            lasso_sklearn.fit(X, y)

            # 计算最优目标值
            solver_ref = LassoSolvers(X, y, lambda_val)
            f_star = solver_ref.objective(lasso_sklearn.coef_)
        except Exception as e:
            print(f"Error in calculating f_star: {e}")
            # 使用一个合理的估计值
            f_star = 0.1

        # --- 运行算法 ---

        # CD
        cd_start = time.time()
        cd_results = coordinate_descent(X, y, n_samples, n_features, lambda_val, max_iter, f_star)
        cd_time = time.time() - cd_start
        results['Coordinate Desc'].append(cd_results)
        performance_stats['Coordinate Desc'].append({
            'final_suboptimality': cd_results[-1] if cd_results else 1.0,
            'time': cd_time
        })

        # 临近点梯度法
        pg_start = time.time()
        pg_results = proximal_gradient(X, y, n_samples, n_features, lambda_val, max_iter, f_star)
        pg_time = time.time() - pg_start
        results['Proximal Gradient'].append(pg_results)
        performance_stats['Proximal Gradient'].append({
            'final_suboptimality': pg_results[-1] if pg_results else 1.0,
            'time': pg_time
        })

        # Huber 类
        hg_start = time.time()
        hg_results = smoothed_gradient(X, y, n_samples, n_features, lambda_val, max_iter, f_star)
        hg_time = time.time() - hg_start
        results['Huber Gradient'].append(hg_results)
        performance_stats['Huber Gradient'].append({
            'final_suboptimality': hg_results[-1] if hg_results else 1.0,
            'time': hg_time
        })

        hga_start = time.time()
        hga_results = huber_gradient_accel(X, y, n_samples, n_features, lambda_val, max_iter, f_star)
        hga_time = time.time() - hga_start
        results['Huber Gradient (Accel)'].append(hga_results)
        performance_stats['Huber Gradient (Accel)'].append({
            'final_suboptimality': hga_results[-1] if hga_results else 1.0,
            'time': hga_time
        })

        hgar_start = time.time()
        hgar_results = huber_gradient_accel_restart(X, y, n_samples, n_features, lambda_val, max_iter, f_star)
        hgar_time = time.time() - hgar_start
        results['Huber (Accel + Restart)'].append(hgar_results)
        performance_stats['Huber (Accel + Restart)'].append({
            'final_suboptimality': hgar_results[-1] if hgar_results else 1.0,
            'time': hgar_time
        })

        # FISTA 类
        fista_start = time.time()
        fista_results = fista(X, y, n_samples, lambda_val, max_iter, f_star)
        fista_time = time.time() - fista_start
        results['FISTA'].append(fista_results)
        performance_stats['FISTA'].append({
            'final_suboptimality': fista_results[-1] if fista_results else 1.0,
            'time': fista_time
        })

        fista_r_start = time.time()
        fista_r_results = fista_restart(X, y, n_samples, lambda_val, max_iter, f_star)
        fista_r_time = time.time() - fista_r_start
        results['FISTA (Restart)'].append(fista_r_results)
        performance_stats['FISTA (Restart)'].append({
            'final_suboptimality': fista_r_results[-1] if fista_r_results else 1.0,
            'time': fista_r_time
        })

        # ADMM 变体 (0.5, 1, 2, 5)
        admm_05_start = time.time()
        admm_05_results = admm(X, y, n_samples, n_features, lambda_val, 0.5, max_iter, f_star)
        admm_05_time = time.time() - admm_05_start
        results['ADMM (rho=0.5)'].append(admm_05_results)
        performance_stats['ADMM (rho=0.5)'].append({
            'final_suboptimality': admm_05_results[-1] if admm_05_results else 1.0,
            'time': admm_05_time
        })

        admm_1_start = time.time()
        admm_1_results = admm(X, y, n_samples, n_features, lambda_val, 1.0, max_iter, f_star)
        admm_1_time = time.time() - admm_1_start
        results['ADMM (rho=1)'].append(admm_1_results)
        performance_stats['ADMM (rho=1)'].append({
            'final_suboptimality': admm_1_results[-1] if admm_1_results else 1.0,
            'time': admm_1_time
        })

        admm_2_start = time.time()
        admm_2_results = admm(X, y, n_samples, n_features, lambda_val, 2.0, max_iter, f_star)
        admm_2_time = time.time() - admm_2_start
        results['ADMM (rho=2)'].append(admm_2_results)
        performance_stats['ADMM (rho=2)'].append({
            'final_suboptimality': admm_2_results[-1] if admm_2_results else 1.0,
            'time': admm_2_time
        })

        admm_5_start = time.time()
        admm_5_results = admm(X, y, n_samples, n_features, lambda_val, 5.0, max_iter, f_star)
        admm_5_time = time.time() - admm_5_start
        results['ADMM (rho=5)'].append(admm_5_results)
        performance_stats['ADMM (rho=5)'].append({
            'final_suboptimality': admm_5_results[-1] if admm_5_results else 1.0,
            'time': admm_5_time
        })

    total_time = time.time() - start_time
    print(f"All {n_trials} trials complete. Total time: {total_time:.2f} seconds.")

    # 绘制云雾图
    plot_convergence_cloud(results, algo_configs, n_samples, n_features, max_iter, n_trials)

    # 绘制性能比较图
    plot_performance_comparison(performance_stats, n_trials)

    # 输出性能统计表格
    print_performance_table(performance_stats)

    return results, performance_stats


def plot_convergence_cloud(results, algo_configs, n_samples, n_features, max_iter, n_trials):
    """绘制收敛云雾图"""
    plt.figure(figsize=(14, 9))
    k_axis = np.arange(max_iter)

    for name, histories in results.items():
        # 检查是否有有效数据
        if not histories or len(histories) == 0:
            print(f"Warning: No data for algorithm {name}")
            continue

        # 确保所有历史记录长度一致
        valid_histories = [h for h in histories if len(h) > 0]
        if not valid_histories:
            print(f"Warning: No valid history for algorithm {name}")
            continue

        min_len = min([len(h) for h in valid_histories])
        if min_len == 0:
            print(f"Warning: Empty history for algorithm {name}")
            continue

        # 创建数据矩阵
        data_matrix = []
        for h in valid_histories:
            if len(h) >= min_len:
                data_matrix.append(h[:min_len])

        if len(data_matrix) == 0:
            print(f"Warning: No valid data for algorithm {name}")
            continue

        data_matrix = np.array(data_matrix)
        current_k_axis = k_axis[:min_len]

        mean_curve = np.mean(data_matrix, axis=0)
        std_curve = np.std(data_matrix, axis=0)

        cfg = algo_configs[name]
        color = cfg['color']
        style = cfg['style']
        width = cfg.get('width', 2)

        # 绘制云雾（所有单次运行）
        for single_run in data_matrix[:min(50, len(data_matrix))]:  # 限制数量防止过密
            plt.plot(current_k_axis, single_run, color=color, alpha=0.05, linewidth=0.3)

        # 绘制平均曲线
        plt.plot(current_k_axis, mean_curve, color=color, linestyle=style,
                 linewidth=width, label=f"{name} (mean)")

        # 绘制±1标准差区域
        plt.fill_between(current_k_axis,
                         mean_curve - std_curve,
                         mean_curve + std_curve,
                         color=color, alpha=0.2)

    plt.yscale('log')
    plt.xlabel('Iteration k', fontsize=14)
    plt.ylabel('Suboptimality $f(x_k) - f^*$', fontsize=14)
    plt.title(f'LASSO Convergence: {n_trials} Randomized Trials\n(n={n_samples}, p={n_features}, λ={0.1})', fontsize=16)

    # 分两列显示图例，避免重叠
    plt.legend(fontsize=10, loc='upper right', framealpha=0.9, ncol=2)

    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.ylim(bottom=1e-10)
    plt.xlim(0, max_iter)

    plt.tight_layout()
    plt.savefig(f'lasso_convergence_cloud_{n_trials}_trials.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_performance_comparison(performance_stats, n_trials):
    """绘制性能比较图"""
    # 提取数据
    algorithms = list(performance_stats.keys())
    final_suboptimalities = []
    times = []
    algo_names = []

    for algo in algorithms:
        stats_list = performance_stats[algo]
        for stat in stats_list:
            final_suboptimalities.append(stat['final_suboptimality'])
            times.append(stat['time'])
            algo_names.append(algo)

    # 创建数据框
    df = pd.DataFrame({
        'Algorithm': algo_names,
        'Final Suboptimality': final_suboptimalities,
        'Time (s)': times
    })

    # 创建图形 - 只保留2个箱线图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. 最终次优性箱线图
    box1 = sns.boxplot(data=df, x='Algorithm', y='Final Suboptimality', ax=axes[0])
    axes[0].set_yscale('log')
    axes[0].set_title(f'Final Suboptimality Distribution\n({n_trials} Trials)')
    axes[0].set_ylabel('Final Suboptimality (log scale)')
    axes[0].set_xlabel('')

    # 旋转x轴标签，避免重叠
    axes[0].tick_params(axis='x', rotation=45)

    # 2. 时间箱线图
    box2 = sns.boxplot(data=df, x='Algorithm', y='Time (s)', ax=axes[1])
    axes[1].set_title(f'Computation Time Distribution\n({n_trials} Trials)')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].set_xlabel('')

    # 旋转x轴标签，避免重叠
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f'lasso_performance_comparison_{n_trials}_trials.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_performance_table(performance_stats):
    """输出性能统计表格"""
    print("\n" + "=" * 100)
    print("ALGORITHM PERFORMANCE SUMMARY")
    print("=" * 100)

    table_data = []

    for algo, stats_list in performance_stats.items():
        if not stats_list:
            continue

        final_suboptimalities = [stat['final_suboptimality'] for stat in stats_list]
        times = [stat['time'] for stat in stats_list]

        table_data.append({
            'Algorithm': algo,
            'Mean Final Suboptimality': np.mean(final_suboptimalities),
            'Std Final Suboptimality': np.std(final_suboptimalities),
            'Median Final Suboptimality': np.median(final_suboptimalities),
            'Mean Time (s)': np.mean(times),
            'Std Time (s)': np.std(times),
            'Median Time (s)': np.median(times)
        })

    # 创建数据框并排序
    df = pd.DataFrame(table_data)

    # 按平均最终次优性排序
    df_sorted_suboptimality = df.sort_values('Mean Final Suboptimality')

    print("\nSorted by Mean Final Suboptimality (Best to Worst):")
    print("-" * 110)
    print(df_sorted_suboptimality.to_string(index=False, float_format='%.6f'))

    # 按平均时间排序
    df_sorted_time = df.sort_values('Mean Time (s)')

    print("\n\nSorted by Mean Time (Fastest to Slowest):")
    print("-" * 110)
    print(df_sorted_time.to_string(index=False, float_format='%.6f'))

    # 计算效率得分（平衡时间和精度）
    df['Efficiency Score'] = 1.0 / (df['Mean Time (s)'] * df['Mean Final Suboptimality'])
    df_sorted_efficiency = df.sort_values('Efficiency Score', ascending=False)

    print("\n\nSorted by Efficiency Score (Time × Accuracy):")
    print("-" * 110)
    print(df_sorted_efficiency[['Algorithm', 'Efficiency Score', 'Mean Time (s)', 'Mean Final Suboptimality']]
          .to_string(index=False, float_format='%.6f'))

    # 保存到CSV
    df.to_csv(f'lasso_performance_summary_{len(stats_list)}_trials.csv', index=False, float_format='%.6f')
    print(f"\nDetailed performance summary saved to 'lasso_performance_summary_{len(stats_list)}_trials.csv'")


# 主程序
if __name__ == "__main__":
    print("开始Lasso算法收敛性比较...")

    # 运行主要比较（低维情形）
    results, performance_stats = run_comparison_with_trials(
        n_trials=100,  # 100次独立重复试验
        n_samples=200,  # 样本数
        n_features=50,  # 特征数
        lambda_val=0.1,  # 正则化参数
        max_iter=75  # 最大迭代次数
    )

    print("\n比较完成!")