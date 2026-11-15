import re
import math
import random
from datetime import datetime

def date_prediction_reward(
    prediction: str, 
    target: str, 
    alpha: float = 0.05,
    use_days: bool = True
) -> float:
    """
    计算日期预测的奖励值，可以精确到天数差异。
    """
    try:
        # 确保输入是字符串
        prediction = str(prediction) if not isinstance(prediction, str) else prediction
        target = str(target) if not isinstance(target, str) else target
        
        # 解析日期
        pred_date = datetime.strptime(prediction, "%Y-%m-%d")
        target_date = datetime.strptime(target, "%Y-%m-%d")
        
        if use_days:
            diff_in_days = abs((pred_date - target_date).days)
            # 半衰期90天：误差90天对应奖励0.5，给早期训练充分的正向梯度
            adjusted_alpha = math.log(2) / 90.0
            reward = math.exp(-adjusted_alpha * diff_in_days)
        else:
            year_diff = pred_date.year - target_date.year
            month_diff = pred_date.month - target_date.month
            diff_in_months = abs(year_diff * 12 + month_diff)
            reward = math.exp(-alpha * diff_in_months)
            
        return reward
        
    except (ValueError, TypeError) as e:
        return 0.0

def has_search_in_solution(solution_str):
    """检查解决方案中是否包含足够的检索操作
    要求：<search>标签和<information>标签都至少出现2次
    """
    # 查找所有 <search> 标签
    search_pattern = r'<search>.*?</search>'
    search_matches = list(re.finditer(search_pattern, solution_str, re.DOTALL))
    search_count = len(search_matches)
    
    # 查找所有 <information> 标签（检索结果）
    info_pattern = r'<information>.*?</information>'
    info_matches = list(re.finditer(info_pattern, solution_str, re.DOTALL))
    info_count = len(info_matches)
    
    # 只有当两个标签都出现至少2次时才返回True
    return search_count >= 2 and info_count >= 2

def normalize_answer(text):
    """规范化答案文本"""
    if not text:
        return ""
    # 如果输入是字节，先解码
    if isinstance(text, bytes):
        try:
            text = text.decode('utf-8')
        except:
            text = str(text)
    elif not isinstance(text, str):
        text = str(text)
    return text.lower().strip()

def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # 确保solution_str是字符串类型
    if isinstance(solution_str, bytes):
        try:
            solution_str = solution_str.decode('utf-8')
        except:
            solution_str = str(solution_str)
    elif not isinstance(solution_str, str):
        solution_str = str(solution_str)
    
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # 修复：如果没有匹配返回None，否则返回最后一个
    if len(matches) <=1:
        return None
    
    # 总是返回最后一个匹配
    return matches[-1].group(1).strip()

def is_valid_date_format(date_str):
    """验证date_str是否符合"YYYY-MM-DD"格式"""
    # 确保date_str是字符串类型
    if not date_str:
        return False
    
    # 如果输入是字节，先解码
    if isinstance(date_str, bytes):
        try:
            date_str = date_str.decode('utf-8')
        except:
            date_str = str(date_str)
    elif not isinstance(date_str, str):
        date_str = str(date_str)
    
    # 清理字符串
    date_str = date_str.strip()
    
    # 更新为正则表达式，支持YYYY-MM-DD格式
    # 年份: 任意4位数字
    # 月份: 01-12
    # 日期: 01-31（简单验证，不检查月份具体天数）
    pattern = r'^(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])-(?P<day>0[1-9]|[12]\d|3[01])$'
    return re.match(pattern, date_str) is not None

def compute_date_metrics(predictions: list, targets: list, search_turns: list = None,
                         prompt_tokens: list = None, response_tokens: list = None,
                         timing_info: list = None, difficulties: list = None):
    """
    计算日期预测的详细评估指标。

    Args:
        predictions: 预测日期列表 (提取后的答案)
        targets: 真实日期列表
        search_turns: 每个样本的检索轮数列表 (从 generation.py 的 valid_search_stats 获取)
        prompt_tokens: 每个样本的 prompt token 数量列表
        response_tokens: 每个样本的 response token 数量列表
        timing_info: 每个样本的耗时列表 (秒)
        difficulties: 每个样本的难度标签列表 (EXP/IMP/DIS)

    Returns:
        tuple: (metrics字典, 正确样本索引列表, 错误样本索引列表, 每个样本的误差天数列表)
    """
    from datetime import datetime

    metrics = {
        'exact_match': 0.0,      # 日期完全正确
        'acc_at_1': 0.0,         # 误差 ≤ 1 天
        'acc_at_7': 0.0,         # 误差 ≤ 7 天
        'acc_at_30': 0.0,        # 误差 ≤ 30 天
        'mae': 0.0,              # 平均绝对天数误差
        'median_ae': 0.0,        # 中位绝对天数误差
        'invalid_rate': 0.0,     # 输出非法日期比例
        'avg_search_turns': 0.0, # 平均检索轮数
        'latency': 0.0,          # 平均耗时
        'avg_prompt_tokens': 0.0,    # 平均 prompt token 数
        'avg_response_tokens': 0.0,  # 平均 response token 数
        'avg_total_tokens': 0.0,     # 平均总 token 数
    }

    # 用于记录正确/错误样本
    correct_indices = []   # 完全正确的样本索引
    wrong_indices = []     # 错误的样本索引（包括无效日期）
    day_error_list = []    # 每个有效样本的误差天数（与 predictions 长度一致，无效样本为 None）

    if not predictions or not targets:
        return metrics, correct_indices, wrong_indices, day_error_list

    n_samples = len(predictions)
    day_errors = []
    valid_count = 0
    search_turns_list = []
    latencies = []
    prompt_token_list = []
    response_token_list = []
    # 用于按难度分组统计
    difficulty_day_errors = {'EXP': [], 'IMP': [], 'DIS': []}
    difficulty_valid_count = {'EXP': 0, 'IMP': 0, 'DIS': 0}
    difficulty_search_turns = {'EXP': [], 'IMP': [], 'DIS': []}
    difficulty_latencies = {'EXP': [], 'IMP': [], 'DIS': []}
    difficulty_prompt_tokens = {'EXP': [], 'IMP': [], 'DIS': []}
    difficulty_response_tokens = {'EXP': [], 'IMP': [], 'DIS': []}
    difficulty_invalid_count = {'EXP': 0, 'IMP': 0, 'DIS': 0}
    difficulty_total_count = {'EXP': 0, 'IMP': 0, 'DIS': 0}

    for i, (pred, target) in enumerate(zip(predictions, targets)):
        # 获取当前样本的难度
        difficulty = difficulties[i] if difficulties and i < len(difficulties) else None
        if difficulty in difficulty_total_count:
            difficulty_total_count[difficulty] += 1

        # 检查预测是否是有效日期
        if not pred or not is_valid_date_format(pred):
            if difficulty in difficulty_invalid_count:
                difficulty_invalid_count[difficulty] += 1
            day_error_list.append(None)  # 无效日期
            wrong_indices.append(i)
            continue

        try:
            pred_date = datetime.strptime(pred, "%Y-%m-%d")
            target_date = datetime.strptime(target, "%Y-%m-%d")
            day_diff = abs((pred_date - target_date).days)
            day_errors.append(day_diff)
            day_error_list.append(day_diff)
            valid_count += 1

            # 记录正确/错误样本索引
            if day_diff == 0:
                correct_indices.append(i)
            else:
                wrong_indices.append(i)

            # 按难度分组收集数据
            if difficulty in difficulty_day_errors:
                difficulty_day_errors[difficulty].append(day_diff)
                difficulty_valid_count[difficulty] += 1
        except (ValueError, TypeError):
            if difficulty in difficulty_invalid_count:
                difficulty_invalid_count[difficulty] += 1
            day_error_list.append(None)  # 无效日期
            wrong_indices.append(i)
            continue

        # 获取检索轮数（从 generation.py 的 valid_search_stats 统计）
        if search_turns and i < len(search_turns):
            search_turns_list.append(search_turns[i])
            if difficulty in difficulty_search_turns:
                difficulty_search_turns[difficulty].append(search_turns[i])

        # 收集耗时信息
        if timing_info and i < len(timing_info):
            latencies.append(timing_info[i])
            if difficulty in difficulty_latencies:
                difficulty_latencies[difficulty].append(timing_info[i])

        # 收集 token 信息
        if prompt_tokens and i < len(prompt_tokens):
            prompt_token_list.append(prompt_tokens[i])
            if difficulty in difficulty_prompt_tokens:
                difficulty_prompt_tokens[difficulty].append(prompt_tokens[i])
        if response_tokens and i < len(response_tokens):
            response_token_list.append(response_tokens[i])
            if difficulty in difficulty_response_tokens:
                difficulty_response_tokens[difficulty].append(response_tokens[i])

    # 计算无效率
    metrics['invalid_rate'] = 1.0 - (valid_count / n_samples) if n_samples > 0 else 1.0

    if day_errors:
        # Exact Match: 误差为0，分母为总样本数
        metrics['exact_match'] = sum(1 for e in day_errors if e == 0) / n_samples

        # Acc@1: 误差 ≤ 1 天，分母为总样本数
        metrics['acc_at_1'] = sum(1 for e in day_errors if e <= 1) / n_samples

        # Acc@7: 误差 ≤ 7 天，分母为总样本数
        metrics['acc_at_7'] = sum(1 for e in day_errors if e <= 7) / n_samples

        # Acc@30: 误差 ≤ 30 天，分母为总样本数
        metrics['acc_at_30'] = sum(1 for e in day_errors if e <= 30) / n_samples

        # MAE: 平均绝对天数误差，无效样本不计入均值（仅统计有效预测的误差）
        metrics['mae'] = sum(day_errors) / len(day_errors)

        # Median AE: 中位绝对天数误差（仅统计有效预测）
        sorted_errors = sorted(day_errors)
        n = len(sorted_errors)
        if n % 2 == 0:
            metrics['median_ae'] = (sorted_errors[n//2 - 1] + sorted_errors[n//2]) / 2
        else:
            metrics['median_ae'] = sorted_errors[n//2]

    # 平均检索轮数
    if search_turns_list:
        metrics['avg_search_turns'] = sum(search_turns_list) / len(search_turns_list)

    # 平均耗时
    if latencies:
        metrics['latency'] = sum(latencies) / len(latencies)

    # Token 开销统计
    if prompt_token_list:
        metrics['avg_prompt_tokens'] = sum(prompt_token_list) / len(prompt_token_list)
    if response_token_list:
        metrics['avg_response_tokens'] = sum(response_token_list) / len(response_token_list)
    if prompt_token_list and response_token_list:
        total_tokens = [p + r for p, r in zip(prompt_token_list, response_token_list)]
        metrics['avg_total_tokens'] = sum(total_tokens) / len(total_tokens)

    # 按难度分组计算指标
    for diff_level in ['EXP', 'IMP', 'DIS']:
        if difficulty_total_count[diff_level] == 0:
            continue

        # 无效率
        invalid_rate = difficulty_invalid_count[diff_level] / difficulty_total_count[diff_level]
        metrics[f'{diff_level}/invalid_rate'] = invalid_rate

        day_errors_diff = difficulty_day_errors[diff_level]
        total_diff = difficulty_total_count[diff_level]
        if day_errors_diff:
            metrics[f'{diff_level}/exact_match'] = sum(1 for e in day_errors_diff if e == 0) / total_diff
            metrics[f'{diff_level}/acc_at_1'] = sum(1 for e in day_errors_diff if e <= 1) / total_diff
            metrics[f'{diff_level}/acc_at_7'] = sum(1 for e in day_errors_diff if e <= 7) / total_diff
            metrics[f'{diff_level}/acc_at_30'] = sum(1 for e in day_errors_diff if e <= 30) / total_diff
            metrics[f'{diff_level}/mae'] = sum(day_errors_diff) / len(day_errors_diff)

            # Median AE
            sorted_errors_diff = sorted(day_errors_diff)
            n_diff = len(sorted_errors_diff)
            if n_diff % 2 == 0:
                metrics[f'{diff_level}/median_ae'] = (sorted_errors_diff[n_diff//2 - 1] + sorted_errors_diff[n_diff//2]) / 2
            else:
                metrics[f'{diff_level}/median_ae'] = sorted_errors_diff[n_diff//2]

        # 平均检索轮数
        if difficulty_search_turns[diff_level]:
            metrics[f'{diff_level}/avg_search_turns'] = sum(difficulty_search_turns[diff_level]) / len(difficulty_search_turns[diff_level])

        # 平均耗时
        if difficulty_latencies[diff_level]:
            metrics[f'{diff_level}/latency'] = sum(difficulty_latencies[diff_level]) / len(difficulty_latencies[diff_level])

        # Token 统计
        if difficulty_prompt_tokens[diff_level]:
            metrics[f'{diff_level}/avg_prompt_tokens'] = sum(difficulty_prompt_tokens[diff_level]) / len(difficulty_prompt_tokens[diff_level])
        if difficulty_response_tokens[diff_level]:
            metrics[f'{diff_level}/avg_response_tokens'] = sum(difficulty_response_tokens[diff_level]) / len(difficulty_response_tokens[diff_level])
        if difficulty_prompt_tokens[diff_level] and difficulty_response_tokens[diff_level]:
            total_tokens_diff = [p + r for p, r in zip(difficulty_prompt_tokens[diff_level], difficulty_response_tokens[diff_level])]
            metrics[f'{diff_level}/avg_total_tokens'] = sum(total_tokens_diff) / len(total_tokens_diff)

    return metrics, correct_indices, wrong_indices, day_error_list


def compute_score_em(solution_str, ground_truth, use_search_bonus=True, reward_type='timeagent', reward_tag='full'):
    """The scoring function for exact match (EM).

    Args:
        use_search_bonus: Whether to add search bonus to the reward. Default True for training.
                          Set to False for validation to use only base_reward * difficulty_multiplier.
        reward_type: timeagent uses date-distance reward; searchr1 uses exact match reward.
        reward_tag: ablation tag, one of use_search_reward / use_date_reward / full.
    """
    # 确保solution_str是字符串类型
    if isinstance(solution_str, bytes):
        try:
            solution_str = solution_str.decode('utf-8')
        except:
            solution_str = str(solution_str)
    elif not isinstance(solution_str, str):
        solution_str = str(solution_str)
    
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    # 获取 difficulty（如果存在）
    difficulty = None
    if isinstance(ground_truth, dict):
        difficulty = ground_truth.get('difficulty', None)
        true_target = ground_truth.get('target')[0]
    else:
        true_target = ground_truth[0]
    
    
    # 检查是否进行了检索
    has_search = has_search_in_solution(solution_str)
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {true_target}")
        print(f"Extracted answer: {answer}")
        print(f"Difficulty: {difficulty}, Has search: {has_search}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return -0.2  # 未输出有效 answer 标签，与格式错误惩罚一致

    if reward_type in ('searchr1', 'naiverag'):
        return 1.0 if normalize_answer(answer) == normalize_answer(true_target) else 0.0

    if reward_type != 'timeagent':
        raise ValueError(f'Unknown reward_type: {reward_type}')

    if not answer or not is_valid_date_format(answer):
        return -0.2  # 格式错误惩罚，与大致猜测的正向收益量级匹配

    if not true_target or not is_valid_date_format(true_target):
        return 0  # 无法评估的情况
    
    # 2. 基础准确性奖励
    base_reward = date_prediction_reward(answer, true_target)
    
    # 3. 难度调整系数（DIS 降至 1.8 为检索 bonus 留出空间）
    difficulty_multiplier = {
        'EXP': 1.0,
        'IMP': 1.5,
        'DIS': 1.8
    }

    # 4. 检索奖励（DIS 检索率低需更强激励；DIS 精确+检索恰好触顶 2.0）
    search_bonus = 0.0
    if has_search:
        if difficulty == 'DIS':
            search_bonus = 0.2
        elif difficulty == 'IMP':
            search_bonus = 0.1
        else:
            # EXP 任务检索反而惩罚（与 base_reward 挂钩，答得越准惩罚越重）
            search_bonus = -base_reward * 0.2
    
    # 5. 综合奖励
    date_reward = base_reward * difficulty_multiplier.get(difficulty, 1.0)
    if reward_tag == 'use_search_reward':
        total_reward = search_bonus if use_search_bonus else 0.0
    elif reward_tag == 'use_date_reward':
        total_reward = date_reward
    elif reward_tag == 'full':
        total_reward = date_reward + search_bonus if use_search_bonus else date_reward
    else:
        raise ValueError(f'Unknown reward_tag: {reward_tag}')

    # 6. 边界处理
    return max(min(total_reward, 2.0), -1.0)  # 限制奖励范围
