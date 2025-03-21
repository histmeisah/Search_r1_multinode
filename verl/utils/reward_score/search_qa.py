# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random

def normalize_answer(s):
    """Normalize answer text for comparison."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    """Check if prediction exactly matches any of the golden answers."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def extract_answer_from_search_response(solution_str):
    """Extract answer from a multi-turn search response."""
    # 首先尝试查找最后的<answer>标签
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    
    if answer_matches:
        # 使用最后一个<answer>标签
        return answer_matches[-1].group(1).strip()
    
    # 如果没有找到<answer>标签，查找最后一段文本
    # 这是一个后备方案，适用于可能格式不正确的情况
    paragraphs = re.split(r'\n\s*\n', solution_str)
    if paragraphs:
        return paragraphs[-1].strip()
    
    return ""


def compute_score(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """
    计算搜索问答任务的奖励分数
    
    Args:
        solution_str: 模型生成的完整响应
        ground_truth: 正确答案(s)
        method: 评估方法 (strict/flexible)
        format_score: 格式正确但答案错误时的分数
        score: 答案正确时的满分
        
    Returns:
        分数值
    """
    # 随机选择一小部分样本打印，用于调试
    do_print = random.randint(1, 64) == 1
    
    # 抽取最终答案
    answer = extract_answer_from_search_response(solution_str)
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if not answer:
        # 如果没有提取到答案，给0分
        return 0.0
    
    # 检查提取的答案是否匹配任何正确答案
    if isinstance(ground_truth, dict) and 'target' in ground_truth:
        # 如果ground_truth是一个带有'target'键的字典
        targets = ground_truth['target']
        if em_check(answer, targets):
            return score
    elif isinstance(ground_truth, (list, tuple)):
        # 如果ground_truth是一个列表或元组
        if em_check(answer, ground_truth):
            return score
    else:
        # 如果ground_truth是一个字符串
        if em_check(answer, ground_truth):
            return score
    
    # 答案不匹配
    return format_score


def reward_with_process_bonus(solution_str, ground_truth, method='strict', search_bonus=0.3, answer_bonus=0.7):
    """
    计算带有过程奖励的搜索问答任务分数。
    这个函数不仅奖励正确答案，还奖励搜索行为。
    
    Args:
        solution_str: 模型生成的完整响应
        ground_truth: 正确答案(s)
        method: 评估方法
        search_bonus: 成功执行搜索的奖励
        answer_bonus: 正确回答的奖励
        
    Returns:
        总分数及其组成部分的字典
    """
    # 检查是否执行了搜索
    search_pattern = r'<search>(.*?)</search>'
    search_count = len(re.findall(search_pattern, solution_str))
    has_search = search_count > 0
    
    # 检查是否有信息区域（表示搜索结果成功获取）
    info_pattern = r'<information>(.*?)</information>'
    info_count = len(re.findall(info_pattern, solution_str))
    has_info = info_count > 0
    
    # 检查是否有最终答案
    answer_pattern = r'<answer>(.*?)</answer>'
    has_answer = bool(re.search(answer_pattern, solution_str))
    
    # 获取答案的准确性分数
    answer_accuracy = 0.0
    if has_answer:
        answer = extract_answer_from_search_response(solution_str)
        if isinstance(ground_truth, dict) and 'target' in ground_truth:
            targets = ground_truth['target']
            answer_accuracy = float(em_check(answer, targets))
        elif isinstance(ground_truth, (list, tuple)):
            answer_accuracy = float(em_check(answer, ground_truth))
        else:
            answer_accuracy = float(em_check(answer, ground_truth))
    
    # 计算过程分数
    process_score = 0.0
    if has_search:
        process_score += 0.2  # 奖励搜索行为
    if has_info:
        process_score += 0.3  # 奖励成功获取信息
    if has_answer:
        process_score += 0.2  # 奖励给出最终答案
    
    # 总分 = 过程分 * 过程权重 + 准确度分 * 准确度权重
    total_score = (process_score * search_bonus) + (answer_accuracy * answer_bonus)
    
    return {
        'total_score': total_score,
        'process_score': process_score,
        'answer_accuracy': answer_accuracy
    } 