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


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """提取答案，支持搜索和非搜索场景"""
    # 首先尝试查找<answer>标签
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    
    if answer_matches:
        # 使用最后一个<answer>标签（对多轮交互有用）
        return answer_matches[-1].group(1).strip()
    
    # 如果没有找到<answer>标签，尝试其他启发式方法
    lines = solution_str.strip().split('\n')
    if lines:
        return lines[-1].strip()
    
    return None


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """统一的EM打分函数，适用于普通QA和搜索QA任务"""
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        if isinstance(ground_truth, dict) and 'target' in ground_truth:
            print(f"Golden answers: {ground_truth['target']}")
        else:
            print(f"Golden answers: {ground_truth}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    
    # 处理不同格式的ground_truth
    if isinstance(ground_truth, dict) and 'target' in ground_truth:
        targets = ground_truth['target']
        if em_check(answer, targets):
            return score
    elif isinstance(ground_truth, (list, tuple)):
        if em_check(answer, ground_truth):
            return score
    else:
        if em_check(answer, ground_truth):
            return score
    
    return format_score


def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score

# 添加一个可选的带过程奖励的计分函数
def compute_score_with_process(solution_str, ground_truth, method='strict', search_bonus=0.3, answer_bonus=0.7):
    """带有过程奖励的评分函数"""
    # 检查搜索行为
    search_pattern = r'<search>(.*?)</search>'
    search_count = len(re.findall(search_pattern, solution_str))
    has_search = search_count > 0
    
    # 检查信息获取
    info_pattern = r'<information>(.*?)</information>'
    info_count = len(re.findall(info_pattern, solution_str))
    has_info = info_count > 0
    
    # 检查是否有最终答案
    answer_pattern = r'<answer>(.*?)</answer>'
    has_answer = bool(re.search(answer_pattern, solution_str))
    
    # 计算基本分数
    base_score = compute_score_em(solution_str, ground_truth)
    
    # 计算过程分数
    process_score = 0.0
    if has_search:
        process_score += 0.2
    if has_info:
        process_score += 0.3
    if has_answer:
        process_score += 0.2
    
    # 总分 = 过程分数 * 过程权重 + 基本分数 * 答案权重
    total_score = (process_score * search_bonus) + (base_score * answer_bonus)
    
    return total_score
