# verl/workers/rollout/vllm_rollout/search_vllm_rollout.py

import re
import torch
import requests
from typing import List, Dict, Any, Tuple
from contextlib import contextmanager
from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.workers.rollout.vllm_rollout.vllm_rollout import vLLMRollout
from vllm import SamplingParams
from omegaconf import OmegaConf
import torch.nn.utils.rnn as rnn_utils
import time

class HTTPSearchClient:
    """HTTP client for remote search API calls"""
    
    def __init__(self, search_url, topk=3):
        self.search_url = search_url
        self.topk = topk
        
    def batch_search(self, queries: List[str]):
        """Call remote search API and return search results"""
        if not queries:
            return []
            
        payload = {
            "queries": queries,
            "topk": self.topk,
            "return_scores": True
        }
        
        try:
            response = requests.post(self.search_url, json=payload, timeout=10).json()
            return response["result"]
        except Exception as e:
            print(f"Search API error: {e}")
            # 返回空结果作为后备
            return [[] for _ in range(len(queries))]
        
    def format_search_results(self, results):
        """Format search results as readable text"""
        formatted_results = []
        
        for result_set in results:
            format_reference = ''
            for idx, doc_item in enumerate(result_set):
                content = doc_item['document']['contents']
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
            formatted_results.append(format_reference)
            
        return formatted_results

class SearchEnabledVLLMRollout(vLLMRollout):
    """vLLM rollout with search capabilities during generation"""
    
    def __init__(self, actor_module, config, tokenizer, model_hf_config, **kwargs):
        # 确保搜索相关配置被保留为OmegaConf类
        self.enable_search = config.get('enable_search', False)
        self.search_url = config.get('search_url', 'http://localhost:8000/retrieve')
        self.search_topk = config.get('search_topk', 3)
        self.max_turns = config.get('max_turns', 5)
        
        # 保存tokenizer对象
        self.tokenizer = tokenizer
        
        # 初始化搜索客户端
        self.search_client = HTTPSearchClient(
            search_url=self.search_url,
            topk=self.search_topk
        )
        
        # 设置搜索相关配置
        self.search_pattern = r'<search>(.*?)</search>'  # 检测搜索查询的正则表达式模式
        self.answer_pattern = r'<answer>(.*?)</answer>'  # 检测最终答案的正则表达式模式
        
        # 调用父类初始化
        super().__init__(actor_module, config, tokenizer, model_hf_config, **kwargs)
        
        # 记录设备信息，避免设备不一致问题
        try:
            self.model_device = next(self.inference_engine.model.parameters()).device
        except:
            self.model_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    def _process_search_in_batch(self, responses):
        """
        Process a batch of responses to identify and execute searches
        
        Returns:
            Tuple of (need_continue_flags, search_results, final_answers)
        """
        batch_size = responses.size(0)
        
        # Decode responses to text
        response_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        
        # Process each response to find search queries or answers
        search_queries = []
        need_continue = [False] * batch_size
        search_results_mapping = {}  # Maps batch index to search result position
        final_answers = [None] * batch_size
        
        for i, text in enumerate(response_texts):
            search_match = re.search(self.search_pattern, text, re.DOTALL)
            answer_match = re.search(self.answer_pattern, text, re.DOTALL)
            
            if search_match:
                # Extract search query and add to batch
                query = search_match.group(1).strip()
                search_queries.append(query)
                search_results_mapping[i] = len(search_queries) - 1
                need_continue[i] = True
            elif answer_match:
                # Extract final answer
                final_answers[i] = answer_match.group(1).strip()
                need_continue[i] = False
            else:
                # No valid operation found
                need_continue[i] = True
        
        # Perform batch search for all queries
        search_results = {}
        if search_queries:
            try:
                results = self.search_client.batch_search(search_queries)
                formatted_results = self.search_client.format_search_results(results)
                
                # Map results back to original batch indices
                for batch_idx, result_idx in search_results_mapping.items():
                    if result_idx < len(formatted_results):
                        search_results[batch_idx] = formatted_results[result_idx]
            except Exception as e:
                print(f"Error during search: {e}")
                # 提供一个默认响应
                for batch_idx in search_results_mapping:
                    search_results[batch_idx] = "No search results found due to an error."
        
        return need_continue, search_results, final_answers
    
    def _create_next_turn_inputs(self, original_inputs, responses, search_results):
        """创建下一轮的输入"""
        batch_size = original_inputs.size(0)
        device = original_inputs.device
        next_inputs = []
        
        for i in range(batch_size):
            # Get the response for this example
            response_text = self.tokenizer.decode(responses[i], skip_special_tokens=True)
            
            # Process the response to extract relevant parts
            for pattern in [r'</search>.*$', r'</answer>.*$']:
                match = re.search(pattern, response_text)
                if match:
                    end_pos = match.start() + len(match.group(0).split('>')[0]) + 1
                    response_text = response_text[:end_pos]
            
            # Create the next context based on whether we have search results
            if i in search_results:
                next_text = f"{response_text}\n\n<information>{search_results[i]}</information>\n\n"
            else:
                next_text = response_text
                
            # Tokenize for next turn and move to correct device
            next_input = self.tokenizer(next_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
            next_inputs.append(next_input)
        
        # 使用PyTorch的pad_sequence函数处理不同长度序列
        padded_inputs = rnn_utils.pad_sequence(next_inputs, batch_first=True, padding_value=self.pad_token_id).to(device)
        
        return padded_inputs
    
    def _safe_indexing(self, tensor, dim, start, end):
        """安全索引，处理边界情况"""
        size = tensor.size(dim)
        if start >= size:
            return torch.tensor([], device=tensor.device, dtype=tensor.dtype)
        elif end <= 0:
            return torch.tensor([], device=tensor.device, dtype=tensor.dtype)
        else:
            start = max(0, start)
            end = min(size, end)
            if start >= end:
                return torch.tensor([], device=tensor.device, dtype=tensor.dtype)
            return tensor.narrow(dim, start, end - start)
    
    @torch.no_grad()
    def generate_sequences_for_validation(self, prompts: DataProto, **kwargs) -> DataProto:
        """生成用于验证的序列，不使用搜索"""
        # 直接调用父类方法，绕过搜索功能
        return super().generate_sequences(prompts, **kwargs)
    
    @torch.no_grad()
    def generate_sequences_with_search(self, prompts: DataProto, **kwargs) -> DataProto:
        """支持多轮交互搜索的生成方法，始终保持固定大小的输出张量"""
        # 检查是否是验证过程
        is_validate = prompts.meta_info.get('validate', False)
        if is_validate:
            print("Running validation, skipping search functionality")
            return self.generate_sequences_for_validation(prompts, **kwargs)
        
        # 获取初始输入和配置信息
        input_ids = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # 获取n值，用于后续处理
        do_sample = prompts.meta_info.get('do_sample', True)
        n_samples = self.sampling_params.n if do_sample else 1
        
        # 从配置获取关键参数
        max_prompt_length = self.config.prompt_length
        max_response_length = self.config.response_length
        
        # 保存原始输入
        original_inputs = input_ids.clone()
        
        # 存储生成的响应和跟踪信息
        all_responses = []
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        final_answers = [None] * batch_size
        
        # 在多轮搜索过程中，维持固定大小的输入
        fixed_size_input_ids = input_ids.clone()
        
        # 多轮生成循环
        for turn in range(self.max_turns):
            if not active_mask.any():
                break
            
            # 获取活跃示例
            num_active = active_mask.sum().item()
            print(f"Turn {turn}, active examples: {num_active}/{batch_size}")
            
            # 准备当前轮次的输入
            active_prompts = DataProto.from_dict({
                'input_ids': fixed_size_input_ids[active_mask],
                'attention_mask': attention_mask[active_mask],
                'position_ids': position_ids[active_mask],
            })
            active_prompts.meta_info = prompts.meta_info.copy()
            
            # 生成响应
            gen_output = super().generate_sequences(active_prompts, **kwargs)
            responses = gen_output.batch['responses']
            
            # 处理形状不匹配问题
            if responses.shape[0] != num_active:
                print(f"Warning: Expected responses shape[0]={num_active}, got {responses.shape[0]}")
                if responses.shape[0] > num_active:
                    responses = responses[:num_active]
            
            # 创建当前轮次的响应张量
            turn_responses = torch.zeros(
                (batch_size, responses.size(1)), 
                dtype=responses.dtype, 
                device=device
            )
            
            # 将活跃示例的响应放入适当位置
            active_indices = torch.where(active_mask)[0]
            for i, idx in enumerate(active_indices):
                if i < responses.size(0):
                    turn_responses[idx] = responses[i]
            
            # 存储响应
            all_responses.append(turn_responses)
            
            # 处理搜索查询并执行搜索
            need_continue, search_results, turn_answers = self._process_search_in_batch(responses)
            
            # 更新活跃掩码和收集最终答案
            new_active_indices = []
            for i, (needs_continuation, answer) in enumerate(zip(need_continue, turn_answers)):
                if i >= len(active_indices):
                    continue
                active_idx = active_indices[i]
                if not needs_continuation:
                    active_mask[active_idx] = False
                    if answer is not None:
                        final_answers[active_idx] = answer
                else:
                    new_active_indices.append(i)
            
            # 如果需要继续，准备下一轮输入
            if active_mask.any() and turn < self.max_turns - 1:
                # 创建带搜索结果的下一轮输入
                active_search_results = {
                    i: search_results[new_active_indices[i]] 
                    for i in range(len(new_active_indices)) 
                    if new_active_indices[i] in search_results
                }
                
                # 创建下一轮输入
                try:
                    next_inputs = self._create_next_turn_inputs(
                        original_inputs[active_mask],
                        responses,
                        active_search_results
                    )
                    
                    # 打印调试信息
                    print(f"Original inputs shape: {original_inputs[active_mask].shape}, Next inputs shape: {next_inputs.shape}")
                    print(f"Total length would be: {original_inputs[active_mask].shape[1] + next_inputs.shape[1]}")
                    
                    # 分配空间：原始输入使用前半部分，搜索结果使用后半部分
                    orig_alloc = max_prompt_length // 2
                    next_alloc = max_prompt_length - orig_alloc
                    
                    # 创建固定大小的新输入张量
                    new_fixed_inputs = torch.full(
                        (next_inputs.shape[0], max_prompt_length),
                        self.pad_token_id,
                        dtype=next_inputs.dtype,
                        device=device
                    )
                    
                    # 填充原始输入（取最后部分）
                    if original_inputs[active_mask].shape[1] <= orig_alloc:
                        new_fixed_inputs[:, :original_inputs[active_mask].shape[1]] = original_inputs[active_mask]
                    else:
                        new_fixed_inputs[:, :orig_alloc] = original_inputs[active_mask][:, -orig_alloc:]
                        print(f"Truncating original inputs from {original_inputs[active_mask].shape[1]} to {orig_alloc}")
                    
                    # 填充下一轮输入（取前部分）
                    if next_inputs.shape[1] <= next_alloc:
                        new_fixed_inputs[:, orig_alloc:orig_alloc+next_inputs.shape[1]] = next_inputs
                    else:
                        new_fixed_inputs[:, orig_alloc:] = next_inputs[:, :next_alloc]
                        print(f"Truncating next inputs from {next_inputs.shape[1]} to {next_alloc}")
                    
                    # 更新活跃示例的fixed_size_input_ids
                    fixed_size_input_ids_clone = fixed_size_input_ids.clone()
                    for i, idx in enumerate(torch.where(active_mask)[0]):
                        if i < new_fixed_inputs.shape[0]:
                            fixed_size_input_ids_clone[idx] = new_fixed_inputs[i]
                    
                    fixed_size_input_ids = fixed_size_input_ids_clone
                    
                except Exception as e:
                    print(f"Error preparing next turn inputs: {e}")
                    # 如果出错，保持原始输入不变
                    print("Keeping original inputs due to error")
        
        # 合并所有响应
        if len(all_responses) == 0:
            combined_responses = torch.zeros(
                (batch_size, 1), 
                dtype=input_ids.dtype, 
                device=device
            )
        else:
            combined_responses = torch.cat(all_responses, dim=1)
        
        # 如果响应长度超过最大值，截断
        if combined_responses.size(1) > max_response_length:
            print(f"Truncating combined responses from {combined_responses.size(1)} to {max_response_length}")
            combined_responses = combined_responses[:, :max_response_length]
        
        # 创建信息掩码
        try:
            state_masking_config = self.config.get('state_masking', {})
            if isinstance(state_masking_config, dict):
                start_marker = state_masking_config.get('start_state_marker', "<information>")
                end_marker = state_masking_config.get('end_state_marker', "</information>")
            else:
                start_marker = "<information>"
                end_marker = "</information>"
            
            info_mask = self._create_info_mask(combined_responses, start_marker, end_marker)
        except Exception as e:
            print(f"Error creating info mask: {e}")
            info_mask = torch.zeros_like(combined_responses, dtype=torch.bool, device=device)
        
        # 构建最终输出
        final_output = {
            'prompts': original_inputs,
            'responses': combined_responses,
            'input_ids': torch.cat([original_inputs, combined_responses], dim=1),
            'attention_mask': torch.cat([
                torch.ones_like(original_inputs, dtype=torch.int, device=device),
                torch.ones_like(combined_responses, dtype=torch.int, device=device)
            ], dim=1),
            'position_ids': torch.arange(
                original_inputs.size(1) + combined_responses.size(1), device=device
            ).expand(batch_size, -1),
            'info_mask': info_mask
        }
        
        # 创建结果
        result = DataProto.from_dict(final_output)
        result.meta_info = prompts.meta_info.copy()
        result.meta_info['final_answers'] = final_answers
        
        # 模拟vLLMRollout中的n参数行为
        if n_samples > 1 and do_sample:
            print(f"Repeating result {n_samples} times to match sampling_params.n={n_samples}")
            result = result.repeat(repeat_times=n_samples, interleave=True)
        
        return result
    
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        Override the generate_sequences method to use the search-enabled version
        """
        # 记录输入信息
        input_batch_size = prompts.batch.batch_size[0] if prompts.batch is not None else None
        input_keys = list(prompts.batch.keys()) if prompts.batch is not None else []
        
        print(f"[DEBUG] generate_sequences - Input batch size: {input_batch_size}")
        print(f"[DEBUG] generate_sequences - Input keys: {input_keys}")
        
        # 检查是否是验证模式
        is_validate = prompts.meta_info.get('validate', False)
        
        # 记录元信息
        print(f"[DEBUG] generate_sequences - Meta info: validate={is_validate}, keys={list(prompts.meta_info.keys())}")
        
        start_time = time.time()
        
        # 检查是否启用搜索
        if self.enable_search and not is_validate:
            print("[DEBUG] generate_sequences - Using search-enabled generation")
            result = self.generate_sequences_with_search(prompts, **kwargs)
        else:
            # 否则，使用父类实现
            print("[DEBUG] generate_sequences - Using standard generation (without search)")
            result = super().generate_sequences(prompts, **kwargs)
        
        end_time = time.time()
        
        # 记录输出信息
        output_batch_size = result.batch.batch_size[0] if result.batch is not None else None
        output_keys = list(result.batch.keys()) if result.batch is not None else []
        
        print(f"[DEBUG] generate_sequences - Output batch size: {output_batch_size}")
        print(f"[DEBUG] generate_sequences - Output keys: {output_keys}")
        print(f"[DEBUG] generate_sequences - Generation time: {end_time - start_time:.2f} seconds")
        
        # 检查批次大小是否匹配
        if input_batch_size != output_batch_size:
            print(f"[DEBUG] generate_sequences - BATCH SIZE MISMATCH: Input={input_batch_size}, Output={output_batch_size}")
            # 检查每个张量的形状
            for key in output_keys:
                if key in prompts.batch:
                    input_shape = tuple(prompts.batch[key].shape)
                    output_shape = tuple(result.batch[key].shape)
                    print(f"[DEBUG] generate_sequences - Shape comparison for '{key}': Input={input_shape}, Output={output_shape}")
        
        return result

    def _create_info_mask(self, responses, start_marker, end_marker):
        """
        创建信息区域掩码，标记<information>和</information>之间的内容
        """
        batch_size, seq_len = responses.size()
        device = responses.device
        info_mask = torch.zeros_like(responses, dtype=torch.bool, device=device)
        
        # 解码响应以查找标记
        response_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=False)
        
        for i, text in enumerate(response_texts):
            # 找到所有信息区域
            start_positions = [m.start() for m in re.finditer(re.escape(start_marker), text)]
            end_positions = [m.start() for m in re.finditer(re.escape(end_marker), text)]
            
            # 确保标记成对出现
            num_regions = min(len(start_positions), len(end_positions))
            
            if num_regions == 0:
                continue
            
            # 对每个区域创建掩码
            for j in range(num_regions):
                start_pos = start_positions[j]
                # 找到对应的结束位置（在开始位置之后的第一个结束标记）
                valid_ends = [pos for pos in end_positions if pos > start_pos]
                if not valid_ends:
                    continue
                end_pos = min(valid_ends)
                
                # 找到对应的token索引
                start_token_idx = len(self.tokenizer.encode(text[:start_pos], add_special_tokens=False))
                end_token_idx = len(self.tokenizer.encode(text[:end_pos + len(end_marker)], add_special_tokens=False))
                
                # 设置掩码，确保索引不超出范围
                start_idx = min(start_token_idx, seq_len-1)
                end_idx = min(end_token_idx, seq_len)
                if start_idx < end_idx:
                    info_mask[i, start_idx:end_idx] = True
        
        return info_mask

    def _create_final_output(self, original_inputs, combined_responses, info_mask=None):
        """创建最终输出，确保尺寸一致并符合长度限制"""
        batch_size = original_inputs.size(0)
        device = original_inputs.device
        
        # 确定最终响应的长度，不超过最大响应长度
        max_response_length = self.config.response_length
        if combined_responses.size(1) > max_response_length:
            combined_responses = combined_responses[:, :max_response_length]
            if info_mask is not None:
                info_mask = info_mask[:, :max_response_length]
        
        # 构建输出
        final_output = {
            'prompts': original_inputs,
            'responses': combined_responses,
            'input_ids': torch.cat([original_inputs, combined_responses], dim=1)
        }
        
        # 创建一致的attention_mask
        final_output['attention_mask'] = torch.cat([
            torch.ones_like(original_inputs, dtype=torch.int, device=device),
            torch.ones_like(combined_responses, dtype=torch.int, device=device)
        ], dim=1)
        
        # 创建一致的position_ids
        final_output['position_ids'] = torch.arange(
            final_output['input_ids'].size(1), device=device
        ).expand(batch_size, -1)
        
        # 添加info_mask如果存在
        if info_mask is not None:
            final_output['info_mask'] = info_mask
        
        return final_output