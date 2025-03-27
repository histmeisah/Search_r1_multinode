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
        self.max_turns = config.get('max_turns', 3)
        
        # 添加max_obs_length参数来控制检索结果的最大长度
        self.max_obs_length = config.get('max_obs_length', 300)
        
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
    
    def _check_tensor_valid(self, tensor):
        """检查张量是否包含非填充标记，避免索引越界错误"""
        if tensor.shape[0] == 0:
            return False
        non_pad = (tensor != self.pad_token_id).any(dim=1)
        return non_pad.all().item()
        
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
                # No valid operation found - 添加无效操作反馈
                need_continue[i] = True
                search_queries.append("invalid query")
                search_results_mapping[i] = len(search_queries) - 1
        
        # Perform batch search for all queries
        search_results = {}
        if search_queries:
            try:
                results = self.search_client.batch_search(search_queries)
                formatted_results = self.search_client.format_search_results(results)
                
                # Map results back to original batch indices
                for batch_idx, result_idx in search_results_mapping.items():
                    if result_idx < len(formatted_results):
                        # 检查是否是无效查询
                        if search_queries[result_idx] == "invalid query":
                            search_results[batch_idx] = "My previous response format was invalid. I should use <search>query</search> to search or <answer>final_answer</answer> to provide my answer."
                        else:
                            search_results[batch_idx] = formatted_results[result_idx]
            except Exception as e:
                print(f"Error during search: {e}")
                # 提供一个默认响应
                for batch_idx in search_results_mapping:
                    search_results[batch_idx] = "No search results found due to an error."
        
        return need_continue, search_results, final_answers
    
    def _create_next_turn_inputs(self, original_inputs, responses, search_results):
        """创建下一轮的输入，限制检索结果长度"""
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
                # 限制检索结果长度
                search_text = search_results[i]
                # 如果搜索结果太长，进行智能截断
                search_tokens = self.tokenizer(search_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
                if len(search_tokens) > self.max_obs_length:
                    # 截断检索结果，保留开头和结尾的重要部分
                    keep_front = self.max_obs_length // 2
                    keep_end = self.max_obs_length - keep_front - 5  # 留5个token放提示信息
                    
                    front_text = self.tokenizer.decode(search_tokens[:keep_front], skip_special_tokens=True)
                    end_text = self.tokenizer.decode(search_tokens[-keep_end:], skip_special_tokens=True)
                    search_text = f"{front_text}\n...[content truncated]...\n{end_text}"
                    print(f"Truncated search result from {len(search_tokens)} to approximately {self.max_obs_length} tokens")
                
                next_text = f"{response_text}\n\n<information>{search_text}</information>\n\n"
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
    
    def _verify_non_empty_input(self, inputs):
        """验证输入是否包含非填充标记，如果没有则添加一个默认标记"""
        # 每一行是否都包含非填充标记
        has_non_pad = (inputs != self.pad_token_id).any(dim=1)
        
        if not has_non_pad.all():
            # 找出全部是填充的行
            problematic_rows = torch.where(~has_non_pad)[0]
            print(f"Warning: Found {len(problematic_rows)} rows with all padding tokens, adding a default token")
            
            # 为这些行添加一个默认的标记（例如，使用eos标记）
            default_token = self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else 0
            for row in problematic_rows:
                inputs[row, 0] = default_token
                
        return inputs
    
    def _safe_pre_process_inputs(self, prompt_token_ids: torch.Tensor) -> List[int]:
        """安全的输入预处理函数，处理全填充输入"""
        non_pad_mask = (prompt_token_ids != self.pad_token_id)
        if not non_pad_mask.any():
            # 如果全是填充标记，返回一个默认序列
            return [self.tokenizer.eos_token_id]
        
        # 使用第一个非填充位置开始
        non_pad_indices = torch.nonzero(non_pad_mask, as_tuple=False)
        if len(non_pad_indices) == 0:
            return [self.tokenizer.eos_token_id]
        
        non_pad_index = non_pad_indices[0][0]
        token_ids = prompt_token_ids[non_pad_index:].tolist()
        return token_ids
    
    @torch.no_grad()
    def generate_sequences_with_search(self, prompts: DataProto, **kwargs) -> DataProto:
        """支持多轮交互搜索的生成方法，使用动态内容长度处理"""
        # 获取初始输入
        input_ids = prompts.batch['input_ids']
        
        # 初始化attention_mask和position_ids属性
        self.attention_mask = prompts.batch['attention_mask']
        self.position_ids = prompts.batch['position_ids']
        
        # 添加安全检查：确保每个输入序列至少包含一个非填充标记
        for i in range(input_ids.size(0)):
            if not (input_ids[i] != self.pad_token_id).any():
                # 如果序列全是填充标记，添加一个非填充标记（如EOS标记）
                input_ids[i, 0] = self.tokenizer.eos_token_id
        
        # 更新prompts对象
        prompts.batch['input_ids'] = input_ids
        
        # 获取相关元信息
        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        
        # 使用和vllm_rollout.py相同的参数处理方式
        search_kwargs = {}
        if not do_sample:
            search_kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        elif is_validate:
            search_kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            }
        
        # 验证阶段可能需要调整搜索相关参数
        if is_validate:
            print("Running validation with search functionality")
            max_turns = min(self.max_turns, 3)  # 验证时限制搜索轮数
        else:
            max_turns = self.max_turns
        
        # 获取n值，用于后续处理
        n_samples = self.sampling_params.n if do_sample else 1
        
        # 从配置获取关键参数
        max_prompt_length = self.config.prompt_length
        max_response_length = self.config.response_length
        
        # 保存原始输入
        original_inputs = input_ids.clone()
        
        # 存储生成的响应和跟踪信息
        all_responses = []
        active_mask = torch.ones(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
        final_answers = [None] * input_ids.size(0)
        
        # 使用动态内容管理方式，而不是固定分配
        rolling_context = input_ids.clone()
        
        # 多轮生成循环
        for turn in range(max_turns):
            if not active_mask.any():
                break
            
            # 获取活跃示例
            num_active = active_mask.sum().item()
            print(f"Turn {turn}, active examples: {num_active}/{input_ids.size(0)}")
            
            # 准备当前轮次的输入 - 动态剪裁到有效长度
            current_context = {}
            for k, v in {'input_ids': rolling_context, 'attention_mask': self.attention_mask, 'position_ids': self.position_ids}.items():
                current_context[k] = v[active_mask]
            
            # 执行动态裁剪，只针对实际存在的键
            keys_to_cut = ['input_ids', 'attention_mask', 'position_ids']
            keys_to_cut = [k for k in keys_to_cut if k in current_context]
            
            if keys_to_cut:
                current_context = self._cut_to_effective_len(
                    current_context, 
                    keys=keys_to_cut,
                    max_length=max_prompt_length
                )
            
            # 确保裁剪后的输入有效
            if not self._check_tensor_valid(current_context['input_ids']):
                print("Warning: Invalid tensors after cutting to effective length, adding safety token")
                current_context['input_ids'] = self._verify_non_empty_input(current_context['input_ids'])
            
            # 准备当前轮次的输入
            active_prompts = DataProto.from_dict({
                'input_ids': current_context['input_ids'],
                'attention_mask': current_context.get('attention_mask', torch.ones_like(current_context['input_ids'], dtype=torch.int)),
                'position_ids': current_context.get('position_ids', torch.arange(current_context['input_ids'].size(1), device=current_context['input_ids'].device).expand(current_context['input_ids'].size(0), -1))
            })
            active_prompts.meta_info = prompts.meta_info.copy()
            
            # 使用自定义的安全生成实现:
            gen_output = self._safe_generate_sequences(active_prompts, **{**kwargs, **search_kwargs})
            responses = gen_output.batch['responses']
            
            # 处理形状不匹配问题
            if responses.shape[0] != num_active:
                print(f"Warning: Expected responses shape[0]={num_active}, got {responses.shape[0]}")
                if responses.shape[0] > num_active:
                    responses = responses[:num_active]
            
            # 创建当前轮次的响应张量
            turn_responses = torch.zeros(
                (input_ids.size(0), responses.size(1)), 
                dtype=responses.dtype, 
                device=input_ids.device
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
            if active_mask.any() and turn < max_turns - 1:
                # 创建带搜索结果的下一轮输入
                active_search_results = {
                    i: search_results[new_active_indices[i]] 
                    for i in range(len(new_active_indices)) 
                    if new_active_indices[i] in search_results
                }
                
                # 创建下一轮输入
                try:
                    next_inputs = self._create_next_turn_inputs(
                        rolling_context[active_mask],
                        responses,
                        active_search_results
                    )
                    
                    # 更新rolling_context - 使用动态上下文管理
                    new_rolling_context = rolling_context.clone()
                    active_rolling_contexts = self._update_rolling_context(
                        rolling_context[active_mask], 
                        responses, 
                        next_inputs, 
                        max_prompt_length
                    )
                    
                    # 将更新后的上下文应用到活跃示例
                    for i, idx in enumerate(torch.where(active_mask)[0]):
                        if i < active_rolling_contexts.shape[0]:
                            new_rolling_context[idx] = active_rolling_contexts[i]
                    
                    rolling_context = new_rolling_context
                    
                    # 更新attention_mask和position_ids
                    self.attention_mask = (rolling_context != self.pad_token_id).to(dtype=torch.int)
                    self.position_ids = torch.cumsum(self.attention_mask, dim=1) - 1
                    self.position_ids = self.position_ids * self.attention_mask  # 确保pad位置的position_id为0
                    
                except Exception as e:
                    print(f"Error preparing next turn inputs: {e}")
                    import traceback
                    traceback.print_exc()
                    # 如果出错，保持原始输入不变
                    print("Keeping original inputs due to error")
        
        # 合并所有响应
        if len(all_responses) == 0:
            combined_responses = torch.zeros(
                (input_ids.size(0), 1), 
                dtype=input_ids.dtype, 
                device=input_ids.device
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
            info_mask = torch.zeros_like(combined_responses, dtype=torch.bool, device=input_ids.device)
        
        # 构建最终输出
        final_output = {
            'prompts': original_inputs,
            'responses': combined_responses,
            'input_ids': torch.cat([original_inputs, combined_responses], dim=1),
            'attention_mask': torch.cat([
                torch.ones_like(original_inputs, dtype=torch.int, device=input_ids.device),
                torch.ones_like(combined_responses, dtype=torch.int, device=input_ids.device)
            ], dim=1),
            'position_ids': torch.arange(
                original_inputs.size(1) + combined_responses.size(1), device=input_ids.device
            ).expand(input_ids.size(0), -1),
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
        
        # 无论是验证还是训练，只要启用了搜索就使用搜索
        if self.enable_search:
            print(f"[DEBUG] generate_sequences - Using search-enabled generation (validate={is_validate})")
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

    def _cut_to_effective_len(self, tensor_dict, keys, max_length=None):
        """裁剪张量到有效长度，避免处理太多填充"""
        effective_len = tensor_dict['attention_mask'].sum(dim=1).max().item()
        if max_length is not None:
            effective_len = min(effective_len, max_length)
        
        result = {}
        for k, v in tensor_dict.items():
            if k in keys:
                result[k] = v[:, -effective_len:]
            else:
                result[k] = v
        
        return result

    def _update_rolling_context(self, input_ids, responses, next_obs_ids, max_length):
        """动态更新上下文，确保有效使用token窗口"""
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # 创建新的上下文张量
        new_contexts = torch.full(
            (batch_size, max_length),
            self.pad_token_id,
            dtype=input_ids.dtype,
            device=device
        )
        
        for i in range(batch_size):
            # 计算各部分的有效内容（去掉padding）
            input_mask = (input_ids[i] != self.pad_token_id)
            resp_mask = (responses[i] != self.pad_token_id)
            obs_mask = (next_obs_ids[i] != self.pad_token_id)
            
            # 提取有效内容
            valid_input = input_ids[i][input_mask]
            valid_resp = responses[i][resp_mask]
            valid_obs = next_obs_ids[i][obs_mask]
            
            # 计算各部分长度
            input_len = valid_input.size(0)
            resp_len = valid_resp.size(0)
            obs_len = valid_obs.size(0)
            
            # 计算总长度
            total_len = input_len + resp_len + obs_len
            
            if total_len <= max_length:
                # 所有内容能放入，直接拼接
                cur_pos = 0
                
                if input_len > 0:
                    new_contexts[i, cur_pos:cur_pos + input_len] = valid_input
                    cur_pos += input_len
                
                if resp_len > 0:
                    new_contexts[i, cur_pos:cur_pos + resp_len] = valid_resp
                    cur_pos += resp_len
                
                if obs_len > 0:
                    new_contexts[i, cur_pos:cur_pos + obs_len] = valid_obs
            else:
                # 内容太多，需要智能截断
                # 分配原则：原始输入30%，响应10%，观察结果60%
                max_input_len = int(max_length * 0.3)
                max_resp_len = int(max_length * 0.1)
                max_obs_len = max_length - max_input_len - max_resp_len
                
                # 如果某部分内容较少，动态调整分配
                if input_len < max_input_len:
                    extra = max_input_len - input_len
                    max_obs_len += extra
                    max_input_len = input_len
                    
                if resp_len < max_resp_len:
                    extra = max_resp_len - resp_len
                    max_obs_len += extra
                    max_resp_len = resp_len
                
                # 进行内容截断
                if input_len > max_input_len:
                    # 保留开头部分
                    valid_input = valid_input[:max_input_len]
                
                if resp_len > max_resp_len:
                    # 优先保留最新响应
                    valid_resp = valid_resp[-max_resp_len:]
                
                if obs_len > max_obs_len:
                    # 重要信息通常在开头和结尾，所以分别保留一部分
                    if max_obs_len > 2:
                        head_size = max_obs_len // 2
                        tail_size = max_obs_len - head_size
                        valid_obs = torch.cat([
                            valid_obs[:head_size],
                            valid_obs[-tail_size:]
                        ])
                    else:
                        # 如果空间极小，只保留开头
                        valid_obs = valid_obs[:max_obs_len]
                
                # 拼接截断后的内容
                cur_pos = 0
                
                if len(valid_input) > 0:
                    end_pos = min(cur_pos + len(valid_input), max_length)
                    new_contexts[i, cur_pos:end_pos] = valid_input[:end_pos-cur_pos]
                    cur_pos = end_pos
                
                if len(valid_resp) > 0 and cur_pos < max_length:
                    end_pos = min(cur_pos + len(valid_resp), max_length)
                    new_contexts[i, cur_pos:end_pos] = valid_resp[:end_pos-cur_pos]
                    cur_pos = end_pos
                
                if len(valid_obs) > 0 and cur_pos < max_length:
                    end_pos = min(cur_pos + len(valid_obs), max_length)
                    new_contexts[i, cur_pos:end_pos] = valid_obs[:end_pos-cur_pos]
        
        # 确保每一行至少有一个非填充token
        new_contexts = self._verify_non_empty_input(new_contexts)
        
        return new_contexts

    def _safe_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """安全版本的生成方法，避免调用super().generate_sequences可能导致的问题"""
        # 如果self.config.free_cache_engine为True，则重建vllm缓存引擎
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # 用于构造attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        # 安全检查：确保输入中没有全填充序列
        for i in range(batch_size):
            if not (idx[i] != self.pad_token_id).any():
                print(f"Warning: Input sequence {i} contains only padding tokens, adding a default token")
                idx[i, 0] = self.tokenizer.eos_token_id

        idx_list = []
        # 从torch.Tensor解析idx为List[List[str]]
        for i in range(batch_size):
            try:
                # 使用安全的预处理方法
                non_pad_mask = (idx[i] != self.pad_token_id)
                if not non_pad_mask.any():
                    idx_list.append([self.tokenizer.eos_token_id])
                    continue
                    
                non_pad_indices = torch.nonzero(non_pad_mask, as_tuple=False)
                if len(non_pad_indices) == 0:
                    idx_list.append([self.tokenizer.eos_token_id])
                    continue
                    
                non_pad_index = non_pad_indices[0][0]
                token_ids = idx[i][non_pad_index:].tolist()
                idx_list.append(token_ids)
            except Exception as e:
                print(f"Error preprocessing input at index {i}: {e}")
                # 提供一个默认值
                idx_list.append([self.tokenizer.eos_token_id])

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        elif is_validate:
            kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            }

        # 使用update_sampling_params上下文管理器
        with self.update_sampling_params(**kwargs):
            try:
                output = self.inference_engine.generate(
                    prompts=None,  # 因为我们已经将其转换为prompt token id
                    sampling_params=self.sampling_params,
                    prompt_token_ids=idx_list,
                    use_tqdm=False)

                # 如果n = 1: (bs, response_length) ; 如果 n > 1: (bs * n, response_length)
                response = output[0].to(idx.device)

                if response.shape[1] < self.config.response_length:
                    from verl.utils.torch_functional import pad_sequence_to_length
                    response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)

                # 使用当前采样参数
                if self.sampling_params.n > 1 and do_sample:
                    idx = idx.repeat_interleave(self.sampling_params.n, dim=0)
                    attention_mask = attention_mask.repeat_interleave(self.sampling_params.n, dim=0)
                    position_ids = position_ids.repeat_interleave(self.sampling_params.n, dim=0)
                    batch_size = batch_size * self.sampling_params.n
                seq = torch.cat([idx, response], dim=-1)
            
            except Exception as e:
                print(f"Error during vLLM generation: {e}")
                # 创建一个空响应作为后备
                response = torch.full((batch_size, self.config.response_length), 
                                      self.pad_token_id, 
                                      device=idx.device, 
                                      dtype=idx.dtype)
                response[:, 0] = self.tokenizer.eos_token_id  # 至少添加一个有意义的标记
                seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # 修复position_ids
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        
        # 创建响应的attention_mask
        from verl.utils.torch_functional import get_eos_mask
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # 所有tp排名应包含相同的数据。所有排名中的数据都有效
        from tensordict import TensorDict
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # 此处input_ids成为整个句子
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # 释放vllm缓存引擎
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)