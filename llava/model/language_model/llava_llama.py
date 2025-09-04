#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        cluster_ids: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if input_ids is not None:
            batch_size = len(input_ids)
        
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        
        # cluster_ids is currently unused; reserved for cluster-conditioned routing
        _ = cluster_ids

        # propagate cluster_ids into MoE MLPs if present
        if cluster_ids is not None:
            try:
                for layer in self.model.model.layers:
                    mlp = getattr(layer, 'mlp', None)
                    if hasattr(mlp, 'set_cluster_ids'):
                        mlp.set_cluster_ids(cluster_ids)
            except Exception:
                pass

        ret_dict = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        if ret_dict.get('loss') is not None:
            text_loss = ret_dict['loss'].mean()
        if ret_dict.get('loss') is not None and self.llm_moe and self.llm_moe_num_experts > 1:
            llm_mlp_routing_probs = torch.stack([r[0] for r in ret_dict.routings], dim=0) # [layer, batch, seq_len, num_experts]
            llm_mlp_routing_idxes = torch.stack([r[1] for r in ret_dict.routings], dim=0).detach()

            llm_mlp_expert_balancing_loss = 0.
            for i in range(batch_size):
                probs_i = llm_mlp_routing_probs[:,i, attention_mask[i].bool()].reshape(-1, self.llm_moe_num_experts)
                idxes_i = llm_mlp_routing_idxes[:,i, attention_mask[i].bool()].reshape(-1, self.llm_moe_num_experts)

                llm_mlp_expert_balancing_loss += (probs_i.mean(0) * idxes_i.mean(0)).sum()

            moe_balance_term = llm_mlp_expert_balancing_loss/batch_size * self.moe_balance_w
            text_loss += moe_balance_term
            # attach simple utilization stats for logging (mean over layers and tokens)
            with torch.no_grad():
                util = llm_mlp_routing_probs.mean(dim=(0,2))  # [batch, num_experts]
                ret_dict['moe_utilization'] = util.detach().cpu()
                # store last stats on the model for callbacks
                try:
                    self._last_moe_utilization = util.detach().mean(0)  # [num_experts]
                    self._last_moe_balance = moe_balance_term.detach().float().cpu()
                except Exception:
                    pass
        if ret_dict.get('loss') is not None:
            ret_dict['loss'] = text_loss
        return ret_dict

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        cluster_ids = kwargs.pop("cluster_ids", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        if cluster_ids is not None:
            inputs['cluster_ids'] = cluster_ids
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
