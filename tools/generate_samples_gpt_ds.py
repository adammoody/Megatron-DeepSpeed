# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Sample Generate GPT"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

import torch
from functools import partial

from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel, GPTModelPipe
from megatron.training import get_model, setup_model_and_optimizer
from megatron.text_generation_utils import generate_and_write_samples_unconditional
from megatron.text_generation_utils import generate_samples_input_from_file
from megatron.text_generation_utils import generate_samples_interactive
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
import os
import subprocess


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    see_memory_usage(f"Before Building Model", force=True)

    args = get_args()
    with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=args.deepspeed_config,
                             enabled=args.zero_stage == 3,
                             mpu=mpu):
        if args.deepspeed:
            model = GPTModelPipe(
                num_tokentypes=0,
                parallel_output=True
            )

            # This is a hack to give us a reference to get_batch_pipe from within training.py
            # We need to call model.set_batch_fn after deepspeed.initialize
            #model._megatron_batch_fn = get_batch_pipe
            model._megatron_batch_fn = get_batch_sample

            # Predompute the attention mask and store it in args. This avoids having to
            # pipeline it as an activation during training. The mask is constant, and thus
            # we can reuse it.
            attention_mask = torch.tril(torch.ones(
                (1, args.seq_length, args.seq_length), device=torch.cuda.current_device())).view(
                    1, 1, args.seq_length, args.seq_length)

            # Convert attention mask to binary:
            attention_mask = (attention_mask < 0.5)
            if args.fp16:
                attention_mask = attention_mask.half()
            elif args.bf16:
                attention_mask = attention_mask.bfloat16()

            # must be bool or the training crashes expecting bool, but getting Half
            args.attn_mask = attention_mask.to(torch.bool)

        else:
            model = GPTModel(
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process
            )
    see_memory_usage(f"After Building Model", force=True)
    return model


class EvalIterable:
    def __init__(self, samples):
        self.idx = 0
        self.samples = samples

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < len(self.samples):
            #vals = get_batch(self.samples[self.idx])
            vals = self.samples[self.idx]
            self.idx += 1
            return vals
        raise StopIteration


def get_batch_sample(tokens):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    data = dict()
    if mpu.get_tensor_model_parallel_rank() == 0:
        data['text'] = torch.zeros((1, args.seq_length + 1), dtype=datatype)
        data['text'][:] = tokenizer.eod
        data['text'][0,:len(tokens)] = torch.tensor(tokens, dtype=datatype)
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return (tokens, position_ids, attention_mask), (labels, loss_mask)


def add_text_generate_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='text generation')

    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--greedy", action='store_true', default=False,
                       help='Use greedy sampling.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=0,
                       help='Top k sampling.')
    group.add_argument("--out-seq-length", type=int, default=1024,
                       help='Size of the output generated text.')
    group.add_argument("--sample-input-file", type=str, default=None,
                       help='Get input from file instead of interactive mode, '
                       'each line is an input.')
    group.add_argument("--sample-output-file", type=str, default=None,
                       help='Output file got from --sample-input-file')
    group.add_argument("--num-samples", type=int, default=0,
                       help='Number of samples to generate unconditionally, '
                       'defaults to 0 and interactive conditional sampling')
    group.add_argument("--genfile", type=str,
                       help='Output file when generating unconditionally')
    group.add_argument("--recompute", action='store_true',
                       help='During generation recompute all attention '
                       'instead of using previously computed keys/values.')

    return parser


def allgather_vocab_logits(args, vocab_parallel_logits):
    """Gather vocab_parallel_logits to all ranks from tensor-parallel ranks from final pipeline stage."""
    rank = mpu.get_tensor_model_parallel_rank()
    world_size = mpu.get_tensor_model_parallel_world_size()

    partition_vocab_size = 0
    if vocab_parallel_logits is not None:
        # Get the partition's vocab indecies
        get_vocab_range = mpu.utils.VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        vocab_start_index, vocab_end_index = get_vocab_range(
            partition_vocab_size, rank, world_size)

    size_tensor = torch.tensor([partition_vocab_size], dtype=torch.int64).to(torch.cuda.current_device()).to(torch.cuda.current_device())
    torch.distributed.all_reduce(size_tensor, op=torch.distributed.ReduceOp.SUM, group=mpu.get_model_parallel_group())

    vocab_logits = torch.zeros((1, args.seq_length, size_tensor[0]), dtype=torch.float32).to(torch.cuda.current_device())
    if vocab_parallel_logits is not None:
        vocab_logits[:, :, vocab_start_index:vocab_end_index] = vocab_parallel_logits.to(torch.float32)
    torch.distributed.all_reduce(vocab_logits, op=torch.distributed.ReduceOp.SUM, group=mpu.get_model_parallel_group())
    return vocab_logits


def get_next_token(args, tokens, vocab_logits):
    numtokens = len(tokens)
    maxval = torch.max(vocab_logits, dim=-1)
    return int(maxval[1][0][numtokens-1])

    #topval = torch.topk(vocab_logits, 2, dim=-1)

    percent = vocab_logits.clone()
    percent = torch.nn.functional.softmax(percent, dim=-1)
    topperc = torch.topk(percent, 5, dim=-1)

    sortedidxs = torch.argsort(vocab_logits, dim=-1, descending=True) 
    if torch.distributed.get_rank() == 0:
        print(sortedidxs[0][numtokens-1])
        print(torch.where(sortedidxs[0][numtokens-1] == tokens[numtokens-1]))
    return int(topperc[1][0][numtokens-1])


def main():
    """Main program."""

    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        os.environ["RANK"] = os.environ['OMPI_COMM_WORLD_RANK']
    if 'OMPI_COMM_WORLD_SIZE' in os.environ:
        os.environ["WORLD_SIZE"] = os.environ['OMPI_COMM_WORLD_SIZE']
    if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        os.environ["LOCAL_RANK"] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']

    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    rank = torch.distributed.get_rank()

    # Set up model and load checkpoint.
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider)

    samples = ["The oakland athletics baseball team is two games behind the houston astros in the american league west division"]

    tokenizer = get_tokenizer()
    context = tokenizer.tokenize(samples[0])

    tokens = context
    while len(tokens) < 100:
        eval_data_iterator = EvalIterable([tokens])

        # ../../../pytorch/deepspeed-bigsci/DeepSpeed/deepspeed/runtime/pipe/engine.py
        output = model[0].eval_batch(eval_data_iterator, compute_loss=False)
        vocab_logits = allgather_vocab_logits(args, output)

        next_token = get_next_token(args, tokens, vocab_logits)
        tokens.append(next_token)
        #if rank == 0:
        #    print(len(tokens), tokenizer.detokenize(tokens), flush=True)
        #    print(len(tokens), tokens, flush=True)

        #next_tokens = gather_output(args, tokens, output)
        #for t in range(len(next_tokens)):
        #    tmptokens = list(tokens)
        #    tmptokens.append(int(next_tokens[t]))
        #    if rank == 0:
        #        #print(next_token, tokens, tokenizer.detokenize(tokens), flush=True)
        #        print(t, tokenizer.detokenize(tmptokens), flush=True)
        #tokens = tmptokens
        #tokens.append(int(next_tokens[0]))
        #if rank == 0:
        #    print(len(tokens), tokenizer.detokenize(tokens), flush=True)
        #    print(len(tokens), tokens, flush=True)

    if rank == 0:
        print(len(tokens), tokenizer.detokenize(tokens), flush=True)

    torch.distributed.barrier()
    quit()


if __name__ == "__main__":

    main()
