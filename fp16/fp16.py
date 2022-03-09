import imp
import importlib
import sys
import os
import argparse
import torch
from transformers import AdamW
# REFERENCE https://github.com/cooelf/AwesomeMRC/blob/master/transformer-mrc/examples/run_squad.py


# optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
# n_gpu = torch.cuda.device_count()

def do_train(args, model, optimizer, n_gpu, step):
    
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
    else:
        pass

    return_dict = model()
    loss = return_dict.loss

    if args.fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    
    if (step + 1) % args.gradient_accumulation_steps == 0:
        if args.fp16:
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    args = parser.parse_args()
