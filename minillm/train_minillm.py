import torch
import os
import json
import torch.distributed as dist
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM, AutoConfig

from arguments import get_args
from utils import print_args, initialize, load_parallel, get_tokenizer, parallel_model_map

from minillm import train, Reward

from peft import PeftModel


def get_teacher_model(args, device):
    """
    加载教师模型,根据配置决定是否使用模型并行和PEFT。
    
    参数:
        args: 训练参数对象，包含模型路径、类型等配置信息
        device: 指定的设备(如GPU)
    
    返回:
        评估模式下的教师模型
    """
    # 从预训练模型路径加载配置
    config = AutoConfig.from_pretrained(args.teacher_model_path)
    # 判断是否使用模型并行
    if args.model_parallel:
        config.is_model_parallel = True
        with init_empty_weights():   # 使用加速器库的空权重初始化方法
            if args.model_type=="qwen":
                model = parallel_model_map[args.model_type](config).to(torch.bfloat16)
            else:
                model = parallel_model_map[args.model_type](config).half()
            # 加载并行模型的权重
        load_parallel(model, args.teacher_model_path)
        model = model.to(device)
    else:
        config.is_model_parallel = False
        model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model_path, 
            config=config, 
            device_map={"": device}, 
            torch_dtype=torch.float16 if args.model_type!="qwen" else torch.bfloat16
        )

        if args.peft is not None:
            if args.peft == "lora":
                assert args.teacher_peft_path is not None
                model = PeftModel.from_pretrained(model, args.peft_path)
            else:
                raise NotImplementedError
        else:
            if dist.get_rank() == 0:
                print(' > number of parameters: {}'.format(
                    sum([p.nelement() for p in model.parameters()])), flush=True)

    model.eval()

    return model


def main():
    # 获取并初始化训练参数
    args = get_args()   # 解析命令行参数
    initialize(args)    # 初始化分布式训练环境

    # 设置CUDA设备
    device = torch.cuda.current_device()
    
    # 创建保存目录并保存参数
    os.makedirs(args.save, exist_ok=True)
    if dist.get_rank() == 0: # 只在主进程中打印和保存
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)

    # 加载和设置DeepSpeed配置       
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    # 设置DeepSpeed训练参数
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000
    
    # 设置精度和模型类型
    args.fp32 = not ds_config["fp16"]["enabled"]
    args.deepspeed_config = None
    if args.teacher_model_type is None:
        args.teacher_model_type = args.model_type

    # 加载模型和tokenizer
    teacher_model = get_teacher_model(args, device)
    tokenizer = get_tokenizer(args)
    
    # 创建奖励对象
    reward = Reward(args, tokenizer, teacher_model)
    
    train(
        args=args,
        tokenizer=tokenizer,
        reward_fn=reward.reward_fn,
        teacher_model=teacher_model,
        ds_config=ds_config,
        prompt_data=args.prompt_data_dir,
        eval_prompt_data=args.prompt_data_dir,
        lm_data=args.lm_data_dir,
        eval_lm_data=args.lm_data_dir,
    )


if __name__ == "__main__":
    main()