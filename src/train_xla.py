from transformers import AutoTokenizer
import torch

from utils import (
    set_manual_seed_all,
    load_checkpoint,
    sync_batch_idx,
)
from train import (
    load_model,
    prepare_train_data,
    prepare_data_loaders,
    init_optimizer,
    init_lr_scheduler,
    init_criterion,
    train_epoch,
)
from validate import validate_epoch
from evaluate import RougeEvaluator

import torch_xla.core.xla_model as xm   
from torch_xla.distributed.parallel_loader import ParallelLoader


def xla_train(index, config):  
    set_manual_seed_all(config.seed)
    device = xm.xla_device()
    print(f"Device {device}--")

    model = load_model(config)
    tokenizer = AutoTokenizer.from_pretrained(config.plm_name)
    setattr(tokenizer, 'decoder_start_token_id', model.config.decoder_start_token_id)

    train_data, valid_data = prepare_train_data(
        f"{config.data_dir}/{config.data_file}", 
        config.valid_ratio, 
        random_split=True,
    )
    train_loader, valid_loader = prepare_data_loaders(
        config,
        train_data, 
        valid_data,
        tokenizer,
        device,
    )
    optimizer = init_optimizer(config, model)
    lr_scheduler = init_lr_scheduler(config, optimizer, len(train_loader))
    criterion = init_criterion(config, model, ignore_id=train_loader.dataset.ignore_id)

    evaluator = RougeEvaluator(config)
    if config.checkpoint is not None:
        load_checkpoint(config, model, optimizer, lr_scheduler, evaluator)
        sync_batch_idx(train_loader, evaluator, sync_wandb=config.wandb)
    elif config.tapt is not None:
        tapt_state_dict = torch.load(f"{config.asset_dir}/{config.tapt}.pth", map_location='cpu')
        model.load_state_dict(tapt_state_dict)

    for epoch in range(evaluator.start_epoch, config.num_epochs):
        if evaluator.end_of_patience():
            break
        
        train_loader = ParallelLoader(train_loader, [device]).per_device_loader(device)
        # train_loader = MpDeviceLoader(train_loader, device)
        epoch_loss = train_epoch(
            config=config,
            dataloader=train_loader, 
            criterion=criterion, 
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
            quiet=False,
            device=device,
        )

        valid_loader = ParallelLoader(valid_loader, [device]).per_device_loader(device)
        # valid_loader = MpDeviceLoader(valid_loader, device)
        epoch_rouge = validate_epoch(
            config=config,
            model=model,
            dataloader=valid_loader,
            evaluator=evaluator,
            epoch=epoch,
            quiet=config.quiet,
            device=device,
        )

        evaluator.epoch += 1
        print(epoch_loss, epoch_rouge)
    