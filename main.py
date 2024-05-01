
from transformers import (
    OwlViTProcessor,
    OwlViTForObjectDetection,
    TrainingArguments,
    Trainer
)
import torch
import os
from src.dataset import (
    get_owl_dataloaders,
    get_owl_datasets,
    get_lvis_datasets,
    get_lvis_dataloaders
)
from src.trainer import CustomTrainer
import yaml


def save_model_checkpoint(model, optimizer, epoch, loss, cfg, wandb_identifier):
    model_filename  = f"{wandb_identifier}_epoch-{epoch}.pt"#datetime.now().strftime("%Y%m%d_%H%M")+"_model.pt"
    path = os.path.join(cfg["model_checkpoint_path"], model_filename)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, path)

def acquire_device(cfg):
    num_gpus = 0
    os.environ['PYTHONPATH'] = cfg['project_path']
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg["gpu"]
        device = torch.device(f'cuda:{cfg["gpu"]}')
        num_gpus = torch.cuda.device_count()
        print(f'Using {num_gpus} GPUs: {cfg["gpu"]}')
    else:
        device = torch.device('cpu')
        print('Use CPU')
    if cfg['is4090']:
        os.environ['NCCL_P2P_DISABLE']="1"
        os.environ['NCCL_IB_DISABLE']="1"
        os.environ['CUDA_LAUNCH_BLOCKING']="1"
        torch.multiprocessing.set_start_method('spawn')
    return device, num_gpus

def build_model(cfg, device, num_gpus):

    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    return model

if __name__ == '__main__':
    with open('config/config.yml','r') as f:
        cfg = yaml.safe_load(f)
    print(cfg)
    device, num_gpus = acquire_device(cfg)

    # bbox => cx,cy,width,height
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    # train_dataset, test_dataset = get_owl_datasets(cfg=cfg, processor=processor)
    train_dataset, test_dataset = get_lvis_datasets(cfg=cfg, processor=processor)
    torch.save(train_dataset, 'output/logs/lvis_train_dataset.pt')
    torch.save(test_dataset, 'output/logs/lvis_test_dataset.pt')
    # train_dataset = torch.load('output/logs/train_dataset.pt')
    # test_dataset = torch.load('output/logs/test_dataset.pt')
    training_args = TrainingArguments(
        output_dir="owlvit-base-patch32_FT",
        per_device_train_batch_size=1,
        num_train_epochs=2,
        fp16=True,
        save_steps=200,
        logging_steps=50,
        learning_rate=1e-5,
        weight_decay=1e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        dataloader_pin_memory=False,
        gradient_accumulation_steps=1
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=processor,
        eval_dataset=test_dataset,
        device = device
    )
    trainer.train()
