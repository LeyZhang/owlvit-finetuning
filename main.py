
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
    get_owl_datasets
)
from src.trainer import CustomTrainer

TRAIN_ANNOTATIONS_FILE = "/home/yang/data/smoke-fire-person-dataset/person/VisDrone/train/train.json"
TRAIN_IMAGES_PATH = "/home/yang/data/smoke-fire-person-dataset/person/VisDrone/train/images"
TEST_ANNOTATIONS_FILE = "/home/yang/data/smoke-fire-person-dataset/person/VisDrone/test/test.json"
TEST_IMAGES_PATH = "/home/yang/data/smoke-fire-person-dataset/person/VisDrone/test/images"


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
        device = torch.device('cuda')
        num_gpus = torch.cuda.device_count()
        print(f'Using {num_gpus} GPUs: {cfg["gpu"]}')
    else:
        device = torch.device('cpu')
        print('Use CPU')
    if cfg['is4090']:
        os.environ['NCCL_P2P_DISABLE']="1"
        os.environ['NCCL_IB_DISABLE']="1"
        torch.multiprocessing.set_start_method('spawn')
    return device, num_gpus

def build_model(cfg, device, num_gpus):

    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    return model

if __name__ == '__main__':
    cfg = {
        "train_images_path": TRAIN_IMAGES_PATH,
        "train_annotations_path": TRAIN_ANNOTATIONS_FILE,
        "test_images_path": TEST_IMAGES_PATH,
        "test_annotations_path": TEST_ANNOTATIONS_FILE,
        "batch_size": 2, 
        "gpu": "0",
        # "gpu": "0,1,2,3",
        "is4090": True,
        "project_path": "home/yang/zj/NLP"
    }
    device, num_gpus = acquire_device(cfg)

    # bbox => cx,cy,width,height
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    train_dataset, test_dataset = get_owl_datasets(cfg=cfg, processor=processor)


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
