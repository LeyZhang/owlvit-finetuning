from transformers import Trainer
from src.matcher import HungarianMatcher
from src.losses import OWLVitLoss
import torch

class CustomTrainer(Trainer):
    def __init__(self, model, args, train_dataset, tokenizer, eval_dataset, device):
        super(CustomTrainer, self).__init__(
            model=model,
            args=args,
            data_collator=self.collate_fn,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset, 
            tokenizer=tokenizer
        )
        self.device = device

    def collate_fn(self, batch):
        input_ids = torch.cat([item["input_ids"] for item in batch],0).int().to(self.device)
        attention_mask = torch.cat([item["attention_mask"] for item in batch], 0).int().to(self.device)
        pixel_values = torch.cat([item["pixel_values"] for item in batch], 0).to(self.device)
        labels = []
        for item in batch:
            for (key, value) in item["labels"].items():
                item["labels"][key] = torch.Tensor(value).to(self.device)
            labels.append(item["labels"])
        batch = {}
        batch["input_ids"] = input_ids
        batch["attention_mask"] = attention_mask
        batch["pixel_values"] = pixel_values
        batch["labels"] = labels
        return batch

    def custom_loss(self, outputs, labels):
        num_classes = 1
        matcher = HungarianMatcher(cost_class = 1, cost_bbox = 5, cost_giou = 2)
        weight_dict = {'loss_ce': 1, 'loss_bg': 1, 'loss_bbox': 5, 'loss_giou': 2}
        losses_dict = ['labels', 'boxes', 'cardinality']
        criterion = OWLVitLoss(num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1, losses_dict=losses_dict)
        criterion.to(self.device)
        torch.save(outputs, 'output/logs/outputs.pt')
        torch.save(labels, 'output/logs/labels.pt')
        torch.save(matcher, 'output/logs/matcher.pt')
        losses = criterion(outputs, labels)
        mean_loss = (losses['loss_ce'] + losses['loss_bbox'] + losses['loss_giou']+ losses['loss_bg']).squeeze()
        return mean_loss, losses

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        if(len(inputs['input_ids'].shape) > 2):
            batch_size, num_max_text_queris, sequence_length = inputs["input_ids"].shape
        else:
            batch_size = 1
            num_max_text_queris, sequence_length = inputs["input_ids"].shape

        inputs["input_ids"] = inputs["input_ids"].view(batch_size * num_max_text_queris, sequence_length)
        inputs["attention_mask"] = inputs["attention_mask"].view(batch_size * num_max_text_queris, sequence_length)
        outputs= model(**inputs, return_dict=True)

        mean_loss, losses = self.custom_loss(outputs, labels)
        print(losses)
        
        return (mean_loss, outputs) if return_outputs else mean_loss

