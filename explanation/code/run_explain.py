# -*- coding: UTF-8 -*-
import torch
import logging
import os
import json
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer, AdamW,
    TrainingArguments,
    set_seed,
)
# from transformers.trainer_utils import get_last_checkpoint
from utils_multiple_choice import processors
from collections import Counter


from Logiformer_explain import Logiformer
from tokenization import arg_tokenizer, our_tokenizer
from utils_multiple_choice import Split, MyMultipleChoiceDataset
from graph_building_blocks.argument_set_punctuation_v4 import punctuations
with open('./graph_building_blocks/explicit_arg_set_v4.json', 'r') as f:
    relations = json.load(f)  # key: relations, value: ignore



logger = logging.getLogger(__name__)

import re
PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")


def get_last_checkpoint(folder):
    content = os.listdir(folder)
    print(content)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    print(checkpoints)
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    dropout: float = field(
        default=0.1,
        metadata={"help": "huggingface RoBERTa config.attention_probs_dropout_prob"}
    )
    init_weights: bool = field(
        default=False,
        metadata={"help": "init weights in Argument NumNet."}
    )

    # training
    roberta_lr: float = field(
        default=5e-6,
        metadata={"help": "learning rate for updating roberta parameters"}
    )
    gt_lr: float = field(
        default=5e-6,
        metadata={"help": "learning rate for updating gcn parameters"}
    )
    proj_lr: float = field(
        default=5e-6,
        metadata={"help": "learning rate for updating fc parameters"}
    )



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(processors.keys())})
    data_dir: str = field(metadata={"help": "Should contain the data files for the task."})
    data_type: str = field(
        default="argument_numnet",
        metadata={
            "help": "data types in utils script. roberta_large | argument_numnet "
        }
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        }
    )
    max_ngram: int = field(
        default=5,
        metadata={"help": "max ngram when pre-processing text and find argument/domain words."}
        )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    demo_data: bool = field(
        default=False,
        metadata={"help": "demo data sets with 100 samples."}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        processor = processors[data_args.task_name]()
        label_list = processor.get_labels()
        num_labels = len(label_list)
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))


    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    if isinstance(relations, tuple): max_rel_id = int(max(relations[0].values()))
    elif isinstance(relations, dict):
        if not len(relations) == 0:
            max_rel_id = int(max(relations.values()))
        else:
            max_rel_id = 0
    else: raise Exception
    model = Logiformer.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        # token_encoder_type="roberta" if "roberta" in model_args.model_name_or_path else "bert",
        token_encoder_type="roberta",
        init_weights=model_args.init_weights,
        max_rel_id=max_rel_id,
        cache_dir=model_args.cache_dir,
        hidden_size=config.hidden_size,
        dropout_prob=model_args.dropout,
    )


    train_dataset = (
        MyMultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            arg_tokenizer=arg_tokenizer,
            relations=relations,
            punctuations=punctuations,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
            demo=data_args.demo_data
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        MyMultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            arg_tokenizer=arg_tokenizer,
            relations=relations,
            punctuations=punctuations,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            max_ngram=data_args.max_ngram,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
            demo=data_args.demo_data
        )
        if training_args.do_eval
        else None
    )
    test_dataset = (
        MyMultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            arg_tokenizer=arg_tokenizer,
            relations=relations,
            punctuations=punctuations,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            max_ngram=data_args.max_ngram,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
            demo=data_args.demo_data
        )
        if training_args.do_predict
        else None
        )


    def compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        return {"acc": simple_accuracy(preds, p.label_ids)}


    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n.startswith("_gt")
                        and not any(nd in n for nd in no_decay)],
            "lr": model_args.gt_lr,
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n.startswith("_gt")
                        and any(nd in n for nd in no_decay)],
            "lr": model_args.gt_lr,
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if n.startswith("roberta")
                        and not any(nd in n for nd in no_decay)],
            "lr": model_args.roberta_lr,
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n.startswith("roberta")
                        and any(nd in n for nd in no_decay)],
            "lr": model_args.roberta_lr,
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if n.startswith("_proj")
                        and not any(nd in n for nd in no_decay)],
            "lr": model_args.proj_lr,
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n.startswith("_proj")
                        and any(nd in n for nd in no_decay)],
            "lr": model_args.proj_lr,
            "weight_decay": 0.0,
        }
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
    )
    # Initialize our Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None)
    )
            
    # Do we need to reload tokenizer??
    
    # state_dict = torch.load(os.path.join(training_args.output_dir, "pytorch_model.bin"), map_location="cpu")
    # If the model is on the GPU, it still works!
    # trainer._load_state_dict_in_model(state_dict)
    
    # load model from last checkpoint
    trainer._load_from_checkpoint(training_args.output_dir)
    
    model.eval()
    # Go through all data and save attention map in file
    
    def decode_spans(spans_ids_list):
        assert len(spans_ids_list) % 4 == 0
        # 8 or 4
        all_span_list = []
        for spans in spans_ids_list:
            span_list = []
            for span in spans:
                span_str = tokenizer.decode(span)
                span_list.append(span_str)
            all_span_list.append(span_list)
        return all_span_list
    
    import sys
    
    def dump_span_importance(dataset):
        dataloader = trainer.get_test_dataloader(dataset)
        data_id_list = []
        length = len(dataset)
        for i in range(length):
            data_id_list.append(dataset[i].example_id)
        # get all id of original dataset
        index = 0
        object_list = []
        for step, inputs in enumerate(dataloader):
            inputs = trainer._prepare_inputs(inputs)
            batch_size = inputs["input_ids"].size(0)
            # print(inputs["input_ids"][0, :, :].size())
            # decoded_str = tokenizer.batch_decode(inputs["input_ids"][0, :, :])
            # print(decoded_str)
            outputs = model(**inputs)
            attn_weight = outputs[2]
            spans_ids_list = outputs[3]
            all_span_list = decode_spans(spans_ids_list)
            attn_weight_all_head = np.sum(attn_weight, axis=1)
            # add all attention heads
            attn_weight_try = np.sum(attn_weight_all_head, axis=1)
            # calculate along the column, which span will have larger impact on the final representation of other spans
            attn_weight_try = attn_weight_try.tolist()
            for tp_id in range(batch_size):
                # for a question
                for pos, option in enumerate(["A", "B", "C", "D"]):
                    # four options
                    cur_id = data_id_list[index] + "-{}".format(option)
                    spans = all_span_list[tp_id * 4 + pos]
                    attention_weight = attn_weight_try[tp_id * 4 + pos]
                    tp_obj = {}
                    tp_obj["id"] = cur_id
                    tp_obj["spans"] = spans
                    tp_obj["importance"] = attention_weight
                    object_list.append(tp_obj)
                index += 1
        return object_list
        # print(all_span_list[1])
        # print(all_span_list[2])
        # print(all_span_list[3])
        # print(attn_weight.size())
        # check the data order, is it just following question-id order? example_id
    
    train_list = dump_span_importance(train_dataset)
    eval_list = dump_span_importance(eval_dataset)
    test_list = dump_span_importance(test_dataset)
    out_file = open("/workspace/Logiformer/Reclor_LogicFormer_span_importance.json", "w")
    all_list = train_list + eval_list + test_list
    json.dump(all_list, out_file)
    out_file.close()
    # Training
    # if training_args.do_train:
    #     trainer.train(
    #         model_path=model_args.mode_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
    #     )
    #     trainer.save_model()
    #     # For convenience, we also re-save the tokenizer to the same directory,
    #     # so that you can share your model easily on huggingface.co/models =)
    #     tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
#     results = {}
#     if training_args.do_eval:
#         logger.info("*** Evaluate ***")

#         result = trainer.evaluate()

#         output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
#         # with open(output_eval_file, "w") as writer:
#         logger.info("***** Eval results *****")
#         for key, value in result.items():
#             logger.info("  %s = %s", key, value)
#             # writer.write("%s = %s\n" % (key, value))

#         results.update(result)

#         eval_result = trainer.predict(eval_dataset)
#         preds = eval_result.predictions  # np array. (1000, 4)
#         pred_ids = np.argmax(preds, axis=1)

#         output_test_file = os.path.join(training_args.output_dir, "eval_predictions.npy")
#         np.save(output_test_file, pred_ids)
#         logger.info("predictions saved to {}".format(output_test_file))

    # Test
#     if training_args.do_predict:
#         if data_args.task_name == "reclor":
#             logger.info("*** Test ***")

#             test_result = trainer.predict(test_dataset)
#             preds = test_result.predictions  # np array. (1000, 4)
#             pred_ids = np.argmax(preds, axis=1)

#             output_test_file = os.path.join(training_args.output_dir, "predictions.npy")
#             np.save(output_test_file, pred_ids)
#             logger.info("predictions saved to {}".format(output_test_file))
#         elif data_args.task_name == "logiqa":
#             logger.info("*** Test ***")

#             test_result = trainer.predict(test_dataset)

#             output_test_file = os.path.join(training_args.output_dir, "test_results.txt")

#             with open(output_test_file, "w") as writer:
#                 logger.info("***** Test results *****")
#                 for key, value in test_result.metrics.items():
#                     logger.info("  %s = %s", key, value)
#                     writer.write("%s = %s\n" % (key, value))

#                 results.update(test_result.metrics)




def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
