---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:1628989
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: Nice little set up within the pub with plenty of seating areas
    outside and ample parking.
  sentences:
  - Mejok Family with children 7 2 United Kingdom Inn
  - Zuc Couple 6 2 Malaysia ApartHotel
  - Zuc Couple 6 1 Greece Guest house
- source_sentence: nice setting, convenient to restaurants and bars, nicely appointed
    and bed was great. Off street parking was also a good bonus.
  sentences:
  - Zuc Couple 2 2 Australia Chalet
  - Nukeye Couple 12 1 United States of America Resort
  - Zuc Couple 2 2 New Zealand Guest house
- source_sentence: The room was very clean and nice. The location was very convenient
    as well.
  sentences:
  - Nukeye Solo traveller 3 2 United States of America Hotel
  - Nukeye Family with children 3 1 United States of America Motel
  - Zuc Family with children 9 1 Ireland Hotel
- source_sentence: Location was great, staff was amazing, rooms were basic. Plugs
    are all in one spot so a little awkward, but otherwise a good stay for the value.
  sentences:
  - Zuc Solo traveller 4 1 Australia Hotel
  - Mejok Group 7 1 United Kingdom Apartment
  - Jof Family with children 8 2 Croatia Bed and Breakfast
- source_sentence: Superb staff, friendly, informative, helpful. Location near city
    centre was spot-on.
  sentences:
  - Mejok Couple 10 2 United Kingdom Hotel
  - Mejok Family with children 8 4 Australia Resort
  - Mejok Family with children 7 4 United Kingdom Apartment
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision fa97f6e7cb1a59073dff9e6b13e2715cf7475ac9 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Superb staff, friendly, informative, helpful. Location near city centre was spot-on.',
    'Mejok Couple 10 2 United Kingdom Hotel',
    'Mejok Family with children 8 4 Australia Resort',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 1,628,989 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                        |
  |:--------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                            |
  | details | <ul><li>min: 4 tokens</li><li>mean: 38.27 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 8 tokens</li><li>mean: 11.34 tokens</li><li>max: 17 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                             | sentence_1                                                                  |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------|
  | <code>location was great and front desk was super nice.</code>                                                                                                                                                                                                                                                         | <code>Nukeye Family with children 6 4 United States of America Hotel</code> |
  | <code>The cleanliness of the hotel suite</code>                                                                                                                                                                                                                                                                        | <code>Made Couple 3 1 South Africa Hotel</code>                             |
  | <code>Everything about Three little birds was perfect, The location, room, food and most of all the service and hospitality from the amazing host and his family.   We would like to say a massive thank you and recommend anyone staying in the Tangalle area to stay.   You won’t regret it!  Thanks again :)</code> | <code>Mejok Couple 11 2 Sri Lanka Guest house</code>                        |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch  | Step  | Training Loss |
|:------:|:-----:|:-------------:|
| 0.0196 | 500   | 4.0029        |
| 0.0393 | 1000  | 3.6188        |
| 0.0589 | 1500  | 3.4281        |
| 0.0786 | 2000  | 3.2779        |
| 0.0982 | 2500  | 3.1707        |
| 0.1179 | 3000  | 3.0808        |
| 0.1375 | 3500  | 3.0183        |
| 0.1572 | 4000  | 2.9192        |
| 0.1768 | 4500  | 2.8872        |
| 0.1964 | 5000  | 2.8342        |
| 0.2161 | 5500  | 2.7852        |
| 0.2357 | 6000  | 2.7625        |
| 0.2554 | 6500  | 2.7143        |
| 0.2750 | 7000  | 2.7054        |
| 0.2947 | 7500  | 2.6831        |
| 0.3143 | 8000  | 2.6514        |
| 0.3339 | 8500  | 2.6262        |
| 0.3536 | 9000  | 2.6215        |
| 0.3732 | 9500  | 2.5992        |
| 0.3929 | 10000 | 2.589         |
| 0.4125 | 10500 | 2.5557        |
| 0.4322 | 11000 | 2.5639        |
| 0.4518 | 11500 | 2.574         |
| 0.4715 | 12000 | 2.5437        |
| 0.4911 | 12500 | 2.5016        |
| 0.5107 | 13000 | 2.5137        |
| 0.5304 | 13500 | 2.5132        |
| 0.5500 | 14000 | 2.4968        |
| 0.5697 | 14500 | 2.5027        |
| 0.5893 | 15000 | 2.4714        |
| 0.6090 | 15500 | 2.4839        |
| 0.6286 | 16000 | 2.4582        |
| 0.6483 | 16500 | 2.4613        |
| 0.6679 | 17000 | 2.45          |
| 0.6875 | 17500 | 2.446         |
| 0.7072 | 18000 | 2.4462        |
| 0.7268 | 18500 | 2.445         |
| 0.7465 | 19000 | 2.4254        |
| 0.7661 | 19500 | 2.4201        |
| 0.7858 | 20000 | 2.4096        |
| 0.8054 | 20500 | 2.4025        |
| 0.8251 | 21000 | 2.4089        |
| 0.8447 | 21500 | 2.4           |
| 0.8643 | 22000 | 2.3931        |
| 0.8840 | 22500 | 2.4075        |
| 0.9036 | 23000 | 2.3897        |
| 0.9233 | 23500 | 2.3913        |
| 0.9429 | 24000 | 2.3969        |
| 0.9626 | 24500 | 2.3691        |
| 0.9822 | 25000 | 2.3844        |
| 1.0018 | 25500 | 2.3594        |
| 1.0215 | 26000 | 2.3407        |
| 1.0411 | 26500 | 2.3346        |
| 1.0608 | 27000 | 2.3302        |
| 1.0804 | 27500 | 2.3342        |
| 1.1001 | 28000 | 2.3402        |
| 1.1197 | 28500 | 2.3441        |
| 1.1394 | 29000 | 2.3314        |
| 1.1590 | 29500 | 2.3401        |
| 1.1786 | 30000 | 2.3193        |
| 1.1983 | 30500 | 2.3263        |
| 1.2179 | 31000 | 2.325         |
| 1.2376 | 31500 | 2.3073        |
| 1.2572 | 32000 | 2.304         |
| 1.2769 | 32500 | 2.3105        |
| 1.2965 | 33000 | 2.3055        |
| 1.3162 | 33500 | 2.3004        |
| 1.3358 | 34000 | 2.3068        |
| 1.3554 | 34500 | 2.2968        |
| 1.3751 | 35000 | 2.2923        |
| 1.3947 | 35500 | 2.2805        |
| 1.4144 | 36000 | 2.2977        |
| 1.4340 | 36500 | 2.2961        |
| 1.4537 | 37000 | 2.2935        |
| 1.4733 | 37500 | 2.2802        |
| 1.4929 | 38000 | 2.2725        |
| 1.5126 | 38500 | 2.3002        |
| 1.5322 | 39000 | 2.2824        |
| 1.5519 | 39500 | 2.2648        |
| 1.5715 | 40000 | 2.2576        |
| 1.5912 | 40500 | 2.28          |
| 1.6108 | 41000 | 2.2694        |
| 1.6305 | 41500 | 2.2645        |
| 1.6501 | 42000 | 2.2714        |
| 1.6697 | 42500 | 2.2723        |
| 1.6894 | 43000 | 2.2591        |
| 1.7090 | 43500 | 2.2559        |
| 1.7287 | 44000 | 2.2463        |
| 1.7483 | 44500 | 2.2502        |
| 1.7680 | 45000 | 2.2605        |
| 1.7876 | 45500 | 2.2454        |
| 1.8073 | 46000 | 2.2472        |
| 1.8269 | 46500 | 2.2632        |
| 1.8465 | 47000 | 2.2368        |
| 1.8662 | 47500 | 2.246         |
| 1.8858 | 48000 | 2.2468        |
| 1.9055 | 48500 | 2.2643        |
| 1.9251 | 49000 | 2.244         |
| 1.9448 | 49500 | 2.2568        |
| 1.9644 | 50000 | 2.2379        |
| 1.9840 | 50500 | 2.2423        |
| 2.0037 | 51000 | 2.2399        |
| 2.0233 | 51500 | 2.2098        |
| 2.0430 | 52000 | 2.2155        |
| 2.0626 | 52500 | 2.2095        |
| 2.0823 | 53000 | 2.1993        |
| 2.1019 | 53500 | 2.2086        |
| 2.1216 | 54000 | 2.2166        |
| 2.1412 | 54500 | 2.2087        |
| 2.1608 | 55000 | 2.1972        |
| 2.1805 | 55500 | 2.2043        |
| 2.2001 | 56000 | 2.2235        |
| 2.2198 | 56500 | 2.188         |
| 2.2394 | 57000 | 2.1865        |
| 2.2591 | 57500 | 2.2031        |
| 2.2787 | 58000 | 2.1902        |
| 2.2984 | 58500 | 2.2131        |
| 2.3180 | 59000 | 2.2097        |
| 2.3376 | 59500 | 2.2027        |
| 2.3573 | 60000 | 2.2083        |
| 2.3769 | 60500 | 2.1955        |
| 2.3966 | 61000 | 2.1951        |
| 2.4162 | 61500 | 2.1997        |
| 2.4359 | 62000 | 2.2076        |
| 2.4555 | 62500 | 2.197         |
| 2.4752 | 63000 | 2.2133        |
| 2.4948 | 63500 | 2.1914        |
| 2.5144 | 64000 | 2.186         |
| 2.5341 | 64500 | 2.1981        |
| 2.5537 | 65000 | 2.2089        |
| 2.5734 | 65500 | 2.1836        |
| 2.5930 | 66000 | 2.2008        |
| 2.6127 | 66500 | 2.1818        |
| 2.6323 | 67000 | 2.1941        |
| 2.6519 | 67500 | 2.1888        |
| 2.6716 | 68000 | 2.1753        |
| 2.6912 | 68500 | 2.2064        |
| 2.7109 | 69000 | 2.2016        |
| 2.7305 | 69500 | 2.1916        |
| 2.7502 | 70000 | 2.2036        |
| 2.7698 | 70500 | 2.1791        |
| 2.7895 | 71000 | 2.2015        |
| 2.8091 | 71500 | 2.1987        |
| 2.8287 | 72000 | 2.1835        |
| 2.8484 | 72500 | 2.2005        |
| 2.8680 | 73000 | 2.179         |
| 2.8877 | 73500 | 2.1898        |
| 2.9073 | 74000 | 2.1825        |
| 2.9270 | 74500 | 2.1932        |
| 2.9466 | 75000 | 2.1762        |
| 2.9663 | 75500 | 2.1941        |
| 2.9859 | 76000 | 2.197         |

</details>

### Framework Versions
- Python: 3.10.12
- Sentence Transformers: 3.3.1
- Transformers: 4.47.1
- PyTorch: 2.5.1+cu121
- Accelerate: 1.2.1
- Datasets: 3.2.0
- Tokenizers: 0.21.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->