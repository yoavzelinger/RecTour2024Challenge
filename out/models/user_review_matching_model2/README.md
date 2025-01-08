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
- source_sentence: Very good value - would stay again. Les Dunes exceeded our expectations
    which were influenced by some of the negative reviews!. Bedroom bright and airy
    and spotless. (Very hot water to bathroom.) Breakfast superb. Restaurant very
    good food at reasonable price. On site private parking. Location perfect for the
    town and ferries. Bedroom furniture slightly &quot;tired&quot;
  sentences:
  - Mejok Couple 9 1 France Hotel
  - Tuhi Solo traveller 8 2 Greece Guest house
  - Nukeye Couple 1 1 Portugal Guest house
- source_sentence: Great location in at Great hotel The person at the desk was so
    kind to let us in early as we had been up early and needed a rest. The place was
    very clean and very comfortable.  It is in the best location for all the great
    tourist venues!! Nothing
  sentences:
  - Fanir Couple 3 4 Italy Bed and Breakfast
  - Mejok Couple 10 1 United Kingdom Hotel
  - Jof Couple 9 1 Turkey Hotel
- source_sentence: 'wonderful :) this is a special place! my time here was amazing
    and forgettable. I found a second home…I wish I could have stayed longer, I really
    didn’t want to leave… the staff is just amazing, the location of the finca, the
    view, the nature, just the whole vibe is wonderful! definetly recommend a stay
    here! will definitely come back :) '
  sentences:
  - Pule Couple 4 1 Poland Homestay
  - Qehoj Solo traveller 2 3 Spain Hostel
  - Qehoj Group 4 1 Jordan ApartHotel
- source_sentence: Overall great brief stay! Room was clean and the breakfast was
    great The tub was slippery so strips could be put down and I didn’t like that
    we had to pay for a hot breakfast because you are already paying a high price
    for the room. Also, smoking should not be allowed in front of the door because
    you have to wait there for your shuttle and can’t go anywhere else.
  sentences:
  - Mejok Group 12 4 Spain Hotel
  - Nukeye Family with children 8 4 United Kingdom Guest house
  - Nukeye Couple 10 1 United Kingdom Hotel
- source_sentence: We loved our stay at Namaste suites! Thank you! We loved the pool!
    They also had free, good,  hot coffee available every morning. Also a continental
    breakfast with toast and fruit was available (although I never ate it). The balconies
    on the second floor had amazing sunset views with a small kitchen and a spot to
    hang out own hammock. There was a nice king size bed, great air conditioning and
    hot water available. Lucia was super friendly and was readily available to answer
    any questions. they are working on a bar upstairs, where the views are gorgeous
    as well. There are quite a few rainbow steps to get up to the property, which
    I actually didn’t mind. we had to park across the street at the park, which was
    also fine too.
  sentences:
  - Jof Couple 1 4 Mexico ApartHotel
  - Dawal Group 7 1 France Hotel
  - Nukeye Group 7 1 Poland ApartHotel
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
    'We loved our stay at Namaste suites! Thank you! We loved the pool! They also had free, good,  hot coffee available every morning. Also a continental breakfast with toast and fruit was available (although I never ate it). The balconies on the second floor had amazing sunset views with a small kitchen and a spot to hang out own hammock. There was a nice king size bed, great air conditioning and hot water available. Lucia was super friendly and was readily available to answer any questions. they are working on a bar upstairs, where the views are gorgeous as well. There are quite a few rainbow steps to get up to the property, which I actually didn’t mind. we had to park across the street at the park, which was also fine too.',
    'Jof Couple 1 4 Mexico ApartHotel',
    'Dawal Group 7 1 France Hotel',
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
  | details | <ul><li>min: 5 tokens</li><li>mean: 60.47 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 8 tokens</li><li>mean: 11.28 tokens</li><li>max: 16 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                    | sentence_1                                                 |
  |:----------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------|
  | <code>Centrally located and excellent service Location  Room Service Cleanliness Loved everything</code>                                      | <code>Mejok Group 10 1 United Kingdom Hotel</code>         |
  | <code> Comfy room, beautiful breakfast room and terrace. Very welcoming staff. The shower pressure and temperature was a bit variable.</code> | <code>Mejok Solo traveller 6 1 United Kingdom Hotel</code> |
  | <code> All meals were excellent but quantity was in excess. </code>                                                                           | <code>Mejok Couple 6 2 United Kingdom Lodge</code>         |
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
| 0.0196 | 500   | 3.8304        |
| 0.0393 | 1000  | 3.4105        |
| 0.0589 | 1500  | 3.1805        |
| 0.0786 | 2000  | 3.0195        |
| 0.0982 | 2500  | 2.889         |
| 0.1179 | 3000  | 2.7914        |
| 0.1375 | 3500  | 2.6757        |
| 0.1572 | 4000  | 2.592         |
| 0.1768 | 4500  | 2.5403        |
| 0.1964 | 5000  | 2.4811        |
| 0.2161 | 5500  | 2.4424        |
| 0.2357 | 6000  | 2.4145        |
| 0.2554 | 6500  | 2.3645        |
| 0.2750 | 7000  | 2.3481        |
| 0.2947 | 7500  | 2.3279        |
| 0.3143 | 8000  | 2.2983        |
| 0.3339 | 8500  | 2.2795        |
| 0.3536 | 9000  | 2.2559        |
| 0.3732 | 9500  | 2.2568        |
| 0.3929 | 10000 | 2.2156        |
| 0.4125 | 10500 | 2.2074        |
| 0.4322 | 11000 | 2.1947        |
| 0.4518 | 11500 | 2.1747        |
| 0.4715 | 12000 | 2.1684        |
| 0.4911 | 12500 | 2.1601        |
| 0.5107 | 13000 | 2.152         |
| 0.5304 | 13500 | 2.1374        |
| 0.5500 | 14000 | 2.1225        |
| 0.5697 | 14500 | 2.1195        |
| 0.5893 | 15000 | 2.1318        |
| 0.6090 | 15500 | 2.1144        |
| 0.6286 | 16000 | 2.0993        |
| 0.6483 | 16500 | 2.073         |
| 0.6679 | 17000 | 2.0833        |
| 0.6875 | 17500 | 2.0931        |
| 0.7072 | 18000 | 2.0747        |
| 0.7268 | 18500 | 2.0669        |
| 0.7465 | 19000 | 2.0572        |
| 0.7661 | 19500 | 2.057         |
| 0.7858 | 20000 | 2.0381        |
| 0.8054 | 20500 | 2.0378        |
| 0.8251 | 21000 | 2.0347        |
| 0.8447 | 21500 | 2.0336        |
| 0.8643 | 22000 | 2.0171        |
| 0.8840 | 22500 | 2.0382        |
| 0.9036 | 23000 | 2.018         |
| 0.9233 | 23500 | 2.0087        |
| 0.9429 | 24000 | 1.9989        |
| 0.9626 | 24500 | 2.0032        |
| 0.9822 | 25000 | 1.9966        |
| 1.0018 | 25500 | 1.9834        |
| 1.0215 | 26000 | 1.9599        |
| 1.0411 | 26500 | 1.9465        |
| 1.0608 | 27000 | 1.9413        |
| 1.0804 | 27500 | 1.9588        |
| 1.1001 | 28000 | 1.9439        |
| 1.1197 | 28500 | 1.9512        |
| 1.1394 | 29000 | 1.9432        |
| 1.1590 | 29500 | 1.9388        |
| 1.1786 | 30000 | 1.9276        |
| 1.1983 | 30500 | 1.9239        |
| 1.2179 | 31000 | 1.9107        |
| 1.2376 | 31500 | 1.904         |
| 1.2572 | 32000 | 1.9207        |
| 1.2769 | 32500 | 1.9186        |
| 1.2965 | 33000 | 1.9233        |
| 1.3162 | 33500 | 1.9186        |
| 1.3358 | 34000 | 1.9227        |
| 1.3554 | 34500 | 1.9029        |
| 1.3751 | 35000 | 1.9125        |
| 1.3947 | 35500 | 1.9066        |
| 1.4144 | 36000 | 1.9036        |
| 1.4340 | 36500 | 1.9139        |
| 1.4537 | 37000 | 1.8885        |
| 1.4733 | 37500 | 1.8864        |
| 1.4929 | 38000 | 1.8904        |
| 1.5126 | 38500 | 1.8833        |
| 1.5322 | 39000 | 1.8896        |
| 1.5519 | 39500 | 1.8812        |
| 1.5715 | 40000 | 1.8833        |
| 1.5912 | 40500 | 1.8933        |
| 1.6108 | 41000 | 1.8808        |
| 1.6305 | 41500 | 1.8724        |
| 1.6501 | 42000 | 1.8924        |
| 1.6697 | 42500 | 1.8577        |
| 1.6894 | 43000 | 1.8753        |
| 1.7090 | 43500 | 1.8695        |
| 1.7287 | 44000 | 1.8827        |
| 1.7483 | 44500 | 1.8767        |
| 1.7680 | 45000 | 1.8706        |
| 1.7876 | 45500 | 1.8593        |
| 1.8073 | 46000 | 1.856         |
| 1.8269 | 46500 | 1.864         |
| 1.8465 | 47000 | 1.8544        |
| 1.8662 | 47500 | 1.8404        |
| 1.8858 | 48000 | 1.8613        |
| 1.9055 | 48500 | 1.859         |
| 1.9251 | 49000 | 1.8644        |
| 1.9448 | 49500 | 1.8533        |
| 1.9644 | 50000 | 1.8357        |
| 1.9840 | 50500 | 1.8581        |
| 2.0037 | 51000 | 1.8302        |
| 2.0233 | 51500 | 1.8084        |
| 2.0430 | 52000 | 1.8277        |
| 2.0626 | 52500 | 1.8407        |
| 2.0823 | 53000 | 1.796         |
| 2.1019 | 53500 | 1.8135        |
| 2.1216 | 54000 | 1.8213        |
| 2.1412 | 54500 | 1.817         |
| 2.1608 | 55000 | 1.8039        |
| 2.1805 | 55500 | 1.8076        |
| 2.2001 | 56000 | 1.8237        |
| 2.2198 | 56500 | 1.8075        |
| 2.2394 | 57000 | 1.8078        |
| 2.2591 | 57500 | 1.8052        |
| 2.2787 | 58000 | 1.803         |
| 2.2984 | 58500 | 1.8132        |
| 2.3180 | 59000 | 1.8042        |
| 2.3376 | 59500 | 1.8013        |
| 2.3573 | 60000 | 1.7967        |
| 2.3769 | 60500 | 1.8105        |
| 2.3966 | 61000 | 1.8003        |
| 2.4162 | 61500 | 1.8148        |
| 2.4359 | 62000 | 1.815         |
| 2.4555 | 62500 | 1.8041        |
| 2.4752 | 63000 | 1.7982        |
| 2.4948 | 63500 | 1.7976        |
| 2.5144 | 64000 | 1.7979        |
| 2.5341 | 64500 | 1.7905        |
| 2.5537 | 65000 | 1.8086        |
| 2.5734 | 65500 | 1.7983        |
| 2.5930 | 66000 | 1.7851        |
| 2.6127 | 66500 | 1.8008        |
| 2.6323 | 67000 | 1.7786        |
| 2.6519 | 67500 | 1.7998        |
| 2.6716 | 68000 | 1.7911        |
| 2.6912 | 68500 | 1.7983        |
| 2.7109 | 69000 | 1.79          |
| 2.7305 | 69500 | 1.7915        |
| 2.7502 | 70000 | 1.7884        |
| 2.7698 | 70500 | 1.7891        |
| 2.7895 | 71000 | 1.8042        |
| 2.8091 | 71500 | 1.7862        |
| 2.8287 | 72000 | 1.7799        |
| 2.8484 | 72500 | 1.7975        |
| 2.8680 | 73000 | 1.785         |
| 2.8877 | 73500 | 1.7891        |
| 2.9073 | 74000 | 1.8029        |
| 2.9270 | 74500 | 1.7764        |
| 2.9466 | 75000 | 1.7938        |
| 2.9663 | 75500 | 1.7879        |
| 2.9859 | 76000 | 1.7804        |

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