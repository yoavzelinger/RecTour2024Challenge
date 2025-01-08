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
- source_sentence: Great staff and location Staff were so welcoming and helpful. We
    arrived 6am from a flight, they were able to offer us a room for half price so
    that we could sleep. Location is walking distance from the Old Town.  10.0 0
  sentences:
  - Mejok Couple 7 1 Romania ApartHotel 7.9 0.0 0 0 1
  - Mejok Solo traveller 11 1 Albania Guest house 8.8 0.0 0 0 0
  - Zuc Couple 8 2 Italy Bed and Breakfast 8.0 0.0 0 0 1
- source_sentence: Homey, relaxing and refreshing B&amp;B after hot days in Iguazu
    Falls Really homey and relaxing B&amp;B in the local neighborhood, where the restaurants
    and town are walkable. Once you step inside, the property and rooms are beautiful
    - nice refreshing pool for day&#47;night dips, beautiful gazebos to hang outside,
    and the A&#47;C is wonderful with hot&#47;humid days.    Pablo and his family
    are very welcoming and right when you first meet, he offers so many helpful tips
    and suggestions. Trust his restaurant tips! Cozy breakfast in the morning. Pablo
    offers himself as a great resource to help book tours, airport transfers, etc
    with if you don&#39;t speak Spanish well!    This is not your typical chain-hotel
    that&#39;s in the center of town. It&#39;s set in a quiet neighborhood that&#39;s
    pretty safe as a female solo traveler, but also at night, I just try to be cautious
    since it&#39;s not familiar territory - but with a friend&#47;partner, you&#39;re
    totally fine! All is great! 10.0 0
  sentences:
  - Panasi Solo traveller 9 1 Greece Hotel 8.1 3.0 0 0 1
  - Mejok Solo traveller 9 3 Vietnam Homestay 9.3 0.0 0 0 0
  - Nukeye Solo traveller 3 2 Argentina Inn 9.5 0.0 0 0 0
- source_sentence: Lovely base for visiting Devon Very friendly, lovely staff who
    really made the stay more enjoyable. Great location- walking distance to beach,
    parks,  town centre etc. Reserve onsite parking. Dated, and quite musty.  More
    a dated B and B than a hotel.  Pool on colder side. 7.0 0
  sentences:
  - Qehoj Solo traveller 3 1 Germany Hotel 8.6 3.0 0 0 0
  - Mejok Family with children 7 3 United Kingdom Hotel 7.7 3.0 0 1 1
  - Pikune Solo traveller 12 2 Turkey Hotel 8.9 3.0 0 0 0
- source_sentence: Nice stay in Bergen Nice hotel in Bergen, close to the city center.
    Room was fine.  9.0 0
  sentences:
  - Mejok Family with children 7 6 Italy Hotel 8.6 4.0 0 1 0
  - Zuc Solo traveller 8 1 Spain Hotel 9.1 5.0 0 0 0
  - Qehoj Solo traveller 6 3 Norway Hotel 8.2 3.0 0 1 1
- source_sentence: ' Large room with jaccuzzie tub. Great location. Second floor,
    no elevator. 8.0 0'
  sentences:
  - Jof Couple 7 1 Canada Motel 8.0 3.0 0 0 0
  - Keyu Family with children 4 3 Italy Bed and Breakfast 9.5 0.0 0 0 0
  - Nen Group 11 1 India Hotel 8.8 3.0 0 1 0
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
    ' Large room with jaccuzzie tub. Great location. Second floor, no elevator. 8.0 0',
    'Jof Couple 7 1 Canada Motel 8.0 3.0 0 0 0',
    'Keyu Family with children 4 3 Italy Bed and Breakfast 9.5 0.0 0 0 0',
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
  |         | sentence_0                                                                         | sentence_1                                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             |
  | details | <ul><li>min: 8 tokens</li><li>mean: 64.14 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 17 tokens</li><li>mean: 20.29 tokens</li><li>max: 26 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | sentence_1                                                                  |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------|
  | <code>Could be much better ! In top of tourist locations and walking distance to Sirkeci square and Sultan Ahmet I strongly recommend everyone to check the reviews in different hotel reservations in details ! 6.0 1</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | <code>Faxul Solo traveller 6 7 Turkey Hotel 7.0 0.0 0 0 1</code>            |
  | <code>Good&#47;comfortable but nothing special. Decent breakfast, comfy bed, 2 little bottles complimentary water, overall pretty clean.  Building is from late 1700s, so no lift, very steep stairs. But the guy at reception, I think his name was Flavius offered to carry some of our luggage.   Breakfast downstairs in the restaurant (outside to the right of reception, down the stairs, and into basement) from 8:30 - 10:00 or 10:30 can’t remember. Bathroom stunk… like chemicals. I think they put the toilet cleaner solution and forgot to actually scrub it.Because the toilet bowl was all green… even though the lid was down with the sanitized little paper strip on the lid. Bathroom is also fairly small, ok for us. But if you’re a bigger person will be very crammed.   We got the deluxe bedroom so we’d have a queen bed(not a full&#47;double), it was on the second story.  It was pretty small for a “deluxe” room, but given the age of the building not surprised.  If you get the attic rooms there is no st...</code> | <code>Pule Group 10 1 Romania Hotel 9.4 4.0 0 0 0</code>                    |
  | <code>Smoke Smell in Non-smoking room. Breakfast was delicious and had lots of variety. We enjoyed our food very well. It was a great start to our day! The coffee wasn&#39;t great, but the food hit the spot! Thank you for a good breakfast, Quality Inn. Our room smelled like smoke! We were placed in a supposedly non-smoking room, but it was definitely very smoky. The only reprieve I had was that I brought my own pillow. Thank god no one in our room had asthma! This was unnacceptible. 6.0 0</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | <code>Nukeye Group 10 4 United States of America Hotel 6.8 2.0 0 0 0</code> |
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
| 0.0196 | 500   | 3.8218        |
| 0.0393 | 1000  | 3.3811        |
| 0.0589 | 1500  | 3.1226        |
| 0.0786 | 2000  | 2.9511        |
| 0.0982 | 2500  | 2.8054        |
| 0.1179 | 3000  | 2.6517        |
| 0.1375 | 3500  | 2.5269        |
| 0.1572 | 4000  | 2.4483        |
| 0.1768 | 4500  | 2.3512        |
| 0.1964 | 5000  | 2.305         |
| 0.2161 | 5500  | 2.2549        |
| 0.2357 | 6000  | 2.2204        |
| 0.2554 | 6500  | 2.1612        |
| 0.2750 | 7000  | 2.1361        |
| 0.2947 | 7500  | 2.111         |
| 0.3143 | 8000  | 2.0876        |
| 0.3339 | 8500  | 2.0617        |
| 0.3536 | 9000  | 2.0359        |
| 0.3732 | 9500  | 2.0249        |
| 0.3929 | 10000 | 2.0074        |
| 0.4125 | 10500 | 1.998         |
| 0.4322 | 11000 | 1.9821        |
| 0.4518 | 11500 | 1.9464        |
| 0.4715 | 12000 | 1.9654        |
| 0.4911 | 12500 | 1.9111        |
| 0.5107 | 13000 | 1.9319        |
| 0.5304 | 13500 | 1.905         |
| 0.5500 | 14000 | 1.9008        |
| 0.5697 | 14500 | 1.8899        |
| 0.5893 | 15000 | 1.8821        |
| 0.6090 | 15500 | 1.868         |
| 0.6286 | 16000 | 1.8739        |
| 0.6483 | 16500 | 1.8468        |
| 0.6679 | 17000 | 1.8433        |
| 0.6875 | 17500 | 1.8543        |
| 0.7072 | 18000 | 1.8289        |
| 0.7268 | 18500 | 1.8288        |
| 0.7465 | 19000 | 1.8356        |
| 0.7661 | 19500 | 1.8175        |
| 0.7858 | 20000 | 1.8138        |
| 0.8054 | 20500 | 1.8067        |
| 0.8251 | 21000 | 1.7997        |
| 0.8447 | 21500 | 1.7938        |
| 0.8643 | 22000 | 1.7928        |
| 0.8840 | 22500 | 1.7726        |
| 0.9036 | 23000 | 1.7671        |
| 0.9233 | 23500 | 1.7732        |
| 0.9429 | 24000 | 1.7695        |
| 0.9626 | 24500 | 1.7539        |
| 0.9822 | 25000 | 1.7722        |
| 1.0018 | 25500 | 1.7636        |
| 1.0215 | 26000 | 1.7202        |
| 1.0411 | 26500 | 1.7287        |
| 1.0608 | 27000 | 1.7146        |
| 1.0804 | 27500 | 1.7093        |
| 1.1001 | 28000 | 1.7042        |
| 1.1197 | 28500 | 1.6893        |
| 1.1394 | 29000 | 1.7127        |
| 1.1590 | 29500 | 1.6924        |
| 1.1786 | 30000 | 1.6935        |
| 1.1983 | 30500 | 1.6857        |
| 1.2179 | 31000 | 1.6933        |
| 1.2376 | 31500 | 1.6835        |
| 1.2572 | 32000 | 1.664         |
| 1.2769 | 32500 | 1.6799        |
| 1.2965 | 33000 | 1.6748        |
| 1.3162 | 33500 | 1.6856        |
| 1.3358 | 34000 | 1.6752        |
| 1.3554 | 34500 | 1.6607        |
| 1.3751 | 35000 | 1.6627        |
| 1.3947 | 35500 | 1.6661        |
| 1.4144 | 36000 | 1.663         |
| 1.4340 | 36500 | 1.6549        |
| 1.4537 | 37000 | 1.6508        |
| 1.4733 | 37500 | 1.6424        |
| 1.4929 | 38000 | 1.6562        |
| 1.5126 | 38500 | 1.657         |
| 1.5322 | 39000 | 1.6495        |
| 1.5519 | 39500 | 1.6341        |
| 1.5715 | 40000 | 1.6529        |
| 1.5912 | 40500 | 1.6239        |
| 1.6108 | 41000 | 1.6343        |
| 1.6305 | 41500 | 1.6379        |
| 1.6501 | 42000 | 1.6343        |
| 1.6697 | 42500 | 1.6219        |
| 1.6894 | 43000 | 1.6216        |
| 1.7090 | 43500 | 1.6292        |
| 1.7287 | 44000 | 1.6278        |
| 1.7483 | 44500 | 1.6352        |
| 1.7680 | 45000 | 1.6282        |
| 1.7876 | 45500 | 1.6299        |
| 1.8073 | 46000 | 1.623         |
| 1.8269 | 46500 | 1.6353        |
| 1.8465 | 47000 | 1.6064        |
| 1.8662 | 47500 | 1.619         |
| 1.8858 | 48000 | 1.6176        |
| 1.9055 | 48500 | 1.6142        |
| 1.9251 | 49000 | 1.599         |
| 1.9448 | 49500 | 1.6207        |
| 1.9644 | 50000 | 1.6152        |
| 1.9840 | 50500 | 1.6137        |
| 2.0037 | 51000 | 1.5985        |
| 2.0233 | 51500 | 1.573         |
| 2.0430 | 52000 | 1.5798        |
| 2.0626 | 52500 | 1.5795        |
| 2.0823 | 53000 | 1.5715        |
| 2.1019 | 53500 | 1.5839        |
| 2.1216 | 54000 | 1.5772        |
| 2.1412 | 54500 | 1.5692        |
| 2.1608 | 55000 | 1.5668        |
| 2.1805 | 55500 | 1.5535        |
| 2.2001 | 56000 | 1.5846        |
| 2.2198 | 56500 | 1.567         |
| 2.2394 | 57000 | 1.5546        |
| 2.2591 | 57500 | 1.5538        |
| 2.2787 | 58000 | 1.5644        |
| 2.2984 | 58500 | 1.5473        |
| 2.3180 | 59000 | 1.5745        |
| 2.3376 | 59500 | 1.5675        |
| 2.3573 | 60000 | 1.5614        |
| 2.3769 | 60500 | 1.5664        |
| 2.3966 | 61000 | 1.5593        |
| 2.4162 | 61500 | 1.5413        |
| 2.4359 | 62000 | 1.5564        |
| 2.4555 | 62500 | 1.5622        |
| 2.4752 | 63000 | 1.5696        |
| 2.4948 | 63500 | 1.5513        |
| 2.5144 | 64000 | 1.5594        |
| 2.5341 | 64500 | 1.5465        |
| 2.5537 | 65000 | 1.5524        |
| 2.5734 | 65500 | 1.5629        |
| 2.5930 | 66000 | 1.5521        |
| 2.6127 | 66500 | 1.5415        |
| 2.6323 | 67000 | 1.5523        |
| 2.6519 | 67500 | 1.5545        |
| 2.6716 | 68000 | 1.5416        |
| 2.6912 | 68500 | 1.5502        |
| 2.7109 | 69000 | 1.5488        |
| 2.7305 | 69500 | 1.5427        |
| 2.7502 | 70000 | 1.5297        |
| 2.7698 | 70500 | 1.5539        |
| 2.7895 | 71000 | 1.5483        |
| 2.8091 | 71500 | 1.5552        |
| 2.8287 | 72000 | 1.5359        |
| 2.8484 | 72500 | 1.5379        |
| 2.8680 | 73000 | 1.557         |
| 2.8877 | 73500 | 1.5389        |
| 2.9073 | 74000 | 1.5313        |
| 2.9270 | 74500 | 1.5525        |
| 2.9466 | 75000 | 1.5403        |
| 2.9663 | 75500 | 1.541         |
| 2.9859 | 76000 | 1.5518        |

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