# llama2-finetune
https://huggingface.co/bidit/llama2-7b-fact_check_v1

# **Declaratively Fine-Tune Large Language Models**
Fine-tuning a large language model **refers to the process of further training the pre-trained model on a specific task or domain using a smaller dataset.** The initial pre-training phase involves training a language model on a massive corpus of text data to learn general language patterns and representations. Fine-tuning, on the other hand, customizes the model to a specific task or domain by exposing it to task-specific data. By fine-tuning a large language model on a specific task, you leverage the pre-trained knowledge of the model while tailoring it to the nuances and requirements of your target task. This typically allows the model to perform better and achieve higher accuracy on the specific task compared to using the pretrained model by itself for your specific task.
#### :arrow_forward: **What is instruction following/tuning? Why should I do it?**
Pre-trained language models are often great at giving general answers, but they struggle when asked to follow specific instructions, especially for tasks in certain domains. To make a pre-trained model do better at these specific tasks, we can train it on examples of those tasks. This is called instruction fine-tuning. We use a dataset with pairs of `{claim, trajectories}` in the domain, and this helps the model learn to give the right response when given those types of claim to validate. This training process typically changes the underlying model weights, but there are also other ways to train it without doing this. When done correctly, this process teaches the model to understand and follow instructions it couldn't handle well before.
#### :arrow_forward: **What will this teach the model?**
We will teach model to reasoning and acting on thought , Action , observation steps.
Here's an example of prompting the base model and an example prompting the fine-tuned model. The model was given all of the text until `Response:`, and it was supposed to continue generating an appropriate response.
**Using the base model (no fine-tuning):** :-1:
```
Follow the instruction "Find fact if Sriti Jha plays the lead role in a TV show."
Sriti Jha does not play the lead role in any popular TV shows. She has acted in several Indian television dramas and movies, including Kundali Bhagya, Tumhari Sulu, and Mere Angne Mein, but she has never played the lead role in a TV show.
```
**After instruction-fine-tuning:** :+1:
```
Follow the instruction "Find fact if Sriti Jha plays the lead role in a TV show."
### Sriti Jha plays the lead role in a TV show. Label: SUPPORTS Thought 1: I should search for Sriti Jha and find out if she plays the lead role in a TV show. Action 1: Search[Sriti Jha] Observation 1: Sriti Jha (born 26 February 1986) is an Indian actress working primarily in Hindi television. She earned wider recognition with her portrayal of Pragya Arora in Kumkum Bhagya, which won her various awards, including Indian Telly Award for Best Actress in a Lead Role and ITA Award for Best Actress Drama.. She made her acting debut in 2007 with Dhoom Machaao Dhoom.[1]. Jha's notable works include portraying Suhani in Shaurya Aur Suhani, Sudha/Devika in Jyoti, Jhanvi/Sia Dobriyal in Dil Se Di Dua... Saubhagyavati Bhava? and Pragya in Kumkum Bhagya and stunt based show Khatron Ke Khiladi.[2] She has also participated in Fear Factor: Khatron Ke Khiladi 12 and became 4th Runner-up in Jhalak Dikhhla Jaa 10.. Thought 2: Based on the evidence, Sriti Jha plays the lead role in the TV show Kumkum Bhagya. So, the claim is indeed supported by the evidence.]
```
There are three different fine-tuning approaches :
1. **Full Fine-Tuning**:
- Involves training the entire pre-trained model on new data from scratch.
- All model layers and parameters are updated during fine-tuning.
- Can lead to high accuracy but requires a significant amount of computational resources and time.
- Runs the risk of catastrophic forgetting: occasionally, since we are updating all of the weights in the model, this process can lead to the algorithm inadvertently losing knowledge of its past tasks, i.e., the knowledge it gained during pretraining. The outcome may vary, with the algorithm experiencing heightened error margins in some cases, while in others, it might completely erase the memory of a specific task leading to terrible performance.
- Best suited when the target task is significantly different from the original pre-training task.
2. **Parameter Efficient Fine-Tuning (PEFT), e.g. LoRA**:
- Focuses on updating only a subset of the model's parameters.
- Often involves freezing certain layers or parts of the model to avoid catastrophic forgetting, or inserting additional layers that are trainable while keeping the original model's weights frozen.
- Can result in faster fine-tuning with fewer computational resources, but might sacrifice some accuracy compared to full fine-tuning.
- Includes methods like LoRA, AdaLoRA and Adaption Prompt (LLaMA Adapter)
- Suitable when the new task shares similarities with the original pre-training task.
3. **Quantization-Based Fine-Tuning (QLoRA)**:
- Involves reducing the precision of model parameters (e.g., converting 32-bit floating-point values to 8-bit or 4-bit integers). This reduces the amount of CPU and GPU memory required by either 4x if using 8-bit integers, or 8x if using 4-bit integers.
- Typically, since we're changing the weights to 8 or 4 bit integers, we will lose some precision/performance.
- This can lead to reduced memory usage and faster inference on hardware with reduced precision support.
- Particularly useful when deploying models on resource-constrained devices, such as mobile phones or edge devices.
**Today, we're going to fine-tune using method 3 since we only have access to a single T4 GPU with 16GiB of GPU VRAM on Colab.** If you have more compute available, give LoRA based fine-tuning or full fine-tuning a try! Typically this requires 4 GPUs with 24GiB of GPU VRAM on a single node multi-GPU cluster and fine-tuning Deepspeeed.
To do this, the new parameters we're introducing are:
- `adapter`: The PEFT method we want to use
- `quantization`: Load the weights in int4 or int8 to reduce memory overhead.
- `trainer`: We enable the `finetune` trainer and can configure a variety of training parameters such as epochs and learning rate.
Some of the examples in the dataset have long sequences, so we set a `global_max_sequence_length` of 512 to ensure that we do not OOM.
We also use 100% of data for training as the evaluation phase takes extra time and we will predict on new examples right afterwards
