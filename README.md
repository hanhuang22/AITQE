<div align="center">
<h3 align="center">
    <a href="https://github.com/hanhuang22/AITQE">
        <img src="figs/logo.png" alt="Logo" height="40">
    </a>
    Beyond Filtering:</br> Adaptive Image-Text Quality Enhancement for MLLM Pretraining
</h3>

  [![Arxiv][arxiv-shield]][arxiv-url]
  [![HuggingFace][model-shield]][model-url]
  [![Issues][issues-shield]][issues-url]
</div>

## Table of Contents
- [🚀 About This Project](#-about-this-project)
- [⚙️ Getting Started](#️-getting-started)
- [🤖 Inference](#-inference)
- [🧪 Methods](#-methods)
- [📖 Citation](#-citation)
- [📧 Contact](#-contact)
- [❤️ Acknowledgments](#️-acknowledgments)


## 🚀 About This Project
[2024.10.12] Release the inference code and pre-trained model of AITQE.

We propose the **A**daptive **I**mage-**T**ext **Q**uality **E**nhancer, **AITQE**, a model that dynamically assesses and enhances the quality of image-text pairs. The conventional method (a) discards low-quality samples in raw data, reducing the amount of pretraining data, while our AITQE (b) enhances low-quality samples, retaining the same volume of data for MLLMs pretraining.

![illus][illus]

Specifically, for pairs exhibiting low quality-such as low semantic similarity between modalities or subpar linguistic quality, AITQE performs text rewriting, generating high-quality text based on the input image and the raw low-quality text.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## ⚙️ Getting Started
### Environment
First setup environment with pip requirement file:
```
pip install -r requirements.txt
```
 

### Model Zoo

| Model | Vision Encoder | Large Language Model |
|:-----:|:--------------:|:----------------------:|
| [AITQE](https://huggingface.co/HymanH/AITQE) | [siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) | [Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B) |


## 🤖 Inference

To run the model, use the inference code below.

If you want AITQE to output all scores and explanations, add '--output_all' argument.

Example command:
```bash
python inference.py \
       --model_path /path/to/AITQE \
       --gpu_id 0 \
       --image_path ./figs/test.png \
       --caption "Some random text to the image like this is a test"
```

Example output of the above command, adding '--output_all':
```json
{"Recaption": "A man stands in front of a checklist of customer service questions, including 'Do you take each customer seriously?' and 'Do you qualify customers properly?'", "Overall Score": "2<Overall>", "Overall Explanation": "The caption is vague and does not accurately describe the image or its content. It lacks detail and relevance to the checklist shown in the image.", "Text Quality Score": 3, "Text Quality Explanation": "The caption is grammatically correct but lacks clarity and relevance to the image. It is vague and does not provide a meaningful description.", "Image-Text Matching Score": 2, "Image-Text Matching Explanation": "The caption does not accurately describe the image, which features a checklist of customer service questions. The caption is unrelated to the content of the image.", "Object Detail Score": 2, "Object Detail Explanation": "The caption does not provide any details about the objects in the image, such as the checklist or the person in the background.", "Semantic Understanding Score": 2, "Semantic Understanding Explanation": "The caption fails to convey any understanding of the image's context or purpose, which is about customer service evaluation.", "Text/Chart Description Score": 2, "Text/Chart Description Explanation": "The caption does not describe the text in the image, which is a checklist of customer service questions."}
```

If you encounter issues connecting to HuggingFace while running the code, you can manually download [Qwen/Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B) and [google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384). After downloading, specify the local paths to these models in the 'config.json' file located inside the aitqe model folder.


<p align="right">(<a href="#readme-top">back to top</a>)</p>


## 🧪 Methods

1. Data collection using GPT-4o in two phases: first scoring raw image-caption pairs with explanation, followed by generation of contrasting quality captions. 

2. AITQE model training using collected data and application of AITQE to enhance MLLM pretraining data.

![main][main]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## 📖 Citation
If you find our project or dataset helpful to your research, please consider citing:

```bibtext
@misc{
}
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>



## 📧 Contact
Github - [New Issue][new-issue]

Han Huang - <han.huang@cripac.ia.ac.cn>


<p align="right">(<a href="#readme-top">back to top</a>)</p>



## ❤️ Acknowledgments
This work was completed during an internship at [Baichuan Inc.](https://www.baichuan-ai.com/home) ([GitHub](https://github.com/baichuan-inc)).

We would like to thank the following projects and their great works: [Qwen](https://github.com/QwenLM/Qwen), [Siglip](https://github.com/google-research/big_vision), [MLM-Filter](https://github.com/Victorwz/MLM_Filter), [ShareGPT4V](https://github.com/ShareGPT4Omni/ShareGPT4V). We would also like to extend our gratitude to all the other related projects and contributors in the community.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



[illus]: figs/illus.png
[main]: figs/main.png

[arxiv-shield]: https://img.shields.io/badge/Arxiv-paper-red?style=for-the-badge&logo=arxiv&logoColor=red
[arxiv-url]: https://arxiv.org/abs/TODO

[model-shield]: https://img.shields.io/badge/HF-Models-yellow?style=for-the-badge&logo=huggingface&logoColor=yellow
[model-url]: https://huggingface.co/HymanH/AITQE

[issues-shield]: https://img.shields.io/github/issues/hanhuang22/AITQE.svg?style=for-the-badge
[issues-url]: https://github.com/hanhuang22/AITQE/issues
[new-issue]: https://github.com/hanhuang22/AITQE/issues/new/choose
