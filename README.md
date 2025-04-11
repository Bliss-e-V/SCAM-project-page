# SCAM: A Real-World Typographic Robustness Evaluation for Multimodal Foundation Models

[Project page](https://bliss-e-v.github.io/SCAM-project-page) - [Dataset](https://huggingface.co/datasets/BLISS-e-V/SCAM) - [Paper](https://arxiv.org/abs/2504.04893) - [Main Repo](https://github.com/Bliss-e-V/SCAM)

> Typographic attacks exploit the interplay between text and visual content in multimodal foundation models, causing misclassifications when misleading text is embedded within images. However, existing datasets are limited in size and diversity, making it difficult to study such vulnerabilities. In this paper, we introduce SCAM, the largest and most diverse dataset of real-world typographic attack images to date, containing images across hundreds of object categories and attack words.  
> Through extensive benchmarking of Vision-Language Models (VLMs) on SCAM, we demonstrate that typographic attacks significantly degrade performance, and identify that training data and model architecture influence the susceptibility to these attacks. Our findings reveal that typographic attacks persist in state-of-the-art Large Vision-Language Models (LVLMs) due to the choice of their vision encoder, though larger Large Language Models (LLMs) backbones help mitigate their vulnerability.

---

This is a static website to showcase the SCAM dataset and the results of the robustness evaluation.

As it's static, a simple local server is sufficient to run it:

```bash
python -m http.server
```


## Structure

* `index.html`: Main page
* `styles.css`: Styles
* `script.js`: JS
* `images/`: Images for the main page
* `data_images/`: Dataset images folder
* `data_images_generator.py`: Script to generate dataset images
* `data_converter_{lvlm,vlm}.py`: Scripts to generate dataset files from the combined results CSV
* `data`: Dataset files (generated - do not edit)
    * `{lvlm,vlm}_models_properties.json`: (L)VLM model properties
    * `{lvlm,vlm}_similarity_metadata.json`: Metadata about similarity scores
    * `{lvlm,vlm}_similarity_data.bin`: Binary file with similarity scores
    * `{lvlm,vlm}_similarity_index.json`: Index file with image metadata, grouped by base image
