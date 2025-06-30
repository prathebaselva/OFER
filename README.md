
### <div align="center">👉 OFER: Occluded Face Expression Reconstruction<div> 
<div align="center">
<a href="https://drive.google.com/file/d/1_GKw8a_pyVnUl-AKc-hjczGkR-lacQcA/view?usp=drive_link"><img src="https://img.shields.io/static/v1?label=Paper&message=CVPR2025:OFER&color=red&logo=arxiv"></a> &ensp;
</div>
Reconstructing 3D face models from a single image is an inherently ill-posed problem, which becomes even more challenging in the presence of occlusions. In addition to fewer available observations, occlusions introduce an extra source of ambiguity where multiple reconstructions can be equally valid. Despite the ubiquity of the problem, very few methods address its multi-hypothesis nature. In this paper we introduce OFER, a novel approach for single-image 3D face reconstruction that can generate plausible, diverse, and expressive 3D faces, even under strong occlusions. Specifically, we train two diffusion models to generate the shape and expression coefficients of a face parametric model, conditioned on the input image. This approach captures the multi-modal nature of the problem, generating a distribution of solutions as output. However, to maintain consistency across diverse expressions, the challenge is to select the best matching shape. To achieve this, we propose a novel ranking mechanism that sorts the outputs of the shape diffusion network based on predicted shape accuracy scores. We evaluate our method using standard benchmarks and introduce CO-545, a new protocol and dataset designed to assess the accuracy of expressive faces under occlusion. Our results show improved performance over occlusion-based methods, while also enabling the generation of diverse expressions for a given image.

![Teaser](OFER_teaser.png)

## Pretrained weights
- Download the neutral and expression model weights from here : [Link](https://drive.google.com/drive/u/0/folders/1TQNyZVucVhwI0n9aMc-s3rO3hCG0V1mj)
- Download model weights trained on FLAME2020 : [Link](https://drive.google.com/drive/folders/1MZutLKw6jvWEfwqRMYiBOggAuKBEY3zn?usp=sharing)
and place it in 'checkpoint' folder
# 🔧 Setup

- Python >= 3.9 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.13.0+cu11.7](https://pytorch.org/)
```bash
conda create -n OFER python=3.9
conda activate OFER
conda env update --file ofer.yml --prune
```
## Dependencies
- Download FaRL pretrained model from [Link](https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep64.pth) and place it in 'pretrained' folder
- Download FLAME2020 model files from [Link](https://flame.is.tue.mpg.de/download.php) and place it in 'data' folder

## Training
- To train IdGen:
    - Training Data:
        - Obtained the data as mentioned in [MICA](https://github.com/Zielon/MICA?tab=readme-ov-file#dataset-and-training) 
    - Training
        ```bash
        python trainIdGen.py --cfg './src/configs/config_flameparamdiffusion_flame23.yml' --toseed 0 
        ```
- To train ExpGen:
    - Follow similar steps to IdGen
    - ```bash
        python trainExpGen.py --cfg './src/configs/config_flameparamdiffusion_exp.yml' --toseed 0 
        ```
- To train IdRank:
    - Training Data:
        - Same data of IdGen
    - Training
        ```bash
        python trainIdRank.py --cfg './src/configs/config_flameparamrank_flame23.yml' --toseed 0 
        ```
## Testing
-To run the demo:
```bash
python test.py --numcheckpoint 3 --cfg './src/configs/config_flameparamdiffusion_flame20.yml' --checkpoint1 'checkpoint/model_idrank.tar' --checkpoint2 'checkpoint/model_idgen_flame20.tar' --checkpoint3 'checkpoint/model_expgen_flame20.tar' --filename 'data/PICKPIK/validation/4.txt' --imagepath 'data/PICKPIK/' --outputpath 'output'
```
-Results will be saved in 'output' folder

## Acknowledgements
- Thanks to [MICA](https://github.com/Zielon/MICA) for their work and code base.

## Citation
Please cite for reference:
```bibtex
@InProceedings{pratheba2025OFER,
Author = {Selvaraju, Pratheba and Abrevaya, Victoria Fernandez and Bolkart, Timo and Akkerman, Rick and Ding, Tianyu and Amjadi, Faezeh and Zharkov, Ilya},
Title = {OFER: Occluded Face Expression Reconstruction},
Year = {2025},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025},
pages =  {26985-26995},
}
```
