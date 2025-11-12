<div align="center">

# Guiding Point Cloud Denoising with Learned Structural Priors



<br/>

![status](https://img.shields.io/badge/status-active-2ea44f)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)
![lightning](https://img.shields.io/badge/Lightning-2.4.0-792ee5?logo=lightning&logoColor=white)
![pytorch](https://img.shields.io/badge/PyTorch-2.3.1-ee4c2c?logo=pytorch&logoColor=white)

</div>


### 🧭 Repository Structure
```
PGD/
├─ models/
│  ├─ pgd.py            
│  ├─ feature.py        
│  ├─ blocks.py         
│  ├─ InfoCD.py         
│  └─ utils.py         
├─ datasets/
│  ├─ pcl.py            
│  └─ patch.py         
├─ utils/
│  ├─ transforms.py     
│  └─ misc.py           
├─ chamfer3d/           
├─ pointops/            
├─ data/                
├─ pretrained/          
├─ logs/                
├─ train.py             
├─ test.py              
├─ evaluate.py          
└─ README.md            
```

---

## 🚀 Usage


### 1) Training

Train with default parameters (PUNet dataset, 3 resolutions, noise range 0.005-0.02):
```bash
python train.py
```

Train with custom parameters:
```bash
python train.py \
  --dataset PUNet \
  --dataset_root ./data \
  --resolutions 10000_poisson 30000_poisson 50000_poisson \
  --noise_min 0.005 \
  --noise_max 0.02 \
  --patch_size 1000 \
  --train_batch_size 20 \
  --lr 5e-4 \
  --log_root ./logs/PGD
```

---

### 2) Testing

**Test with 1% and 2% noise** (2 iterations):
```bash
python test.py \
  --ckpt pretrained/PGD.ckpt \
  --resolutions 10000_poisson 50000_poisson \
  --noise_lvls 0.01 0.02 \
  --niters 2
```

**Test with 2.5% noise** (3 iterations):
```bash
python test.py \
  --ckpt pretrained/PGD.ckpt \
  --resolutions 10000_poisson 50000_poisson \
  --noise_lvls 0.025 \
  --niters 3
```

---

### 3) Evaluation

The test script automatically runs evaluation. You can also evaluate existing results separately:

**Evaluate 1% noise results:**
```bash
python evaluate.py \
  --output_pcl_dir ./data/results/PGD/PUNet_Ours2x__50000_poisson_0.01 \
  --dataset_root ./data \
  --dataset PUNet \
  --summary_dir ./data/results/PGD \
  --experiment_name PUNet_Ours2x__50000_poisson_0.01 \
  --device cuda \
  --res_gts 50000_poisson
```

**Evaluate 2% noise results:**
```bash
python evaluate.py \
  --output_pcl_dir ./data/results/PGD/PUNet_Ours2x__50000_poisson_0.02 \
  --dataset_root ./data \
  --dataset PUNet \
  --summary_dir ./data/results/PGD \
  --experiment_name PUNet_Ours2x__50000_poisson_0.02 \
  --device cuda \
  --res_gts 50000_poisson
```

**Evaluate 2.5% noise results:**
```bash
python evaluate.py \
  --output_pcl_dir ./data/results/PGD/PUNet_Ours3x__50000_poisson_0.025 \
  --dataset_root ./data \
  --dataset PUNet \
  --summary_dir ./data/results/PGD \
  --experiment_name PUNet_Ours3x__50000_poisson_0.025 \
  --device cuda \
  --res_gts 50000_poisson
```

---

## 🛠 Environment & Dependencies

### Step 1: Create Conda Environment

```bash
conda create -n pgd python=3.10
conda activate pgd
```

---

### Step 2: Install PyTorch

Install PyTorch 2.3.1 with CUDA 11.8:

```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

> 🔗 Official guide: [PyTorch Get Started](https://pytorch.org/get-started/locally/)

---

### Step 3: Install PyTorch Lightning

```bash
pip install lightning==2.4.0
```

> 🔗 Official guide: [PyTorch Lightning Installation](https://lightning.ai/docs/pytorch/stable/starter/installation.html)

---

### Step 4: Install iopath

```bash
pip install iopath==0.1.10
```

---

### Step 5: Install PyTorch3D

**Option 1: Install from Anaconda Cloud (Recommended for Linux with CUDA)**

```bash
conda install pytorch3d -c pytorch3d
```

**Option 2: Install from source (Recommended for compatibility)**

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

> 🔗 Official guide: [PyTorch3D Installation](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)  
> ⚠️ **Note**: If Option 1 fails, use Option 2. Installation may take several minutes when building from source.

---

### Step 6: Install PyTorch Geometric

Install PyTorch Geometric and extension libraries:

```bash
pip install torch-geometric==2.6.1
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.1+cu118.html
```

> 🔗 Official guide: [PyTorch Geometric Installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)  
> 💡 For other PyTorch/CUDA versions, visit the [PyG installation page](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

---

### Step 7: Install Point Cloud Utilities

```bash
pip install point-cloud-utils==0.31.0
```

> 🔗 Repository: [point-cloud-utils GitHub](https://github.com/fwilliams/point-cloud-utils)

---

### Step 8: Install Additional Dependencies

```bash
pip install pandas tensorboard tqdm scipy
```

---

### Step 9: Compile Custom CUDA Operators

**Compile Chamfer3D:**

```bash
cd chamfer3d
python setup.py install
cd ..
```

**Compile PointOps:**

```bash
cd pointops
python setup.py install
cd ..
```

> ⚠️ **Note**: Ensure CUDA toolkit is properly installed and `nvcc` is in your PATH.


---

## 📖 Citation

If you find this work useful for your research, please consider citing:

```bibtex
@inproceedings{pgd2026,
  title={Guiding Point Cloud Denoising with Learned Structural Priors},
  author={Your Name and Co-authors},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

---


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star!**

Made with ❤️ for advancing point cloud denoising research

</div>


