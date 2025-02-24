# project

Installation
Clone the repo 
```bash
git clone https://github.com/TUM-3DML-Project/project.git
```

Create a conda envrionment and install dependencies.
```bash
conda env create -f environment.yml --solver classic
conda activate env_fastpartslip
```
Install PyTorch3D
We utilize PyTorch3D for rendering point clouds. Please install it by the following commands or its official guide:
```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2"
```

Install Grounding-Dino


1.Clone the GroundingDINO repository from GitHub.

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
```

2. Change the current directory to the GroundingDINO folder.

```bash
cd GroundingDINO/
```

3. Install the required dependencies in the current directory.

```bash
pip install -e .
```

4. Download pre-trained model weights.

```bash
mkdir weights
cd weights
wget -q https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth
cd ../..
```

Get SAM weights
```bash
mkdir SAM
cd SAM
mkdir weights
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ../..
```
