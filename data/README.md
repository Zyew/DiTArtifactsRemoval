# Artifacts Simulation
Artifacts are generated from two reliable clinical CT datasets:
[CQ500](（https://www.kaggle.com/datasets/crawford/qureai-headct) for brain, 
[PANCREATIC-CT-CBCT-SEG](https://www.cancerimagingarchive.net/collection/pancreatic-ct-cbct-seg/) for abdomen. 
Each CT volume is treated as a true object and projected into the sinogram domain. Artifacts are then added directly in the sinogram, and the corrupted sinograms are reconstructed to obtain artifact volumes. 
The projection and reconstruction operations are performed using [DiffCT](https://github.com/sypsyp97/diffct).
