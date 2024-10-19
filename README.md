# MIFGCN
Multi-Source Information Fusion Graph Convolution Network (MIFGCN) is a novel framework designed for spatial-temporal traffic flow prediction. MIFGCN integrates multi-source data (traffic flow, speed, and weather conditions) with dynamic graph convolution techniques to improve prediction accuracy and robustness. This project is based on the research published in Expert Systems With Applications.

Overview
MIFGCN introduces a dynamic graph module that captures evolving spatial-temporal dependencies by integrating traffic speed, weather data, and spatial correlations. Additionally, it incorporates an attention-based temporal interaction module that learns multiscale temporal correlations to enhance long-term prediction accuracy.

Datasets
MIFGCN is evaluated on four real-world traffic datasets collected by the Caltrans Performance Measurement System (PeMS), including PeMS03, PeMS04, PeMS07, and PeMS08. These datasets cover various traffic scenarios and include auxiliary data such as traffic speed and weather conditions.

Citation
If you find this work useful, please consider citing the original paper:

@article{li2024multi,
  title={Multi-Source Information Fusion Graph Convolution Network for traffic flow prediction},
  author={Li, Qin and Xu, Pai and He, Deqiang and Wu, Yuankai and Tan, Huachun and Yang, Xuan},
  journal={Expert Systems with Applications},
  volume={252},
  pages={124288},
  year={2024},
  publisher={Elsevier}
}
