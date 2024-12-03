# MIFGCN
Multi-Source Information Fusion Graph Convolution Network (MIFGCN) is a novel framework designed for spatial-temporal traffic flow prediction. MIFGCN integrates multi-source data (traffic flow, speed, and weather conditions) with dynamic graph convolution techniques to improve prediction accuracy and robustness. This project is based on the research published in Expert Systems With Applications.

Overview
MIFGCN introduces a dynamic graph module that captures evolving spatial-temporal dependencies by integrating traffic speed, weather data, and spatial correlations. Additionally, it incorporates an attention-based temporal interaction module that learns multiscale temporal correlations to enhance long-term prediction accuracy.

Datasets
MIFGCN is evaluated on four real-world traffic datasets collected by the Caltrans Performance Measurement System (PeMS), including PeMS03, PeMS04, PeMS07, and PeMS08. These datasets cover various traffic scenarios and include auxiliary data such as traffic speed and weather conditions.
The weather data for this article is sourced from the following websites and has been crawled and preprocessed accordingly. The website is as follows: http://tianqi.2345.com/Pc/GetHistory?areaInfo [reaId]=349727&reaInfo [reaType]=1&date [year]=2021&date [month]=2. AreaId can be found in the following two files. http://tianqi.2345.com/tqpcimg/tianqiimg/theme4/js/citySelectData2.js
http://tianqi.2345.com/tqpcimg/tianqiimg/theme4/js/interCitySelectData2.js
For Chinese Mainland data, please look for the first file, and look for the second file in other regions. The areaType parameter is ignored. Please adjust the year and month by yourself.The collection period for the PeMS03 dataset is from September 2018 to November 2018, for the PeMS04 dataset from January 2018 to February 2018, for the PeMS07 dataset from May 2017 to August 2017, and for the PeMS08 dataset from July 2016 to August 2016. The data preprocessing process has been described in the article, including the use of one-hot encoding for weather types and representing wind direction as trigonometric functions. Readers are also encouraged to explore new weather features to investigate the contributions of different combinations to multi-source data. 
Traffic flow and speed data can be obtained by logging into https://pems.dot.ca.gov/ and clicking on "Data Clearinghouse" in the tools section. In the new page, select "station 5-minute" from the type dropdown and search for the corresponding region. The PeMS08 dataset was collected in San Bernardino, while the other datasets were collected in the San Francisco Bay Area. Then, choose the appropriate time period for download. The data files are saved in txt format, with the 10th and 11th features on each line representing flow and speed, respectively.

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
