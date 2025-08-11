Deforestation and forest fires are among the most pressing environmental crises facing our planet. Early and accurate detection of forest cover loss and active fires is essential for mitigating irreversible ecological damage and ensuring informed decision-making. In this work, we present a high-performance deep- learning framework that integrates the DeepLabV3+ architecture with an EfficientNet-B08 backbone to effectively tackle both deforestation detection and wildfire monitoring using satellite imagery. The proposed system leverages advanced multi-scale feature extraction and Group Normalization, enabling robust segmentation under diverse atmospheric conditions and complex forest structures. We thoroughly evaluate our framework on two representative datasets: (1) Bragagnolo et al.’s forest segmentation dataset, where it achieves a best validation Intersection over Union (IoU) of 0.9100 and a pixel accuracy of 0.9605, and (2) Farhat et al.’s FireDataset_20m forest fire dataset, which poses extreme class imbalance between fire and non-fire pixels. Despite this challenge, our model attains 99.95% accuracy, 93.16% precision, and 91.47% recall, underscoring its capacity to effectively identify active fires while minimizing false alarms. Qualitative analyses further demonstrate precise boundary delineation for forest cover and accurate localization of fire hotspots. This work thus offers a scalable, dual-purpose solution for environmental monitoring, paving the way for enhanced resource allocation, policy-making, and targeted interventions . 


In addition, we have did a comparsion to other models, HRNET, FASTFCN, PSPNET, as the script is upload to github. Also, we have the script to generate the charts to calculate the peformance metric.
The e12best is the script used to analysis of the deforestation datset, and e12bestv2 is for preloading the images in RAM to reduce training time.
The fire16.py is for the forest fire training.

![image](https://github.com/user-attachments/assets/312999d1-7dee-46e8-8b51-6043e400dab8)

 Figure1: illustrates a comprehensive pipeline that begins with a training dataset containing paired satellite images and their corresponding ground truth labels. These labels are represented as binary masks where deforested areas are clearly highlighted. The architecture demonstrates the systematic process through which the neural network processes input satellite imagery using an encoder-decoder structure, denoted as f(x;θ), to generate predictions. A crucial component of this system is the error feedback loop that continuously compares predictions against target labels, enabling the model to update its parameters (θ) through iterative training to improve accuracy over time.





 


![image](https://github.com/user-attachments/assets/a0bf72bf-c740-41d3-8efa-036de36a13ee)

Figure2: Comparsion of Image, Ground Truth, Prediction, and Difference Map

![image](https://github.com/user-attachments/assets/41098dd6-c755-485c-851d-272de0a3cc04)

Figure3: Comparsion of Image and Overlay.


Citations:
1. Bragagnolo L, da Silva RV, Grzybowski JMV. Amazon and Atlantic Forest image datasets for semantic segmentation [Data set]. Zenodo; 2021. Available from: https://doi.org/10.5281/zenodo.4498086

2.Model and dataset for multi-spectral detection of forest fires on board satellites. In: National Conference on Artificial Intelligence (CNIA 2022); 2022 Jun; Saint-Étienne, France. ffhal-03866412.
Link to download of dataset: https://www.actuia.com/en/news/ciar-project-deep-learning-to-detect-forest-fires-from-satellites/


