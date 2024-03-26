# DGNIS
[MSSP 2022] A domain generalization network combing invariance and specificity towards real-time intelligent fault diagnosis


## Paper

Paper link: [A domain generalization network combing invariance and specificity towards real-time intelligent fault diagnosis](https://www.sciencedirect.com/science/article/pii/S0888327022001686)

## Abstract

Domain adaptation-based fault diagnosis (DAFD) methods have been explored to address cross- domain fault diagnosis problems, where distribution discrepancy exists between the training and testing data. However, the indispensable priori target distribution needed by DAFD methods hinders their application on real-time cross-domain fault diagnosis, where target data are not accessible in advance. To tackle this challenge, this paper proposes a novel domain generalization network for fault diagnosis under unknown working conditions. The main idea is to exploit domain invariance and retain domain specificity simultaneously, enabling deep models to benefit from the universal applicability of domain-invariant features while retaining the predictive power of specialized domain structures. Global distribution alignment and local class cluster are implemented to learn domain-invariant knowledge and obtain discriminant representations. Predictions of multiple task classifiers that preserve domain structures are optimally merged based on selected similarities for final diagnostic decisions. Extensive cross-domain fault diag- nostic experiments validated the effectiveness of the proposed method

##  Proposed Network 


![image](https://github.com/CHAOZHAO-1/DGNIS/blob/main/IMG/F1.png)

##  BibTex Citation

If you like our paper or code, please use the following BibTex:

@article{zhao2022domain,
  title={A domain generalization network combing invariance and specificity towards real-time intelligent fault diagnosis},
  author={Zhao, Chao and Shen, Weiming},
  journal={Mechanical Systems and Signal Processing},
  volume={173},
  pages={108990},
  year={2022},
  publisher={Elsevier}
}
