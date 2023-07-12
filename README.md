# Relevance Feedback using SVM Active Learning in Python

A Python implementation of SVM-Active-based Relevance Feedback based on paper [Image Retrieval with Relevance Feedback using SVM Active Learning](https://www.researchgate.net/publication/316508249_Image_Retrieval_with_Relevance_Feedback_using_SVM_Active_Learning)

## Requirements
1. Image Dataset
- I use Corel dataset that is very popular in CBIR to demo. The data folder is compressed at [Corel.zip](https://github.com/HoangPham3003/SVM-Active-Learning-for-Relevance-Feedback/blob/main/db/Corel.zip) in my repository.
- Paths of images are saved as PKL file [paths.pkl]at (https://github.com/HoangPham3003/SVM-Active-Learning-for-Relevance-Feedback/blob/main/db/features/paths.pkl).
2. Features Database
- All Corel images are extracted to 4096d-vectors features using CNN. I use VGG19 as a feature extractor. The features database is compressed at [features.zip](https://github.com/HoangPham3003/SVM-Active-Learning-for-Relevance-Feedback/blob/main/db/features/features.zip) in my repository.