# SP25-690-Vaddempudi
Detecting Data Leakage in Deep Learning Using CNN and Vision Transformer Representations
 

1. Why This Problem Matters

In deep learning, we usually assume that the training data and test data are completely separate. But in real cases, this is not always true. Sometimes, the same or very similar samples appear in both sets. This is called data leakage. Because of this, models can show very high accuracy even though they are not really learning properly. This creates a false idea of good performance. Also, it is not easy to find this problem manually. So in this project, the idea is to build a deep learning–based system that can automatically detect if such leakage is present. This will help make model evaluation more reliable and trustworthy.

 

2. What the System Will Do

The main goal of this project is to check whether a dataset split has leakage or not. The model will take samples from both training and test sets as input and give an output saying whether leakage exists. It can also give a confidence score showing how likely leakage is. The system should be able to detect both exact duplicates and slightly changed (near-duplicate) samples. The performance will be measured using accuracy, precision, recall, and F1-score. A good result means the model can correctly detect even small or hidden leakage.

 

3. Data and How It Will Be Prepared

This project will use a standard dataset like CIFAR-10. To test the system properly, we will create different versions of the dataset. Some versions will have no leakage, while others will have controlled leakage. This includes copying exact images into both training and test sets, and also creating similar images using small changes like noise, rotation, or cropping. Different levels of leakage will be created so we can test how well the model performs in each case. Proper separation will be maintained so experiments are fair and controlled.

 

4. How the Model Will Be Built

This project will use two deep learning models from the course: a Convolutional Neural Network (CNN) and a Vision Transformer (ViT). Both models will be trained to learn features (representations) from images. After that, these features will be compared between training and test samples using similarity measures like cosine similarity. Then, a small neural network (like an MLP) will take these similarity values and predict whether leakage is present. By comparing CNN and transformer features, we can understand which type of model is better at identifying overlap between datasets.

 

5. How It Will Be Compared with Other Methods

To check if the proposed method is useful, it will be compared with simple approaches. These include random guessing and basic duplicate checking using raw image comparison. The CNN-based model will act as a main baseline, and the Vision Transformer will be used as a more advanced comparison model. This will help us see whether better feature learning leads to better leakage detection.

 

6. How Experiments Will Be Done

Different dataset versions with different leakage levels will be used for experiments. The models will be trained using standard methods like Adam optimizer. A validation set will be used to tune parameters. Testing will be done on separate data to measure final performance. Experiments will be repeated multiple times to make sure results are stable. We will also carefully change one factor at a time (like model type or leakage level) to understand its effect clearly.

 

7. What Results We Expect

Simple methods will likely work only when there are exact duplicate images. But they may fail when the images are slightly different. The deep learning models should perform better in these cases because they can learn deeper patterns. The Vision Transformer might perform better in some cases because it captures global information. Results will be shown using tables and graphs to clearly compare performance under different conditions.

 

8. Where the Model Might Fail

The model may sometimes confuse very similar but different images as leakage, leading to false positives. It may also miss leakage if the duplicate images are heavily modified. Another issue is that the model’s performance depends on how well it learns features. These problems will be studied by looking at wrong predictions and understanding why they happened.

 

9. Ethical and Practical Considerations

Detecting data leakage is very important because it affects how we trust machine learning results. If leakage is not found, models may look better than they actually are. However, automatic systems are not perfect, and wrong decisions may also happen. So results should always be checked carefully. Clear reporting and honest evaluation are important to use such systems responsibly.

 

10. Project Feasibility

This project is practical and can be completed within the course time. It uses standard datasets and well-known deep learning models. The implementation can be done using tools like PyTorch and can run on platforms like Google Colab. The work can be done step by step, starting from simple methods and then moving to more advanced ones. This makes the project manageable while still allowing meaningful results.
