# On the Automatic Generation of Medical Imaging Reports
<B> CheXNet </B> is a powerful vision-language pre-training model capable of handling image-captioning tasks efficiently. This project utilizes the <B> MIMIC-CXR </B> Dataset for generating captions from Radiology X-ray images, specifically applying <B> CheXNet </B> with the <B> Transformer decoder </B>.
<hr>

# Dataset
The <a href="https://physionet.org/content/mimic-cxr/2.0.0/" target="_blank"> MIMIC-CXR </a> dataset, a comprehensive collection of radiology images and their corresponding textual descriptions, provides valuable data for research in medical imaging and natural language processing, facilitating advancements in AI-driven diagnostic tools.
<hr>

# Features
<ul>
 <li> <B> Feature Extractor </B> : <B> CheXNet </B>, a DenseNet121-based architecture trained on large chest X-ray images, is used as the feature extractor for accurately identifying and localizing objects within images. Composed of multiple convolutional layers, <B> CheXNet </B> is capable of extracting intricate features from radiology images, ensuring reliable and efficient performance for medical image analysis. This model provides robust classification results, making it suitable for real-time applications in medical imaging. </li> </p>
  <li> <B> Image Captioning </B> : The model incorporates a <B> Transformer decoder </B>, leveraging the power of Transformers and Multi-Head Attention for image feature extraction and generating descriptive captions. By effectively capturing intricate visual details through auto-regressive generation, the system creates coherent and contextually relevant descriptions that reflect the detected objects and their interactions within the image. </li>
</ul>
<hr>

# To Start
<B> Requirements : </B>
<ul>
 <li> Python 3.8+ </li>
 <li> PyTorch (with CUDA support if using a GPU) </li>
</ul> </p>
<B> Installation : </B>



 
