Image or vision classification is a fundamental task
in computer vision, critical for interpreting and understanding
visual data. In recent years, image classification models have seen
significant advancements. In this paper, a detailed taxonomy is
developed to identify the key categories of image classification
models: CNN, Transformer, and Hybrid models. Subsequently,
we performed an extensive evaluation of various machine learning models for pneumonia detection using a chest X-ray (CXR)
dataset. Our study compared the performance of three categories,
focusing on key metrics such as accuracy, precision, recall, and
F1-score as well as average inference time per image. Understanding the trade-offs between inference time and performance is
crucial for selecting the appropriate model for specific application
requirements. The results demonstrated that Transformer-based
models, particularly SwinV2, consistently outperformed other
models across multiple metrics. SwinV2 achieved the highest
accuracy (95.03%), recall (94.66%), and F1-score (94.70%),
indicating its robustness and reliability in pneumonia detection.
However, longer inference time (133.58 ms) for SwinV2 model
is a trade-off which is not ignorable. Although CNN models
like EfficientNetB0 and ResNet50 showed the highest precision,
they did not perform as well in other metrics, highlighting the
importance of a balanced evaluation across different performance
aspects. The consistent performance of convolutional transformer
models, like CvT, across all metrics supports integration of
convolutional layers with transformer architectures to leverage
the strengths of both approaches. However, the results reveal that
no specific model category demonstrates superior performance
over the others, although CNN models exhibit significantly faster
temporal efficiency compared to transformer and hybrid models.