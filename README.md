# Cat-Kanye-Pikachu Classifier Backend

A Flask API that leverages the convolutional base of Xception model pre-trained on the ImageNet dataset to classify images of cats, Kanye and Pikachu. 

> The classifier on top of the base is trained on ~ 500 images of each of the targets. As a result of the three-class training, the model will give confident scores on images that resemble but don't belong to the target classes (for example: any human in place of Kanye). Further, the classifier uses global average pooling which intensifies this effect. Global pooling is perfectly suited to the use case of this three-way classification since there is a large variance **between** the classes.

## Usage

Send a POST request (encoded as multipart/form-data) with the image data ('image' field). The front-end code can be found [here](https://github.com/sahilshaheen/cat-kanye-pikachu-classifier-frontend) for reference.