# Student Assessment: Denoising Probabilistic Diffusion Models

# Introduction & Goals

This assessment will test your ability to extend the concepts learned in the course. You will be tasked with generating images using a pre-trained diffusion model, evaluating the results using CLIP Score and Frechet Inception Distance, and then using MLOps tools to track and visualize your work. A new component of this assessment is to analyze the internal representations of the U-Net model to gain deeper insights into the generation process.

**Goals:**

1. Generate high-quality images of flowers using a text-to-image diffusion model.  
2. Extract intermediate embeddings from the U-Net's downsampling path during generation.  
3. Evaluate the generated images using CLIP Score and Fréchet Inception Distance (FID).  
4. Create a FiftyOne dataset to visualize and analyze your generated images, their metadata, and their embeddings.  
5. Use FiftyOne Brain to compute uniqueness and representativeness scores for your generated images.  
6. Log your experiment, including hyperparameters, evaluation metrics, and analysis results, using Weights & Biases (wandb).

## Setup & Installation

First, let's install the necessary libraries.

| %pip install fiftyone wandb open-clip-torch |
| :---- |

## Part 1: Image Generation and Embedding Extraction

In this section, you will load the pre-trained U-Net model from notebook `05_CLIP.ipynb`, generate images of flowers, and extract embeddings from the model's bottleneck.

| import torchfrom utils import UNet\_utils, ddpm\_utils\# TODO: Initialize the U-Net model and load the pre-trained weights from notebook 05\.\# Make sure to use the same architecture as in the notebook.device \= torch.device("cuda" if torch.cuda.is\_available() else "cpu")model \= UNet\_utils.UNet(    T=400, img\_ch=3, img\_size=32, down\_chs=(256, 256, 512), t\_embed\_dim=8, c\_embed\_dim=512).to(device)\# model.load\_state\_dict(torch.load('path\_to\_your\_model.pth')) \# You need to provide the path to your trained modelmodel.eval()\# TODO: Define a list of text prompts to generate images for.text\_prompts \= \[    "A photo of a red rose",    "A photo of a white daisy",    "A photo of a yellow sunflower"\]\# \--- Embedding Extraction using Hooks \---\# We will use PyTorch hooks to extract the output of the 'down2' layer (the bottleneck).embeddings\_storage \= {}def get\_embedding\_hook(name):    def hook(model, input, output):        embeddings\_storage\[name\] \= output.detach()    return hook\# TODO: Register a forward hook on the \`down2\` layer of the U-Net model.\# The hook should store the output of the layer in the \`embeddings\_storage\` dictionary.\# model.down2.register\_forward\_hook(get\_embedding\_hook('down2'))\# TODO: Modify the \`sample\_flowers\` function from notebook 05 to generate images \# and store the extracted embeddings.\# You will need to run the generation process and then access the \`embeddings\_storage\`\# to get the embeddings for each generated image.\# generated\_images, \_ \= sample\_flowers(text\_prompts)\# extracted\_embeddings \= embeddings\_storage\['down2'\] |
| :---- |

## Part 2: Evaluation with CLIP Score and Frechet Inception Distance

Now, evaluate the quality of your generated images using the measures described in the **Metrics Calculation Guide** section.

| import open\_clip\# TODO: Calculate the CLIP score for each generated image against its prompt.\# You can use the \`calculate\_clip\_score\` function from the evaluation guide.\# TODO: Calculate the FID score for the set of generated images.\# You will need the \`calculate\_fid\` function and the Inception model from the evaluation guide.\# You will also need to load the real TF-Flowers dataset to compare against. |
| :---- |

## Part 3: Embedding Analysis with FiftyOne Brain

In this section, you will use FiftyOne to analyze the embeddings you extracted from the U-Net.

| import fiftyone as foimport fiftyone.brain as fob\# TODO: Create a new FiftyOne dataset.dataset \= fo.Dataset(name="generated\_flowers\_with\_embeddings")\# TODO: Iterate through your generated images and add them to the dataset.\# For each image, create a fiftyone.Sample and add the following metadata:\# \- The file path to the saved image.\# \- The text prompt (as a \`fo.Classification\` label).\# \- The CLIP score (as a custom field).\# \- The extracted U-Net embedding (as a custom field).\# TODO: Compute uniqueness for the dataset.\# fob.compute\_uniqueness(dataset)\# TODO: Compute representativeness using the extracted U-Net embeddings.\# fob.compute\_representativeness(dataset, embeddings="unet\_embedding")\# TODO: Launch the FiftyOne App to visualize your dataset and analyze the results.\# session \= fo.launch\_app(dataset) |
| :---- |

## Part 4: Logging with Weights & Biases

Log your experiment and results to Weights & Biases for tracking and comparison.

| import wandb\# TODO: Login to wandb.\# wandb.login()\# TODO: Initialize a new wandb run.\# run \= wandb.init(project="diffusion\_model\_assessment\_v2")\# TODO: Log your hyperparameters (e.g., guidance weight \`w\`, number of steps \`T\`).\# TODO: Log your evaluation metrics (CLIP Score and FID).\# TODO: Create a wandb.Table to log your results. The table should include:\# \- The generated image.\# \- The text prompt.\- The CLIP score.\# \- The uniqueness score.\# \- The representativeness score.\# TODO: Finish the wandb run.\# run.finish() |
| :---- |

## Scoring Rubric

- **20 points:** Correctly generate images from text prompts using the model.  
- **15 points:** Successfully extract intermediate embeddings from the U-Net's downsampling path.  
- **20 points:** Correctly calculate and report the CLIP Score and FID for the generated images.  
- **25 points:** Successfully create a FiftyOne dataset, add embeddings, and compute uniqueness and representativeness using FiftyOne Brain.  
- **20 points:** Successfully log the experiment, including all required metrics and analysis results, to Weights & Biases and a published FiftyOne dataset on HuggingFace.

### Metric Calculation Guide

**CLIP Score** This metric measures the semantic alignment between a text prompt and a generated image. Higher scores indicate that the image content matches the text description more closely.

| import torchimport open\_clipfrom PIL import Imagedef calculate\_clip\_score(image\_path, text\_prompt):    \# Load model    model, \_, preprocess \= open\_clip.create\_model\_and\_transforms('ViT-B-32', pretrained='laion2b\_s34b\_b79k')        \# Preprocess inputs    image \= preprocess(Image.open(image\_path)).unsqueeze(0)    tokenizer \= open\_clip.get\_tokenizer('ViT-B-32')    text \= tokenizer(\[text\_prompt\])    \# Compute features and similarity    with torch.no\_grad():        image\_features \= model.encode\_image(image)        text\_features \= model.encode\_text(text)                \# Normalize features        image\_features /= image\_features.norm(dim=\-1, keepdim=True)        text\_features /= text\_features.norm(dim=\-1, keepdim=True)                \# Calculate dot product        score \= (image\_features @ text\_features.T).item()            return score |
| :---- |

**Fréchet Inception Distance (FID)** FID measures the distance between the feature distributions of real images and generated images. Lower scores indicate that the generated images possess visual quality and diversity similar to the real dataset. Note that this [metric](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance) is defined through the InceptionV3 model and you have to use an ImageNet pre-trained InceptionV3 model to compute it. Here's a [demo notebook](https://colab.research.google.com/drive/1-OCf05KRfnhmw2JJqtt808dfSIsFHpoU?usp=sharing) to do this.

| import numpy as npfrom scipy.linalg import sqrtmdef calculate\_fid(real\_embeddings, gen\_embeddings):    \# real\_embeddings and gen\_embeddings should be Numpy arrays of shape (N, 2048\)     \# extracted from an InceptionV3 model        \# Calculate mean and covariance    mu1, sigma1 \= real\_embeddings.mean(axis=0), np.cov(real\_embeddings, rowvar=False)    mu2, sigma2 \= gen\_embeddings.mean(axis=0), np.cov(gen\_embeddings, rowvar=False)    \# Calculate sum squared difference between means    ssdiff \= np.sum((mu1 \- mu2)\*\*2)    \# Calculate sqrt of product of covariances    covmean \= sqrtm(sigma1.dot(sigma2))        \# Handle numerical errors    if np.iscomplexobj(covmean):        covmean \= covmean.real    \# Final FID calculation    fid \= ssdiff \+ np.trace(sigma1 \+ sigma2 \- 2.0 \* covmean)    return fid |
| :---- |

The metrics must be computed on every sample and saved on the FiftyOne dataset. 

## Bonus Task: Building a Classifier with an "IDK" Option (20 points)

For the bonus, you will modify an MNIST-digit classifier to include an "I don't know" (IDK) option. This is useful in real-world applications where a model should defer to a human when it is not confident in its prediction.

You will modify the [MNIST generation and classification notebook](https://colab.research.google.com/github/andandandand/practical-computer-vision/blob/main/notebooks/Denoising_Diffusion_Probabilistic_Model_U_net_MNIST_Generation.ipynb) for this task. Create and publish a FiftyOne dataset with your experiments’ results. 

| from run\_assessment import get\_classifier\# TODO: Load the pre-trained MNIST classifier from notebook 06\.mnist\_classifier \= get\_classifier()mnist\_classifier.eval()\# TODO: Create a function that takes an image and a confidence threshold.\# The function should return the predicted class if the model's confidence is above the threshold,\# and "IDK" otherwise.\# Hint: Use the softmax output of the classifier to get the confidence scores.def predict\_with\_idk(image, model, threshold):    \# Your code here    Pass  |
| :---- |

##    