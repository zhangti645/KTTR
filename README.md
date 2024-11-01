# KTTR
Clustered Key-Reference Frame Selection with Compact Temporal Trajectory for Talking Face Video Compression
##Abstract
 we propose a generative compression framework for talking face videos. The framework incorporates a compact feature extraction network, a clustering-based key-reference frame selection algorithm, and advanced encoding schemes, enabling ultra-low transmission bandwidth with high visual fidelity. To enhance reconstruction, the framework employs a sparse-to-dense estimation strategy, generating dense motion and occlusion maps to facilitate precise motion alignment and realistic frame synthesis Quality 
 ## Comparisons at Similar Bitrate
![compare](https://github.com/user-attachments/assets/f016cc00-f4a7-41c4-8ad3-5111f909763f)
## Training
To train a model on VoxCeleb dataset, please follow the instruction from https://github.com/AliaksandrSiarohin/video-preprocessing. When finishing the downloading and pre-processing the dataset, you can train the model.
```
python run.py
```
The code will generate a folder in the log directory, creating a new timestamped directory for each run. Checkpoints will be stored in this folder. To monitor the loss values during training, refer to log.txt. Additionally, you can find the training data reconstructions in the train-vis subfolder. Training settings can be modified in the ./config/vox.yaml file.
## Inference
To encode a sequence, please put testing sequence in ./testing_data/ file and run

```
python K-Means_Encoder.py
```
we use a context-based feature encoding method that computes compact feature residuals relative to key-reference frames. This process involves calculating differences, quantizing the residuals, encoding them with signed zero-order exponential-Golomb coding, and compressing the binary codes using context-based arithmetic coding with Prediction by Partial Matching (PPM). The resulting bitstream is transmitted to the decoder for reconstruction. The Golomb coding process is illustrated in the following figure.
![image](https://github.com/user-attachments/assets/a2bdf4db-0ba1-495a-a941-08e556de9510)

After obtaining the bistream , please run
```
python k-Means_Decoder.py
```


## Evaluate
In ./evaluate/multiMetric.py file, we provide the corresponding quality measures, including DISTS, LPIPS, PSNR and SSIM.
# Additional notes
## Reference
The arithmetic-coding refers to https://github.com/nayuki/Reference-arithmetic-coding.

