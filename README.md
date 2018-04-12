 # ImageDecomposition-DECT

This reposity is organized mainly for an image decomposition algorithm which is proposed to solve the material decomposition problem in Dual-energy Computed Tomography (DECT). <br>

The aloghrithm is designed based on deep learning paradigm. For more theoretical details, please go to [Deep Learning](http://www.deeplearningbook.org/) and [Material Decomposition Using DECT](https://pubs.rsna.org/doi/10.1148/rg.2016150220).<br>
  
All have been tested with python 3.6 and tensorflow 1.4.0 in Linux. <br>
  * checkpoint: the checkpoint path for the model trained with tensorflow. The [pre-trained model](https://pan.baidu.com/s/1r1OTjid2muWWZfxURB8Pjw) was trained on a dataset which contained totally 2,454,300 samples. Each sample is a 65*65 image patch extracted from 5987 image slices.
  * data: contains 2 path.
    * test: two test data files, 'test_data_cranial.mat' and 'test_data_pleural.mat'.
    * train: we only provide a sub-set (90,000 training samples) in the 'training_samples_90000.rar' file which can be download from [here](https://pan.baidu.com/s/1r1OTjid2muWWZfxURB8Pjw). <br>
  * result: save decomposition result.
  * src: the codes for three decomposition algorithms:
    * Direct matrix inversion (matrix_inversion.m)
    * Iterative decomposition (iterative_decomposition.m). Related paper: [Iterative image-domain decomposition for dual-energy CT](https://aapm.onlinelibrary.wiley.com/doi/abs/10.1118/1.4889338)
    * The proposed deep model (main.py). After download the pre-trained mode, you can use the following command to run the algorithm. <br>
    >> " python main.py --dataset="../data/test/test_data_crainal.mat" --model="feedforward" --model_name="your-saved-result-name" --checkpoint="../checkpoint/FCN_trained_model " <br>
    
    
Author email: vastcyclone@yeah.net



