# MCSF (Multi-Source Chunk and Stride Fusion)


![MCSF](/imgs/mcsf.png)

## Get started (Requirements and Setup)

python 3.6


```
cd MCSF
conda create -n mcsf python=3.6
conda activate mcsf  
pip install -r requirements.txt
```
## Project Structure
```
Directory: 
- /data
        - /plc_365 (places features  for summe and tvsum)
        - /splits (original and non-overlapping splits)
        - /SumMe (processed dataset h5)
        - /TVSum (processed dataset h5)
- /csnet (implementation of csnet method)
- /mcsf-places365-early-fusion 
- /mcsf-places365-late-fusion 
- /mcsf-places365-intermediate-fusion
- /src/evaluation (evaluation using F1-score)
- /src/visualization 
- /sum-ind (implementation of SUM-Ind method)


```
## Datasets
Structured h5 files with the video features and annotations of the SumMe and TVSum datasets are available within the "data" folder. The GoogleNet features of the video frames were extracted by [Ke Zhang](https://github.com/kezhang-cs) and [Wei-Lun Chao] and the h5 files were obtained from [Kaiyang Zhou](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce). 


### Download
```
wget https://zenodo.org/record/4884870/files/datasets.tar

```

### Files Structure

The implemented models use the provided h5 files which have the following structure:
```
/key
    /features                 2D-array with shape (n_steps, feature-dimension)
    /gtscore                  1D-array with shape (n_steps), stores ground truth improtance score (used for training, e.g. regression loss)
    /user_summary             2D-array with shape (num_users, n_frames), each row is a binary vector (used for test)
    /change_points            2D-array with shape (num_segments, 2), each row stores indices of a segment
    /n_frame_per_seg          1D-array with shape (num_segments), indicates number of frames in each segment
    /n_frames                 number of frames in original video
    /picks                    positions of subsampled frames in original video
    /n_steps                  number of subsampled frames
    /gtsummary                1D-array with shape (n_steps), ground truth summary provided by user (used for training, e.g. maximum likelihood)
    /video_name (optional)    original video name, only available for SumMe dataset
```
Original videos and annotations for each dataset are also available in the authors' project webpages:

**TVSum dataset**: [https://github.com/yalesong/tvsum](https://github.com/yalesong/tvsum) 


**SumMe dataset**: [https://gyglim.github.io/me/vsum/index.html#benchmark](https://gyglim.github.io/me/vsum/index.html#benchmark)


<br/>

## MCSF Variations and CSNet
We used [SUM-GAN](https://github.com/j-min/Adversarial_Video_Summary) method as a starting point for the implementation.

<br/>

### How to train

 Run main.py file with the configurations specified in configs.py to train the model.
 In config.py you find argument parameters for training:

| Parameter        | type           | default  |
| ------------- |:-------------:| -----:|
| mode     | string  possible values (train, test) | train
| verbose      | boolean      |   true |
| video_type | string (summe or tvsum)      |    summe |
| input_size | int     |    1024 |
| hidden_size | int     |    500 |
| split_index | int     |    0 |
| n_epochs | int     |    20 |
| m | int  (number of divisions used for chunk and stride network)   |    4 |

<br/>
<br/>


For training the model using a single split, run:

```
python main.py --split_index N (with N being the index of the split)
```
<br/>



## SUM-Ind

Train and test codes are written in `main.py`. To see the detailed arguments, please do `python main.py -h`.

<br/>

#### How to train

```bash
python main.py -d datasets/eccv16_dataset_summe_google_pool5.h5 -s datasets/summe_splits.json -m summe --gpu 0 --save-dir log/summe-split0 --split-id 0 --verbose
```

#### How to test

```bash
python main.py -d datasets/eccv16_dataset_summe_google_pool5.h5 -s datasets/summe_splits.json -m summe --gpu 0 --save-dir log/summe-split0 --split-id 0 --evaluate --resume path_to_your_model.pth.tar --verbose --save-results
```
<br/>

## Citation
```
@article{kanafani2021MCSF, 
   title={MCSF (Multi-Source Chunk and Stride Fusion)},
   author={Kanafani, Hussain and Ghauri, Junaid Ahmed and Hakimov, Sherzod and Ewerth, Ralph}, 
   Conference={ACM International Conference on Multimedia Retrieval (ICMR)}, 
   year={2021} 
}
```
