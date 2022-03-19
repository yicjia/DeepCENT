# DeepCENT

DeepCENT implements a deep neural network to predict the individual time to an event for right censoring data. It utilizes an innovative loss function that combines the mean square error and the concordance index. DeepCENT can also handle competing risks. 

Please refer to our paper for more details: https://arxiv.org/abs/2202.05155


### From source

Download a local copy of DeepCENT and install from the directory:

	git clone https://github.com/yicjia/DeepCENT.git
	cd DeepCENT
	pip install .

### Dependencies

torch, lifelines, sklearn and all of their respective dependencies. 



## Example

### Using Google Colab
Here is a tutorial on Google Colab
- Regular survival data: <a href="https://colab.research.google.com/drive/13BZj4r4SaTcr7n2MKCCq9SCpJRsHm4QE?usp=sharing">
  	<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
	</a>
  
- Competing risks data: 

## Reference

### If you use DeepCENT in your research, please consider citing:=
Jia, Y., &  Jeong, J. H. (2022). DeepCENT: Prediction of Censored Event Time via Deep Learning. *arXiv preprint arXiv:2202.05155*.
```
@article{jia2022deepcent,
  title={Deep{CENT}: Prediction of Censored Event Time via Deep Learning},
  author={Jia, Yichen and Jeong, Jong-Hyeon},
  journal={arXiv preprint arXiv:2202.05155},
  year={2022}
}
