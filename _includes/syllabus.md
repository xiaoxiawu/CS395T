# CS 395T - Deep learning seminar - Fall 2016

**meets** TTh 2:00 - 3:30pm in WAG 308

**instructor** [Philipp Krähenbühl](http://www.philkr.net)<br/>
office [GDC 4.824](http://facilitiesservices.utexas.edu/buildings/UTM/0152)<br/>
office hours by appointment (send email)

**TA** Huihuang Zheng<br/>
email huihuang@utexas.edu<br/>
office [GDC 6.802](http://facilitiesservices.utexas.edu/buildings/UTM/0152)<br/>
TA hours We 9:30-10:30am at TA station desk 2  

Please use [piazza](http://www.piazza.com/utexas/fall2016/cs395tdeeplearning) for assignment questions.

### Prerequisites
 * 391L - Intro Machine learning  (or equivalent)
 * 311 or 311H - Discrete math for computer science (or equivalent)
 * **proficiency in Python**, high level C++ understanding
   * All projects are in Python with Caffe, Tensorflow or Theano as recommended deep learning backends. It is also recommended to familiarize yourself with numpy, scipy, scikit-learn and matplotlib as additional libraries.
   * NOTE: It is possible to use other languages, but the course staff cannot provide support.
 * Basic **deep learning background**
   * You should be familiar with at least one deep learning package (Caffe, Tensorflow, Torch, Matconvnet, ...). You should have **trained at least one deep network** with one of these packages. I'd encourage the use of Caffe, Tensorflow or Theano for projects (if you want help from the course staff), but it is not required.


### Class overview
 * The class reads (rates and reviews) two recent research papers per class
 * Each paper is presented by one student (two per class), and discussed by entire class.
 * Two projects in teams up to 3 students (teams can change between projects)
 * Auditing allowed if there is space (no homework or presentation, but participation required)

This class covers advanced topics in deep learning, ranging from optimization to computer vision, computer graphics and unsupervised feature learning, and touches on deep language models, as well as deep learning for games. This is meant to be a very interactive class for upper level students (MS or PhD). For every class we read two recent research papers (most no older than two years), which we will discuss in class.

#### Goals of the class
After this class you should be able to

 * Read, rate and review deep learning papers
 * Create and give an interesting presentation on in deep learning
 * Devise and execute a research project in deep learning (at the level of a top tier workshop publication: CVPR, ICCV, ICML, NIPS, ACL, SIGGRAPH or equivalent)

### Grading
 * 30%  paper presentation
 * 30%  project 1 (10% presentation, 20% project)
 * 40%  project 2 (10% presentation, 30% project)
 * (optional) 12.5%  volunteering for second presentation


To map percentages to letter grades we use the following python script

```python
def grade(p):
  from math import floor
  if p < 50: return 'F'
  v = (100-p) * 4 / (50 + 1e-5)
  return chr(ord('A')+int(v)) + ['+','','','-'][int((v-floor(v))*4)]
```

### Schedule

| Date | Topic | Papers | Presenters | Notes and due dates |
| --- | --- | --- | --- | --- |
| Aug 25 | Administrative and intro (Linear models) | | Philipp | |
| Aug 30 | Gradient based optimization | [Large-scale machine learning with stochastic gradient descent, Bottou 2010](http://leon.bottou.org/publications/pdf/compstat-2010.pdf) | Philipp | |
| Sep 1 | Deep networks and backpropagation |  [Deep learning, LeCun, Bengio and Hinton 2015](http://www.nature.com/nature/journal/v521/n7553/full/nature14539.html) | Philipp | **paper selection** Th Sep 1, **6am** email TA |
| Sep 06 | Dropout and batch normalization | [Dropout: A Simple Way to Prevent Neural Networks from Overfitting, Srivastava etal. 2014](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)<br/><br/>[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, Ioffe etal. 2015](https://arxiv.org/pdf/1502.03167v3) | | **project 1 out** |
| Sep 08 | Advanced optimization and initialization | [Adam: A Method for Stochastic Optimization, Kingma and Ba 2015](https://arxiv.org/pdf/1412.6980v8.pdf) <br/><br/>[Data-dependent Initializations of Convolutional Neural Networks, Krähenbühl etal. 2016](https://arxiv.org/pdf/1511.06856v2.pdf) | | |
| Sep 13 | Convolutional Networks for image classification | [Gradient-based learning applied to document recognition, LeCun etal. 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)<br/><br/>[ImageNet Classification with Deep Convolutional Neural Networks, Krizhevsky etal. 2012](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) | | |
| Sep 15 | Convolutional Networks for object detection | [Rich feature hierarchies for accurate object detection and semantic segmentation, Girshick etal. 2014](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf)<br/><br/>[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, Ren etal. 2015](https://arxiv.org/pdf/1506.01497v3.pdf) | | |
| Sep 20 | Convolutional networks for pixel-wise prediction | [Fully Convolutional Networks for Semantic Segmentation, Long etal 2015](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)<br/><br/>[Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs, Chen etal. 2015](http://arxiv.org/pdf/1412.7062v4.pdf) | | Project 1 QA |
| Sep 22 | Advanced deep network architectures | [Very Deep Convolutional Networks for Large-Scale Image Recognition, Simonyan etal. 2015](http://arxiv.org/pdf/1409.1556v6)<br/><br/>[Deep Residual Learning for Image Recognition, He etal. 2016](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) | | |
| Sep 27 | Visualizing deep networks | [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps, Simonyan etal. 2014](http://arxiv.org/pdf/1312.6034v2)<br/><br/>[Inverting visual representations with convolutional networks, Dosovitskiy 2016](http://lmb.informatik.uni-freiburg.de/Publications/2016/DB16/Invert_AlexNet_final.pdf) | | |
| Sep 29 | Image manipulation | [Understanding Deep Image Representations by Inverting Them, Mahendran etal. 2015](http://www.robots.ox.ac.uk/~vgg/publications/2015/Mahendran15/mahendran15.pdf)<br/><br/>[A Neural Algorithm of Artistic Style, Gatys 2015](http://arxiv.org/pdf/1508.06576v2) | | |
| Oct 04 | Project 1 presentations | Format TBD | | Project 1 due **6am** |
| Oct 06 | Project 1 presentations | Format TBD | | Project 2 out |
| Oct 11 | Stereo, Flow | [Computing the Stereo Matching Cost with a Convolutional Neural Network, Zbontar 2015](http://arxiv.org/pdf/1409.4326v2.pdf)<br/><br/>[EpicFlow: Edge-Preserving Interpolation of Correspondences for Optical Flow, Revaud etal 2015](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Revaud_EpicFlow_Edge-Preserving_Interpolation_2015_CVPR_paper.pdf) | | |
| Oct 13 | Monocular depth and normal estimation | [Depth Map Prediction from a Single Image using a Multi-Scale Deep Network, Eigen etal. 2014](http://www.cs.nyu.edu/~deigen/depth/depth_nips14.pdf)<br/><br/>[Designing Deep Networks for Surface Normal Estimation, Wang etal. 2015](http://www.cs.cmu.edu/~dfouhey/2015/deep3d/deep3d.pdf) | | |
| Oct 18 | Image generation | [Auto-Encoding Variational Bayes, Klingma etal. 2014](https://arxiv.org/pdf/1312.6114v10)<br/><br/>[Generative Adversarial Nets, Goodfellow etal. 2014](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) | | |
| Oct 20 | Recurrent drawing models | [Pixel Recurrent Neural Networks, Oord etal 2016](https://arxiv.org/pdf/1601.06759v3) | | Project 2 flash presentations (2min) |
| Oct 25 | Auto-encoders | [Extracting and Composing Robust Features with Denoising Autoencoders, Vincent etal 2008](http://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf)<br/><br/>[Context Encoders: Feature Learning by Inpainting, Pathak etal. 2016](https://arxiv.org/pdf/1604.07379v1) | | |
| Oct 27 | Self-supervision | [Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles, Noroozi etal 2016](http://arxiv.org/pdf/1603.09246v2)<br/><br/>[Colorful Image Colorization, Zhang etal. 2016](https://arxiv.org/pdf/1603.08511v3) | | |
| Nov 01 | Recurrent language models | [Sequence to sequence learning with neural networks, Sutskever etal. 2014](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)<br/><br/>[Learning longer memory in recurrent neural networks, Mikolov etal. 2014](http://arxiv.org/pdf/1412.7753) | | |
| Nov 03 | Image and language | [From Captions to Visual Concepts and Back, Rang etal. 2015](http://arxiv.org/pdf/1411.4952v3)<br/><br/>[Learning to Compose Neural Networks for Question Answering, Andreas etal. 2016](http://arxiv.org/pdf/1601.01705.pdf) | | |
| Nov 08 | Atari games | [Playing Atari with Deep Reinforcement Learning, Mhin etal. 2013](https://arxiv.org/pdf/1312.5602.pdf)<br/><br/>[Human-level control through deep reinforcement learning, Mhin etal. 2015](http://www.nature.com/nature/journal/v518/n7540/abs/nature14236.html) | | |
| Nov 10 | Alpha GO | [Mastering the game of Go with deep neural networks and tree search, Silver etal. 2016](http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html) | | |
| Nov 15 | Data Collection | [ImageNet: A Large-Scale Hierarchical Image Database, Deng etal. 2009](http://www.image-net.org/papers/imagenet_cvpr09.pdf)<br/><br/> [Microsoft COCO: Common Objects in Context, Lin etal. 2014](http://arxiv.org/pdf/1405.0312v3) | | |
| Nov 17 | Alternative data collection | [FlowNet: Learning Optical Flow with Convolutional Networks, Fischer etal. 2015](http://arxiv.org/pdf/1504.06852v2.pdf)<br/><br/>[Playing for Data: Ground Truth from Computer Games, Richter etal. 2016](http://download.visinf.tu-darmstadt.de/data/from_games/data/eccv-2016-richter-playing_for_data.pdf) | | |
| Nov 22 | TBD | <br/><br/> TBD | | |
| Nov 24 | No Class | Thanksgiving holidays | | |
| Nov 29 | Final project presentations | Format TBD | | Project 2 due **6am** |
| Dec 01 | Final project presentations | Format TBD | | |


### Expected workload
Estimates of required effort to pass the class are:

 * 2-4 hours per week reading / reviewing papers
 * 7 hours per semester (1/2 hour per week) preparing paper presentations
 * 2-10 hours per week of programming

### General tips

 * Start the projects early
   * most deep neural networks take 1 day to train in a GPU
   * let us know early if you don't have GPU access (first or second week)
 * read your assigned papers early and prepare the slides early
   * a bad presentation will waste 30 min of your fellow students lives
   * you have the option to get feedback on your slides ahead of time from the instructor

#### Tips for reading/reviewing a paper

 * Just reading the paper is **not** sufficient
   * Do **more** than merely summarizing the paper
   * No paper is *trivial*
 * Question any decision and claim made by the authors
   * It is the authors responsibility to convince you that their approach works better than prior (or simpler) alternatives
   * If a claim is not backed by experiments or a citation (or backed by a wrong citation), you may assume it's wrong
 * Think about how this fits with other peoples findings
   * Is there a larger theme across a series of papers?
   * Does it contradict other paper you know?
 * Use colored markers
   * Mark important things in one color
   * Mark things you disagree with (you think are wrong) in a different color

#### Tips for presentations

 * Have a story
   * Motivate well
   * Provide context
 * Make the presentation **visual**
   * Use at least one picture/figure/graph per slide
   * Walls of bulleted text are unacceptable
   * Most mathematical concepts can first be expressed in a figure
 * Feel free to make things interactive
   * as long as it fits into the time budget
 * Show your slides to the instructor a week before the presentation

### Notes
Syllabus subject to change.
