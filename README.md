
<h1 align="center">Neon Drive Reinforcement Learning</h1>
<h3 align="center">In this work I seek to develop a reinforcement learning algorithm, based solely on the screen image. The algorithm must be able to stay 60 seconds alive in the chosen environment. This environment, is a game available on the steam platform called Neon Drive.</h3>

<p align="center"> 
  <img src="https://img.shields.io/badge/PyTorch-v1.4.0-blue"/>
  <img src="https://img.shields.io/badge/TorchVision-v0.5.0-blue"/>
  <img src="https://img.shields.io/badge/OpenCV-v4.2.0-blue"/>
  <img src="https://img.shields.io/badge/Numpy-v1.18.2-blue"/>
  <img src="https://img.shields.io/badge/Matplotlib-v3.1.2-blue"/>
  <img src="https://img.shields.io/badge/Scikit_Image-v0.16.2-blue"/>
  <img src="https://img.shields.io/badge/mss-v5.0.0-blue"/>
  <img src="https://img.shields.io/badge/Tqdm-v4.42.1-blue"/>
</p>
<br/>

# Environment
<p align="justify"> 
  <img src="https://media.giphy.com/media/XBuh0LZxoCoNgx1g1M/giphy.webp" alt="Neon Drive" align="right" width="320">
  <a>Neon Drive is a slick retro-futuristic and '80s inspired arcade game. This game has a very simple purpose, to deviate from fixed abstacles over time using 3 types of discrete actions: left, right and straight. In this case, Neon Drive will serve as the environment for our reinforcement learning algorithm.</a>
  <a>To use this algorithm you need to open the game and enter in 'endurence' mode. Let the car hit an obstacle, in the restart screen run the following command:</a>
</p>
  
```shell
sudo python3 dqn.py --resolution 1920x1080 --train policy_net.pth
```

<p align="justify"> 
 <a>Then go back to the game screen and let the algorithm work. In case there's any doubt, you need to run with sudo because of the keyboard module.</a>
</p>

>**Obs**: If you have problems with terminal environment variables please add -E after sudo.

# Setup
<p align="justify"> 
 <a>All of requirements is show in the badgets above, but if you want to install all of them, enter the repository and execute the following line of code:</a>
</p>

```shell
pip3 install -r requirements.txt
```

# Deep Q-Network 
<p align="justify" float="left">
  Neural networks can usually solve tasks just by looking at the location, so let's use a piece of the screen centered on the car as an input. By using only image our task becomes much more difficult. Since we cannot render multiple environments at the same time, we need a lot of training time. Strictly speaking, we will present the state as the difference between
the current screen patch and the previous one. This will allow the agent to take the velocity of the obstacles into account from one image.
    
  Our model will be a convolutional neural network that takes in the difference between the current and previous screen patches. It has three outputs, representing ![equation](https://latex.codecogs.com/gif.latex?Q(s,&space;\mathrm{left})), ![equation](https://latex.codecogs.com/gif.latex?Q(s,&space;\mathrm{right})) and ![equation](https://latex.codecogs.com/gif.latex?Q(s,&space;\mathrm{straight})) where ![equation](https://latex.codecogs.com/gif.latex?s) is the input to the network. In effect, the network is trying to predict the *expected return* of taking each action given the current input.
</p>

# Image Processing
<p align="justify"> 
  <a>The image processing performed in this work is quite simple, but it is very important for the overall functioning of the algorithm. Through the mss module the screen was captured and transformed into a numpy array variable. With the BGR screen saved, we applied a color filter available in the OpenCV module to transform everything to grayscale. We cut 53.84% of the upper pixels, 20% of the lower pixels, and 20% of the left and   right pixels. After that, we applied the triangle threshhold function to transform the image to black and white. Finally, we resize the final image to 160x90 pixels using area interpolation and invert all of the binary pixels. You can follow the steps of this process in the following images:</a>
</p>

<p align="center"> 
  <img src="media/image_process_comparation.gif" alt="Image Process Comparation"/>
</p>

<p align="center"> 
  <a>In order, the respective images are: normal input image, image converted to grayscale, cropped image with triangle threshold and lastly the cropped image with triangle threshold and all the binary pixels inverted.</a>
</p>

# Reward Function
<p align="justify" float="left"> 
  <img src="https://media.giphy.com/media/jS8vrTLvwG1Oox4kLH/giphy.webp" alt="Neon Drive" align="right" width="320">
  As the agent observes the current state of the environment and chooses an action, the environment transitions to a new state, and also returns a reward that indicates the consequences of  the action. In this task, rewards are +1 for every incremental timestep and the environment terminates if the car hits the obstacle. This means better performing scenarios will run for longer duration, accumulating larger return.

  Our aim will be to train a policy that tries to maximize the discounted, cumulative reward ![equation](https://latex.codecogs.com/gif.latex?R_{t_0}&space;=&space;\sum_{t=t_0}^{\infty}&space;\gamma^{t&space;-&space;t_0}&space;r_t) , where ![equation](https://latex.codecogs.com/gif.latex?R_{t_0}) is also known as the return. The discount, ![equation](https://latex.codecogs.com/gif.latex?\gamma) , should be a constant between 0 and 1 that ensures the sum converges. It makes rewards from the uncertain far future less important for our agent than the ones in the near future that it can be fairly confident about.
</p>

# Results
<p align="justify"> 
  <a>After the training is done a file called <em>data.csv</em> is generated. From this file we can show some important information such as: reward history, number of steps per epochs and noise interference on the data. To view your graphs run the following command:</a>
</p>

```shell
python3 data_visualization.py --file data.csv
```

<p align="justify"> 
  <a>The following images show the graphics generated by the <em>data.csv</em> file already present in the repository.</a>
</p>

<p align="center"> 
  <img src="https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2Fwww.mitzvahmarket.com%2Fwp-content%2Fuploads%2F2013%2F06%2FYour-Logo-Here-Black-2-e1371130716893.jpg&f=1&nofb=1" alt="Neon Drive">
  <img src="https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2Fwww.mitzvahmarket.com%2Fwp-content%2Fuploads%2F2013%2F06%2FYour-Logo-Here-Black-2-e1371130716893.jpg&f=1&nofb=1" alt="Neon Drive">
  <img src="https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2Fwww.mitzvahmarket.com%2Fwp-content%2Fuploads%2F2013%2F06%2FYour-Logo-Here-Black-2-e1371130716893.jpg&f=1&nofb=1" alt="Neon Drive">
</p>

<p align="justify"> 
  <a><em>If you liked this repository, please don't forget to starred it!</em></a>  <img src="https://img.shields.io/github/stars/victorkich/Neon-Drive-Reinforcement-Learning?style=social" align="center"/>
</p>
