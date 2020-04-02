## Neon Drive - A Steam Game

<p align="justify"> 
  <img src="https://media.giphy.com/media/XBuh0LZxoCoNgx1g1M/giphy.webp" alt="Neon Drive" align="right">
  <a>Neon Drive is a slick retro-futuristic and '80s inspired arcade game. This game has a very simple purpose, to deviate from fixed abstacles over time using 3 types of discrete actions: left, right and no movement. In this case, Neon Drive will serve as the environment for our reinforcement learning algorithm.</a>
  <a>To use this algorithm you need to open the game and enter 'endurence' mode. Let the car hit an obstacle, in the restart screen run the following command:</a></p>
  
```shell
sudo python3 DQN.py --resolution 1920x1080 --train policy_net.pt --save new_policy_net.pt
```

<p align="justify"> 
 <a>Then go back to the game screen and let the algorithm work. In case there's any doubt, you need to run with sudo because of the keyboard module.</a>
</p>

_Obs: If you have problems with terminal environment variables please add -E after sudo._

#### Requirements

All of requirements is show in the shields bellow, but if you want to install all of then run the following line:
```shell
pip3 install -r requirements.txt
```

<p align="center"> 
  <img src="https://img.shields.io/badge/Python-v3.6.9-blue"/>
  <img src="https://img.shields.io/badge/PyTorch-v1.4.0-blue"/>
  <img src="https://img.shields.io/badge/TorchVision-v0.5.0-blue"/>
  <img src="https://img.shields.io/badge/OpenCV-v4.2.0-blue"/>
  <img src="https://img.shields.io/badge/Numpy-v1.18.2-blue"/>
  <img src="https://img.shields.io/badge/Matplotlib-v3.1.2-blue"/>
  <img src="https://img.shields.io/badge/Argparse-v1.1-blue"/>
  <img src="https://img.shields.io/badge/mss-v5.0.0-blue"/>
  <img src="https://img.shields.io/badge/Tqdm-v4.42.1-blue"/>
</p>



### Deep Q-Network architecture

### Image Processing

### Reward Function

### Results

If you liked this repository, please don't forget to starred it!   <img src="https://img.shields.io/github/stars/victorkich/Neon-Drive-Reinforcement-Learning?style=social"/>
