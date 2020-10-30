import numpy as np
import time
import random
import gym
import gym_tetris
import matplotlib.pyplot as plt
import threading
import os
import cv2
from tensorflow import set_random_seed
from keras import activations
from keras.models import Model
from keras.layers import Dense, Input, Convolution2D, Flatten
from keras.optimizers import Adam, SGD
from keras.layers import LeakyReLU
from collections import deque
from tkinter import *
from PIL import Image, ImageTk

env = gym.make('Tetris-v0')
env.seed(42)


episodes = 200000
steps = 1000
lr = 0.01
lrdecay = 0.01
gamma = 0.95
batchSize = 32

epsilon = 0.8
min_epsilon = 0.01
max_epsilon = 0.8
decay = 0.005

preMemory = deque(maxlen=100000)
model = makeModel()

highScore = -0

currentImage = []
render = False
gui = GUI()
stats = [0,0,0]

scoreQue = deque(maxlen=100)
summary = []
solved = False
finishedAfter=None

if "preData.npy" not in os.listdir():
    memory = deque(maxlen=100000)
    while len(memory) < 5000:
        preRunOnce()
    np.save("preData",memory)
else:
    memory = deque(np.load("tetrisPreData.npy"),maxlen=100000)
for i in range(50):
    print(i)
    replay()

print("Start:")
for ep in range(episodes):
    stats[0] = ep
    score = runOnce()
    if score > highScore:
        highScore = score
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay*ep)
    scoreQue.append(score)
    rm = np.mean(scoreQue)
    print("Episode: ",ep, "| Score: " ,score,"| HighScore: ",highScore, "| Epsilon: ", epsilon, "Running Mean: ", np.mean(scoreQue))
    stats[2] = highScore
    if ep%10==0 and ep !=0:
        summary.append((ep, score, rm))
    if rm >= 195.0 and not solved:
        print("Done after ",ep," episodes") 
        solved = True
        finishedAfter = ep

summaryAndQuit()

===========================================================================

def makeModel():
    activation = lambda x: activations.relu(x, alpha=0.3)
    inputs = Input(shape=(86,66,3))
    conv1 = Convolution2D(32, kernel_size=(5,5),strides=(2,2), activation='relu')(inputs)
    conv2 = Convolution2D(32, kernel_size=(3,3), activation='relu')(conv1)
    flatten = Flatten()(conv2)
    dense1 = Dense(48, activation=activation)(flatten)
    out = Dense(env.action_space.n,activation=activation)(dense1)
    model = Model(inputs=inputs, outputs=out)
    model.summary()
    model.compile(loss='mse',optimizer=Adam(lr=lr, decay=lrdecay))
    return model

def preProc(obs):
    scale = cv2.resize(obs,dsize=(66,86), interpolation=cv2.INTER_NEAREST)
    if render: 
        gui.updateFrame(ImageTk.PhotoImage(Image.fromarray(scale)))
    proc = scaled / 255
    return proc

def getAction(state):
    exploit= random.uniform(0,1)
    if exploit > epsilon:
        prediction = model.predict(np.array([state]))
        return prediction.argmax()
    else:
        return env.action_space.sample()

def replay():
    minibatch = random.sample(memory,batchSize)
    states = []
    targets = []
    for state, action, reward, nextState, done in minibatch:
        states.append(state)
        if done:
            target = reward
        else:
            target = reward + gamma*np.amax(model.predict(np.array([nextState])))
        target_f = model.predict(np.array([state]))[0]
        target_f[action]=target
        targets.append(target_f)
    model.train_on_batch(np.array(states),np.array(targets))

def preRunOnce():
    score = 0
    state = preProc(env.reset())
    for step in range(totalSteps):
        action = getAction(state)
        newState, reward, done, _ = env.step(action)
        
        score += reward
        state[1] = score
        
        newState = preProc(newState)
        preMemory.append((state,action,reward,newState,done))
        
        if done:
            break

        state = newState
    if score > 0:
        memory.extend(preMemory)
        print("dump")
    preMemory.clear()
    
def runOnce():
    score = 0
    state = preProc(env.reset())
    for step in range(totalSteps):
        action = getAction(state)
        newState, reward, done, _ = env.step(action)
        
        score += reward
        state[1] = score
        
        newState = preProc(newState)
        preMemory.append((state,action,reward,newState,done))
        
        if done:
            break
        
        if len(memory)>=batchSize:
            replay()
        
        state = newState
    if score > 0:
        memory.extend(preMemory)
    preMemory.clear()
    
    return score

class GUI(threading.Thread):
    
    def __init__(self):
        threading.Thread.__init__(self)
        self.start()
           
    def run(self):
        self.root = Tk()
        switcher = Button(self.root,text="Switch Rendering", command=switchRender)
        switcher.pack()
        quitButton = Button(self.root,text="Quit", command=summaryAndQuit)
        quitButton.pack()
        self.picLabel = Label(self.root)
        self.picLabel.pack(side="left", fill="both", expand="yes")
        self.statsLabel=Label(self.root,text="Episode: 0\nScore: 0\nHighscore: 0")
        self.statsLabel.pack(side="right")
        self.root.mainloop()
    
    def updateFrame(self,img):
        self.picLabel.config(image = img)
        self.picLabel.image = img
        self.statsLabel['text']="Episode: "+str(stats[0])+"\nScore: "+str(stats[1])+"\nHighscore: "+str(stats[2])
    
def switchRender():
    global render
    render = not render

def summaryAndQuit():
    global summary
    for t in summary:
        print(t)
    if finishedAfter:
        print("finished after: ", finishedAfter)
    os._exit(1)
