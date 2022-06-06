import gym
import ray
import psutil
import os
import time

ray.init()

def memory_used():
    return psutil.Process(os.getpid()).memory_info().rss * 1e-6  # To megabyte

@ray.remote
class Env:
    def __init__(self):
        self.env = gym.make('CarRacing-v0')
        self.env.unwrapped.verbose = 0

    def reset(self):
        self.env.reset()


#################### Multiple Env

e = [Env.remote() for i in range(6)]  # multiple envs in parallel 
for i in range(10000000):
    ray.get([a.reset.remote() for a in e])

    if i % 200 == 0:
        print('Ram Used: %f' % psutil.virtual_memory().percent)

#################### Single Env

#e = Env.remote()  # Single env
#
#for i in range(10000000):
#    ray.get(e.reset.remote())
#
#    if i % 1000 == 0:
#        print('Ram Used: %f' % psutil.virtual_memory().percent)


#################### Without Ray

#e = gym.make('CarRacing-v0')
#e.unwrapped.verbose = 0
#
#for i in range(1000000):
#    e.reset()
#
#    if i % 1000 == 0:
#        print('Ram Used: %f' % memory_used())
