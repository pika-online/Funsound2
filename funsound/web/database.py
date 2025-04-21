from funsound.utils import * 

class User:
    def __init__(self,userId,cache_dir='./cache'):
        self.userId = userId
        mkdir(cache_dir)
        self.cache_dir = cache_dir
        self.volume = 10000