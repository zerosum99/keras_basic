# -*- coding: utf-8 -*-
from __future__ import division, print_function
import collections
import numpy as np
import pygame
import random
import os

class MyWrappedGame(object):
    
    def __init__(self):
        # pygame을 헤드리스 모드로 실행
        #        os.environ["SDL_VIDEODRIVER"] = "dummy"
        
        pygame.init()
        pygame.key.set_repeat(10, 100)
        
        # 환경 설정
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_BLACK = (0, 0, 0)
        self.GAME_WIDTH = 400
        self.GAME_HEIGHT = 400
        self.BALL_WIDTH = 20
        self.BALL_HEIGHT = 20
        self.PADDLE_WIDTH = 50
        self.PADDLE_HEIGHT = 10
        self.GAME_FLOOR = 350
        self.GAME_CEILING = 10
        self.BALL_VELOCITY = 10
        self.PADDLE_VELOCITY = 20
        self.FONT_SIZE = 30
        # self.MAX_TRIES_PER_GAME = 100
        self.MAX_TRIES_PER_GAME = 1
        self.CUSTOM_EVENT = pygame.USEREVENT + 1
        self.font = pygame.font.SysFont("Comic Sans MS", self.FONT_SIZE)
        

    def reset(self):
        self.frames = collections.deque(maxlen=4)
        self.game_over = False
        # 공과 패들을 시작 지점으로 옮긴다.
        self.paddle_x = self.GAME_WIDTH // 2
        self.game_score = 0
        self.reward = 0
        self.ball_x = random.randint(0, self.GAME_WIDTH)
        self.ball_y = self.GAME_CEILING
        self.num_tries = 0
        # 화면과 시계 등을 초기화한다.
        self.screen = pygame.display.set_mode(
                (self.GAME_WIDTH, self.GAME_HEIGHT))
        self.clock = pygame.time.Clock()
    
    def step(self, action):
        pygame.event.pump()
        
        if action == 0:   # 패들을 왼쪽으로 옮긴다.
            self.paddle_x -= self.PADDLE_VELOCITY
            if self.paddle_x < 0:
                # 벽을 친 경우에는 오른쪽으로 옮긴다.
                self.paddle_x = self.PADDLE_VELOCITY
        elif action == 2: # 패들을 오른쪽으로 옮긴다.
            self.paddle_x += self.PADDLE_VELOCITY
            if self.paddle_x > self.GAME_WIDTH - self.PADDLE_WIDTH:
                # 벽을 친 경우에는 왼쪽으로 옮긴다.
                self.paddle_x = self.GAME_WIDTH - self.PADDLE_WIDTH - self.PADDLE_VELOCITY
        else:             # 패들을 움직이지 않는다.

            pass

        self.screen.fill(self.COLOR_BLACK)
        score_text = self.font.render("Score: {:d}/{:d}, Ball: {:d}"
            .format(self.game_score, self.MAX_TRIES_PER_GAME,
                    self.num_tries), True, self.COLOR_WHITE)
#        self.screen.blit(score_text, 
#            ((self.GAME_WIDTH - score_text.get_width()) // 2,
#             (self.GAME_FLOOR + self.FONT_SIZE // 2)))
                
        # 공 위치를 업데이트
        self.ball_y += self.BALL_VELOCITY
        ball = pygame.draw.rect(self.screen, self.COLOR_WHITE,
                                pygame.Rect(self.ball_x, self.ball_y,
                                            self.BALL_WIDTH,
                                            self.BALL_HEIGHT))
        # 패들 위치를 업데이트
        paddle = pygame.draw.rect(self.screen, self.COLOR_WHITE,
                                  pygame.Rect(self.paddle_x, 
                                              self.GAME_FLOOR,
                                              self.PADDLE_WIDTH,
                                              self.PADDLE_HEIGHT))

        # 충돌 여부를 확인하고 보상을 준다.
        self.reward = 0
        if self.ball_y >= self.GAME_FLOOR - self.BALL_WIDTH // 2:
            if ball.colliderect(paddle):
                self.reward = 1
            else:
                self.reward = -1
                
            self.game_score += self.reward
            self.ball_x = random.randint(0, self.GAME_WIDTH)
            self.ball_y = self.GAME_CEILING
            self.num_tries += 1
            
        pygame.display.flip()

        # 마지막 네 프레임을 저장한다.
        self.frames.append(pygame.surfarray.array2d(self.screen))
        
        if self.num_tries >= self.MAX_TRIES_PER_GAME:
            self.game_over = True
            
        self.clock.tick(30)
        return self.get_frames(), self.reward, self.game_over
        

    def get_frames(self):
        return np.array(list(self.frames))
    

if __name__ == "__main__":   
    game = MyWrappedGame()

    NUM_EPOCHS = 10
    for e in range(NUM_EPOCHS):
        print("Epoch: {:d}".format(e))
        game.reset()
        input_t = game.get_frames()
        game_over = False
        while not game_over:
            action = np.random.randint(0, 3, size=1)[0]
            input_tp1, reward, game_over = game.step(action)
            print(action, reward, game_over)
        
    