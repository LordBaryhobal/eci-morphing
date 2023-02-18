#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pygame
from math import atan2, sin, cos, sqrt, pi, degrees
import numpy as np

WIDTH, HEIGHT = 1200, 400

class Vec:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.m = self.mag()
    
    def __add__(self, v):
        return Vec(self.x+v.x, self.y+v.y)
    
    def __radd__(self, v):
        return Vec(self.x+v.x, self.y+v.y)
    
    def __sub__(self, v):
        return Vec(self.x-v.x, self.y-v.y)
    
    def __rsub__(self, v):
        return Vec(v.x-self.x, v.y-self.y)
    
    def __mul__(self, o):
        if isinstance(o, Vec):
            return self.x*o.x + self.y*o.y
        
        return Vec(self.x*o, self.y*o)
    
    def __rmul__(self, o):
        if isinstance(o, Vec):
            return self.x*o.x + self.y*o.y
        
        return Vec(self.x*o, self.y*o)
    
    def __truediv__(self, n):
        return Vec(self.x/n, self.y/n)
    
    def mag(self):
        return sqrt(self.x**2 + self.y**2)
    
    def norm(self):
        mag = self.mag()
        if mag == 0: return Vec()
        return self/mag
    
    def __repr__(self):
        return f"Vec({self.x}, {self.y})"
    
    def angle(self):
        return atan2(self.y, self.x)
    
    def __getitem__(self, item):
        return [self.x, self.y][item]
    
    def __len__(self):
        return 2
    
    def to_pol(self):
        return PolVec(self.angle(), self.mag())

class Poly:
    DRAW_NORMALS = False
    DRAW_CENTER = False
    
    def __init__(self, pts=None, normals=None, col=(255,255,255)):
        if pts is None: pts = []
        if normals is None: normals = []
        self.pts = pts
        self.normals = normals
        self.col = col
        self.center = Vec()
    
    def calc_center(self):
        xs, ys = list(zip(*self.pts))
        x, y = (max(xs)+min(xs))/2, (max(ys)+min(ys))/2
        self.center = Vec(x, y)
    
    def calc_normals(self):
        self.normals = []
        for i in range(len(self.pts)):
            p1, p2 = self.pts[i-1], self.pts[i]
            d = p2-p1
            n = Vec(-d.y, d.x)
            
            self.normals.append(n)
    
    def render(self, surf):
        if len(self.pts) > 2:
            pygame.draw.polygon(surf, self.col, self.pts, 1)
            
            if self.DRAW_NORMALS:
                for i in range(len(self.pts)):
                    p = (self.pts[i]+self.pts[i-1])/2
                    n = self.normals[i]/10
                    pygame.draw.line(surf, (255,255,255), p, p+n)
        
        if self.DRAW_CENTER:
            pygame.draw.circle(surf, self.col, self.center, 5)
    
    def translate(self, v):
        for pt in self.pts:
            pt.x += v.x
            pt.y += v.y

class PolVec:
    def __init__(self, angle=0, mag=0):
        self.angle = angle
        self.mag = mag
    
    def to_cart(self):
        return Vec(self.mag*cos(self.angle), self.mag*sin(self.angle))

class ECI:
    def __init__(self, poly):
        self.poly = poly
    
    def render(self, surf, center, r):
        weights = [n.mag() for n in self.poly.normals]
        max_w = max(weights)
        
        for n in self.poly.normals:
            p = center + n.norm()*r
            pygame.draw.line(surf, self.poly.col, p, p+n/10)
            
            
            a = n.angle()
            x = WIDTH/4 * (a%(2*pi))/(2*pi)
            x = 17*WIDTH/24 + x
            y1 = HEIGHT-HEIGHT/24
            y2 = y1-n.mag()/max_w * (HEIGHT-HEIGHT/12)
            pygame.draw.line(surf, self.poly.col, [x,y1], [x,y2])

class Pair:
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2
        
        self.x0, self.y0 = v1.x, v1.y
        self.x, self.y = (v2.x-v1.x)/100, (v2.y-v1.y)/100
    
    def lerp(self, pct):
        return Vec(self.y0 + pct*self.y, -(self.x0 + pct*self.x))

class Morph:
    ECI_CENTER = Vec(WIDTH/2, HEIGHT/2)
    ECI_RADIUS = HEIGHT/6
    LAMBDA = 20
    SIGMA = 0.2
    
    def __init__(self, poly1, poly2):
        self.poly1 = poly1
        self.poly2 = poly2
        self.poly3 = Poly(col=(255,255,0))
        
        self.poly1.calc_center()
        self.poly2.calc_center()
        
        self.eci1 = ECI(self.poly1)
        self.eci2 = ECI(self.poly2)
        self.eci3 = ECI(self.poly3)
        
        self.pct = 50
        self.calc_data()
        self.calc_morph()
    
    def render(self, surf):
        self.poly1.render(surf)
        self.poly2.render(surf)
        self.poly3.render(surf)
        
        pygame.draw.circle(surf, (255,255,255), self.ECI_CENTER, self.ECI_RADIUS, 1)
        self.eci1.render(surf, self.ECI_CENTER, self.ECI_RADIUS)
        self.eci2.render(surf, self.ECI_CENTER, self.ECI_RADIUS)
        self.eci3.render(surf, self.ECI_CENTER, self.ECI_RADIUS)
        
        #pygame.draw.circle(surf, (100,100,100), (self.poly2.center-self.poly1.center)*self.pct/100 + self.poly1.center, 3)
    
    def angle_diff(self, a1, a2):
        if a1 > a2:
            return min(a1-a2, 2*pi+a2-a1)
        
        return min(a2-a1, 2*pi+a1-a2)
    
    def calc_data(self):
        norms1, norms2 = [(n.mag(), n.angle()) for n in self.poly1.normals], [(n.mag(), n.angle()) for n in self.poly2.normals]
        weights1, angles1 = list(zip(*norms1))
        weights2, angles2 = list(zip(*norms2))
        
        I, J = len(self.poly1.pts), len(self.poly2.pts)
        
        max_w_diff = max(max(weights1)-min(weights2), max(weights2)-min(weights1))
        
        weights = np.zeros([I, J])
        source_sum = np.zeros([I])
        target_sum = np.zeros([J])
        
        self.normals = []
        
        for i in range(I):
            for j in range(J):
                angle_diff = self.angle_diff(angles1[i], angles2[j]) / pi
                weight_diff = abs(weights2[j]-weights1[i])/max_w_diff
                weights[i, j] = self.SIGMA * (weight_diff + self.LAMBDA*angle_diff)
        
        weights = np.exp(-weights)
        
        for i in range(I):
            source_sum[i] = 1/weights[i, :].sum()
        
        for j in range(J):
            target_sum[j] = 1/weights[:, j].sum()
        
        for i in range(I):
            for j in range(J):
                self.normals.append(Pair(
                    self.poly1.normals[i] * (weights[i, j] * source_sum[i]),
                    self.poly2.normals[j] * (weights[i, j] * target_sum[j])
                ))
    
    def calc_morph(self):
        srted = []
        for n in self.normals:
            srted.append(n.lerp(self.pct))
        
        self.sort(srted)
        poly = self.poly3
        poly.pts = []
        poly.normals = []
        
        poly.pts.append(Vec(0,0))
        
        poly.normals.append(srted[0])
        
        poly.pts.append(srted[0]*-1)
        
        for n in srted[1:-1]:
            poly.normals.append(Vec(-n.y, n.x))
            p = poly.pts[-1]
            poly.pts.append(p - n)
        
        p1, p2 = poly.pts[-1], poly.pts[0]
        d = p2-p1
        poly.normals.append(Vec(-d.y, d.x))
        poly.calc_center()
        poly.translate((self.poly2.center-self.poly1.center)*self.pct/100 - poly.center + self.poly1.center)
        poly.calc_center()
    
    def sort(self, lst):
        changed = True
        while changed:
            changed = False
            for i in range(len(lst)-1):
                a1 = lst[i].angle()
                a2 = lst[i+1].angle()
                if a1 > a2:
                    lst[i], lst[i+1] = lst[i+1], lst[i]
                    changed = True

# http://web.mit.edu/manoli/ecimorph/www/ecimorph.html
if __name__ == "__main__":
    pygame.init()
    
    w = pygame.display.set_mode([WIDTH, HEIGHT])
    
    clock = pygame.time.Clock()
    
    
    poly1 = Poly(pts=[
        Vec(152, 159),
        Vec(90, 211),
        Vec(365, 329),
        Vec(344, 143)
    ], col=(255,0,0))
    poly2 = Poly(pts=[
        Vec(349, 191),
        Vec(54, 66),
        Vec(101, 353),
        Vec(201, 370)
    ], col=(0,255,0))
    
    """poly1 = Poly(pts=[
        Vec(40, 30),
        Vec(30, 60),
        Vec(60, 90),
        Vec(100, 110),
        Vec(130, 50)
    ], col=(255,0,0))
    poly2 = Poly(pts=[
        Vec(30, 30),
        Vec(20, 110),
        Vec(140, 110),
        Vec(130, 30),
        Vec(70, 20)
    ], col=(0,255,0))"""
    
    poly1.calc_normals()
    poly2.calc_normals()
    
    morph = Morph(poly1, poly2)
    running = True
    playing = False
    reverse = False
    
    while running:
        pygame.display.set_caption(f"Morhping - {clock.get_fps():.2f}fps - {morph.pct}%")
        events = pygame.event.get()
        
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    morph.pct = min(100, morph.pct+5)
                    morph.calc_morph()
                
                elif event.button == 5:
                    morph.pct = max(0, morph.pct-5)
                    morph.calc_morph()
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                
                elif event.key == pygame.K_SPACE:
                    playing = not playing
            
            elif event.type == pygame.QUIT:
                running = False
        
        if playing:
            if reverse:
                morph.pct -= 1
                if morph.pct == 0:
                    reverse = False
            
            else:
                morph.pct += 1
                if morph.pct == 100:
                    reverse = True
            
            morph.calc_morph()
        
        w.fill(0)
        pygame.draw.line(w, (150,150,150), [WIDTH/3, 0], [WIDTH/3, HEIGHT])
        pygame.draw.line(w, (150,150,150), [2*WIDTH/3, 0], [2*WIDTH/3, HEIGHT])
        
        morph.render(w)
        
        pygame.display.flip()
        
        clock.tick(60)
