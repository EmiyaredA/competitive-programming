# ==================================================
# 🧩 Python 通用算法竞赛模板
# 适用场景：蓝桥杯 / CCPC / ACM / 学校算法竞赛
# Author: moritaOliver
# ==================================================

import sys
import math
from collections import defaultdict, Counter, deque
from itertools import accumulate, combinations, permutations
from heapq import heappush, heappop, heapify
import bisect

# -----------------------------
# ⚙️ 快速输入输出（推荐）
# -----------------------------
input = sys.stdin.readline
def inp(): return int(input().strip()) # 快速读取一个整数输入。
def inlt(): return list(map(int, input().strip().split())) # 读取一行多个整数，返回为列表。
def insr(): return list(input().strip()) # 读取一行字符串，返回字符列表。
def inlsts(n): return [inlt() for _ in range(n)] # 连续读入 n 行整数列表。

# -----------------------------
# 🧮 常用工具函数
# -----------------------------
def ceil_div(a, b): return (a + b - 1) // b
def gcd(a, b): return math.gcd(a, b) # 最大公约数
def lcm(a, b): return a * b // math.gcd(a, b) # 最小公倍数
def prefix_sum(arr): return list(accumulate(arr, initial=0)) # 前缀和数组
def binary_search(arr, x): # 二分查找（返回第一个等于 x 的索引）
    i = bisect.bisect_left(arr, x)
    return i if i < len(arr) and arr[i] == x else -1

# -----------------------------
# 🔁 基础算法模板区
# -----------------------------

# --- BFS 模板 ---
def bfs(start, graph):
    q = deque([start])
    visited = {start}
    while q:
        u = q.popleft()
        for v in graph[u]:
            if v not in visited:
                visited.add(v)
                q.append(v)
    return visited

# --- DFS 模板 ---
def dfs(u, graph, visited):
    visited.add(u)
    for v in graph[u]:
        if v not in visited:
            dfs(v, graph, visited)

# --- 二分查找（判定模板）---
def binary_search_check(lo, hi, check):
    while lo < hi:
        mid = (lo + hi) // 2
        if check(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

# --- 并查集模板 ---
class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
        self.rank = [0]*n

    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        if self.rank[xr] < self.rank[yr]:
            self.p[xr] = yr
        elif self.rank[xr] > self.rank[yr]:
            self.p[yr] = xr
        else:
            self.p[yr] = xr
            self.rank[xr] += 1
        return True

# --- Dijkstra 最短路径 ---
def dijkstra(n, graph, start):
    dist = [float('inf')] * n
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[v] > d + w:
                dist[v] = d + w
                heappush(pq, (dist[v], v))
    return dist

# --- 0/1 背包模板 ---
def knapsack(n, w, weights, values):
    dp = [0] * (w + 1)
    for i in range(n):
        for j in range(w, weights[i] - 1, -1):
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
    return dp[w]

# --- 最长递增子序列 LIS ---
def LIS(nums):
    dp = []
    for x in nums:
        i = bisect.bisect_left(dp, x)
        if i == len(dp): dp.append(x)
        else: dp[i] = x
    return len(dp)

# -----------------------------
# 🧠 核心解题函数
# -----------------------------
def solve():
    # 示例：读取输入
    # n, m = inlt()
    # arr = inlt()
    # g = defaultdict(list)
    # for _ in range(m):
    #     u, v = inlt()
    #     g[u].append(v)
    #
    # 逻辑示例：
    # print(LIS(arr))
    #
    # 你在这里写每个题的核心逻辑 👇
    pass

# -----------------------------
# 🚀 主入口
# -----------------------------
if __name__ == "__main__":
    # 单组样例：
    solve()

    # 多组样例（若题目说明有多组）：
    # t = inp()
    # for _ in range(t):
    #     solve()
