import sys
import math
from collections import defaultdict, Counter, deque
from itertools import accumulate, combinations, permutations
from heapq import heappush, heappop, heapify
from typing import List
import bisect

input = sys.stdin.readline
def inp(): return int(input().strip()) # 快速读取一个整数输入。
def inlt(): return list(map(int, input().strip().split())) # 读取一行多个整数，返回为列表。
def insr(): return list(input().strip()) # 读取一行字符串，返回字符列表。
def inlsts(n): return [inlt() for _ in range(n)] # 连续读入 n 行整数列表。

def ceil_div(a, b): return (a + b - 1) // b
def gcd(a, b): return math.gcd(a, b) # 最大公约数
def lcm(a, b): return a * b // math.gcd(a, b) # 最小公倍数
def prefix_sum(arr): return list(accumulate(arr, initial=0)) # 前缀和数组
def binary_search(arr, x): # 二分查找（返回第一个等于 x 的索引）
    i = bisect.bisect_left(arr, x)
    return i if i < len(arr) and arr[i] == x else -1

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

def dfs(u, graph, visited):
    visited.add(u)
    for v in graph[u]:
        if v not in visited:
            dfs(v, graph, visited)

# 二分查找
def binary_search_check(lo, hi, check):
    while lo < hi:
        mid = (lo + hi) // 2
        if check(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

# 并查集
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

# 从一个源点到图中其他所有节点的最短路径
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

# 0-1 背包问题
def knapsack(n, w, weights, values):
    dp = [0] * (w + 1)
    for i in range(n):
        for j in range(w, weights[i] - 1, -1):
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
    return dp[w]

def LIS(nums):
    dp = []
    for x in nums:
        i = bisect.bisect_left(dp, x)
        if i == len(dp): dp.append(x)
        else: dp[i] = x
    return len(dp)

class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        items = set(nums)
        st_sqeuence = {} # key是序列开始值，value是对应结束值
        end_sequence = {} # key是序列的结束值，value是对应的起始值
        
        max_len = 0
        for x in items:
            
            if str(x-1) in end_sequence:
                end_sequence[str(x)] = end_sequence[str(x-1)]
                del end_sequence[str(x-1)]
                st_sqeuence[end_sequence[str(x)]] = str(x)
            else:
                end_sequence[str(x)] = str(x)
            max_len = max(max_len, x - int(end_sequence[str(x)]) + 1)
                
            if str(x+1) in st_sqeuence:
                st_sqeuence[str(x)] = st_sqeuence[str(x+1)]
                del st_sqeuence[str(x+1)]
                end_sequence[st_sqeuence[str(x)]] = str(x)
            else:
                st_sqeuence[str(x)] = str(x)
            max_len = max(max_len, int(st_sqeuence[str(x)]) - x + 1)
            
            if str(x) in st_sqeuence and str(x) in end_sequence:
                st_index = end_sequence[str(x)]
                end_index = st_sqeuence[str(x)]
                del end_sequence[str(x)]
                del st_sqeuence[str(x)]
                st_sqeuence[st_index] = end_index
                end_sequence[end_index] = st_index
                max_len = max(max_len, int(end_index)-int(st_index)+1)
        
        return max_len
        

if __name__ == "__main__":
    # solve()
    solve = Solution()
    res = solve.longestConsecutive([4,0,-4,-2,2,5,2,0,-8,-8,-8,-8,-1,7,4,5,5,-4,6,6,-3])
    print(res)
    
