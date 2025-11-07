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
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        num_items = dict()
        for index, num in enumerate(nums):
            target_num = target - num
            if str(target_num) in num_items:
                return [num_items[str(target_num)], index]
            num_items[str(num)] = index
    

if __name__ == "__main__":
    # solve()
    solve = Solution()
    res = solve.twoSum([2,7,11,15], 9)
    print(res)
    
