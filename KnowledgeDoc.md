好的，以下是常见数据结构（如队列、栈、字典等）和它们常用的 API 方法的文档。这个文档主要涵盖了 Python 中 `deque`、`list`、`set`、`dict` 和 `queue.Queue` 等常见数据结构的常用方法。

---

## 常见数据结构 API 文档

### 1. **队列：`deque`（双端队列）**

`deque` 是 Python 中 `collections` 模块中的一种数据结构，支持从两端高效的插入和删除操作，适用于队列和栈操作。

**导入**：

```python
from collections import deque
```

**常用 API**：

* `deque([iterable])`：创建一个双端队列。可以用迭代器初始化。
* `q.append(x)`：将元素 `x` 加入队列的右端（尾部）。
* `q.appendleft(x)`：将元素 `x` 加入队列的左端（头部）。
* `q.pop()`：移除并返回队列右端的元素（尾部）。
* `q.popleft()`：移除并返回队列左端的元素（头部）。
* `q.extend(iterable)`：将一个迭代器的元素添加到队列的右端。
* `q.extendleft(iterable)`：将一个迭代器的元素添加到队列的左端。
* `q.remove(x)`：从队列中移除第一个值为 `x` 的元素。
* `q.rotate(n)`：将队列中的元素循环右移 `n` 步（如果 `n` 为负数则为左移）。

**示例**：

```python
q = deque([1, 2, 3])
q.append(4)         # [1, 2, 3, 4]
q.appendleft(0)     # [0, 1, 2, 3, 4]
q.popleft()         # 0, 队列变为 [1, 2, 3, 4]
```

---

### 2. **栈：`list`（列表）**

Python 中的 `list` 可以作为栈使用，栈是先进后出（LIFO）的数据结构。

**常用 API**：

* `list.append(x)`：将元素 `x` 推入栈顶（相当于入栈）。
* `list.pop()`：移除并返回栈顶元素（相当于出栈）。
* `list[-1]`：访问栈顶元素（不移除）。

**示例**：

```python
stack = []
stack.append(1)     # [1]
stack.append(2)     # [1, 2]
stack.pop()         # 返回 2, 栈变为 [1]
```

---

### 3. **集合：`set`（集合）**

`set` 是一个无序、不重复的数据集，适用于去重和快速查找。

**常用 API**：

* `set.add(x)`：将元素 `x` 加入集合。
* `set.remove(x)`：从集合中移除元素 `x`，若元素不存在，则抛出 KeyError。
* `set.discard(x)`：从集合中移除元素 `x`，如果元素不存在，不会抛出异常。
* `set.pop()`：移除并返回集合中的任意一个元素（无序）。
* `set.clear()`：移除集合中的所有元素。
* `set.union(other_set)` 或 `set | other_set`：返回集合的并集。
* `set.intersection(other_set)` 或 `set & other_set`：返回集合的交集。
* `set.difference(other_set)` 或 `set - other_set`：返回集合的差集。
* `set.issubset(other_set)`：判断集合是否是另一个集合的子集。
* `set.issuperset(other_set)`：判断集合是否是另一个集合的超集。

**示例**：

```python
s = {1, 2, 3}
s.add(4)            # {1, 2, 3, 4}
s.remove(2)         # {1, 3, 4}
s.discard(5)        # {1, 3, 4}, 不抛异常
```

---

### 4. **字典：`dict`（字典）**

`dict` 是 Python 中的哈希表，支持快速的键值对查找和更新。

**常用 API**：

* `dict[key] = value`：设置键 `key` 的值为 `value`，如果键已存在则覆盖。
* `dict.get(key)`：获取键 `key` 对应的值，如果键不存在返回 `None`（或自定义默认值）。
* `dict.pop(key)`：移除并返回键 `key` 对应的值，如果键不存在则抛出 KeyError。
* `dict.popitem()`：移除并返回字典中的任意键值对（无序）。
* `dict.keys()`：返回字典中所有的键。
* `dict.values()`：返回字典中所有的值。
* `dict.items()`：返回字典中所有的键值对。
* `dict.update(other_dict)`：将 `other_dict` 中的键值对更新到当前字典中。
* `dict.setdefault(key, default)`：如果键 `key` 不在字典中，则插入该键并设置默认值 `default`，否则返回该键的值。

**示例**：

```python
d = {'a': 1, 'b': 2}
d['c'] = 3            # {'a': 1, 'b': 2, 'c': 3}
d.pop('b')            # 返回 2, d 变为 {'a': 1, 'c': 3}
d.get('a')            # 返回 1
d.update({'d': 4})    # {'a': 1, 'c': 3, 'd': 4}
```

---

### 5. **队列（线程安全）：`queue.Queue`**

`queue.Queue` 是 Python 的一个线程安全的队列，常用于多线程编程中。

**导入**：

```python
from queue import Queue
```

**常用 API**：

* `queue.put(item)`：将元素 `item` 放入队列。
* `queue.get()`：从队列中获取并移除一个元素。
* `queue.qsize()`：返回队列中元素的数量。
* `queue.empty()`：如果队列为空，则返回 `True`。
* `queue.full()`：如果队列已满，则返回 `True`。

**示例**：

```python
q = Queue()
q.put(1)              # 队列变为 [1]
q.put(2)              # 队列变为 [1, 2]
q.get()               # 返回 1, 队列变为 [2]
```

---

## 总结

这些数据结构和 API 提供了在算法和程序设计中的基本构建块。它们的特点和适用场景可以帮助你更好地选择合适的数据结构来提高效率。在具体算法实现中，这些数据结构常常是用来简化逻辑、加速查找或保证线程安全的关键工具。
