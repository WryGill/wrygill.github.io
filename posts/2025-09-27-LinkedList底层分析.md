---
title: LinkedList底层分析
date: 2025-09-27 14:45:22
tags: ["集合", "底层"]
categories: [JavaSE]
---

# LinkedList底层分析
`LinkedList`是Java集合框架中基于**双向链表** 实现的List接口，它同时实现了`List`和`Deque`接口，既可以作为列表使用，也可以作为双端队列使用。与ArrayList的数组实现不同，LinkedList通过节点间的指针连接实现动态存储。
## 一、底层数据结构
### 双向链表结构
从JDK 1.7开始，LinkedList采用**双向链表** 实现（早期版本使用循环链表），每个节点都包含指向前驱和后继节点的引用。
![image-20250927224800473](https://youke1.picui.cn/s1/2025/10/10/68e8ffe042359.png)
### Node节点定义
    
    private static class Node<E> {
        E item;           // 存储的数据元素
        Node<E> next;     // 指向下一个节点的引用
        Node<E> prev;     // 指向上一个节点的引用
    
        Node(Node<E> prev, E element, Node<E> next) {
            this.item = element;
            this.next = next;
            this.prev = prev;
        }
    }
**设计特点** ：
  * **泛型支持** ：使用泛型`<E>`确保类型安全
  * **双向指针** ：每个节点都维护前驱和后继引用，支持双向遍历
  * **内存布局** ：每个节点除了存储数据外，还需要额外的两个引用空间


## 二、核心方法实现
### add方法 - 尾部插入
LinkedList的add方法默认在链表尾部插入元素，这是最高效的插入方式。
    
    public boolean add(E e) {
        linkLast(e);
        return true;
    }
    
    /**
     * Links e as last element.
     */
    void linkLast(E e) {
        final Node<E> l = last;                    // 保存当前尾节点
        final Node<E> newNode = new Node<>(l, e, null);  // 创建新节点
        last = newNode;                            // 更新尾节点引用
        if (l == null)                             // 如果原链表为空
            first = newNode;                       // 新节点也是头节点
        else
            l.next = newNode;                      // 原尾节点的next指向新节点
        size++;                                    // 更新元素个数
        modCount++;                                // 更新修改次数（用于fail-fast）
    }
**实现要点** ：
  * **时间复杂度** ：O(1)，只需要修改指针引用
  * **边界处理** ：正确处理空链表的情况
  * **状态维护** ：同时更新size和modCount


### addFirst方法 - 头部插入
    
    public void addFirst(E e) {
        linkFirst(e);
    }
    
    private void linkFirst(E e) {
        final Node<E> f = first;
        final Node<E> newNode = new Node<>(null, e, f);
        first = newNode;
        if (f == null)
            last = newNode;
        else
            f.prev = newNode;
        size++;
        modCount++;
    }
### add(int index, E element) - 指定位置插入
    
    public void add(int index, E element) {
        checkPositionIndex(index);
        
        if (index == size)
            linkLast(element);
        else
            linkBefore(element, node(index));
    }
    
    void linkBefore(E e, Node<E> succ) {
        final Node<E> pred = succ.prev;
        final Node<E> newNode = new Node<>(pred, e, succ);
        succ.prev = newNode;
        if (pred == null)
            first = newNode;
        else
            pred.next = newNode;
        size++;
        modCount++;
    }
### get方法 - 随机访问
LinkedList的get方法体现了双向链表的查找优化策略。
    
    public E get(int index) {
        checkElementIndex(index);
        return node(index).item;
    }
    
    Node<E> node(int index) {
        // assert isElementIndex(index);
    
        if (index < (size >> 1)) {        // 如果索引在前半段
            Node<E> x = first;
            for (int i = 0; i < index; i++)
                x = x.next;
            return x;
        } else {                          // 如果索引在后半段
            Node<E> x = last;
            for (int i = size - 1; i > index; i--)
                x = x.prev;
            return x;
        }
    }
**优化策略** ：
  * **双向查找** ：根据索引位置选择从头或从尾开始遍历
  * **时间复杂度** ：平均O(n/2)，最坏O(n)
  * **性能对比** ：比ArrayList的O(1)随机访问慢，但比单向链表的O(n)快


### remove方法 - 元素删除
    
    public E remove(int index) {
        checkElementIndex(index);
        return unlink(node(index));
    }
    
    E unlink(Node<E> x) {
        final E element = x.item;
        final Node<E> next = x.next;
        final Node<E> prev = x.prev;
    
        if (prev == null) {
            first = next;
        } else {
            prev.next = next;
            x.prev = null;
        }
    
        if (next == null) {
            last = prev;
        } else {
            next.prev = prev;
            x.next = null;
        }
    
        x.item = null;
        size--;
        modCount++;
        return element;
    }
**删除要点** ：
  * **时间复杂度** ：O(1)（找到节点后），但查找节点需要O(n)
  * **内存清理** ：将节点的引用置为null，帮助GC回收
  * **边界处理** ：正确处理头节点和尾节点的删除


## 三、性能分析与对比
### 时间复杂度对比
操作 | ArrayList | LinkedList | 说明  
---|---|---|---  
随机访问 | O(1) | O(n) | ArrayList通过索引直接访问  
头部插入 | O(n) | O(1) | LinkedList只需修改指针  
尾部插入 | O(1) | O(1) | 两者都很快  
中间插入 | O(n) | O(n) | LinkedList查找+插入，但插入更快  
删除元素 | O(n) | O(n) | LinkedList查找+删除，但删除更快  
### 内存使用对比
方面 | ArrayList | LinkedList  
---|---|---  
内存布局 | 连续内存块 | 分散的节点  
额外开销 | 无 | 每个节点2个引用  
内存利用率 | 高 | 相对较低  
缓存友好性 | 好 | 差  
### 适用场景分析
**推荐使用LinkedList的场景** ：
  * **频繁的头部/尾部操作** ：作为栈或队列使用
  * **频繁的插入删除** ：中间位置的操作较多
  * **内存充足** ：对内存使用不敏感
  * **需要双端队列功能** ：实现Deque接口


**推荐使用ArrayList的场景** ：
  * **频繁的随机访问** ：通过索引访问元素
  * **内存敏感** ：需要节省内存空间
  * **缓存友好** ：需要良好的局部性
  * **简单列表操作** ：主要是尾部添加和遍历


## 四、常见面试题
### 1\. ArrayList和LinkedList有什么区别？
**数据结构** ：
  * ArrayList基于动态数组，内存连续
  * LinkedList基于双向链表，内存分散


**性能特点** ：
  * ArrayList随机访问O(1)，插入删除O(n)
  * LinkedList随机访问O(n)，插入删除O(1)（找到位置后）


**内存使用** ：
  * ArrayList内存利用率高，缓存友好
  * LinkedList每个元素需要额外存储两个引用


**使用建议** ：
  * 频繁查询用ArrayList
  * 频繁插入删除用LinkedList


### 2\. 为什么LinkedList实现了Deque接口？
LinkedList的双向链表结构天然支持双端操作：
  * 可以在头部和尾部高效地进行插入、删除操作
  * 支持栈（LIFO）和队列（FIFO）的操作模式
  * 提供了丰富的双端队列方法：addFirst、addLast、removeFirst、removeLast等